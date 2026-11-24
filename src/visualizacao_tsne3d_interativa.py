# -*- coding: utf-8 -*-
"""TSNE13 3D: PNG para o TCC e HTML Plotly totalmente offline.

Modulo grafico sem qualquer fit, acesso ao parquet, leitura de F10 ou ajuste
no modelo. Recebe somente o embedding e as predicoes JA calculadas.
A finalizacao oficial t-SNE 3D permanece bloqueada ate a fase de integracao.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

NOME = '13_tsne_mapa_TSNE1_TSNE2_TSNE3'
PNG = NOME + '.png'
HTML = NOME + '_INTERATIVO.html'
CSV = NOME + '_amostra_visualizacao.csv'
MANIFESTO = NOME + '_manifesto.json'
COLUNAS = ('TSNE1', 'TSNE2', 'TSNE3')


def _exigir(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise ValueError(mensagem)


def _vetor(x, n: int, nome: str, dtype=float) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype).reshape(-1)
    _exigir(len(arr) == n and np.isfinite(arr).all(), f'{nome}: tamanho ou valores invalidos.')
    return arr


def _dados_validos(coordenadas, classe_real, *, scores=None, limiar_oof=None,
                   responsabilidades=None, linha_original=None, origem='desenvolvimento',
                   fracao_f10=None):
    _exigir(isinstance(coordenadas, pd.DataFrame), 'Coordenadas devem ser DataFrame.')
    _exigir(all(c in coordenadas.columns for c in COLUNAS), 'Faltam TSNE1, TSNE2, TSNE3.')
    z = coordenadas.loc[:, COLUNAS].to_numpy(dtype=float)
    n = len(z)
    _exigir(z.shape == (n, 3) and n >= 4 and np.isfinite(z).all(),
            'O embedding 3D esta vazio ou contem valores invalidos.')
    y_validar = _vetor(classe_real, n, 'classe_real', float)
    _exigir(np.isin(y_validar, [0, 1]).all(), 'Classe real deve ser 0 ou 1.')
    y = y_validar.astype(np.uint8)
    _exigir(origem in ('desenvolvimento', 'f10', 'simulado'), 'Origem desconhecida.')
    if origem == 'f10':
        _exigir(fracao_f10 is not None and np.isfinite(float(fracao_f10))
                and 0 < float(fracao_f10) <= 1, 'F10 exige fracao registrada no YAML.')
    else:
        _exigir(fracao_f10 is None, 'Fracao F10 somente na visualizacao F10.')
    p = None
    if scores is not None:
        p = _vetor(scores, n, 'scores')
        _exigir(((0 <= p) & (p <= 1)).all(), 'Scores fora de [0,1].')
    tau = None
    if limiar_oof is not None:
        tau = float(limiar_oof)
        _exigir(p is not None and np.isfinite(tau) and 0 <= tau <= 1,
                'Limiar MCC OOF requer scores validos.')
    gamma = None
    if responsabilidades is not None:
        gamma = np.asarray(responsabilidades, dtype=float)
        _exigir(gamma.ndim == 2 and gamma.shape[0] == n and gamma.shape[1] >= 1
                and np.isfinite(gamma).all() and (gamma >= 0).all()
                and np.allclose(gamma.sum(axis=1), 1, atol=1e-5, rtol=1e-5),
                'Responsabilidades exigem matriz (n,K), K>=1, nao negativa, somando 1.')
    if linha_original is None and 'linha_original' in coordenadas.columns:
        linha_original = coordenadas['linha_original']
    if linha_original is None:
        linhas = np.arange(n, dtype=np.int64)
        tipo_id = 'indice_local_visualizacao'
    else:
        ids_verificar = _vetor(linha_original, n, 'linha_original', float)
        _exigir(np.all(ids_verificar == np.floor(ids_verificar)) and
                (ids_verificar <= np.iinfo(np.int64).max).all(),
                'Identificadores de linha devem ser inteiros validos.')
        linhas = ids_verificar.astype(np.int64)
        _exigir((linhas >= 0).all() and len(np.unique(linhas)) == n,
                'Identificadores de linha devem ser nao negativos e unicos.')
        tipo_id = 'linha_original'
    return z, y, p, tau, gamma, linhas, tipo_id


def _amostra_estratificada(y: np.ndarray, *, max_pontos: int, semente: int) -> np.ndarray:
    """Amostra exclusiva para desenhar; nao interfere em predicoes/metrica.

    Preserva todas as fraudes quando couberem; nao chama nenhum fit ou predict.
    """
    n = len(y)
    _exigir(type(max_pontos) is int and max_pontos >= 4, 'max_pontos deve ser inteiro >= 4.')
    _exigir(type(semente) is int and semente >= 0, 'Semente invalida.')
    if n <= max_pontos:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(semente)
    positivos = np.flatnonzero(y == 1)
    negativos = np.flatnonzero(y == 0)
    if not len(positivos) or not len(negativos):
        return np.sort(rng.choice(n, size=max_pontos, replace=False)).astype(int)
    n_pos = min(len(positivos), max(1, round(max_pontos * len(positivos) / n)))
    if len(positivos) <= max_pontos - 1:
        n_pos = len(positivos)  # Classes raras integralmente preservadas, se couber.
    n_pos = min(n_pos, max_pontos - 1)
    n_neg = max_pontos - n_pos
    if n_neg > len(negativos):
        n_neg = len(negativos)
        n_pos = min(max_pontos - n_neg, len(positivos))
    ip = rng.choice(positivos, size=n_pos, replace=False)
    ine = rng.choice(negativos, size=n_neg, replace=False)
    return np.sort(np.r_[ip, ine]).astype(int)


def _figura_html(z, y, p, tau, gamma, ids, *, origem, fracao_f10, tipo_id):
    fig = go.Figure()
    modos: list[tuple[str, list[int]]] = []
    estilo = {0: ('Legitimas', '#1f77b4'), 1: ('Fraudes', '#d62728')}

    def adicionar(nome, idx, *, cor, tamanho, opacidade, colorbar=False):
        ix = np.asarray(idx, dtype=int)
        if not len(ix):
            return None
        props = dict(size=tamanho, opacity=opacidade, color=cor)
        if colorbar:
            props.update(colorscale='Viridis', cmin=0, cmax=1,
                         colorbar={'title': 'Risco estimado', 'thickness': 14})
        personalizados = np.stack((ids[ix].astype(str), y[ix].astype(str),
                                   np.full(len(ix), 'n/d') if p is None
                                   else np.char.mod('%.7f', p[ix])), axis=1)
        fig.add_trace(go.Scatter3d(
            x=z[ix, 0], y=z[ix, 1], z=z[ix, 2], mode='markers', name=nome,
            marker=props, customdata=personalizados,
            hovertemplate=(f'{tipo_id}: %{{customdata[0]}}<br>'
                           'Classe real: %{customdata[1]}<br>'
                           'Risco estimado: %{customdata[2]}<br>'
                           'TSNE1: %{x:.7f}<br>TSNE2: %{y:.7f}<br>'
                           'TSNE3: %{z:.7f}<extra>%{fullData.name}</extra>'),
            visible=False,
        ))
        return len(fig.data) - 1

    indices = []
    for classe in (0, 1):
        grupo = np.flatnonzero(y == classe)
        label, cor = estilo[classe]
        indice = adicionar(f'{label} (n={len(grupo)})', grupo, cor=cor,
                           tamanho=2.1 if classe == 0 else 4.1,
                           opacidade=.40 if classe == 0 else .88)
        if indice is not None:
            indices.append(indice)
    modos.append(('Classe real', indices))

    if p is not None:
        indices = []
        indice = adicionar('Risco de fraude estimado', np.arange(len(y)), cor=p,
                           tamanho=2.5, opacidade=.72, colorbar=True)
        if indice is not None:
            indices.append(indice)
        modos.append(('Risco estimado', indices))

    if p is not None and tau is not None:
        indices = []
        pred = (p >= tau).astype(int)
        for classe in (0, 1):
            grupo = np.flatnonzero(pred == classe)
            label = 'Predicao legitima' if classe == 0 else 'Predicao fraude'
            _, cor = estilo[classe]
            indice = adicionar(f'{label} (n={len(grupo)})', grupo, cor=cor,
                               tamanho=2.1 if classe == 0 else 4.1,
                               opacidade=.45 if classe == 0 else .90)
            if indice is not None:
                indices.append(indice)
        modos.append((f'Predicao tau* OOF ({tau:.7f})', indices))

    if gamma is not None:
        palette = ['#8e44ad', '#e67e22', '#16a085', '#e74c3c']
        # Preserva as quatro cores historicas; K maiores usam HSL deterministico.
        dominante = np.argmax(gamma, axis=1)
        indices = []
        for k in range(gamma.shape[1]):
            cor = palette[k] if k < len(palette) else f'hsl({int((k*137.508)%360)}, 72%, 40%)'
            grupo = np.flatnonzero(dominante == k)
            indice = adicionar(f'G{k + 1} dominante (n={len(grupo)})', grupo,
                               cor=cor, tamanho=2.8, opacidade=.70)
            if indice is not None:
                indices.append(indice)
        modos.append(('Gaussiana dominante', indices))

    visiveis = [False] * len(fig.data)
    for i in modos[0][1]:
        visiveis[i] = True
    for i, estado in enumerate(visiveis):
        fig.data[i].visible = estado

    buttons = []
    for nome, tracos in modos:
        vis = [i in tracos for i in range(len(fig.data))]
        buttons.append(dict(label=nome, method='update',
                            args=[{'visible': vis}, {'title': 't-SNE 3D | ' + nome}]))

    fig.update_layout(
        title='t-SNE 3D | Classe real',
        template='plotly_white',
        scene=dict(xaxis_title='TSNE1', yaxis_title='TSNE2', zaxis_title='TSNE3',
                   aspectmode='data', bgcolor='rgba(245,248,252,1)'),
        legend=dict(orientation='h', y=-.05),
        height=780, margin=dict(l=0, r=0, t=95, b=75),
        updatemenus=[dict(type='dropdown', direction='down', active=0, showactive=True,
                          x=.01, y=1.115, xanchor='left', buttons=buttons)],
    )
    return fig, [m[0] for m in modos]


def _figura_png(z, y, *, origem, n_total, dpi):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    for classe, nome, cor, tamanho, alpha in ((0, 'Legitimas', '#1f77b4', 6, .28),
                                              (1, 'Fraudes', '#d62728', 18, .83)):
        ix = y == classe
        if ix.any():
            ax.scatter(z[ix, 0], z[ix, 1], z[ix, 2], s=tamanho, c=cor,
                       alpha=alpha, depthshade=False, rasterized=True,
                       label=f'{nome} (n={int(ix.sum())} exibidas)')
    ax.set(xlabel='TSNE1', ylabel='TSNE2', zlabel='TSNE3')
    ax.view_init(elev=22, azim=-59)
    ax.legend(loc='upper left', fontsize=9)
    fig.suptitle('Projecao tridimensional t-SNE1 x t-SNE2 x t-SNE3',
                 fontsize=15, y=.96)
    fig.subplots_adjust(top=.9, bottom=.055)
    return fig


def gerar_tsne13_3d(coordenadas: pd.DataFrame, classe_real, destino: str | Path, *,
                    scores=None, limiar_oof=None, responsabilidades=None,
                    linha_original=None, origem='desenvolvimento', fracao_f10=None,
                    max_pontos=5000, semente=42, dpi=180) -> dict:
    """Produz PNG, HTML Plotly inline, CSV e manifesto sem sobrescrever arquivos.

    A entrada de F10 deve ser a projecao *ja congelada*, sem qualquer refit.
    Os limites max_pontos/estratificacao afetam APENAS o desenho, jamais metricas.
    Retorna paths e nomes; orquestrador/galerias integrarao em fase posterior.
    """
    z, y, p, tau, gamma, ids, tipo_id = _dados_validos(
        coordenadas, classe_real, scores=scores, limiar_oof=limiar_oof,
        responsabilidades=responsabilidades, linha_original=linha_original,
        origem=origem, fracao_f10=fracao_f10)
    _exigir(type(dpi) is int and 72 <= dpi <= 600, 'dpi fora de 72..600.')
    idx = _amostra_estratificada(y, max_pontos=max_pontos, semente=semente)
    dest = Path(destino)
    _exigir(dest.is_dir(), 'Diretorio destino deve existir previamente.')
    nomes = (PNG, HTML, CSV, MANIFESTO)
    _exigir(not any((dest / nome).exists() for nome in nomes),
            'TSNE13 3D ja existe: nao sobrescrever resultados anteriores.')
    zz, yy, ii = z[idx], y[idx], ids[idx]
    pp = None if p is None else p[idx]
    gg = None if gamma is None else gamma[idx]
    with tempfile.TemporaryDirectory(prefix='.tsne13_3d_', dir=dest) as temporario:
        tmp = Path(temporario)
        fig, modos = _figura_html(zz, yy, pp, tau, gg, ii,
                                 origem=origem, fracao_f10=fracao_f10, tipo_id=tipo_id)
        fig.write_html(str(tmp / HTML), include_plotlyjs=True, full_html=True,
                       config={'responsive': True, 'displaylogo': False})
        figura_png = _figura_png(zz, yy, origem=origem, n_total=len(y), dpi=dpi)
        try:
            figura_png.savefig(tmp / PNG, dpi=dpi, bbox_inches='tight')
        finally:
            plt.close(figura_png)
        df = pd.DataFrame({tipo_id: ii, 'classe_real': yy,
                           'TSNE1': zz[:, 0], 'TSNE2': zz[:, 1], 'TSNE3': zz[:, 2]})
        if pp is not None:
            df['prob_fraude'] = pp
        if tau is not None:
            df['predicao_tau_mcc_oof'] = (pp >= tau).astype(int)
        if gg is not None:
            df['gaussiana_dominante'] = 1 + np.argmax(gg, axis=1)
        df.to_csv(tmp / CSV, index=False, float_format='%.17g')
        hashes = {nome: sha256((tmp / nome).read_bytes()).hexdigest()
                  for nome in (PNG, HTML, CSV)}
        _exigir((tmp / PNG).read_bytes().startswith(b'\x89PNG\r\n\x1a\n'),
                'PNG invalido.')
        txt = (tmp / HTML).read_text(encoding='utf-8')
        _exigir(len(txt) > 1000000 and 'plotly.js' in txt.lower()
                and 'scatter3d' in txt.lower() and 'TSNE3' in txt
                and not re.search(r'<script[^>]+\bsrc\s*=', txt, flags=re.IGNORECASE),
                'HTML Plotly nao e independente de servidor/CDN.')
        manifesto = {
            'etapa': 'FASE_7_2_RENDERIZADOR_ISOLADO',
            'origem': origem, 'f10_exploratorio': origem == 'f10',
            'fracao_f10_dentro_do_fold': fracao_f10,
            'n_total_entrada': int(len(y)), 'n_exibidos': int(len(idx)),
            'n_fraudes_entrada': int(y.sum()), 'n_fraudes_exibidas': int(yy.sum()),
            'semente_amostragem_visual': int(semente),
            'gmm_n_components': None if gamma is None else int(gamma.shape[1]),
            'limiar_oof_congelado': tau,
            'modos_disponiveis': modos,
            'sha256_arquivos': hashes,
            'modelo_retreinado': False, 'metrica_recalculada': False,
        }
        (tmp / MANIFESTO).write_text(json.dumps(manifesto, indent=2, ensure_ascii=False,
                                               allow_nan=False) + '\n', encoding='utf-8')
        # Publicacao apos validar TODOS os arquivos. Nenhuma entrada e modificada.
        for nome in nomes:
            _exigir(not (dest / nome).exists(), 'Conflito ao publicar ' + nome)
        for nome in nomes:
            (tmp / nome).replace(dest / nome)
    return {nome: dest / nome for nome in nomes}


def _demo_sintetica(saida: Path) -> dict:
    """Previa visual SEM dados reais; nunca executa pipeline/t-SNE/GMM."""
    saida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 850
    y = np.zeros(n, dtype=np.uint8)
    y[:36] = 1
    rng.shuffle(y)
    z = rng.normal(0, 1, (n, 3))
    z[y == 1] += np.array([1.0, 0.8, -0.65])
    coord = pd.DataFrame(z, columns=COLUNAS)
    coord.insert(0, 'linha_original', np.arange(n) + 50000)
    p = np.clip(.018 + .025 * rng.random(n) + .024 * y, 0, 1)
    gamma = rng.dirichlet([.8, 1.3, .9, 1.2], n)
    return gerar_tsne13_3d(
        coord, y, saida, scores=p, limiar_oof=.045,
        responsabilidades=gamma, origem='simulado', semente=42,
        max_pontos=600, dpi=150,
    )


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='TSNE13 3D — somente previa sintetica offline.')
    parser.add_argument('--demo', action='store_true', help='Gera apenas dados sinteticos.')
    parser.add_argument('--saida', type=Path, help='Diretorio de saida, sem sobrescrita.')
    args = parser.parse_args(argv)
    if not args.demo or args.saida is None:
        parser.error('Use --demo --saida CAMINHO. Nenhum treinamento sera feito.')
    arquivos = _demo_sintetica(args.saida)
    for nome, caminho in arquivos.items():
        print('[TSNE13 3D DEMO]', nome, '->', caminho.resolve(), flush=True)
    print('PREVIA SINTETICA, NAO SAIDA DO TCC REAL.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
