"""Biplot PCA 1D/2D/3D do desenvolvimento, sem novos fits ou acesso ao F10.

A sobreposicao de classe usa rotulos apenas na apresentacao. As correlacoes
usam os coeficientes do PCA ajustado no mesmo X_pre e nao usam a classe.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import visualizacao_oof as vis

PREFIXO = 'PCA_BIPLOT_'  # Nome PROVISORIO; numero oficial pendente dos catalogos.


def _entrada(data):
    """Extrai (X_pre, scores, PCA) do MESMO ajuste do desenvolvimento."""
    red = data.get('redutor')
    if red is None or not hasattr(red, 'modelo_') or not hasattr(red, 'colunas_'):
        raise ValueError('Biplot PCA exige RedutorPCA ja ajustado.')
    x = data.get('pre')
    z = data.get('coord')
    if not isinstance(x, pd.DataFrame) or not isinstance(z, pd.DataFrame):
        raise ValueError('X preprocessado e scores devem ser DataFrames.')
    if x.empty or not x.index.equals(z.index) or list(x.columns) != list(red.colunas_):
        raise ValueError('X, scores e colunas do PCA estao desalinhados.')
    modelo = red.modelo_
    d = int(modelo.n_components_)
    if d not in (1, 2, 3) or z.shape != (len(x), d):
        raise ValueError('Biplot aceita somente PCA 1D, 2D ou 3D efetivamente ajustado.')
    if list(z.columns) != [f'PC{i+1}' for i in range(d)]:
        raise ValueError('Eixos PC nao correspondem ao PCA treinado.')
    xv, zv = x.to_numpy(copy=False), z.to_numpy(copy=False)
    if not np.isfinite(xv).all() or not np.isfinite(zv).all():
        raise ValueError('Coordenadas ou variaveis nao finitas.')
    if len(x) < 2:
        raise ValueError('Biplot exige ao menos duas observacoes.')
    y = np.asarray(data.get('y'))
    if y.shape != (len(x),) or not np.isin(y, [0, 1]).all():
        raise ValueError('Rotulos descritivos desalinhados ou invalidos.')
    return x, z, modelo, y


def tabela_correlacoes(data):
    """Correlacoes Pearson X_j vs PC_i a partir do PCA ja ajustado.

    Formula vale tambem para whiten=True (escala positiva das PCs).
    Colunas constantes ou autovalores nulos recebem correlacao 0 e sao
    identificadas explicitamente por `constante`.
    """
    x, z, modelo, _ = _entrada(data)
    a = np.asarray(modelo.components_, dtype=float)
    ev = np.asarray(modelo.explained_variance_, dtype=float)
    d, p = a.shape
    if d != z.shape[1] or p != x.shape[1]:
        raise ValueError('Matriz de componentes incompatível com X e scores.')
    if not np.isfinite(a).all() or not np.isfinite(ev).all() or np.any(ev < 0):
        raise ValueError('Parametros PCA nao finitos ou variancias negativas.')
    std = np.std(x.to_numpy(copy=False), axis=0, ddof=1, dtype=np.float64)
    valido = np.isfinite(std) & (std > 1e-12)
    corr = np.zeros((p, d), dtype=float)
    np.divide((a * np.sqrt(ev)[:, None]).T, std[:, None], out=corr,
              where=valido[:, None])
    if not np.isfinite(corr).all() or np.any(np.abs(corr) > 1.0001):
        raise ValueError('Correlacoes PCA incoerentes: conferir ajuste e dados.')
    corr = np.clip(corr, -1.0, 1.0)
    colunas = [f'corr_PC{i+1}' for i in range(d)]
    tabela = pd.DataFrame(corr, columns=colunas)
    tabela.insert(0, 'variavel_preprocessada', list(x.columns))
    tabela['constante'] = ~valido
    tabela['relevancia_no_espaco'] = np.linalg.norm(corr, axis=1)
    return tabela


def _principais(tabela, limite):
    if not 1 <= limite <= 40:
        raise ValueError('max_setas deve estar entre 1 e 40.')
    return tabela.sort_values(['relevancia_no_espaco', 'variavel_preprocessada'],
                              ascending=[False, True], kind='stable').head(limite)


def _estilo(config, y, idx):
    estilos = config['visualizacao']['classes']
    for grupo, chave in ((0, 'legitima'), (1, 'fraude')):
        ids = idx[y[idx] == grupo]
        if len(ids):
            yield ids, estilos[chave]


def _scatter(ax, valores, y, idx, config, *, d):
    for ids, e in _estilo(config, y, idx):
        ax.scatter(*([valores[ids, j] for j in range(d)]),
                   color=e['cor'], alpha=e.get('alpha', .5), marker=e['marcador'],
                   s=e.get('tamanho', 16), label=e.get('nome', ''), rasterized=True)


def _setas_2d(ax, selecionadas, spans, *, labels=True):
    norm = .30 * float(min(spans))
    for i, (_, r) in enumerate(selecionadas.iterrows(), start=1):
        u, v = (float(r[f'corr_PC{k}']) for k in (1, 2))
        ax.annotate('', xy=(u * norm, v * norm), xytext=(0, 0),
                    arrowprops={'arrowstyle': '->', 'color': '#222222', 'lw': 1.4})
        if labels:
            ax.text(u * norm * 1.10, v * norm * 1.10,
                    f'V{i}', fontsize=8, ha='center', va='center')
    return norm


def _circulo(ax, selecionadas):
    t = np.linspace(0, 2 * np.pi, 240)
    ax.plot(np.cos(t), np.sin(t), '--', c='grey', lw=1)
    ax.axhline(0, color='grey', lw=.7); ax.axvline(0, color='grey', lw=.7)
    for i, (_, r) in enumerate(selecionadas.iterrows(), start=1):
        u, v = float(r['corr_PC1']), float(r['corr_PC2'])
        ax.annotate('', xy=(u, v), xytext=(0, 0),
                    arrowprops={'arrowstyle': '->', 'color': '#274d83', 'lw': 1.5})
        ax.text(u * 1.10, v * 1.10, f'V{i}', fontsize=8,
                ha='center', va='center')
    ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), xlabel='corr(X, PC1)', ylabel='corr(X, PC2)',
           title='Circulo de correlacoes (escala real)')
    ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.2)


def _fig_1d(z, y, idx, config, selecao, variancia):
    fig, (ax, av) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [1.1, 1.0]})
    rng = np.random.default_rng(int(config['experimento']['semente']))
    jitter = rng.uniform(-.11, .11, len(idx))
    for ids, e in _estilo(config, y, idx):
        mask = np.isin(idx, ids)
        ax.scatter(z[ids, 0], jitter[mask], color=e['cor'], marker=e['marcador'],
                   s=e.get('tamanho', 16), alpha=e.get('alpha', .5), label=e['nome'])
    ax.set(xlabel=f'PC1 ({variancia[0]:.2%} variancia explicada)',
           ylabel='Jitter puramente visual (sem PC2)', title='Observacoes sobre PC1')
    ax.set_yticks([]); ax.legend(loc='best'); ax.grid(axis='x', alpha=.2)
    labels = list(selecao['variavel_preprocessada'])
    vals = np.asarray(selecao['corr_PC1'], float)
    yy = np.arange(len(labels))
    av.axvline(0, color='grey', lw=1)
    for pos, valor in zip(yy, vals):
        cor = '#138276' if valor >= 0 else '#8051a8'
        av.annotate('', xy=(valor, pos), xytext=(0, pos),
                    arrowprops={'arrowstyle': '-|>', 'lw': 2.0, 'color': cor, 'mutation_scale': 12})
        av.text(valor + (.04 if valor >= 0 else -.04), pos, f'{valor:+.2f}',
                ha='left' if valor >= 0 else 'right', va='center', color=cor, fontsize=9)
    av.set(yticks=yy, yticklabels=labels, xlim=(-1.18, 1.18),
           xlabel='Correlacao Pearson de cada variavel com PC1',
           title='Direcao e intensidade das variaveis em PC1')
    av.invert_yaxis(); av.grid(axis='x', alpha=.2)
    fig.suptitle('Biplot PCA 1D adaptado — desenvolvimento F1–F9', fontweight='bold')
    fig.tight_layout(); return fig


def _fig_2d(z, y, idx, config, selecionadas, variancia):
    fig, (ax, av) = plt.subplots(1, 2, figsize=(16, 7.8))
    _scatter(ax, z, y, idx, config, d=2)
    spans = [np.ptp(z[idx, j]) for j in range(2)]
    fator = _setas_2d(ax, selecionadas, spans)
    ax.set(xlabel=f'PC1 ({variancia[0]:.2%})', ylabel=f'PC2 ({variancia[1]:.2%})',
           title=f'Scores PCA + setas ampliadas (fator visual {fator:.3g})')
    ax.axhline(0, lw=.5, color='grey'); ax.axvline(0, lw=.5, color='grey')
    ax.legend(loc='best'); ax.grid(alpha=.2)
    _circulo(av, selecionadas)
    itens = [f'V{i}={r["variavel_preprocessada"]}'
             for i, (_, r) in enumerate(selecionadas.iterrows(), start=1)]
    legenda = '\n'.join('   |   '.join(itens[j:j+3]) for j in range(0, len(itens), 3))
    fig.text(.5,.02,legenda,ha='center',fontsize=8,wrap=True)
    fig.suptitle('Biplot PCA 2D — desenvolvimento F1–F9', fontweight='bold')
    fig.tight_layout(rect=[0,.08,1,.95]); return fig


def _fig_3d(z, y, idx, config, selecionadas, variancia):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(17, 8.5))
    ax = fig.add_subplot(121, projection='3d')
    av = fig.add_subplot(122, projection='3d')
    _scatter(ax, z, y, idx, config, d=3)
    factor = .27 * min(float(np.ptp(z[idx, j])) for j in range(3))
    for i, (_, r) in enumerate(selecionadas.iterrows(), start=1):
        u, v, w = [float(r[f'corr_PC{k}']) for k in (1, 2, 3)]
        ax.quiver(0, 0, 0, u * factor, v * factor, w * factor,
                  color='#262626', arrow_length_ratio=.12, lw=1.3)
        ax.text(u * factor * 1.1, v * factor * 1.1, w * factor * 1.1,
                f'V{i}', fontsize=8)
        av.quiver(0, 0, 0, u, v, w, color='#274d83', lw=1.5,
                  arrow_length_ratio=.10)
        av.text(u * 1.1, v * 1.1, w * 1.1, str(r['variavel_preprocessada']), fontsize=7)
    u = np.linspace(0, 2*np.pi, 35)
    v = np.linspace(0, np.pi, 18)
    av.plot_wireframe(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                      np.outer(np.ones_like(u), np.cos(v)), alpha=.12, linewidth=.4)
    ax.set(xlabel=f'PC1 ({variancia[0]:.2%})', ylabel=f'PC2 ({variancia[1]:.2%})',
           zlabel=f'PC3 ({variancia[2]:.2%})', title=f'Scores + setas (fator {factor:.3g})')
    ax.legend(loc='best')
    av.set(xlabel='corr(X, PC1)', ylabel='corr(X, PC2)', zlabel='corr(X, PC3)',
           xlim=(-1.05,1.05), ylim=(-1.05,1.05), zlim=(-1.05,1.05),
           title='Esfera de correlacoes (escala real)')
    itens = [f'V{i}={r["variavel_preprocessada"]}: ' +
             ','.join(f'{r[f"corr_PC{k}"]:+.2f}' for k in (1,2,3))
             for i, (_, r) in enumerate(selecionadas.iterrows(), start=1)]
    tabela = '\n'.join('   |   '.join(itens[j:j+3]) for j in range(0, len(itens), 3))
    fig.text(.50,.025,'Correlacoes (PC1, PC2, PC3): ' + tabela,
             ha='center', va='bottom', fontsize=7, wrap=True)
    fig.suptitle('Biplot PCA 3D — desenvolvimento F1–F9', fontweight='bold')
    fig.tight_layout(rect=[0,.07,1,.95]); return fig


def gerar(config, data, destino, ext, dpi, csv):
    """Produz grafico correspondente a dimensionalidade PCA e CSV completo."""
    x, coord, modelo, y = _entrada(data)
    opts = config['visualizacao']['graficos']['pca_biplot']
    max_setas = int(opts.get('max_setas', 6))
    tab = tabela_correlacoes(data)
    selecao = _principais(tab, max_setas)
    limite = int(config['visualizacao']['geometria'].get('pontos_maximos_amostra', 20000))
    # Amostragem apenas visual; indices das observacoes permanecem alinhados.
    from src.visualizacao_geometrica import _amostra_plot
    idx = _amostra_plot(len(y), y, config, limite)
    z = coord.to_numpy(dtype=float)
    dim = int(z.shape[1])
    variancia = modelo.explained_variance_ratio_
    figura = (_fig_1d(z, y, idx, config, selecao, variancia) if dim == 1 else
              _fig_2d(z, y, idx, config, selecao, variancia) if dim == 2 else
              _fig_3d(z, y, idx, config, selecao, variancia))
    base = f'{PREFIXO}{dim}D'
    vis._salvar(figura, Path(destino) / f'{base}.{ext}', dpi=dpi)
    if csv:
        vis._csv_protegido(tab, Path(destino) / f'{base}_correlacoes.csv')
        coords = coord.iloc[idx].copy()
        coords['classe_real'] = y[idx]
        coords.insert(0, 'indice_linha_desenvolvimento', np.asarray(coord.index)[idx])
        vis._csv_protegido(coords, Path(destino) / f'{base}_scores_amostra.csv')
    return {'arquivo': f'{base}.{ext}', 'dimensoes': dim, 'n_obs_total': len(z),
            'n_obs_amostra': len(idx), 'n_variaveis': x.shape[1],
            'n_setas': len(selecao), 'origem': 'desenvolvimento_F1_F9', 'f10_utilizado': False}
