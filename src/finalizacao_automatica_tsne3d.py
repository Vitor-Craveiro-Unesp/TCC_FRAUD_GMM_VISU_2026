# -*- coding: utf-8 -*-
"""Finalizacao oficial t-SNE 3D com graficos F10 interativos GMM04/GMM07/GMM08.

Implementacao inspirada em finalizacao_automatica_pca3d.py, preservando o
contrato central do F10: modelo e tau* congelados em OOF F1-F9, sem refit no
F10 e sem uso de F10 na selecao.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from html import escape
import json

import numpy as np
import pandas as pd
import yaml

from src.folds import hash_arquivo
from src.amostragem_f10 import fracao_amostra_f10, selecionar_f10_estratificado
from src.ponte_oof_final_tsne import verificar_oof_concluido_tsne
from src.treinamento_final_tsne import carregar_pacote_confiavel
from src.finalizacao_controlada_tsne import executar_ajuste_final_tsne_em_memoria, persistir_finalizacao_controlada
from src.avaliacao_final_tsne import avaliar_f10_em_memoria
from src.finalizacao_automatica_pca1d import ROOT, exigir, ler_json, sha, salvar_json_novo


def validar_automacao(config: dict) -> bool:
    """Valida sem abrir dados nem criar saidas; permissao exclusiva TSNE 3D."""
    fracao_amostra_f10(config)
    if str(config.get('reducao', {}).get('metodo', '')).upper() != 'TSNE':
        raise ValueError('Finalizador t-SNE 3D nao aceita PCA.')
    dimensao = config.get('tsne', {}).get('n_components')
    if dimensao not in (3, [3]):
        raise ValueError('Finalizador t-SNE 3D aceita somente dimensao 3.')
    fracao = config.get('dados', {}).get('fracao_base')
    if type(fracao) not in (int, float) or float(fracao) not in (0.01, 0.03, 0.1):
        raise ValueError('Finalizacao aceita desenvolvimento 1%, 3% ou 10%.')
    if (config.get('divisao', {}).get('total_folds') != 10 or
            config['divisao'].get('qual_fold_teste') != 10 or
            config['divisao'].get('quantidade_folds_teste') != 1):
        raise ValueError('O F10 deve permanecer o unico fold de teste.')
    gmm = config.get('gmm', {})
    k = gmm.get('n_components')
    if k not in ([4], 4) or gmm.get('n_init') != 5 or str(gmm.get('covariance_type', '')).lower() != 'full':
        raise ValueError('Requer GMM K4 FULL n_init5.')
    am = config.get('amostragem', {})
    if am.get('modo') != 'RUS' or am.get('razao_negativos_positivos') != 16:
        raise ValueError('Requer RUS 16:1.')
    if config.get('experimento', {}).get('semente') != 42:
        raise ValueError('Requer seed42.')
    if config.get('execucao', {}).get('proteger_resultados') is not True:
        raise ValueError('Protecao de resultados obrigatoria.')
    # Dependencias da galeria/geometria 3D.
    for arq in (
        'src/visualizacao_geometrica.py',
        'src/visualizacao_gmm04.py',
        'src/visualizacao_gmm07.py',
        'src/visualizacao_gmm08_3d.py',
    ):
        exigir((ROOT/arq).is_file(), f'Dependencia ausente: {arq}')
    return True


def _ler_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding='utf-8-sig'))


def _salvar_json_novo(p: Path, conteudo: dict) -> None:
    with p.open('x', encoding='utf-8') as f:
        json.dump(conteudo, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def _amostra_f10(folds, y, *, config: dict) -> np.ndarray:
    return selecionar_f10_estratificado(folds, y, config)


def _dados_geometria_tsne(pacote, pred, coords):
    exigir(list(coords.columns) == ['linha_original', 'TSNE1', 'TSNE2', 'TSNE3'],
           'Coordenadas t-SNE 3D invalidas.')
    ids = coords['linha_original'].to_numpy(dtype=np.int64)
    exigir(np.array_equal(ids, pred['linha_original'].to_numpy(dtype=np.int64)),
           'Predicoes e coordenadas F10 desalinhadas.')
    c = coords[['TSNE1', 'TSNE2', 'TSNE3']].copy()
    c.columns = ['PC1', 'PC2', 'PC3']  # reutiliza os modulos 3D especializados
    gamma = pacote.gmm.modelo_.predict_proba(c.to_numpy(dtype=float))
    return {
        'coord': c,
        'y': pred['y_true'].to_numpy(dtype=np.uint8),
        'scores': pred['prob_fraude'].to_numpy(dtype=float),
        'gamma': pd.DataFrame(gamma, columns=list(pacote.gmm.saidas_)),
        'gmm': pacote.gmm,
        'rh': pacote.risco,
        'redutor': None,
        'limiar_oof': float(pacote.limiar_oof),
    }


def _normalizar_rotulos_figura_tsne(fig):
    from matplotlib.text import Text
    trocas = (
        ('PCA 3D', 't-SNE 3D'),
        ('PC1', 'TSNE1'),
        ('PC2', 'TSNE2'),
        ('PC3', 'TSNE3'),
        ('F10 não utilizado', 'F10 exploratório (modelo F1–F9 congelado)'),
        ('F10 nao utilizado', 'F10 exploratório (modelo F1–F9 congelado)'),
        ('Observações do desenvolvimento', 'Observações F10'),
        ('Observações da simulação', 'Observações F10'),
        ('média do desenvolvimento', 'média das projeções F10'),
        ('média das projeções desenvolvimento', 'média das projeções F10'),
        ('Rotulos de desenvolvimento', 'Rótulos F10'),
    )
    for obj in fig.findobj(match=Text):
        txt = obj.get_text()
        novo = txt
        for a, b in trocas:
            novo = novo.replace(a, b)
        if novo != txt:
            obj.set_text(novo)


def _normalizar_html_interativo_tsne(destino: Path):
    for path in sorted(destino.glob('GMM*_INTERATIVA.html')):
        html = path.read_text(encoding='utf-8-sig')
        trocas = (
            ('PCA 3D', 't-SNE 3D'),
            ('PC1', 'TSNE1'),
            ('PC2', 'TSNE2'),
            ('PC3', 'TSNE3'),
            ('Observações do desenvolvimento', 'Observações F10'),
            ('Observações da simulação', 'Observações F10'),
            ('F10 não utilizado.', 'F10 exploratório repetido, modelo F1–F9 congelado.'),
            ('F10 não utilizado', 'F10 exploratório repetido, modelo F1–F9 congelado'),
            ('F10 nao utilizado', 'F10 exploratório repetido, modelo F1–F9 congelado'),
            ('média do desenvolvimento', 'média das projeções F10'),
            ('Rotulos de desenvolvimento', 'Rótulos F10'),
        )
        novo = html
        for antes, depois in trocas:
            novo = novo.replace(antes, depois)
        if novo != html:
            path.write_text(novo, encoding='utf-8')


def _desenhar_observacoes_tsne(figs: Path, cfg: dict, pred: pd.DataFrame, coords: pd.DataFrame):
    from matplotlib import pyplot as plt
    from src import visualizacao_oof as vo
    from src import visualizacao_geometrica as geo

    dpi = int(cfg['visualizacao']['padrao'].get('dpi', 180))
    dados_y = pred['y_true'].to_numpy(dtype=np.uint8)
    ids = coords['linha_original'].to_numpy(dtype=np.int64)
    z = coords[['TSNE1', 'TSNE2', 'TSNE3']].to_numpy(dtype=float)
    lim = int(cfg['visualizacao']['geometria'].get('pontos_maximos_amostra', 20000))
    iy = geo._amostra_plot(len(ids), dados_y, cfg, lim)
    fig = plt.figure(figsize=(11.5, 8.0))
    ax = fig.add_subplot(111, projection='3d')
    cores = cfg['visualizacao']['classes']
    for classe, chave in ((0, 'legitima'), (1, 'fraude')):
        sel = iy[dados_y[iy] == classe]
        if not len(sel):
            continue
        estilo = cores[chave]
        ax.scatter(z[sel, 0], z[sel, 1], z[sel, 2],
                   c=estilo.get('cor', '#4389c2'), s=float(estilo.get('tamanho', 12)),
                   alpha=float(estilo.get('alpha', .55)), marker=estilo.get('marcador', 'o'),
                   label=f"{estilo.get('nome', chave)} (n={len(sel)})", rasterized=True)
    ax.set_xlabel('TSNE1')
    ax.set_ylabel('TSNE2')
    ax.set_zlabel('TSNE3')
    ax.view_init(elev=22, azim=-64)
    ax.set_box_aspect((1, 1, .85))
    ax.set_title('OBSERVAÇÕES F10 — t-SNE 3D')
    ax.legend(loc='upper left')
    fig.subplots_adjust(left=.06, right=.94, bottom=.08, top=.91)
    vo._salvar(fig, figs/'TSNE_OBSERVACOES_TSNE1_TSNE2_TSNE3.png', dpi=dpi)
    pd.DataFrame({
        'linha_original_f10': ids[iy],
        'TSNE1': z[iy, 0], 'TSNE2': z[iy, 1], 'TSNE3': z[iy, 2],
        'classe_real': dados_y[iy],
    }).to_csv(figs/'TSNE_OBSERVACOES_TSNE1_TSNE2_TSNE3.csv', index=False, float_format='%.17g')


def _desenhar_geometria_tsne(folder: Path, state: dict, pacote, pred: pd.DataFrame, coords: pd.DataFrame, met: dict):
    from matplotlib.figure import Figure
    from src import visualizacao_geometrica as geo
    from src import visualizacao_gmm04 as gmm04
    from src import visualizacao_gmm07 as gmm07
    from src import visualizacao_gmm08_3d as gmm08_3d

    figs = folder/'vencedor/teste_final/graficos_gerados'
    figs.mkdir(parents=True, exist_ok=False)
    cfg = state['config']
    dpi = int(cfg['visualizacao']['padrao'].get('dpi', 180))
    dados = _dados_geometria_tsne(pacote, pred, coords)
    _desenhar_observacoes_tsne(figs, cfg, pred, coords)

    mc = met['comparacao_limiares']
    mcc05 = float(mc['tau_0_50']['mcc'])
    mccstar = float(mc['tau_mcc_oof']['mcc'])
    exigir(mccstar == float(met['mcc_limiar_oof']) and
           float(mc['tau_mcc_oof']['limiar']) == float(pacote.limiar_oof),
           'Valores F10 divergem do limiar e MCC congelados.')

    avaliacao = {'origem': 'f10_amostral_congelado'}
    guardar = geo._guardar
    avaliar_cortes = geo._avaliar_cortes_oof
    rotulo07 = gmm07._rotulo_mcc
    rotulo08 = gmm08_3d._origem_mcc
    salvar_figura = Figure.savefig

    def cortes(obj, tau):
        exigir(tau == pacote.limiar_oof and obj.get('origem') == 'f10_amostral_congelado',
               'Origem ou limiar F10 incorretos na geometria t-SNE.')
        return mcc05, mccstar

    def rotulo_f10(obj):
        exigir(obj.get('origem') == 'f10_amostral_congelado',
               'Rotulo MCC sem origem F10.')
        return ('MCC F10', 'MCC F10 exploratório; τ* selecionado exclusivamente em OOF F1–F9.')

    def fig_salvar(fig, *args, **kwargs):
        _normalizar_rotulos_figura_tsne(fig)
        return salvar_figura(fig, *args, **kwargs)

    def geo_guardar(fig, *args, **kwargs):
        _normalizar_rotulos_figura_tsne(fig)
        return guardar(fig, *args, **kwargs)

    geo._guardar = geo_guardar
    geo._avaliar_cortes_oof = cortes
    gmm07._rotulo_mcc = rotulo_f10
    gmm08_3d._origem_mcc = rotulo_f10
    Figure.savefig = fig_salvar
    try:
        gmm04.gerar(cfg, dados, figs, 'png', dpi, True)
        escolha = {'limiar_mcc_oof': pacote.limiar_oof}
        gmm07.gerar(cfg, dados, escolha, figs, 'png', dpi, True, avaliacao=avaliacao)
        geo._fronteira(cfg, dados, escolha, figs, 'png', dpi, True, avaliacao=avaliacao)
    finally:
        Figure.savefig = salvar_figura
        geo._guardar = guardar
        geo._avaliar_cortes_oof = avaliar_cortes
        gmm07._rotulo_mcc = rotulo07
        gmm08_3d._origem_mcc = rotulo08

    _normalizar_html_interativo_tsne(figs)
    print('[TSNE 3D] Geometria F10 com GMM04/GMM07/GMM08 e HTMLs interativos concluída.', flush=True)


_GRAFICOS = (
    ('TSNE_OBSERVACOES_TSNE1_TSNE2_TSNE3.png', 't-SNE — Observações F10 em 3D'),
    ('GMM04_dominancia_3d.png', 'GMM04 — Dominância 3D'),
    ('GMM07_sensibilidade_3d_comparativo.png', 'GMM07 — Sensibilidade 3D'),
    ('GMM08_mapa_risco_3d_comparativo.png', 'GMM08 — Risco e geometria 3D 95%'),
)

_INTERATIVOS = (
    ('GMM07_sensibilidade_3d_comparativo_INTERATIVA.html', 'GMM07 — Sensibilidade 3D interativa'),
    ('GMM08_mapa_risco_3d_comparativo_INTERATIVA.html', 'GMM08 — Mapa de risco 3D interativo'),
)


def _galeria_final_tsne(folder: Path, ids: np.ndarray, pacote, met: dict):
    dest = folder/'vencedor/teste_final'
    figs = dest/'graficos_gerados'
    gal = dest/'galeria'
    for nome, _ in _GRAFICOS:
        exigir((figs/nome).is_file(), f'Figura obrigatoria faltando: {nome}')
    for nome, _ in _INTERATIVOS:
        exigir((figs/nome).is_file(), f'HTML interativo obrigatorio faltando: {nome}')
    cards = ''.join(
        f'<article><a href="../graficos_gerados/{escape(n, quote=True)}" target="_blank">'
        f'<img src="../graficos_gerados/{escape(n, quote=True)}" loading="lazy" alt="{escape(label)}">'
        f'<p>{escape(label)}</p></a></article>'
        for n, label in _GRAFICOS
    )
    interativos = ''.join(
        f'<a href="../graficos_gerados/{escape(n, quote=True)}" target="_blank">{escape(label)}</a>'
        for n, label in _INTERATIVOS
    )
    arquivos = ['../predicoes_f10.csv', '../metricas_f10.json', '../indices_f10_amostra.npy', '../coordenadas_TSNE1_TSNE2_TSNE3_f10.csv', '../modelo_final_f1_f9/manifesto_modelo_final.json']
    links = ''.join(f'<a href="{escape(a, quote=True)}">{escape(Path(a).name)}</a>' for a in arquivos)
    html = f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>t-SNE 3D F10 — galeria completa</title><style>body{{font:15px system-ui,sans-serif;background:#f5f7fa;color:#172236;max-width:1400px;margin:auto;padding:20px}}
.grade{{display:grid;grid-template-columns:repeat(auto-fit,minmax(315px,1fr));gap:16px}}article{{background:#fff;border:1px solid #dae1e9;border-radius:10px;padding:12px;box-shadow:0 2px 8px #0001}}article img{{width:100%;aspect-ratio:4/3;object-fit:contain}}
a{{color:#285783;text-decoration:none}}article p{{font-weight:600}}.aviso{{background:#e8edf4;border-left:4px solid #355877;padding:14px}}.links{{display:flex;gap:16px;flex-wrap:wrap}}</style></head><body>
<h1>t-SNE 3D / GMM K4 n_init5 — Teste final F10</h1><p class="aviso">Amostra F10 de {len(ids)} observações, {fracao_amostra_f10(pacote.configuracao):.2%} do fold, EXPLORATÓRIO repetido. F10 não é nova validação independente. τ* = {pacote.limiar_oof:.17g}, escolhido em OOF F1–F9. Modelo, encoder, scaler, redutor t-SNE e GMM congelados.</p>
<p>F10: AP={met['ap']:.7f} | MCC em τ*={met['mcc_limiar_oof']:.7f} | LogLoss={met['logloss']:.7f} | Score={met['score_composto_limiar_oof']:.7f}</p>
<h2>Gráficos tridimensionais interativos</h2><div class="links">{interativos}</div>
<h2>Gráficos estáticos F10 ({len(_GRAFICOS)})</h2><div class="grade">{cards}</div>
<h2>Resultados e proveniência</h2><div class="links">{links}</div></body></html>'''
    gal.mkdir(parents=True, exist_ok=True)
    with (gal/'index.html').open('x', encoding='utf-8') as f:
        f.write(html)


def concluir_ensaio(pasta_experimento: str | Path, *, raiz: str | Path | None = None) -> Path:
    pasta = Path(pasta_experimento).resolve()
    base = Path(raiz).resolve() if raiz is not None else Path(__file__).resolve().parents[1]
    if not pasta.is_relative_to((base / 'outputs').resolve()):
        raise ValueError('Finalizacao so aceita experimento dentro de outputs/.')
    tdir = pasta / 'vencedor' / 'teste_final'
    if not tdir.is_dir():
        raise FileNotFoundError('Pasta teste_final do OOF ausente.')
    trava = tdir / '.finalizacao_tsne_3d.lock'
    try:
        with trava.open('x', encoding='utf-8') as f:
            f.write('Finalizacao t-SNE 3D em curso. Nao executar duas vezes em paralelo.\n')
    except FileExistsError as exc:
        raise RuntimeError('Finalizacao t-SNE ja em andamento; trava encontrada.') from exc
    try:
        return _concluir_ensaio_sem_lock(pasta, raiz=base)
    finally:
        trava.unlink(missing_ok=True)


def _concluir_ensaio_sem_lock(pasta_experimento: str | Path, *, raiz: str | Path | None = None) -> Path:
    from src import orquestracao_oof as orq

    base = Path(raiz).resolve() if raiz is not None else Path(__file__).resolve().parents[1]
    pasta = Path(pasta_experimento).resolve()
    if not pasta.is_relative_to((base / 'outputs').resolve()):
        raise ValueError('Finalizacao so aceita experimento dentro de outputs/.')
    pre = verificar_oof_concluido_tsne(pasta)
    validar_automacao(pre.configuracao)
    manifest = _ler_json(pasta / 'manifesto.json')
    yaml_salvo = pasta / 'configuracao.yaml'
    if yaml.safe_load(yaml_salvo.read_text(encoding='utf-8-sig')) != pre.configuracao:
        raise RuntimeError('Configuracao OOF alterada.')
    pq = base / pre.configuracao['dados']['arquivo']
    folds, arq_folds, hash_parquet = orq._identificar_folds(base, pre.configuracao, pq)
    if hash_parquet != manifest['sha256_parquet'] or hash_arquivo(arq_folds) != manifest['sha256_folds']:
        raise RuntimeError('Parquet ou folds divergem do OOF.')
    with np.load(pasta/'vencedor/oof/predicoes.npz', allow_pickle=False) as arr:
        ids_dev = np.asarray(arr['linha_original'], dtype=np.int64)
        fold_dev = np.asarray(arr['fold'], dtype=np.uint8)
        y_dev = np.asarray(arr['y_true'], dtype=np.uint8)
    if not np.array_equal(folds[ids_dev], fold_dev) or np.any(fold_dev == 10):
        raise RuntimeError('Indices de desenvolvimento nao pertencem a F1-F9.')
    teste = pasta/'vencedor/teste_final'
    modelo_dir = teste/'modelo_final_f1_f9'
    marcador = teste/'manifesto_finalizacao_tsne.json'
    if marcador.exists():
        status = _ler_json(marcador)
        if status.get('galeria_f10_concluida') is True:
            print('[RETOMADA TSNE] F10 e galeria 3D ja concluidos.', flush=True)
            return pasta
        print('[RETOMADA TSNE] F10 ja avaliado; concluindo graficos e galeria.', flush=True)
    
    if modelo_dir.is_dir():
        pacote = carregar_pacote_confiavel(modelo_dir)
        if (pacote.escolha_oof != pre.escolha_oof or pacote.configuracao != pre.configuracao or
                pacote.proveniencia['sha256_parquet'] != hash_parquet or
                pacote.proveniencia['sha256_folds'] != hash_arquivo(arq_folds)):
            raise RuntimeError('Modelo final existente diverge do OOF. Nao sobrescrever.')
        print('[RETOMADA TSNE] Modelo F1-F9 existente e verificado.', flush=True)
    else:
        df = orq._ler_desenvolvimento(pq, ids_dev)
        if len(df) != len(ids_dev) or not np.array_equal(df[pre.configuracao['dados']['coluna_alvo']].to_numpy(), y_dev):
            raise RuntimeError('Rotulos do desenvolvimento divergentes do OOF.')
        resultado = executar_ajuste_final_tsne_em_memoria(
            df, fold_dev, pre.configuracao, pre.escolha_oof,
            {**pre.proveniencia, 'sha256_predicoes_oof': hash_arquivo(pasta/'vencedor/oof/predicoes.npz')},
            registrar_historico_em=True,
        )
        persistir_finalizacao_controlada(resultado, modelo_dir)
        pacote = resultado.pacote
        print('[TSNE 3D] Modelo final F1-F9 salvo.', flush=True)

    pred_path = teste/'predicoes_f10.csv'
    coords_path = teste/'coordenadas_TSNE1_TSNE2_TSNE3_f10.csv'
    met_path = teste/'metricas_f10.json'
    idx_path = teste/'indices_f10_amostra.npy'

    if pred_path.is_file() and coords_path.is_file() and met_path.is_file() and idx_path.is_file():
        pred = pd.read_csv(pred_path)
        coords = pd.read_csv(coords_path)
        met = _ler_json(met_path)
        with idx_path.open('rb') as f:
            ids_f10 = np.load(f, allow_pickle=False)
    else:
        publicados = ('predicoes_f10.csv', 'metricas_f10.json', 'indices_f10_amostra.npy',
                      'manifesto_amostra_f10.json', 'coordenadas_TSNE1_TSNE2_TSNE3_f10.csv')
        if any((teste/nome).exists() for nome in publicados):
            raise RuntimeError('Saida F10 parcial existente: nao reavaliar nem sobrescrever.')
        y = orq._ler_alvo(pq, pre.configuracao['dados']['coluna_alvo'])
        if y.shape != folds.shape:
            raise RuntimeError('Tamanho da base difere dos folds.')
        fracao_f10 = fracao_amostra_f10(pre.configuracao)
        ids_f10 = _amostra_f10(folds, y, config=pre.configuracao)
        if not np.all(folds[ids_f10] == 10):
            raise RuntimeError('Indices de teste nao pertencem exclusivamente a F10.')
        from src.finalizacao_automatica_pca1d import ler_linhas_f10
        df_f10 = ler_linhas_f10(pq, ids_f10)
        if not np.array_equal(df_f10[pre.configuracao['dados']['coluna_alvo']].to_numpy(), y[ids_f10]):
            raise RuntimeError('Rotulos F10 nao correspondem ao parquet/folds.')
        pred, met, coords = avaliar_f10_em_memoria(pacote, df_f10, folds[ids_f10], ids_f10, retornar_coordenadas=True)
        if (list(coords.columns) != ['linha_original', 'TSNE1', 'TSNE2', 'TSNE3'] or
                not np.array_equal(coords['linha_original'].to_numpy(dtype=np.int64), ids_f10) or
                not np.array_equal(pred['linha_original'].to_numpy(dtype=np.int64), ids_f10) or
                not np.isfinite(coords[['TSNE1', 'TSNE2', 'TSNE3']].to_numpy(dtype=np.float64)).all()):
            raise RuntimeError('Coordenadas F10 inconsistentes com as predicoes; nao publicar resultados.')
        met['carater'] = 'F10_EXPLORATORIO_REPETIDO_SEM_NOVA_INDEPENDENCIA'
        met['fracao_f10_dentro_do_fold'] = fracao_f10
        met['fracao_desenvolvimento_f1_f9'] = float(pre.configuracao['dados']['fracao_base'])
        with pred_path.open('x', encoding='utf-8') as f:
            pred.to_csv(f, index=False, float_format='%.17g')
        with coords_path.open('x', encoding='utf-8') as f:
            coords.to_csv(f, index=False, float_format='%.17g')
        _salvar_json_novo(met_path, met)
        with idx_path.open('xb') as f:
            np.save(f, ids_f10, allow_pickle=False)
        _salvar_json_novo(teste/'manifesto_amostra_f10.json', {
            'fracao_dentro_fold_f10': fracao_f10, 'semente': 42,
            'n': len(ids_f10), 'n_fraudes': int(pred['y_true'].sum()),
            'sha256_indices': sha256(ids_f10.tobytes()).hexdigest(),
            'metodo': 'TSNE', 'dimensao': 3, 'reutilizacao_f10': 'exploratorio',
            'limiar_congelado_em': 'OOF_F1_F9', 'sem_refit_no_f10': True,
        })
        _salvar_json_novo(marcador, {
            'etapa': 'F10_AVALIADO_GALERIA_PENDENTE', 'metodo': 'TSNE',
            'dimensao': 3, 'id_candidato_oof': pre.escolha_oof['id_candidato'],
            'fracao_desenvolvimento': float(pre.configuracao['dados']['fracao_base']),
            'fracao_f10_dentro_do_fold': fracao_f10,
            'galeria_f10_concluida': False,
            'sha256_modelo': hash_arquivo(modelo_dir/'modelo_final_f1_f9.joblib'),
            'sha256_predicoes': hash_arquivo(pred_path),
            'sha256_coordenadas': hash_arquivo(coords_path),
            'sha256_metricas': hash_arquivo(met_path),
            'sha256_indices_arquivo': hash_arquivo(idx_path),
        })
        print('[TSNE 3D] F10 avaliado; renderizando graficos 3D...', flush=True)

    figs = teste/'graficos_gerados'
    if not figs.exists():
        _desenhar_geometria_tsne(pasta, {'config': pre.configuracao}, pacote, pred, coords, met)
    if not (teste/'galeria/index.html').is_file():
        _galeria_final_tsne(pasta, ids_f10, pacote, met)

    status = _ler_json(marcador)
    status.update({
        'etapa': 'F10_AVALIADO_GALERIA_CONCLUIDA',
        'galeria_f10_concluida': True,
        'sha256_galeria_index': hash_arquivo(teste/'galeria/index.html'),
        'graficos_sha256': {n: hash_arquivo(figs/n) for n, _ in _GRAFICOS if (figs/n).is_file()},
        'interativos_sha256': {n: hash_arquivo(figs/n) for n, _ in _INTERATIVOS if (figs/n).is_file()},
    })
    tmp = marcador.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(marcador)
    print('[TSNE 3D] F10 avaliado, graficos 3D e galeria concluídos.', flush=True)
    return pasta
