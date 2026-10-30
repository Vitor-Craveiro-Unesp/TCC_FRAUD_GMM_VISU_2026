# -*- coding: utf-8 -*-
"""PCA 1D F10, geometria PARTE 1: PC1, observacoes, GMM01, GMM04.

--verificar: audita arquivos F10 ja salvos, SHA, ordenacao e modelos SEM abrir Parquet,
            joblib ou gravar arquivos. Nao recalcula metricas.
--gerar: carrega SOMENTE o modelo local confiavel de F1-F9 e desenha quatro graficos
         usando as funcoes originais do projeto. Nao chama .fit, nem produz novas
         probabilidades de fraude, nem reavalia F10, nem acessa Parquet ou TARGET.

Saida: dentro de vencedor/teste_final/graficos_gerados, sem sobrescrever nada.
HTML de revisao separado. Galeria principal existente permanece intacta.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

ENSAIO = RAIZ / 'outputs/ensaio_pca_gmm_20261029_173123'
MODELO = RAIZ / 'outputs/treinamento_final_pca_10pct_20261029_174241/PCA_1D'
TESTE = ENSAIO / 'vencedor/teste_final'
GEOMETRIA = TESTE / 'dados_geometria'
FIGURAS = TESTE / 'graficos_gerados'
GALERIA = TESTE / 'galeria/complemento_geometria_pca1d_f10.html'
MANIFESTO = TESTE / 'manifesto_complemento_geometria_pca1d_f10.json'

NOMES = (
    '12_pca_projecao_PC1',
    'PCA_OBSERVACOES_PC1',
    'GMM01_densidade_marginal_1d_PC1',
    'GMM04_dominancia_1d',
)


def exigir(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def ler_json(path: Path) -> dict:
    exigir(path.is_file(), f'JSON ausente: {path}')
    result = json.loads(path.read_text(encoding='utf-8-sig'))
    exigir(isinstance(result, dict), f'JSON invalido: {path}')
    return result


def hash_arquivo(path: Path) -> str:
    exigir(path.is_file(), f'Arquivo ausente: {path}')
    digest = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validar_alinhamento(coords: pd.DataFrame, preds: pd.DataFrame, n_esperado: int) -> None:
    exigir(list(coords.columns) == ['linha_original', 'PC1'], 'CSV PC1 nao corresponde ao schema congelado.')
    exigir(list(preds.columns) == ['linha_original', 'fold', 'y_true', 'prob_fraude',
                                  'predicao_tau_0_50', 'predicao_tau_mcc_oof'],
           'Schema de predicoes F10 inesperado.')
    exigir(len(coords) == len(preds) == n_esperado, 'Numero de registros F10 incompatível.')
    ids = coords['linha_original'].to_numpy()
    exigir(np.issubdtype(ids.dtype, np.integer) and np.all(ids >= 0) and
           np.all(np.diff(ids) > 0), 'IDs PCA desordenados ou repetidos.')
    exigir(np.array_equal(ids, preds['linha_original'].to_numpy()), 'F10: PC1 e scores desalinhados.')
    exigir((preds['fold'] == 10).all(), 'F10 nao e o unico fold.')
    exigir(preds['y_true'].isin([0, 1]).all(), 'Rotulos F10 nao binarios.')
    exigir(preds['predicao_tau_0_50'].isin([0, 1]).all() and
           preds['predicao_tau_mcc_oof'].isin([0, 1]).all(), 'Predicoes binarias invalidas.')
    x = coords['PC1'].to_numpy(dtype=float)
    p = preds['prob_fraude'].to_numpy(dtype=float)
    exigir(np.isfinite(x).all() and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(),
           'Valores PC1 ou scores F10 invalidos.')


def validar_cortes(preds: pd.DataFrame, tau: float) -> None:
    exigir(np.isfinite(tau) and 0 <= tau <= 1, 'Tau OOF invalido.')
    p = preds['prob_fraude'].to_numpy(dtype=float)
    exigir(np.array_equal((p >= 0.5).astype(int), preds['predicao_tau_0_50'].to_numpy()),
           'Predicao de corte fixo diverge do F10 salvo.')
    exigir(np.array_equal((p >= tau).astype(int), preds['predicao_tau_mcc_oof'].to_numpy()),
           'Predicao no tau OOF diverge do F10 salvo.')


def checar_destino(formato: str) -> None:
    exigir(formato == 'png', 'Este complemento suporta PNG apenas; YAML diferente.')
    exigir(not MANIFESTO.exists() and not GALERIA.exists(),
           'Este complemento ja foi iniciado ou concluido. Nao sobrescrever.')
    for nome in NOMES:
        for ext in ('png', 'csv'):
            exigir(not (FIGURAS / f'{nome}.{ext}').exists(),
                   f'Arquivo existente, preservado: {nome}.{ext}')


def conferir() -> dict:
    print('=== PCA 1D F10 — CONFERENCIA GEOMETRIA PARTE 1 ===', flush=True)
    g = ler_json(GEOMETRIA / 'manifesto_geometria_f10.json')
    m = ler_json(MODELO / 'manifesto_modelo_final.json')
    e = ler_json(ENSAIO / 'vencedor/metricas/melhor_configuracao.json')
    esp = ler_json(TESTE / 'manifesto_espelho_graficos_f10.json')
    exigir(g['dimensao'] == m['dimensao_pca'] == 1, 'Modelo nao e PCA 1D.')
    exigir(g['n'] == esp['n'] == 18523 and m['n_treino_original'] == 166716,
           'Amostra F10/modelo divergentes.')
    exigir(g['tipo'] == 'EXTRACAO_DESCRITIVA_F10_JA_AVALIADO' and
           g['parquet_target_lido'] is False and
           g['nova_avaliacao_f10'] is False and g['novo_treinamento'] is False,
           'Origem das coordenadas nao autorizada.')
    tau = float(g['limiar_mcc_oof_congelado'])
    exigir(tau == float(m['limiar_mcc_oof_congelado']) ==
           float(e['limiar_mcc_oof']) == float(esp['limiar_mcc_congelado_oof']),
           'Limiar MCC OOF mudou.')
    exigir(str(e['id_candidato']) == m['id_candidato_oof'], 'Candidato PCA 1D alterado.')
    exigir(hash_arquivo(MODELO / 'modelo_final_f1_f9.joblib') ==
           m['sha256_modelo_final'] == g['sha256_modelo_final'], 'Hash do modelo divergente.')
    exigir(hash_arquivo(TESTE / 'predicoes_f10.csv') == esp['predicoes_sha256'],
           'Predicoes F10 mudaram.')
    exigir(hash_arquivo(TESTE / 'metricas_f10.json') == esp['metricas_sha256'],
           'Metricas F10 mudaram.')
    for name in ('coordenadas_PC1_f10.csv', 'resumo_padronizacao_f10.csv'):
        exigir(hash_arquivo(GEOMETRIA/name) == g['arquivos'][name],
               f'Arquivo de geometria alterado: {name}')
    config = yaml.safe_load((ENSAIO / 'configuracao.yaml').read_text(encoding='utf-8-sig'))
    exigir(config['reducao']['metodo'].upper() == 'PCA' and config['pca']['n_components'] == [1],
           'Configuracao OOF nao corresponde a PCA 1D.')
    exigir(config['visualizacao']['graficos']['pca_projecoes']['habilitar'] and
           config['visualizacao']['graficos']['pca_biplot']['habilitar'] and
           config['visualizacao']['graficos']['densidades_marginais']['habilitar'] and
           config['visualizacao']['graficos']['regioes_dominancia']['habilitar'],
           'Um dos graficos esta desativado na configuracao do ensaio.')
    coords = pd.read_csv(GEOMETRIA / 'coordenadas_PC1_f10.csv')
    preds = pd.read_csv(TESTE / 'predicoes_f10.csv')
    validar_alinhamento(coords, preds, g['n'])
    validar_cortes(preds, tau)
    checar_destino(str(config['visualizacao']['padrao']['formato']).lower())
    print(f'[APROVADO] {len(coords)} linhas F10, PCs, scores e IDs alinhados.', flush=True)
    print(f'[APROVADO] Modelo/OOF congelados, tau*={tau:.17g}.', flush=True)
    print('[APROVADO] Nenhuma avaliacao, retreinamento ou leitura do Parquet.', flush=True)
    print('[APROVADO] Destinos novos; graficos 00..05 intocados.', flush=True)
    return {'config': config, 'coords': coords, 'preds': preds,
            'geom_meta': g, 'modelo_meta': m, 'tau': tau}


def preparar_dados_figuras(pacote, estado: dict) -> dict:
    # A responsabilidade gaussiana eh puramente geometrica; NAO eh probabilidade de fraude.
    pc1 = estado['coords']['PC1'].to_numpy(dtype=float)
    coords = pd.DataFrame({'PC1': pc1})
    g = np.asarray(pacote.gmm.modelo_.predict_proba(coords.to_numpy(dtype=float)), dtype=float)
    exigir(g.ndim == 2 and g.shape[0] == len(coords) and
           np.isfinite(g).all() and np.allclose(g.sum(axis=1), 1, atol=1e-6),
           'Responsabilidades gaussianas invalidas.')
    cols = list(pacote.gmm.saidas_)
    exigir(len(cols) == g.shape[1] == 4, 'GMM esperado com K=4.')
    gamma = pd.DataFrame(g, columns=cols)
    return {'coord': coords, 'y': estado['preds']['y_true'].to_numpy(dtype=np.uint8),
            'scores': estado['preds']['prob_fraude'].to_numpy(dtype=float),
            'gamma': gamma, 'gmm': pacote.gmm, 'rh': pacote.risco,
            'redutor': pacote.pca, 'limiar_oof': estado['tau']}


def corrigir_csv_projecao(path: Path, ids: np.ndarray) -> None:
    tabela = pd.read_csv(path)
    exigir(len(tabela) == len(ids) and 'risco_estimado_treino' in tabela.columns,
           'CSV PCA12 inesperado; nao foi feita alteracao.')
    tabela = tabela.rename(columns={'risco_estimado_treino': 'prob_fraude_f10'})
    tabela.insert(0, 'linha_original_f10', ids)
    tabela.to_csv(path, index=False, float_format='%.17g')


def html_revisao(n: int, tau: float, hashes: dict) -> str:
    titulos = {
        NOMES[0]: '12 — Projeção PCA PC1 (F10)',
        NOMES[1]: 'Observações reais sobre PC1 (F10)',
        NOMES[2]: 'GMM01 — Densidade do modelo congelado no domínio F10',
        NOMES[3]: 'GMM04 — Dominância das gaussianas no domínio F10',
    }
    cards = ''.join(
        f'<article><h2>{html.escape(titulos[nome])}</h2>'
        f'<a href="../graficos_gerados/{nome}.png" target="_blank">'
        f'<img src="../graficos_gerados/{nome}.png" alt="{html.escape(titulos[nome])}"></a>'
        f'<p><a href="../graficos_gerados/{nome}.csv">CSV do gráfico</a></p></article>'
        for nome in NOMES
    )
    return ('<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>PCA 1D — Geometria F10</title><style>'
            'body{font:15px system-ui,sans-serif;background:#f5f7fa;color:#172236;max-width:1300px;margin:auto;padding:24px}'
            'article{background:white;border:1px solid #d7dee6;border-radius:10px;padding:14px;margin:20px 0}'
            'img{width:100%;max-height:700px;object-fit:contain}a{color:#245386}'
            '.nota{background:#e8edf4;border-left:4px solid #355877;padding:13px}'
            '</style></head><body><h1>PCA 1D — Complemento geométrico F10</h1>'
            f'<p class="nota">F10 exploratório, n={n:,}. O modelo GMM e os transformadores foram ajustados exclusivamente em F1–F9. '
            f'τ* OOF permanece fixo em {tau:.17g}. Nenhum ajuste, reamostragem, predição ou reavaliação F10 neste complemento. '
            'As densidades e dominâncias GMM01/GMM04 descrevem os componentes do GMM congelado, não a prevalência de fraude no F10. '
            'Os limites dos eixos geométricos usam coordenadas PC1 do F10. As funções de desenho são as originais do OOF.</p>'
            f'{cards}<p>Hashes de origem registrados no manifesto. Figuras 00–05 e galeria principal preservadas.</p>'
            '</body></html>')


def gerar(estado: dict) -> None:
    from src.treinamento_final_pca import carregar_pacote_confiavel
    from src import visualizacao_geometrica as geo
    from src import visualizacao_gmm01 as gmm01
    from src import visualizacao_gmm04 as gmm04
    from src import visualizacao_pca_biplot as biplot
    from src import visualizacao_oof as vis

    pacote = carregar_pacote_confiavel(MODELO)
    exigir(pacote.n_treino_original == 166716 and pacote.limiar_oof == estado['tau'],
           'Modelo carregado nao coincide com a selecao F10.')
    dados = preparar_dados_figuras(pacote, estado)
    config = estado['config']
    estilo = config['visualizacao']['padrao']
    dpi = int(estilo.get('dpi', 180))
    ids = estado['coords']['linha_original'].to_numpy(dtype=np.int64)
    print('=== GERANDO 4 GRAFICOS COM ROTINAS OOF APROVADAS ===', flush=True)

    TESTE.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.parcial_graficos_geometria_f10_', dir=TESTE))
    try:
        # A funcao 12 possui flag legada 'desenvolvimento/simulado' apenas para marca
        # visual de dados sinteticos. O CSV gerado e corrigido explicitamente para F10.
        op12 = config['visualizacao']['graficos']['pca_projecoes']
        geo._pca_histograma_1d(config, dados, stage, 'png', dpi, True, op12)
        corrigir_csv_projecao(stage/f'{NOMES[0]}.csv', ids)
        print(f'[GERADO] {NOMES[0]}.png e CSV (F10)', flush=True)

        limite = int(config['visualizacao']['geometria'].get('pontos_maximos_amostra', 20000))
        idx = geo._amostra_plot(len(ids), dados['y'], config, limite)
        figura = biplot._fig_observacoes_pc1(dados['coord'].iloc[idx,0].to_numpy(),
                                               dados['y'][idx], config)
        vis._salvar(figura, stage/f'{NOMES[1]}.png', dpi=dpi)
        pd.DataFrame({'linha_original_f10': ids[idx], 'PC1': dados['coord'].iloc[idx,0].to_numpy(),
                      'classe_real': dados['y'][idx]}).to_csv(
            stage/f'{NOMES[1]}.csv', index=False, float_format='%.17g')
        print(f'[GERADO] {NOMES[1]}.png e CSV (F10)', flush=True)

        gmm01.gerar(config, dados, stage, 'png', dpi, True)
        print(f'[GERADO] {NOMES[2]}.png e CSV (modelo congelado)', flush=True)
        gmm04.gerar(config, dados, stage, 'png', dpi, True)
        print(f'[GERADO] {NOMES[3]}.png e CSV (modelo congelado)', flush=True)

        files = [stage/f'{name}.{ext}' for name in NOMES for ext in ('png', 'csv')]
        exigir(all(p.is_file() and p.stat().st_size > 0 for p in files),
               'Nem todos os quatro graficos geraram PNG e CSV; nada publicado.')
        hashes = {p.name: hash_arquivo(p) for p in files}
        html_text = html_revisao(len(ids), estado['tau'], hashes)
        info = {
            'versao': 'GEOMETRIA_F10_PCA1D_P1_V1',
            'modo': 'APLICACAO_VISUAL_DA_GEOMETRIA_CONGELADA',
            'n_f10': len(ids), 'tau_star_oof': estado['tau'],
            'sem_treinamento': True, 'sem_ajuste_pca': True,
            'sem_ajuste_gmm': True, 'sem_novos_scores_fraude': True,
            'sem_nova_avaliacao_f10': True, 'sem_leitura_parquet': True,
            'origem_modelo': str((MODELO/'modelo_final_f1_f9.joblib').relative_to(RAIZ)).replace('\\','/'),
            'sha256_modelo': estado['modelo_meta']['sha256_modelo_final'],
            'sha256_coordenadas_f10': estado['geom_meta']['arquivos']['coordenadas_PC1_f10.csv'],
            'sha256_predicoes_f10': hash_arquivo(TESTE/'predicoes_f10.csv'),
            'fontes_codigos': {n:hash_arquivo(RAIZ/'src'/n) for n in (
                'visualizacao_geometrica.py','visualizacao_gmm01.py','visualizacao_gmm04.py',
                'visualizacao_pca_biplot.py')},
            'graficos': hashes,
            'nota': 'GMM01/GMM04 exibem densidade/dominancia de componentes aprendidos em F1–F9; dominio baseado em PC1 F10.',
            'pendentes': ['GMM07_F10','GMM08_F10','PRE01_F10'],
        }
        (stage/MANIFESTO.name).write_text(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')
        (stage/GALERIA.name).write_text(html_text,encoding='utf-8')
        checar_destino('png')
        FIGURAS.mkdir(parents=True, exist_ok=True)
        GALERIA.parent.mkdir(parents=True, exist_ok=True)
        # Publicar somente arquivos novos. Se ocorrer falha operacional, os arquivos
        # parciais continuam visiveis e a reexecucao fica bloqueada por colisao.
        for p in files:
            os.rename(p, FIGURAS/p.name)
        os.rename(stage/GALERIA.name, GALERIA)
        os.rename(stage/MANIFESTO.name, MANIFESTO)
    finally:
        if stage.exists():
            if any(stage.iterdir()):
                print('[AVISO] Arquivos de preparo preservados para inspecao:', stage,flush=True)
            else:
                stage.rmdir()

    print(f'[CONCLUIDO] 4 PNG + 4 CSV em: {FIGURAS}', flush=True)
    print(f'[REVISAO] {GALERIA}', flush=True)
    print(f'[MANIFESTO] {MANIFESTO}', flush=True)
    print('[F10] Nenhum novo treino, reamostragem ou avaliacao.', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--verificar', action='store_true')
    group.add_argument('--gerar', action='store_true')
    a = ap.parse_args()
    estado = conferir()
    if a.gerar:
        gerar(estado)


if __name__ == '__main__':
    main()
