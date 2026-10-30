# -*- coding: utf-8 -*-
"""Espelha gráficos do campeão OOF no F10 salvo, sem repetir inferência.

Execução na raiz TCC_EXPERIMENTAL_V3:
  python scripts/execucao/ESPELHAR_VISUAIS_F10_NO_ENSAIO.py --verificar
  python scripts/execucao/ESPELHAR_VISUAIS_F10_NO_ENSAIO.py --gerar

Faz SOMENTE leitura de CSV/JSON da avaliação já concluída e do YAML do
experimento OOF de origem. Não lê Parquet/folds nem abre modelos joblib.
Os arquivos gerados são armazenados em vencedor/teste_final/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, log_loss, matthews_corrcoef

RAIZ = Path(__file__).resolve().parents[2]
# Arquivos em scripts/execucao/ sao executados por caminho, fora do pacote.
# Tornar a raiz importavel sem depender do PYTHONPATH do terminal.
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
FONTE_F10 = RAIZ / 'outputs/avaliacao_final_f10_pca_10pct_20261029T232726363532Z'
ETAPA9 = RAIZ / 'scripts/execucao/ETAPA9_GERAR_GALERIA_F10_PCA.py'

GRAFICOS = (
    '00_painel_score_composto.png',
    '01_curva_precision_recall.png',
    '02_mcc_por_limiar.png',
    '03_matriz_confusao_05.png',
    '04_matriz_confusao_mcc.png',
    '05_distribuicao_probabilidades.png',
)


def exigir(condicao: bool, mensagem: str):
    if not condicao:
        raise RuntimeError(mensagem)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(bloco)
    return digest.hexdigest()


def ler_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def carregar_conferidor():
    exigir(ETAPA9.is_file(), 'O verificador da Etapa 9 nao foi encontrado; mantenha o arquivo original no projeto.')
    spec = importlib.util.spec_from_file_location('conferidor_f10_etapa9', ETAPA9)
    exigir(spec is not None and spec.loader is not None, 'Impossivel carregar a Etapa 9.')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def conferir_vinculo(manifesto: dict, ensaio_arg: Path | None) -> tuple[Path, dict]:
    """Somente vincula F10 ao OOF que originou os modelos congelados."""
    arquivo_pre = (RAIZ / manifesto['referencia_protocolo']).resolve()
    pre = ler_json(arquivo_pre)
    entradas = [x for x in pre['modelos'] if int(x['pca_dimensao']) == 1]
    exigir(len(entradas) == 1, 'Protocolo sem referencia PCA 1D unica.')
    referencia = entradas[0]
    original = (RAIZ / referencia['experimento_oof']).resolve()
    exigir(original.is_dir(), 'Ensaio original do modelo F10 nao existe.')
    if ensaio_arg is not None:
        alvo = (RAIZ / ensaio_arg).resolve()
        exigir(alvo == original,
               'F10 JA AVALIADO PERTENCE A OUTRO ENSAIO: ' + str(original) +
               ' . Nao o vincule artificialmente ao ensaio ' + str(alvo))
    exigir(original.is_relative_to((RAIZ / 'outputs').resolve()), 'Ensaio fora de outputs.')
    escolha_path = original / 'vencedor/metricas/melhor_configuracao.json'
    exigir(escolha_path.is_file(), 'Escolha OOF do ensaio original nao encontrada.')
    escolha = ler_json(escolha_path)
    exigir(str(escolha['id_candidato']) == str(referencia['candidato_oof']),
           'ID de candidato do OOF diferente do modelo congelado para F10.')
    exigir(float(escolha['limiar_mcc_oof']) == float(referencia['limiar_mcc_oof_congelado']),
           'Limiar OOF alterado.')
    return original, referencia


def verificar(ensaio_arg: Path | None = None) -> dict[str, Any]:
    etapa9 = carregar_conferidor()
    manifesto, resultados = etapa9.conferir_fonte(FONTE_F10)
    ensaio, referencia = conferir_vinculo(manifesto, ensaio_arg)
    config_path = ensaio / 'configuracao.yaml'
    exigir(config_path.is_file(), 'Configuracao da execucao OOF nao encontrada.')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
    exigir(config['reducao']['metodo'].upper() == 'PCA' and config['pca']['n_components'] == [1],
           'Ensaio de origem nao corresponde a PCA 1D.')
    tabela = resultados[1]['tabela']
    metricas = resultados[1]['metricas']
    y = tabela['y_true'].to_numpy(dtype=int)
    p = tabela['prob_fraude'].to_numpy(dtype=float)
    tau = float(referencia['limiar_mcc_oof_congelado'])
    mcc = float(matthews_corrcoef(y, p >= tau))
    ap = float(average_precision_score(y, p))
    ll = float(log_loss(y, p, labels=[0, 1]))
    score = (ap + (mcc + 1) / 2 + np.exp(-ll)) / 3
    for nome, atual, esperado in [
        ('AP', ap, metricas['ap']),
        ('MCC no limiar OOF', mcc, metricas['mcc_limiar_oof']),
        ('LogLoss', ll, metricas['logloss']),
        ('Score', score, metricas['score_composto_limiar_oof']),
    ]:
        exigir(np.isclose(atual, esperado, atol=1e-9, rtol=1e-7), f'{nome} diferente do F10 salvo.')
    exigir(len(tabela) == metricas['n'] and set(tabela['fold'].unique()) == {10},
           'Tabela nao corresponde a uma amostra do F10.')
    escolha = ler_json(ensaio / 'vencedor/metricas/melhor_configuracao.json')
    print(f'[VINCULADO] Ensaio PCA 1D: {ensaio.relative_to(RAIZ)}')
    print(f'[F10] Amostra previamente avaliada: n={len(tabela)}; fraudes={int(y.sum())}; tau OOF={tau:.17g}')
    print('[APROVADO] Conferencia completa de hashes, metricas e origem. Sem Parquet, modelo ou nova inferencia.')
    return dict(ensaio=ensaio, referencia=referencia, config=config, escolha=escolha,
                met=metricas, tabela=tabela, origem=FONTE_F10, manifesto=manifesto)


def alterar_textos_figura(fig):
    """Troca apenas os rótulos de população; preserva estilos e coordenadas."""
    from matplotlib.text import Text
    substituicoes = [
        ('Resultados da validação OOF', 'Resultados do teste F10 (τ* OOF congelado)'),
        ('Curva Precision–Recall do modelo selecionado em OOF',
         'Curva Precision–Recall do modelo avaliado em F10'),
        ('AP calculada sobre as predições OOF.',
         'AP calculada sobre as predições F10.'),
        ('MCC calculado no limiar ótimo das predições OOF.',
         'MCC calculado em F10 com limiar τ* congelado em OOF.'),
        ('LogLoss calculado sobre as predições OOF.',
         'LogLoss calculado sobre as predições F10.'),
        ('OOF F1–F9', 'F10 estratificado 10%'),
        ('F10 não utilizado', 'F10 avaliado sem reajustar modelo'),
        ('Limiar ótimo do MCC (τ*)', 'Limiar τ* congelado em OOF'),
        ('Limiar ótimo (τ*)', 'Limiar τ* OOF (fixo)'),
        ('MCC no limiar ótimo', 'MCC no limiar OOF'),
        ('Limiar ótimo τ*', 'Limiar OOF τ*'),
        ('τ* MCC', 'τ* OOF (fixo)'),
    ]
    for texto in fig.findobj(match=Text):
        valor = texto.get_text()
        if not valor:
            continue
        novo = valor
        for antes, depois in substituicoes:
            novo = novo.replace(antes, depois)
        if novo != valor:
            texto.set_text(novo)
    # A origem deve ficar visivel sem adulterar scores nem ponto operacional.
    fig.text(.985, .010, 'F10 • AMOSTRAL • EXPLORATÓRIO', ha='right',
             color='#8497af', fontsize=7.8)


def figuras_usando_mesmo_codigo(dados: dict, pasta_figuras: Path):
    """Reutiliza EXATAMENTE as rotinas aprovadas do OOF, sem mudar src."""
    from src import visualizacao_oof as vo
    config = dados['config']
    tabela = dados['tabela']
    met = dados['met']
    y = tabela['y_true'].to_numpy(dtype=np.int8)
    p = tabela['prob_fraude'].to_numpy(dtype=float)
    escolha = {'limiar_mcc_oof': float(dados['escolha']['limiar_mcc_oof'])}
    vencedor = pd.Series({
        'ap': float(met['ap']),
        'mcc': float(met['mcc_limiar_oof']),
        'logloss': float(met['logloss']),
        'score': float(met['score_composto_limiar_oof']),
    })
    op_salvar, op_csv = vo._salvar, vo._csv_protegido

    def salvar(fig, destino, *, dpi):
        alterar_textos_figura(fig)
        return op_salvar(fig, destino, dpi=dpi)

    def csv_f10(tabela_csv, destino):
        tabela_csv = tabela_csv.copy()
        tabela_csv.rename(columns={'n_oof': 'n_f10', 'prevalencia_oof': 'prevalencia_f10'}, inplace=True)
        for coluna in tabela_csv.select_dtypes(include=['object', 'string']).columns:
            tabela_csv[coluna] = tabela_csv[coluna].replace({
                'OOF_F1_F9': 'F10_AMOSTRAL_10PCT',
            })
        return op_csv(tabela_csv, destino)

    # Modificação TEMPORÁRIA apenas no processo atual. Ao sair, restaura.
    vo._salvar = salvar
    vo._csv_protegido = csv_f10
    try:
        ext, dpi = 'png', int(config['visualizacao']['padrao'].get('dpi', 180))
        for nome, func in [
            ('00_painel_score_composto', lambda: vo._plot_painel_00(config,y,p,vencedor,escolha,pasta_figuras,ext,dpi,True)),
            ('01_curva_precision_recall', lambda: vo._plot_pr(config,y,p,vencedor,escolha,pasta_figuras,ext,dpi,True)),
            ('02_mcc_por_limiar', lambda: vo._plot_mcc(config,y,p,vencedor,escolha,pasta_figuras,ext,dpi,True)),
            ('03_matriz_confusao_05', lambda: vo._matriz(config,y,p,.5,'matriz_confusao_05',pasta_figuras,ext,dpi,True)),
            ('04_matriz_confusao_mcc', lambda: vo._matriz(config,y,p,escolha['limiar_mcc_oof'],'matriz_confusao_mcc',pasta_figuras,ext,dpi,True)),
            ('05_distribuicao_probabilidades',lambda: vo._distribuicao(config,y,p,escolha,pasta_figuras,ext,dpi,True)),
        ]:
            print('[GRAFICO]', nome, flush=True)
            func()
    finally:
        vo._salvar = op_salvar
        vo._csv_protegido = op_csv
    exigir(all((pasta_figuras / a).is_file() for a in GRAFICOS), 'Nem todos os seis graficos foram gerados.')


def copiar_sem_sobrescrever(origem: Path, destino: Path):
    if destino.is_file():
        exigir(sha256(origem) == sha256(destino),
               f'Arquivo existente com conteudo diferente: {destino}. Preservado.')
        return
    exigir(not destino.exists(), 'Destino ocupado por arquivo incompativel: ' + str(destino))
    shutil.copy2(origem, destino)
    exigir(sha256(origem) == sha256(destino), 'Falha de integridade ao copiar: '+str(destino))


def gerar_html(ensaio: Path, destino: Path, csv_hash: str, met_hash: str) -> str:
    import html
    lista = ''.join(
        f'<article><h3>{html.escape(nome.replace("_", " ").replace(".png", ""))}</h3>'
        f'<div class="par">'
        f'<figure><img src="../../visualizacao/oof/{nome}" alt="Gráfico OOF"><figcaption>Campeão OOF • F1–F9</figcaption></figure>'
        f'<figure><img src="../graficos_gerados/{nome}" alt="Gráfico F10"><figcaption>Teste final F10 • 10% estratificado</figcaption></figure>'
        f'</div></article>' for nome in GRAFICOS
    )
    return f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PCA 1D — OOF × F10</title><style>body{{font:15px system-ui,sans-serif;background:#f5f7fa;color:#172236;max-width:1440px;margin:auto;padding:24px}}h1{{font-size:26px}}h3{{margin:12px 0}}
article{{background:white;border:1px solid #d7dee6;border-radius:10px;padding:14px;margin:20px 0}}.par{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}figure{{margin:2px;min-width:0}}img{{width:100%;height:auto;object-fit:contain}}figcaption{{text-align:center;color:#496179;font-weight:600;margin:6px}}.alerta{{background:#e8edf4;padding:14px;border-left:4px solid #355877}}@media(max-width:900px){{.par{{grid-template-columns:1fr}}}}</style></head>
<body><h1>PCA 1D — Comparação visual OOF × F10</h1><p class="alerta">Mesma função gráfica, resultados distintos. F10 exploratório (10%, 18.523 registros), sem novo treino, sem reotimização de τ* e sem seleção de modelo. Os diagnósticos do treinamento permanecem apenas na galeria OOF.</p><p>Predições F10 SHA-256: <code>{csv_hash}</code> · Métricas SHA-256: <code>{met_hash}</code></p>{lista}</body></html>'''


def gerar(dados: dict):
    ensaio = dados['ensaio']
    final = ensaio / 'vencedor/teste_final'
    figuras = final / 'graficos_gerados'
    galeria = final / 'galeria'
    arq_html = galeria / 'index.html'
    origem_p = dados['origem'] / 'PCA_1D/predicoes_f10.csv'
    origem_m = dados['origem'] / 'PCA_1D/metricas_f10.json'
    manifesto_saida = final / 'manifesto_espelho_graficos_f10.json'
    # Bloqueio antes da primeira escrita, inclusive reexecuções.
    exigir(not manifesto_saida.exists(), 'Esta exportacao ja foi concluida. Nao repetir: ' + str(manifesto_saida))
    exigir(not arq_html.exists(), 'Galeria F10 existente. Preservada sem sobrescrever.')
    for nome in GRAFICOS:
        exigir(not (figuras / nome).exists(), f'Figura F10 ja existe e nao sera sobrescrita: {nome}')
    print('[PREPARADO] Saida exclusiva do F10:', final, flush=True)
    figuras.mkdir(parents=True, exist_ok=True)
    galeria.mkdir(parents=True, exist_ok=True)
    copiar_sem_sobrescrever(origem_p, final / 'predicoes_f10.csv')
    copiar_sem_sobrescrever(origem_m, final / 'metricas_f10.json')
    figuras_usando_mesmo_codigo(dados, figuras)
    html_galeria = gerar_html(ensaio, final, sha256(origem_p), sha256(origem_m))
    arq_html.write_text(html_galeria, encoding='utf-8')
    manifest = {
        'versao': 'F10_LAYOUT_OOF_ESPELHO_V2',
        'fase': 'teste_final',
        'modelo_origem_oof': str(ensaio.relative_to(RAIZ)).replace('\\','/'),
        'amostra': 'F10_10PCT_ESTRATIFICADA_EXPLORATORIA',
        'n': int(dados['met']['n']),
        'limiar_mcc_congelado_oof': float(dados['escolha']['limiar_mcc_oof']),
        'sem_retreino': True,
        'sem_nova_inferencia': True,
        'sem_reotimizacao_limiar': True,
        'original_f10': str(dados['origem'].relative_to(RAIZ)).replace('\\','/'),
        'predicoes_sha256': sha256(final / 'predicoes_f10.csv'),
        'metricas_sha256': sha256(final / 'metricas_f10.json'),
        'graficos': {nome:sha256(figuras / nome) for nome in GRAFICOS},
        'galeria': 'galeria/index.html',
        'nao_reproduzidos_com_scores_apenas': [
            '06_comparacao_hiperparametros', '07_metricas_por_fold',
            '09_painel_parametros_gaussianos', '11_pca_cotovelo',
            '12_pca_projecao_PC1', 'GMM01_densidade_marginal_1d_PC1',
            'GMM04_dominancia_1d', 'GMM07_sensibilidade_1d_comparativo',
            'GMM08_mapa_risco_1d_comparativo', 'GMM18_convergencia_em_inicializacoes',
            'PCA_BIPLOT_1D', 'PCA_COEFICIENTES_PC1', 'PCA_OBSERVACOES_PC1',
        ],
    }
    manifesto_saida.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print('[CONCLUIDO] Figuras padrao OOF com dados F10: ' + str(figuras))
    print('[GALERIA] ' + str(arq_html))
    print('[MANIFESTO] ' + str(manifesto_saida))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modo = parser.add_mutually_exclusive_group(required=True)
    modo.add_argument('--verificar', action='store_true')
    modo.add_argument('--gerar', action='store_true')
    parser.add_argument('--ensaio', type=Path, default=None)
    args = parser.parse_args()
    dados = verificar(args.ensaio)
    if args.verificar:
        print('[SOMENTE VERIFICACAO] Nenhum arquivo alterado, nenhum F10 reavaliado.')
    else:
        gerar(dados)


if __name__ == '__main__':
    main()
