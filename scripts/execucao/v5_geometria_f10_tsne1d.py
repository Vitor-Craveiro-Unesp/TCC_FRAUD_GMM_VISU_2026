# -*- coding: utf-8 -*-
"""V5: geometria GMM01/GMM04/GMM07/GMM08 t-SNE1D no F10 ja avaliado.

Read-only em --planejar. --executar utiliza exclusivamente CSV/JSON V3 e
modelo persistido V2. Nao le Parquet/folds, nao ajusta encoder/TSNE/GMM/RH,
nao altera OOF, PCA, V2/V3/V4 e nao procura novo limiar no F10.

Os parametros GMM sao de F1-F9; somente os pontos/metricas sao F10.
Destino transacional novo: teste_final/graficos_geometricos_v5.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from scripts.execucao import v4_graficos_preditivos_tsne1d_f10 as v4
from src.treinamento_final_tsne import carregar_pacote_confiavel

ROOT = Path(__file__).resolve().parents[2]
DEST_NAME = 'graficos_geometricos_v5'
PNG_FILES = (
    'GMM01_densidade_marginal_1d_TSNE1.png',
    'GMM04_dominancia_1d.png',
    'GMM07_sensibilidade_1d_comparativo.png',
    'GMM08_mapa_risco_1d_comparativo.png',
)
LOCK_NAME = '.graficos_tsne_f10_v5_em_andamento.lock'


def exigir(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def sha(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def ler_json(path):
    with Path(path).open('r', encoding='utf-8-sig') as f:
        obj = json.load(f)
    exigir(isinstance(obj, dict), f'JSON invalido: {path}')
    return obj


def preflight():
    # A V4 valida integralmente hashes e identidade do V3/V2 e fonte PCA.
    q = v4.preflight()
    v4.saida_valida(q)
    cfg = q['cfg']
    escolha = q['escolha']
    info = q['info']
    exigir(cfg['reducao']['metodo'].upper() == 'TSNE' and
           escolha['configuracao']['reducao']['n_components'] == 1,
           'V5 exclusiva do TSNE+GMM 1D.')
    exigir(q['v3_meta']['n_f10'] == 18523 and
           q['v3_meta']['nao_reotimizou_limiar_f10'] is True,
           'V3 deve estar aprovado e com limiar congelado.')
    for nome in ('densidades_marginais', 'regioes_dominancia',
                 'sensibilidade_fronteira', 'mapa_risco_fronteira'):
        exigir(nome in cfg['visualizacao']['graficos'],
               f'Configuracao original nao contem grafico {nome}.')
    destino = info['output'].parent / DEST_NAME
    return {
        'q': q, 'info': info, 'cfg': cfg, 'escolha': escolha,
        'v3': q['v3'], 'v4': q['out'], 'out': destino,
        'staging': destino.parent / ('.' + DEST_NAME + '_incompleta'),
        'lock': ROOT / 'outputs' / LOCK_NAME,
    }


def mostrar_plano(st):
    q = st['q']
    print('=== V5 - GEOMETRIA GMM F10 TSNE 1D ===', flush=True)
    print('EXPERIMENTO_TSNE=' + str(st['info']['info']['pasta']), flush=True)
    print('FONTE_V3=' + str(st['v3']), flush=True)
    print('FONTE_V4=' + str(st['v4']), flush=True)
    print('CANDIDATO_CONGELADO=' + st['escolha']['id_candidato'], flush=True)
    print('TAU_OOF_CONGELADO=' + format(st['escolha']['limiar_mcc_oof'], '.17g'), flush=True)
    print('N_F10_PAREADO=' + str(q['v3_meta']['n_f10']), flush=True)
    print('MODELO_V2_SHA256=' + st['info']['modelo_hash'], flush=True)
    print('DESTINO_V5=' + str(st['out']), flush=True)
    print('GRAFICOS_PLANEJADOS=' + ','.join(PNG_FILES), flush=True)
    print('AJUSTE_TSNE_GMM=NAO', flush=True)
    print('OOF_REEXECUTADO=NAO', flush=True)
    print('NOVAS_PREDICOES_F10=NAO', flush=True)
    print('LIMIAR_F10_OTIMIZADO=NAO', flush=True)


def verificar_saida(st):
    pasta = st['out']
    manifesto = pasta / 'manifesto_v5.json'
    exigir(manifesto.is_file(), 'Pasta V5 presente sem manifesto; nao sobrescrever.')
    m = ler_json(manifesto)
    exigir(m.get('status') == 'APROVADA' and
           m.get('id_candidato_oof') == st['escolha']['id_candidato'] and
           m.get('n_f10') == 18523 and
           m.get('limiar_mcc_oof_congelado') == st['escolha']['limiar_mcc_oof'] and
           m.get('sha256_modelo_v2') == st['info']['modelo_hash'] and
           m.get('sha256_predicoes_v3') == sha(st['v3'] / 'predicoes_f10.csv') and
           m.get('sha256_coordenadas_v3') == sha(st['v3'] / 'coordenadas_TSNE1_f10.csv') and
           m.get('sha256_metricas_v3') == sha(st['v3'] / 'metricas_f10.json'),
           'V5 nao pertence aos resultados F10 aprovados.')
    arquivos = m.get('sha256_arquivos', {})
    exigir(isinstance(arquivos, dict) and set(PNG_FILES).issubset(arquivos),
           'Manifesto V5 nao lista os quatro PNGs.')
    for nome, esperado in arquivos.items():
        arq = pasta / nome
        exigir(arq.is_file() and sha(arq) == esperado, f'Arquivo V5 alterado: {nome}')
    pngs = {p.name for p in pasta.glob('*.png')}
    exigir(pngs == set(PNG_FILES), 'Conjunto de PNGs V5 nao corresponde a 4/4.')
    return m


def carregar_artefatos(st):
    pred = pd.read_csv(st['v3'] / 'predicoes_f10.csv')
    coord = pd.read_csv(st['v3'] / 'coordenadas_TSNE1_f10.csv')
    met = ler_json(st['v3'] / 'metricas_f10.json')
    nomes_pred = {'linha_original', 'fold', 'y_true', 'prob_fraude',
                  'predicao_tau_0_50', 'predicao_tau_mcc_oof'}
    exigir(nomes_pred.issubset(pred.columns), 'V3 predicoes sem colunas obrigatorias.')
    exigir(list(coord.columns) == ['linha_original', 'TSNE1'],
           'V3 coordenadas nao sao TSNE1 no formato original.')
    n = st['q']['v3_meta']['n_f10']
    ids = st['info']['ids']
    tau = float(st['escolha']['limiar_mcc_oof'])
    exigir(len(pred) == len(coord) == len(ids) == n and
           np.array_equal(pred['linha_original'].to_numpy(dtype=np.int64), ids) and
           np.array_equal(coord['linha_original'].to_numpy(dtype=np.int64), ids) and
           np.all(pred['fold'].to_numpy(dtype=np.int64) == 10),
           'IDs/ordem F10 divergentes entre previsoes e TSNE1.')
    y = pred['y_true'].to_numpy(dtype=np.uint8)
    p = pred['prob_fraude'].to_numpy(dtype=np.float64)
    z = coord['TSNE1'].to_numpy(dtype=np.float64)
    exigir(set(np.unique(y).tolist()) == {0, 1} and
           np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all() and
           np.isfinite(z).all(), 'Y/scores/coordenadas nao finitos ou invalidos.')
    exigir(np.array_equal(pred['predicao_tau_0_50'].to_numpy(dtype=np.uint8),
                            (p >= .5).astype(np.uint8)) and
           np.array_equal(pred['predicao_tau_mcc_oof'].to_numpy(dtype=np.uint8),
                          (p >= tau).astype(np.uint8)),
           'Classes V3 diferentes dos limiares originais.')
    exigir(met.get('limiar_otimizado_no_f10') is False and
           met.get('limiar_mcc_oof_congelado') == tau and
           met.get('n') == n and
           np.isclose(matthews_corrcoef(y, p >= tau), met['mcc_limiar_oof'], atol=1e-12),
           'Metricas e limiar V3 divergentes.')
    return pd.DataFrame({'TSNE1': z}), y, p, met


@contextmanager
def adaptar_rotulos_f10(y, p, tau):
    """Reusa as figuras oficiais e troca SOMENTE os rotulos dos NOVOS PNGs.

    Nunca altera o modulo no disco; funcoes originais sao restauradas no finally.
    """
    from matplotlib.text import Text
    from src import visualizacao_geometrica as geo
    from src import visualizacao_gmm07 as gmm07
    original_mcc = geo._avaliar_cortes_oof
    original_save = geo._guardar
    original_rotulo = gmm07._rotulo_mcc
    mcc_fixo = float(matthews_corrcoef(y, p >= .5))
    mcc_tau = float(matthews_corrcoef(y, p >= tau))

    def conferir_av(av, corte):
        exigir(isinstance(av, dict) and
               av.get('origem') == 'f10_amostral_congelado' and
               corte == tau, 'GMM07/08 tentou usar outro corte/origem.')
        return mcc_fixo, mcc_tau

    def rotulo_f10(av):
        exigir(av.get('origem') == 'f10_amostral_congelado',
               'GMM07 sem origem F10 explicita.')
        return ('MCC F10', 'MCC calculado em F10; tau* congelado no OOF.')

    substituicoes = (
        ('B — Corte ótimo MCC OOF', 'B — Limiar τ* congelado OOF'),
        ('B — Corte ótimo MCC', 'B — Limiar τ* congelado OOF'),
        ('LIMIAR ÓTIMO DO MCC', 'LIMIAR τ* CONGELADO NO OOF'),
        ('Corte ótimo MCC OOF', 'Limiar τ* congelado OOF'),
        ('MCC OOF', 'MCC F10'),
        ('PC1 |', 'TSNE1 |'),
    )

    def salvar_visual(fig, dest, base, ext, dpi):
        for t in fig.findobj(match=Text):
            antigo = t.get_text()
            novo = antigo
            for origem, alvo in substituicoes:
                novo = novo.replace(origem, alvo)
            if novo != antigo:
                t.set_text(novo)
        return original_save(fig, dest, base, ext, dpi)

    geo._avaliar_cortes_oof = conferir_av
    geo._guardar = salvar_visual
    gmm07._rotulo_mcc = rotulo_f10
    try:
        yield mcc_fixo, mcc_tau
    finally:
        geo._avaliar_cortes_oof = original_mcc
        geo._guardar = original_save
        gmm07._rotulo_mcc = original_rotulo


def executar(st):
    mostrar_plano(st)
    if st['out'].exists():
        verificar_saida(st)
        print('V5_STATUS=JA_APROVADA_SAIDA_VALIDADA', flush=True)
        return 0
    exigir(not st['staging'].exists(), 'Pasta V5 de staging ja existe; preservar e auditar.')
    exigir(not st['lock'].exists(), 'Lock V5 existente; verificar processo antes de repetir.')
    with st['lock'].open('x', encoding='utf-8') as fd:
        fd.write('V5 graficos geometricos TSNE1D, sem fit; pasta protegida.\n')
    try:
        z, y, p, met = carregar_artefatos(st)
        print('[V5] Carregando somente o pacote V2 persistido, sem fit.', flush=True)
        pacote = carregar_pacote_confiavel(st['info']['modelo_dir'])
        exigir(pacote.n_treino_original == 166716 and
               pacote.escolha_oof['id_candidato'] == st['escolha']['id_candidato'] and
               pacote.limiar_oof == st['escolha']['limiar_mcc_oof'] and
               list(pacote.gmm.colunas_) == ['TSNE1'] and
               pacote.gmm.modelo_.covariance_type == 'full' and
               len(pacote.gmm.saidas_) == 4,
               'Pacote V2 nao corresponde ao TSNE1D/GMM FULL K4 congelado.')
        # Recalculo apenas de conferencia numerica, sem alterar V3 nem salvar novos scores.
        gamma = pacote.gmm.responsabilidades(z)
        from src.probabilidade import calcular_probabilidades
        p_conferencia = calcular_probabilidades(gamma, pacote.risco.riscos_).to_numpy(dtype=np.float64)
        max_dif = float(np.max(np.abs(p_conferencia - p)))
        exigir(max_dif <= 1e-6 and
               np.array_equal(p_conferencia >= pacote.limiar_oof, p >= pacote.limiar_oof) and
               np.array_equal(p_conferencia >= .5, p >= .5),
               f'Scores na geometria divergem do V3: max_dif={max_dif:.3g}.')
        dados = {
            'coord': z,
            'y': y,
            'scores': p,
            'gamma': gamma,
            'gmm': pacote.gmm,
            'rh': pacote.risco,
            'redutor': pacote.tsne,
            'limiar_oof': pacote.limiar_oof,
        }
        from src import visualizacao_geometrica as geo
        from src import visualizacao_gmm01 as gmm01
        from src import visualizacao_gmm04 as gmm04
        from src import visualizacao_gmm07 as gmm07
        dpi = int(st['cfg']['visualizacao']['padrao'].get('dpi', 180))
        # Pasta nova: nunca modificar as imagens OOF nem V4.
        st['staging'].mkdir(exist_ok=False)
        print('[V5] Renderizando GMM01 / GMM04 com gaussianas F1..F9.', flush=True)
        gmm01.gerar(st['cfg'], dados, st['staging'], 'png', dpi, True)
        gmm04.gerar(st['cfg'], dados, st['staging'], 'png', dpi, True)
        print('[V5] Renderizando GMM07 / GMM08 com MCC F10 e tau* OOF.', flush=True)
        with adaptar_rotulos_f10(y, p, pacote.limiar_oof) as (mcc_fixo, mcc_tau):
            escolha = {'limiar_mcc_oof': pacote.limiar_oof}
            avaliacao = {'origem': 'f10_amostral_congelado'}
            gmm07.gerar(st['cfg'], dados, escolha, st['staging'], 'png', dpi, True,
                        avaliacao=avaliacao)
            geo._fronteira(st['cfg'], dados, escolha, st['staging'], 'png', dpi, True,
                           avaliacao=avaliacao)
        # Conferir precisamente as quatro imagens PNG oficiais da V5.
        pngs = {p.name for p in st['staging'].glob('*.png')}
        exigir(pngs == set(PNG_FILES),
               f'PNG V5 esperado 4/4, encontrado {sorted(pngs)}.')
        for arq in PNG_FILES:
            path = st['staging'] / arq
            exigir(path.is_file() and path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n',
                   f'PNG V5 invalido: {arq}.')
            print('[V5 GRAFICO] ' + arq, flush=True)
        arquivos = sorted(p for p in st['staging'].iterdir() if p.is_file())
        hashes = {p.name: sha(p) for p in arquivos}
        resumo = {
            'versao': 'V5_GMM_GEOMETRIA_F10_TSNE1D_V1', 'status': 'APROVADA',
            'id_candidato_oof': st['escolha']['id_candidato'],
            'n_f10': int(len(y)), 'n_fraudes_f10': int(y.sum()),
            'limiar_mcc_oof_congelado': float(pacote.limiar_oof),
            'sha256_modelo_v2': st['info']['modelo_hash'],
            'sha256_predicoes_v3': sha(st['v3'] / 'predicoes_f10.csv'),
            'sha256_coordenadas_v3': sha(st['v3'] / 'coordenadas_TSNE1_f10.csv'),
            'sha256_metricas_v3': sha(st['v3'] / 'metricas_f10.json'),
            'modelo_gmm': 'FULL_K4_F1_F9_CONGELADO',
            'geometria_pontos': 'TSNE1_OOS_F10',
            'tipo_avaliacao': 'AMOSTRAL_EXPLORATORIO_REUTILIZADO',
            'n_preditos_positivos_tau_oof': int(np.count_nonzero(p >= pacote.limiar_oof)),
            'n_preditos_positivos_050': int(np.count_nonzero(p >= .5)),
            'mcc_f10_050': float(mcc_fixo),
            'mcc_f10_tau_oof': float(mcc_tau),
            'score_f10_max': float(p.max()),
            'max_diferenca_scores_geometria_v3': max_dif,
            'sem_ajuste_modelo': True,
            'sem_reotimizacao_limiar_f10': True,
            'graficos_png': list(PNG_FILES),
            'sha256_arquivos': hashes,
            'data_utc': datetime.now(timezone.utc).isoformat(),
            'observacao': ('GMM01 densidade nao e probabilidade de fraude. '
                           'GMM01/GMM04/GMM07/GMM08 usam parametros congelados F1-F9; '
                           'F10 fornece apenas observacoes OOS e MCC, nao ajusta modelos. '
                           'Intervalos gaussianos 95% nao sao fronteiras de classe.'),
        }
        with (st['staging'] / 'manifesto_v5.json').open('x', encoding='utf-8') as fd:
            json.dump(resumo, fd, ensure_ascii=False, indent=2, allow_nan=False)
            fd.write('\n')
        exigir(not st['out'].exists(), 'Destino V5 apareceu durante execucao; nao sobrescrever.')
        os.replace(st['staging'], st['out'])
        verificar_saida(st)
        print('V5_MCC_F10_TAU_OOF=' + format(mcc_tau, '.7f'), flush=True)
        print('V5_MCC_F10_050=' + format(mcc_fixo, '.7f'), flush=True)
        print('V5_PRED_POSITIVOS_TAU_OOF=' + str(int(np.sum(p >= pacote.limiar_oof))), flush=True)
        print('V5_SCORE_GEOMETRIA_DIF_MAX=' + format(max_dif, '.12g'), flush=True)
        print('V5_GRAFICOS_VERIFICADOS=4/4', flush=True)
        print('V5_STATUS=APROVADA', flush=True)
        print('MODELO_F1_F9_REAJUSTADO=NAO', flush=True)
        print('PREDICOES_V3_MODIFICADAS=NAO', flush=True)
        return 0
    finally:
        st['lock'].unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mod = parser.add_mutually_exclusive_group(required=True)
    mod.add_argument('--planejar', action='store_true')
    mod.add_argument('--executar', action='store_true')
    args = parser.parse_args(argv)
    try:
        st = preflight()
        mostrar_plano(st) if args.planejar else None
        if args.planejar:
            print('V5_STATUS=PLANEJAMENTO_APROVADO_SEM_DADOS_F10', flush=True)
            return 0
        return executar(st)
    except Exception as e:
        print('V5_STATUS=BLOQUEADA', file=sys.stderr, flush=True)
        print('V5_MOTIVO=' + str(e), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
