# -*- coding: utf-8 -*-
"""V4: seis graficos preditivos TSNE+GMM 1D para o F10 JA avaliado na V3.

--planejar: apenas contratos/metadados e hashes, sem ler scores do F10.
--executar: somente CSVs/JSONs de F10 congelados pela V3; NAO le Parquet,
NAO executa fit/transform, NAO reotimiza limiar, NAO modifica PCA/OOF/V3.

Saida transacional e nova: vencedor/teste_final/graficos_preditivos_v4.
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

from scripts.execucao import v3_avaliacao_f10_tsne1d as v3
from src import visualizacao_oof as vo
from src.avaliacao_final_tsne import avaliar_scores_com_limiar_oof_congelado

ROOT = Path(__file__).resolve().parents[2]
DEST_NAME = 'graficos_preditivos_v4'
PNG_FILES = (
    '00_painel_score_composto.png',
    '01_curva_precision_recall.png',
    '02_mcc_por_limiar.png',
    '03_matriz_confusao_05.png',
    '04_matriz_confusao_mcc.png',
    '05_distribuicao_probabilidades.png',
)


def exigir(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def sha(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ler_json(path):
    with Path(path).open('r', encoding='utf-8-sig') as f:
        return json.load(f)


def preflight():
    info = v3.preflight()  # valida identidades V1/V2, SHA do modelo e indices do PCA.
    meta_v3 = v3.verify_existing(info)  # valida todos hashes V3, sem ler transacoes.
    cfg = info['pre'].configuracao
    escolha = info['pre'].escolha_oof
    exigir(cfg['reducao']['metodo'].upper() == 'TSNE' and
           escolha['configuracao']['reducao']['n_components'] == 1,
           'V4 exclusiva de TSNE 1D.')
    exigir(meta_v3['status'] == 'APROVADA' and
           meta_v3['limiar_tau_mcc_oof_congelado'] == escolha['limiar_mcc_oof'],
           'V3 nao validada ou limiar diferente do OOF.')
    exigir(meta_v3['n_f10'] == 18523 and meta_v3['nao_reotimizou_limiar_f10'] is True,
           'Contrato amostral F10 diferente de V3.')
    data = info['output']
    out = data.parent / DEST_NAME
    lock = ROOT / 'outputs' / '.graficos_tsne_f10_v4_em_andamento.lock'
    return {'info': info, 'v3_meta': meta_v3, 'cfg': cfg,
            'escolha': escolha, 'v3': data, 'out': out, 'lock': lock}


def saida_valida(info):
    out = info['out']
    meta_path = out / 'manifesto_v4.json'
    exigir(meta_path.is_file(), 'Saida V4 ja existe sem manifesto. Nao sobrescrever.')
    m = ler_json(meta_path)
    exigir(m.get('status') == 'APROVADA' and
           m.get('id_candidato_oof') == info['escolha']['id_candidato'] and
           m.get('limiar_mcc_oof_congelado') == info['escolha']['limiar_mcc_oof'] and
           m.get('sha256_predicoes_v3') == sha(info['v3'] / 'predicoes_f10.csv') and
           m.get('sha256_metricas_v3') == sha(info['v3'] / 'metricas_f10.json'),
           'Manifesto V4 nao corresponde ao V3 congelado.')
    files = m.get('sha256_arquivos', {})
    exigir(all(name in files for name in PNG_FILES), 'V4 sem os seis PNGs obrigatorios.')
    for name, hash_expected in files.items():
        path = out / name
        exigir(path.is_file() and sha(path) == hash_expected,
               f'Arquivo V4 ausente/alterado: {name}')
    return m


def planejar(info):
    print('=== V4 - GRAFICOS PREDITIVOS TSNE+GMM 1D F10 ===', flush=True)
    print('EXPERIMENTO_TSNE=' + str(info['info']['info']['pasta']), flush=True)
    print('FONTE_V3=' + str(info['v3']), flush=True)
    print('MODELO_V2_SHA256=' + info['info']['modelo_hash'], flush=True)
    print('CANDIDATO_OOF=' + info['escolha']['id_candidato'], flush=True)
    print('TAU_OOF_CONGELADO=' + format(info['escolha']['limiar_mcc_oof'], '.17g'), flush=True)
    print('N_F10=' + str(info['v3_meta']['n_f10']), flush=True)
    print('DESTINO_GRAFICOS=' + str(info['out']), flush=True)
    print('GRAFICOS_PLANEJADOS=' + ','.join(PNG_FILES), flush=True)
    print('MODELO_REAJUSTADO=NAO', flush=True)
    print('SCORES_RECALCULADOS=NAO', flush=True)
    print('LIMIAR_F10_OTIMIZADO=NAO', flush=True)


@contextmanager
def rotulos_f10():
    """Apenas alteracao de textos de figuras/CSV na V4, sem tocar codigo OOF."""
    from matplotlib.text import Text
    original_save = vo._salvar
    original_csv = vo._csv_protegido
    original_ops = vo._pontos_operacionais_mcc
    pares = (
        ('Curva Precision–Recall do modelo selecionado em OOF',
         'Curva Precision–Recall do modelo em F10'),
        ('Resultados da validação OOF', 'Resultados do teste F10 exploratório'),
        ('AP calculada sobre as predições OOF.', 'AP calculada sobre as predições F10.'),
        ('MCC calculado no limiar ótimo das predições OOF.',
         'MCC no F10 com limiar τ* congelado em OOF.'),
        ('LogLoss calculado sobre as predições OOF.',
         'LogLoss calculado sobre as predições F10.'),
        ('OOF F1–F9', 'F10 exploratório'),
        ('Limiar ótimo do MCC (τ*)', 'Limiar τ* congelado em OOF'),
        ('Limiar ótimo (τ*)', 'Limiar τ* congelado em OOF'),
        ('MCC no limiar ótimo', 'MCC em τ* OOF congelado'),
        ('LIMIAR ÓTIMO DO MCC:', 'LIMIAR τ* CONGELADO EM OOF:'),
    )

    def salvar_figura(fig, dest, *, dpi):
        for text in fig.findobj(match=Text):
            atual = text.get_text()
            alterado = atual
            for antigo, novo in pares:
                alterado = alterado.replace(antigo, novo)
            if alterado != atual:
                text.set_text(alterado)
        return original_save(fig, dest, dpi=dpi)

    def salvar_csv(tab, dest):
        tab = tab.copy()
        tab.rename(columns={'n_oof': 'n_f10', 'prevalencia_oof': 'prevalencia_f10'},
                   inplace=True)
        if 'origem' in tab:
            tab['origem'] = 'F10_EXPLORATORIO_REUTILIZADO_TAU_OOF_CONGELADO'
        if 'corte' in tab:
            tab['corte'] = tab['corte'].replace({'MCC OOF': 'Tau MCC OOF congelado'})
        return original_csv(tab, dest)

    def operacional_mcc(y, p, tau):
        ops = original_ops(y, p, tau)
        for item in ops:
            item['origem'] = 'F10_EXPLORATORIO_REUTILIZADO_TAU_OOF_CONGELADO'
            if item['corte'] == 'MCC OOF':
                item['corte'] = 'Tau MCC OOF congelado'
        return ops

    vo._salvar = salvar_figura
    vo._csv_protegido = salvar_csv
    vo._pontos_operacionais_mcc = operacional_mcc
    try:
        yield
    finally:
        vo._salvar = original_save
        vo._csv_protegido = original_csv
        vo._pontos_operacionais_mcc = original_ops


def dados_v3(info):
    df = pd.read_csv(info['v3'] / 'predicoes_f10.csv')
    met = ler_json(info['v3'] / 'metricas_f10.json')
    n = int(info['v3_meta']['n_f10'])
    exigir(len(df) == n, 'N de predicoes V3 diferente de manifesto.')
    exigir({'linha_original', 'fold', 'y_true', 'prob_fraude',
            'predicao_tau_0_50', 'predicao_tau_mcc_oof'}.issubset(df.columns),
           'CSV V3 com colunas obrigatorias ausentes.')
    ids = df['linha_original'].to_numpy(dtype=np.int64)
    y = df['y_true'].to_numpy(dtype=np.uint8)
    p = df['prob_fraude'].to_numpy(dtype=np.float64)
    tau = float(info['escolha']['limiar_mcc_oof'])
    exigir(np.array_equal(ids, info['info']['ids']) and
           np.all(df['fold'].to_numpy(dtype=int) == 10) and
           np.all(np.isin(y, (0, 1))) and
           np.all(np.isfinite(p)) and
           np.all((p >= 0) & (p <= 1)),
           'V3: indices, folds, labels ou scores invalidos.')
    exigir(np.array_equal(df['predicao_tau_0_50'].to_numpy(dtype=np.uint8),
                            (p >= .5).astype(np.uint8)) and
           np.array_equal(df['predicao_tau_mcc_oof'].to_numpy(dtype=np.uint8),
                          (p >= tau).astype(np.uint8)),
           'Predicoes V3 nao correspondem a limiares fixos.')
    recalculado = avaliar_scores_com_limiar_oof_congelado(y, p, limiar_mcc_oof=tau)
    for chave in ('ap', 'mcc_limiar_oof', 'logloss', 'score_composto_limiar_oof'):
        exigir(np.isclose(recalculado[chave], met[chave], atol=1e-10, rtol=1e-10),
               f'Metrica V3 divergente: {chave}.')
    exigir(met['limiar_otimizado_no_f10'] is False and
           met['limiar_mcc_oof_congelado'] == tau,
           'Limiar alterado no V3.')
    return y, p, met, recalculado


def executar(info):
    planejar(info)
    if info['out'].exists():
        m = saida_valida(info)
        print('V4_STATUS=JA_APROVADA_SAIDA_VALIDADA', flush=True)
        print('V4_MCC_TAU_OOF=' + format(m['mcc_f10_tau_oof'], '.7f'), flush=True)
        return 0
    exigir(not info['lock'].exists(), 'Lock V4 existente. Inspecione antes de remover.')
    with info['lock'].open('x', encoding='utf-8') as l:
        l.write('Visualizacao V4 TSNE F10; nao executar em paralelo.\n')
    try:
        y, p, met, recalculado = dados_v3(info)
        n = len(y)
        tau = float(info['escolha']['limiar_mcc_oof'])
        n_pred_05 = int(np.count_nonzero(p >= .5))
        n_pred_tau = int(np.count_nonzero(p >= tau))
        mcc = float(matthews_corrcoef(y, p >= tau))
        figs = info['out']
        staging = info['out'].parent / ('.' + DEST_NAME + '_incompleta')
        exigir(not staging.exists(), 'Pasta staging V4 anterior existe; preservar e investigar.')
        staging.mkdir(parents=False, exist_ok=False)
        cfg = info['cfg']
        escolha = {'limiar_mcc_oof': tau}
        vencedor = pd.Series({
            'ap': float(met['ap']),
            'mcc': float(met['mcc_limiar_oof']),
            'logloss': float(met['logloss']),
            'score': float(met['score_composto_limiar_oof']),
            'prevalencia': float(met['prevalencia']),
        })
        dpi = int(cfg['visualizacao']['padrao'].get('dpi', 180))
        print('[V4] Gerando seis graficos a partir de V3, sem fit.', flush=True)
        with rotulos_f10():
            graficos = (
                ('00_painel_score_composto',
                 lambda: vo._plot_painel_00(cfg, y, p, vencedor, escolha, staging, 'png', dpi, True)),
                ('01_curva_precision_recall',
                 lambda: vo._plot_pr(cfg, y, p, vencedor, escolha, staging, 'png', dpi, True)),
                ('02_mcc_por_limiar',
                 lambda: vo._plot_mcc(cfg, y, p, vencedor, escolha, staging, 'png', dpi, True)),
                ('03_matriz_confusao_05',
                 lambda: vo._matriz(cfg, y, p, .5, 'matriz_confusao_05', staging, 'png', dpi, True)),
                ('04_matriz_confusao_mcc',
                 lambda: vo._matriz(cfg, y, p, tau, 'matriz_confusao_mcc', staging, 'png', dpi, True)),
                ('05_distribuicao_probabilidades',
                 lambda: vo._distribuicao(cfg, y, p, escolha, staging, 'png', dpi, True)),
            )
            for nome, func in graficos:
                func()
                path = staging / (nome + '.png')
                exigir(path.is_file() and path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n',
                       f'PNG V4 ausente/invalido: {nome}')
                print('[V4 GRAFICO] ' + nome, flush=True)
        outputs = sorted(path for path in staging.iterdir() if path.is_file())
        pngs = [p for p in outputs if p.suffix.lower() == '.png']
        exigir(set(p.name for p in pngs) == set(PNG_FILES),
               'Conjunto PNG diferente dos seis graficos aprovados.')
        sha_files = {p.name: sha(p) for p in outputs}
        matrix = recalculado['comparacao_limiares']['tau_mcc_oof']['matriz_confusao']
        report = {
            'versao': 'V4_GRAFICOS_PREDITIVOS_TSNE_F10_V1',
            'status': 'APROVADA',
            'tipo': 'VISUALIZACAO_F10_EXPLORATORIO_REUTILIZADO',
            'origem': 'V3_F10_18523_SEM_RECOMPUTAR_PROBABILIDADES',
            'n_f10': n,
            'id_candidato_oof': info['escolha']['id_candidato'],
            'limiar_mcc_oof_congelado': tau,
            'sha256_modelo_v2': info['info']['modelo_hash'],
            'sha256_predicoes_v3': sha(info['v3'] / 'predicoes_f10.csv'),
            'sha256_metricas_v3': sha(info['v3'] / 'metricas_f10.json'),
            'n_fraudes': int(np.sum(y)),
            'n_preditos_positivos_tau_oof': n_pred_tau,
            'n_preditos_positivos_0_50': n_pred_05,
            'mcc_f10_tau_oof': mcc,
            'matriz_confusao_tau_oof': matrix,
            'score_min': float(p.min()),
            'score_max': float(p.max()),
            'score_mediana': float(np.median(p)),
            'graficos_png': list(PNG_FILES),
            'sha256_arquivos': sha_files,
            'sem_refit': True, 'sem_reotimizacao_limiar_f10': True,
            'data_utc': datetime.now(timezone.utc).isoformat(),
        }
        with (staging / 'manifesto_v4.json').open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
        exigir(not figs.exists(), 'Destino final V4 ja existe; nao sobrescrever.')
        os.replace(staging, figs)
        saida_valida(info)
        print('V4_N_PREVISTOS_POSITIVOS_TAU_OOF=' + str(n_pred_tau), flush=True)
        print('V4_N_PREVISTOS_POSITIVOS_050=' + str(n_pred_05), flush=True)
        print('V4_CM_TN_FP_FN_TP=' + ','.join(str(matrix[k]) for k in ('tn','fp','fn','tp')), flush=True)
        print('V4_SCORE_MIN=' + format(p.min(), '.7f'), flush=True)
        print('V4_SCORE_MAX=' + format(p.max(), '.7f'), flush=True)
        print('V4_MCC_TAU_OOF=' + format(mcc, '.7f'), flush=True)
        print('V4_GRAFICOS_VERIFICADOS=6/6', flush=True)
        print('V4_STATUS=APROVADA', flush=True)
        print('OOF_REEXECUTADO=NAO', flush=True)
        print('V3_PREDICOES_MODIFICADAS=NAO', flush=True)
        return 0
    finally:
        info['lock'].unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--planejar', action='store_true')
    mode.add_argument('--executar', action='store_true')
    args = parser.parse_args(argv)
    try:
        info = preflight()
        if args.planejar:
            planejar(info)
            print('V4_STATUS=PLANEJAMENTO_APROVADO_SEM_LEITURA_DE_SCORES_F10', flush=True)
            return 0
        return executar(info)
    except Exception as e:
        print('V4_STATUS=BLOQUEADA', file=sys.stderr, flush=True)
        print('V4_MOTIVO=' + str(e), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
