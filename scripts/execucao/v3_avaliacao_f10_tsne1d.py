# -*- coding: utf-8 -*-
"""V3: avaliacao F10 TSNE1D com os MESMOS 18523 indices F10 PCA.

--planejar: somente artefatos congelados/modelo e metadados, sem Parquet ou fit.
--executar: verifica Parquet/folds mestre, le somente F10 selecionado, realiza
inferencia OOS em lotes sem fit, calcula metricas com tau MCC OOF CONGELADO.

A saida V3 e transacional em uma pasta nova dentro de teste_final, para
preservar resultados anteriores e simplificar V4-V7.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

from src import orquestracao_oof as orq
from src.avaliacao_final_tsne import avaliar_scores_com_limiar_oof_congelado
from src.probabilidade import PreditorProbabilisticoGMM
from src.treinamento_final_tsne import carregar_pacote_confiavel
from scripts.execucao import v2_ajuste_final_tsne1d as v2

ROOT = Path(__file__).resolve().parents[2]
PCA_FOLDER = 'ensaio_pca_gmm_20261102_132101'
EXPECTED_SHA_INDICES = '0c767c6f55d36cc9620caa75c9605ec30adc65a63846a36e82d17b6dedb85a0c'
EXPECTED_SHA_MODELO = '770c9bb37d05416a8cac48a33e849a3944518d8801c6867b37e049f01834aeee'
EXPECTED_N_F10 = 18523
EXPECTED_N_DEV = 166716
EXPECTED_CANDIDATE = '757edec8b54d68'
V3_DIR = 'avaliacao_f10_v3'
LOCK = '.avaliacao_tsne_f10_v3_em_andamento.lock'


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as fd:
        data = json.load(fd)
    require(isinstance(data, dict), f'JSON invalido: {path}')
    return data


def sha_file(path):
    h = sha256()
    with Path(path).open('rb') as fd:
        for block in iter(lambda: fd.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_indices(pca):
    target = pca / 'vencedor' / 'teste_final'
    ids_path = target / 'indices_f10_amostra.npy'
    meta_path = target / 'manifesto_amostra_f10.json'
    require(ids_path.is_file() and meta_path.is_file(), 'Amostra PCA F10/manifesto ausente.')
    meta = read_json(meta_path)
    arr = np.load(ids_path, allow_pickle=False)
    require(arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer), 'Indices PCA F10 invalidos.')
    ids = np.asarray(arr, dtype=np.int64)
    require(ids.size == EXPECTED_N_F10 and bool(np.all(ids >= 0)) and bool(np.all(np.diff(ids) > 0)),
            'Amostra F10 mudou de quantidade/ordem.')
    digest = sha256(ids.tobytes()).hexdigest()
    require(digest == EXPECTED_SHA_INDICES and meta.get('sha256_indices') == digest,
            'SHA da amostra PCA F10 divergiu de V1.')
    require(meta.get('n') == EXPECTED_N_F10 and float(meta.get('fracao', -1)) == .1
            and int(meta.get('semente', -1)) == 42, 'Amostragem F10 nao e 10% seed42.')
    return ids, digest, ids_path


def preflight():
    info = v2.preflight()
    pre = info['pre']
    assert_folder = (ROOT / 'outputs' / PCA_FOLDER)
    require(assert_folder.is_dir(), 'Pasta PCA referencia ausente.')
    require(pre.plano.id_candidato_oof == EXPECTED_CANDIDATE and
            pre.plano.dimensao_tsne == 1 and pre.n_predicoes_oof == EXPECTED_N_DEV,
            'Candidato t-SNE/OOF diferente do que V1 aprovou.')
    require(pre.plano.fold_teste_reservado == 10 and pre.plano.folds_treino == tuple(range(1, 10)),
            'Divisao de folds alterada.')
    model_dir = info['destino']
    model_file = model_dir / 'modelo_final_f1_f9.joblib'
    model_manifest = model_dir / 'manifesto_modelo_final.json'
    require(model_file.is_file() and model_manifest.is_file(), 'V2 nao concluiu modelo+manifesto.')
    manifest = read_json(model_manifest)
    require(manifest.get('tipo') == 'TREINAMENTO_FINAL_TSNE_F1_F9' and
            manifest.get('teste_f10_executado') is False and
            manifest.get('id_candidato_oof') == EXPECTED_CANDIDATE and
            manifest.get('dimensao_tsne') == 1 and
            manifest.get('n_treino_original') == EXPECTED_N_DEV and
            manifest.get('limiar_mcc_oof_congelado') == pre.plano.limiar_mcc_oof_congelado,
            'Manifesto V2 nao corresponde ao modelo esperado.')
    hash_model = sha_file(model_file)
    require(hash_model == EXPECTED_SHA_MODELO and manifest.get('sha256_modelo_final') == hash_model,
            'SHA256 do modelo V2 diverge do aprovado no terminal.')
    prov = manifest.get('proveniencia', {})
    for key, value in pre.proveniencia.items():
        require(prov.get(key) == value, f'Proveniencia do modelo V2 divergente: {key}.')
    require(prov.get('sha256_predicoes_oof') == sha_file(info['npz_oof']),
            'Modelo V2 nao corresponde as predicoes OOF congeladas.')
    ids, digest, ids_path = checked_indices(assert_folder)
    root = info['destino'].parent
    require(root.is_dir(), 'Diretorio teste_final TSNE nao encontrado.')
    return {
        'info': info, 'pre': pre, 'modelo_dir': model_dir, 'modelo_hash': hash_model,
        'pca': assert_folder, 'ids': ids, 'sha_indices': digest,
        'ids_path': ids_path, 'output': root / V3_DIR,
        'staging': root / '.avaliacao_f10_v3_incompleta',
        'lock': ROOT / 'outputs' / LOCK,
    }


def verify_existing(info):
    dest = info['output']
    manifest_path = dest / 'manifesto_v3.json'
    require(manifest_path.is_file(), 'Saida V3 existente mas manifesto ausente; nao sobrescrever.')
    m = read_json(manifest_path)
    require(m.get('status') == 'APROVADA' and m.get('f10_exploratorio_reutilizado') is True,
            'Manifesto V3 anterior incompleto ou invalido.')
    require(m.get('id_candidato_oof') == EXPECTED_CANDIDATE and
            m.get('sha256_modelo_v2') == info['modelo_hash'] and
            m.get('sha256_indices_f10_pca') == info['sha_indices'] and
            m.get('n_f10') == EXPECTED_N_F10, 'Identidade do F10 V3 anterior divergiu.')
    files = m.get('sha256_arquivos', {})
    needed = {'predicoes_f10.csv', 'metricas_f10.json', 'coordenadas_TSNE1_f10.csv', 'indices_f10_amostra.npy'}
    require(set(files) == needed, 'Lista de artefatos V3 incompleta.')
    for name, digest in files.items():
        path = dest / name
        require(path.is_file() and sha_file(path) == digest, f'Arquivo V3 alterado: {name}')
    require(np.array_equal(np.load(dest / 'indices_f10_amostra.npy', allow_pickle=False), info['ids']),
            'Indices V3 diferentes do PCA.')
    return m


def show_plan(info):
    pre = info['pre']
    print('=== V3 - AVALIACAO F10 TSNE+GMM 1D ===', flush=True)
    print('EXPERIMENTO_TSNE=' + str(info['info']['pasta']), flush=True)
    print('PCA_REFERENCIA=' + str(info['pca']), flush=True)
    print('CANDIDATO_CONGELADO=' + pre.plano.id_candidato_oof, flush=True)
    print('TAU_OOF_CONGELADO=' + format(pre.plano.limiar_mcc_oof_congelado, '.17g'), flush=True)
    print('MODELO_V2_SHA256=' + info['modelo_hash'], flush=True)
    print('N_F10_PAREADO=' + str(len(info['ids'])), flush=True)
    print('SHA256_INDICES_F10=' + info['sha_indices'], flush=True)
    print('SAIDA_V3=' + str(info['output']), flush=True)
    print('TREINO_TSNE_GMM_REEXECUTADO=NAO', flush=True)
    print('LIMIAR_OTIMIZADO_NO_F10=NAO', flush=True)
    print('AVALIACAO_F10_EXPLORATORIA_REUTILIZADA=SIM', flush=True)


def infer_batched(pacote, frame, ids, target, batch_n, tau):
    count = len(frame)
    scores = np.empty(count, dtype=np.float64)
    tsne1 = np.empty(count, dtype=np.float64)
    # Nenhum fit ou fit_transform; reaproveitar SOMENTE o modelo congelado.
    pred = PreditorProbabilisticoGMM(pacote.gmm, pacote.risco)
    excluded = [target, pacote.configuracao['dados'].get('coluna_id'),
                *pacote.configuracao['dados'].get('colunas_excluir', [])]
    excluded = [c for c in excluded if c]
    for start in range(0, count, batch_n):
        stop = min(count, start + batch_n)
        x = frame.iloc[start:stop].drop(columns=excluded, errors='ignore').copy()
        x.index = pd.RangeIndex(len(x))
        require(list(x.columns) == pacote.encoder.colunas_, 'Ordem/nomes features diferentes do modelo V2.')
        enc = pacote.encoder.transform(x)
        prep = pacote.imputador.transform(enc)
        coord = pacote.tsne.transform(prep)
        require(list(coord.columns) == ['TSNE1'] and len(coord) == len(x),
                'Embedding OOS nao corresponde a TSNE1.')
        p = pred.prever_probabilidade(coord)
        arr = np.asarray(p, dtype=np.float64)
        zz = coord['TSNE1'].to_numpy(dtype=np.float64)
        require(arr.shape == zz.shape == (stop-start,) and np.isfinite(arr).all()
                and (arr >= 0).all() and (arr <= 1).all() and np.isfinite(zz).all(),
                'Scores/embedding F10 invalidos.')
        scores[start:stop] = arr
        tsne1[start:stop] = zz
        print(f'[V3 OOS] {stop}/{count} registros projetados sem fit.', flush=True)
    return scores, tsne1


def run(info, batch_n):
    if info['output'].exists():
        verify_existing(info)
        print('V3_STATUS=JA_APROVADA_SAIDA_VALIDADA', flush=True)
        return 0
    require(not info['staging'].exists(),
            'Pasta temporaria V3 parcial encontrada. Preserve e investigue antes de repetir.')
    lock = info['lock']
    with lock.open('x', encoding='utf-8') as fh:
        fh.write('V3 avaliacao TSNE OOS F10 pareada.\n')
    try:
        pre = info['pre']
        cfg = pre.configuracao
        parquet = ROOT / cfg['dados']['arquivo']
        print('[V3] Revalidando hashes reais do Parquet e do fold mestre.', flush=True)
        folds, fold_file, parquet_hash = orq._identificar_folds(ROOT, cfg, parquet)
        manifest = info['info']['manifesto_oof']
        require(parquet_hash == manifest['sha256_parquet'] and
                orq.hash_arquivo(fold_file) == manifest['sha256_folds'],
                'Fonte Parquet/folds alterados desde OOF; bloquear V3.')
        ids = info['ids']
        require(ids[-1] < len(folds) and np.all(folds[ids] == 10),
                'Amostra F10 PCA nao pertence integralmente ao fold 10 mestre.')
        print('F10_PERTENCIMENTO_FOLD_MESTRE=CONFIRMADO', flush=True)
        print('[V3] Lendo somente as transacoes selecionadas do F10.', flush=True)
        df = orq._ler_desenvolvimento(parquet, ids)
        require(len(df) == EXPECTED_N_F10, 'F10 lido incompleto.')
        target = cfg['dados']['coluna_alvo']
        y = df[target].to_numpy(dtype=np.uint8)
        require(np.isin(y, [0, 1]).all() and len(np.unique(y)) == 2,
                'Classes F10 invalidas.')
        # Conferir que os mesmos indices correspondem aos mesmos rotulos PCA.
        pca_predictions = info['pca'] / 'vencedor' / 'teste_final' / 'predicoes_f10.csv'
        require(pca_predictions.is_file(), 'Predicoes PCA F10 nao encontradas para auditoria pareada.')
        pca_tab = pd.read_csv(pca_predictions, usecols=['linha_original', 'y_true'])
        require(len(pca_tab) == len(ids)
                and np.array_equal(pca_tab['linha_original'].to_numpy(dtype=np.int64), ids)
                and np.array_equal(pca_tab['y_true'].to_numpy(dtype=np.uint8), y),
                'Indices ou rotulos F10 diferentes do PCA.')
        print('PCA_TSNE_F10_INDICES_ROTULOS_IDENTICOS=SIM', flush=True)
        print('[V3] Carregando modelo V2 confiavel; nenhuma etapa de treinamento.', flush=True)
        pacote = carregar_pacote_confiavel(info['modelo_dir'])
        require(pacote.n_treino_original == EXPECTED_N_DEV and
                pacote.escolha_oof['id_candidato'] == EXPECTED_CANDIDATE and
                pacote.limiar_oof == pre.plano.limiar_mcc_oof_congelado,
                'Modelo V2 recarregado diverge do candidato aprovado.')
        require(int(pacote.tsne.parametros['n_components']) == 1,
                'Modelo persistido nao e TSNE1D.')
        print('[V3] Projetando F10 OOS em lotes, sem reajuste t-SNE/GMM.', flush=True)
        p, z = infer_batched(pacote, df, ids, target, batch_n, pacote.limiar_oof)
        tau = pacote.limiar_oof
        del pacote
        gc.collect()
        metrics = avaliar_scores_com_limiar_oof_congelado(y, p, limiar_mcc_oof=tau)
        metrics.update({
            'metodo': 'TSNE', 'dimensao_tsne': 1, 'id_candidato_oof': EXPECTED_CANDIDATE,
            'classe_avaliacao': 'F10_EXPLORATORIO_REUTILIZADO_PAREADO_PCA',
            'f10_reutilizado': True, 'indices_pca_congelados': True,
            'modelo_reajustado_no_f10': False,
        })
        require(metrics['limiar_otimizado_no_f10'] is False and
                metrics['limiar_mcc_oof_congelado'] == tau and metrics['n'] == len(ids),
                'Metricas fora da especificacao V3.')
        tab = pd.DataFrame({
            'linha_original': ids, 'fold': np.full(len(ids), 10, dtype=np.uint8),
            'y_true': y, 'prob_fraude': p,
            'predicao_tau_0_50': (p >= .5).astype(np.uint8),
            'predicao_tau_mcc_oof': (p >= tau).astype(np.uint8),
        })
        coords = pd.DataFrame({'linha_original': ids, 'TSNE1': z})
        out = info['staging']
        require(not out.exists(), 'Pasta temporaria V3 ja existe.')
        out.mkdir(parents=False, exist_ok=False)
        tab.to_csv(out / 'predicoes_f10.csv', index=False, float_format='%.17g')
        coords.to_csv(out / 'coordenadas_TSNE1_f10.csv', index=False, float_format='%.17g')
        with (out / 'metricas_f10.json').open('x', encoding='utf-8') as fd:
            json.dump(metrics, fd, ensure_ascii=False, indent=2, allow_nan=False)
            fd.write('\n')
        np.save(out / 'indices_f10_amostra.npy', ids, allow_pickle=False)
        digest_files = {name: sha_file(out / name) for name in (
            'predicoes_f10.csv', 'metricas_f10.json',
            'coordenadas_TSNE1_f10.csv', 'indices_f10_amostra.npy')}
        meta = {
            'versao': 'V3_TSNE1D_F10_OOS_PAREADO_V1', 'status': 'APROVADA',
            'n_f10': len(ids), 'n_fraudes': int(y.sum()),
            'id_candidato_oof': EXPECTED_CANDIDATE,
            'sha256_modelo_v2': info['modelo_hash'],
            'sha256_indices_f10_pca': info['sha_indices'],
            'parquet_sha256': parquet_hash,
            'folds_sha256': orq.hash_arquivo(fold_file),
            'limiar_tau_mcc_oof_congelado': tau,
            'f10_exploratorio_reutilizado': True,
            'modelo_ajustado_somente_f1_f9': True,
            'nao_reotimizou_limiar_f10': True,
            'inferência': 'openTSNE transform OOS em lotes; nenhum fit no F10',
            'tamanho_lote': batch_n,
            'pca_f10_referencia': str(info['pca']),
            'sha256_arquivos': digest_files,
        }
        with (out / 'manifesto_v3.json').open('x', encoding='utf-8') as fd:
            json.dump(meta, fd, indent=2, ensure_ascii=False, allow_nan=False)
            fd.write('\n')
        # Verificar fonte e arquivo em disco antes do commit atomico da pasta.
        require(np.array_equal(pd.read_csv(out / 'predicoes_f10.csv', usecols=['linha_original'])
                               ['linha_original'].to_numpy(dtype=np.int64), ids),
                'CSV de saida nao preserva os indices de F10.')
        require(np.array_equal(pd.read_csv(out / 'coordenadas_TSNE1_f10.csv', usecols=['linha_original'])
                               ['linha_original'].to_numpy(dtype=np.int64), ids),
                'CSV de coordenadas F10 desalinhado.')
        require(not info['output'].exists(), 'Saida V3 ja existe; nao sobrescrever.')
        os.replace(out, info['output'])
        verify_existing(info)
        print(f'V3_AP={metrics["ap"]:.7f}', flush=True)
        print(f'V3_MCC_TAU_OOF={metrics["mcc_limiar_oof"]:.7f}', flush=True)
        print(f'V3_LOGLOSS={metrics["logloss"]:.7f}', flush=True)
        print(f'V3_SCORE_COMPOSTO={metrics["score_composto_limiar_oof"]:.7f}', flush=True)
        print('V3_DADOS_FINAL=' + str(info['output']), flush=True)
        print('V3_STATUS=APROVADA', flush=True)
        print('OOF_REEXECUTADO=NAO', flush=True)
        print('F10_LIMIAR_REOTIMIZADO=NAO', flush=True)
        return 0
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--planejar', action='store_true')
    g.add_argument('--executar', action='store_true')
    parser.add_argument('--lote', type=int, default=500,
                        help='Tamanho do lote de inferencia OOS (padrao=500).')
    args = parser.parse_args(argv)
    require(10 <= args.lote <= 2000, 'Lote deve ficar entre 10 e 2000.')
    info = preflight()
    show_plan(info)
    if args.planejar:
        print('V3_STATUS=PLANEJAMENTO_APROVADO_SEM_F10_DADOS', flush=True)
        return 0
    return run(info, args.lote)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('V3_STATUS=BLOQUEADA', file=sys.stderr, flush=True)
        print('V3_MOTIVO=' + str(error), file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
