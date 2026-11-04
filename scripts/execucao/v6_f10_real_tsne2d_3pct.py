# -*- coding: utf-8 -*-
"""V6: avaliacao F10 REAL TSNE+GMM 2D (3%), sem refit nem graficos.

Modos: --planejar, --executar, --validar.
Usa SOMENTE o modelo V5 aprovado e o limiar MCC OOF congelado.
Nao chama qualquer funcao de ajuste OOF, ajuste do modelo ou gera graficos.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import logging
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

METODO = 'tsne'
PASTA_OOF = 'ensaio_tsne_gmm_20261103_135207'
ID_VENCEDOR = 'd517359b0f6645'
N_DESENVOLVIMENTO = 50013
N_GMM_RUS = 4437
SHA_MODELO_V5 = 'd9fb29d0c3f900a856f87724315a48e223968cfa0ea61d88bf3ee53546eff588'
NOME_MODELO = 'modelo_final_f1_f9'
NOME_F10 = 'avaliacao_f10_3pct'
NOME_LOCK = '.ensaio_real_3pct_2d_tsne.lock'


def exigir(condicao, motivo):
    if not condicao:
        raise RuntimeError(motivo)


def conferir_estrutura_teste_final(pasta):
    """Permite somente modelo final e, apos V6, avaliacao F10."""
    teste = pasta / 'vencedor' / 'teste_final'
    exigir(teste.is_dir(), 'Pasta teste_final da V5 ausente.')
    nomes = {p.name for p in teste.iterdir()}
    nao_permitidos = nomes - {NOME_MODELO, NOME_F10}
    exigir(not nao_permitidos,
           'Teste final contem itens inesperados: ' + repr(sorted(nao_permitidos)) +
           '. Nao excluir; envie o erro para analise.')
    modelo = teste / NOME_MODELO
    exigir(modelo.is_dir(), 'Pacote de modelo final V5 ausente.')
    destino = teste / NOME_F10
    if destino.exists():
        exigir(destino.is_dir(), 'Destino F10 existe mas nao e diretorio.')
        return modelo, destino, 'F10_PRESENTE'
    return modelo, destino, 'F10_AINDA_NAO_EXECUTADO'


def conferir_base():
    """Reusa a V5 so na conferencia F1..F9; NAO chama validador V5 que proibe F10."""
    from scripts.execucao import v5_modelo_final_tsne2d_3pct as v5
    drv, cfg, pasta, info = v5.conferir_base()
    exigir(pasta.name == PASTA_OOF, 'Pasta OOF da V4/V5 nao corresponde a aprovada.')
    exigir(len(info['ids']) == N_DESENVOLVIMENTO and
           info['escolha']['id_candidato'] == ID_VENCEDOR and
           info['escolha']['teste_final_usado'] is False and
           int(info['escolha']['configuracao']['reducao']['n_components']) == 2,
           'OOF vencedor F1..F9 mudou depois da V5.')
    modelo, destino, estado = conferir_estrutura_teste_final(pasta)
    man = drv.validar_modelo(modelo, cfg, info['escolha'], METODO, N_DESENVOLVIMENTO)
    exigir(man.get('tipo') == 'TREINAMENTO_FINAL_TSNE_F1_F9' and
           man.get('metodo') == 'TSNE' and
           man.get('teste_f10_executado') is False and
           int(man.get('dimensao_tsne', -1)) == 2 and
           int(man.get('n_treino_original', -1)) == N_DESENVOLVIMENTO and
           int(man.get('n_ajuste_gmm', -1)) == N_GMM_RUS and
           man.get('id_candidato_oof') == ID_VENCEDOR and
           man.get('limiar_mcc_oof_congelado') == info['escolha']['limiar_mcc_oof'] and
           man.get('sha256_modelo_final') == SHA_MODELO_V5,
           'Modelo final ou proveniencia V5 divergentes do registro aprovado.')
    for chave in ('sha256_parquet', 'sha256_folds', 'sha256_yaml'):
        exigir(man.get('proveniencia', {}).get(chave) == info['manifesto'].get(chave),
               'Proveniencia do modelo diverge do OOF: ' + chave)
    exigir(Path(man.get('proveniencia', {}).get('experimento_oof', '')).resolve() == pasta.resolve(),
           'Modelo final nao pertence a esta execucao OOF.')
    arquivo = modelo / 'modelo_final_f1_f9.joblib'
    exigir(arquivo.is_file() and drv.hash_arquivo(arquivo) == SHA_MODELO_V5,
           'SHA-256 fisico do modelo V5 foi alterado; F10 bloqueado.')
    exigir(int(cfg['divisao']['qual_fold_teste']) == 10 and
           cfg['reducao']['metodo'] == 'TSNE' and
           cfg['tsne']['n_components'] == [2] and
           float(cfg['dados']['fracao_base']) == .03,
           'Configuracao congelada TSNE 2D 3% divergente.')
    return drv, cfg, pasta, info, modelo, destino, estado


def imprimir_plano(drv, pasta, info, modelo, destino, estado):
    print('=== V6 | PLANEJAMENTO F10 TSNE + GMM 2D ===', flush=True)
    print('V6_EXPERIMENTO=' + str(pasta), flush=True)
    print('V6_MODELO_F1_F9=' + str(modelo / 'modelo_final_f1_f9.joblib'), flush=True)
    print('V6_SHA256_MODELO=' + SHA_MODELO_V5, flush=True)
    print('V6_N_OOF=' + str(len(info['ids'])), flush=True)
    print('V6_CANDIDATO_OOF=' + ID_VENCEDOR, flush=True)
    print('V6_TAU_OOF_CONGELADO=' + f'{float(info["escolha"]["limiar_mcc_oof"]):.7f}', flush=True)
    print('V6_DESTINO_F10=' + str(destino), flush=True)
    print('V6_ESTADO_F10=' + estado, flush=True)
    print('V6_AMOSTRA_F10=3%_ESTRATIFICADA_SEMENTE_42', flush=True)
    print('V6_F10_EXPLORATORIO_REUTILIZADO_NO_PROJETO=SIM', flush=True)
    print('V6_OOF_REEXECUTADO=NAO', flush=True)
    print('V6_MODELO_REAJUSTADO=NAO', flush=True)
    print('V6_GRAFICOS_GERADOS=NAO', flush=True)
    print('V6_PLANEJAMENTO=APROVADO_SEM_LER_F10', flush=True)


def conferir_resultados(drv, cfg, pasta, info, modelo, destino):
    """Confere os arquivos F10 SEM reabrir modelo para fit e SEM recalcular limiar."""
    exigir(destino.is_dir(), 'Resultados F10 ainda nao foram gerados.')
    caminho, mf = drv.validar_f10(pasta, cfg, info, METODO, modelo)
    exigir(caminho == destino, 'Destino validado F10 divergente.')
    exigir(mf.get('metodo') == 'TSNE' and mf.get('status') == 'APROVADA' and
           mf.get('sem_fit_no_f10') is True and
           mf.get('sem_otimizacao_de_tau_no_f10') is True and
           mf.get('avaliacao_exploratoria') is True and
           mf.get('sha256_parquet') == info['manifesto']['sha256_parquet'] and
           mf.get('sha256_folds') == info['manifesto']['sha256_folds'] and
           mf.get('id_candidato_oof') == ID_VENCEDOR,
           'Manifesto F10 nao respeita a proveniencia / isolamento de F1..F9.')
    arquivos = {'predicoes_f10.csv', 'coordenadas_TSNE1_TSNE2_f10.csv',
                'indices_f10_amostra.npy', 'metricas_f10.json', 'manifesto_f10_3pct.json'}
    exigir({p.name for p in destino.iterdir()} == arquivos,
           'Arquivos na avaliacao F10 diferentes dos cinco arquivos esperados.')
    pred = pd.read_csv(destino / 'predicoes_f10.csv')
    coor = pd.read_csv(destino / 'coordenadas_TSNE1_TSNE2_f10.csv')
    ids = np.load(destino / 'indices_f10_amostra.npy', allow_pickle=False)
    met = drv.ler_json(destino / 'metricas_f10.json')
    exigir(set(pred.columns) == {'linha_original','fold','y_true','prob_fraude',
                                'predicao_tau_0_50','predicao_tau_mcc_oof'} and
           set(coor.columns) == {'linha_original','TSNE1','TSNE2'},
           'Colunas do F10 ou das coordenadas nao correspondem ao contrato.')
    exigir(ids.ndim == 1 and len(ids) > 0 and ids.dtype.kind in 'iu' and
           len(ids) == len(pred) == len(coor) == int(mf['n_f10']) == int(met['n']) and
           np.array_equal(ids, pred['linha_original'].to_numpy(dtype=np.int64)) and
           np.array_equal(ids, coor['linha_original'].to_numpy(dtype=np.int64)) and
           np.all(np.diff(ids) > 0),
           'Amostra F10 desalinhada, repetida ou com contagem divergente.')
    exigir(ids[-1] < len(info['folds_master']) and
           np.all(info['folds_master'][ids] == 10) and
           np.all(pred['fold'].to_numpy() == 10) and
           len(np.intersect1d(ids, info['ids'])) == 0,
           'F10 contem registros fora do fold reservado ou compartilhados com F1..F9.')
    y = pred['y_true'].to_numpy(dtype=np.int64)
    p = pred['prob_fraude'].to_numpy(dtype=np.float64)
    exigir(set(y.tolist()) == {0, 1} and np.isfinite(p).all() and
           ((p >= 0) & (p <= 1)).all() and
           np.isfinite(coor[['TSNE1', 'TSNE2']].to_numpy(dtype=np.float64)).all() and
           int(y.sum()) == int(mf['n_fraudes_f10']) == int(met['n_fraudes']) and
           int(met['n_legitimas']) + int(met['n_fraudes']) == len(y),
           'Rotulos, scores, coordenadas ou contagens F10 invalidos.')
    tau = float(info['escolha']['limiar_mcc_oof'])
    exigir(float(met['limiar_mcc_oof_congelado']) == tau and
           float(mf['tau_oof_congelado']) == tau and
           met['limiar_otimizado_no_f10'] is False and
           met.get('dimensao') == 2,
           'Limiar, dimensionalidade ou natureza da avaliacao F10 divergente.')
    exigir(np.array_equal(pred['predicao_tau_0_50'].to_numpy(dtype=np.int64), (p >= .5).astype(np.int64)) and
           np.array_equal(pred['predicao_tau_mcc_oof'].to_numpy(dtype=np.int64), (p >= tau).astype(np.int64)),
           'Predicoes por limiar divergentes dos scores gravados.')
    exigir(sha256(ids.tobytes()).hexdigest() == mf['sha256_indices_f10'],
           'Hash dos indices F10 nao confere com o manifesto.')
    from src.avaliacao_final_tsne import avaliar_scores_com_limiar_oof_congelado
    recalculo = avaliar_scores_com_limiar_oof_congelado(y, p, limiar_mcc_oof=tau)
    for chave in ('ap', 'mcc_limiar_oof', 'logloss', 'exp_menos_logloss',
                  'score_composto_limiar_oof', 'prevalencia'):
        exigir(abs(float(recalculo[chave]) - float(met[chave])) < 1e-10,
               'Metrica F10 divergente: ' + chave)
    for corte in ('tau_0_50', 'tau_mcc_oof'):
        for k in ('tn', 'fp', 'fn', 'tp'):
            exigir(int(recalculo['comparacao_limiares'][corte]['matriz_confusao'][k]) ==
                   int(met['comparacao_limiares'][corte]['matriz_confusao'][k]),
                   'Matriz de confusao F10 divergente: ' + corte + '.' + k)
    print('=== V6 | AVALIACAO REAL F10 TSNE + GMM 2D ===', flush=True)
    print('V6_STATUS=F10_TSNE_2D_3PCT_APROVADO', flush=True)
    print('V6_EXPERIMENTO=' + str(pasta), flush=True)
    print('V6_F10_DESTINO=' + str(destino), flush=True)
    print('V6_N_F10=' + str(len(y)), flush=True)
    print('V6_N_FRAUDES_F10=' + str(int(y.sum())), flush=True)
    print('V6_CANDIDATO_OOF=' + ID_VENCEDOR, flush=True)
    print('V6_TAU_MCC_OOF=' + f'{tau:.7f}', flush=True)
    print('V6_AP_F10=' + f'{float(met["ap"]):.7f}', flush=True)
    print('V6_MCC_F10_TAU_OOF=' + f'{float(met["mcc_limiar_oof"]):.7f}', flush=True)
    print('V6_LOGLOSS_F10=' + f'{float(met["logloss"]):.7f}', flush=True)
    print('V6_SCORE_COMPOSTO_F10=' + f'{float(met["score_composto_limiar_oof"]):.7f}', flush=True)
    print('V6_MODELO_SHA256=' + SHA_MODELO_V5, flush=True)
    print('V6_F10_NAO_OTIMIZADO=SIM', flush=True)
    print('V6_OOF_REEXECUTADO=NAO', flush=True)
    print('V6_MODELO_REAJUSTADO=NAO', flush=True)
    print('V6_PCA_2D_EXECUTADO=NAO', flush=True)
    print('V6_GRAFICOS_GERADOS=NAO', flush=True)
    print('V6_F10_E_EXPLORATORIO=SIM', flush=True)
    return met


def executar(drv, cfg, pasta, info, modelo, destino, estado):
    if estado == 'F10_PRESENTE':
        print('[V6] F10 ja existente: somente validar; nao recalcular scores.', flush=True)
        conferir_resultados(drv, cfg, pasta, info, modelo, destino)
        return
    lock = drv.ROOT / 'outputs' / NOME_LOCK
    exigir(not lock.exists(), 'Lock de outra execucao TSNE 2D presente: ' + str(lock))
    with lock.open('x', encoding='utf-8') as arquivo:
        arquivo.write('V6: somente inferencia e avaliacao F10, modelo TSNE 2D congelado.\n')
    try:
        _modelo, _destino, estado_novo = conferir_estrutura_teste_final(pasta)
        exigir(estado_novo == 'F10_AINDA_NAO_EXECUTADO',
               'F10 apareceu durante aquisicao do lock; nao sobrescrever.')
        print('[V6] Inferencia F10 real: 3% estratificado por classe. Sem fit.', flush=True)
        criado = drv.executar_f10(pasta, cfg, info, METODO, modelo)
        exigir(criado == destino and destino.is_dir(), 'Caminho F10 retornado divergente.')
        conferir_resultados(drv, cfg, pasta, info, modelo, destino)
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument('--planejar', action='store_true')
    m.add_argument('--executar', action='store_true')
    m.add_argument('--validar', action='store_true')
    args = ap.parse_args(argv)
    try:
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        drv, cfg, pasta, info, modelo, destino, estado = conferir_base()
        if args.planejar:
            imprimir_plano(drv, pasta, info, modelo, destino, estado)
        elif args.executar:
            executar(drv, cfg, pasta, info, modelo, destino, estado)
        else:
            conferir_resultados(drv, cfg, pasta, info, modelo, destino)
        print('EXIT_CODE_V6=0', flush=True)
        return 0
    except Exception as e:
        print('V6_STATUS=BLOQUEADO_SEM_REAJUSTE', file=sys.stderr, flush=True)
        print('V6_MOTIVO=' + str(e), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
