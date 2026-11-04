# -*- coding: utf-8 -*-
"""Novo ensaio independente PCA-GMM e t-SNE-GMM 3D, 3% real.

Reusa o executor 1D ja homologado, apenas com um identificador NOVO de
experimento, sem editar configuracao original nem pastas de execucoes antigas.

Use o PowerShell EXECUTAR_RETESTE_REAL_1D_3PCT.ps1 para executar os dois
metodos em processos diferentes e, depois, abrir as 2 pastas e os 4 HTMLs.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import re
import sys
import traceback

import numpy as np
import pandas as pd
import yaml

from scripts.execucao import experimentos_reais_3pct_3d as base

ROOT = Path(__file__).resolve().parents[2]
LOTE_PADRAO = 'RETESTE_20261103_3D_V1'
N_OOF_ESPERADO = 50013
N_F10_ESPERADO = 5557
N_FRAUDES_F10 = 29
PRE01_REFERENCIAS_1D = {
    'pca': ROOT / 'outputs' / 'ensaio_pca_gmm_20261103_162549',
    'tsne': ROOT / 'outputs' / 'ensaio_tsne_gmm_20261103_163631',
}
BASELINES = {
    'pca': ROOT / 'outputs' / 'ensaio_pca_gmm_20261103_155155',
    'tsne': ROOT / 'outputs' / 'ensaio_tsne_gmm_20261103_135207',
}


def exigir(condicao, msg):
    if not condicao:
        raise RuntimeError(msg)


def sha256_arquivo(path: Path) -> str:
    dig = sha256()
    with path.open('rb') as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b''):
            dig.update(bloco)
    return dig.hexdigest()


def configurar(lote: str):
    exigir(bool(re.fullmatch(r'[A-Za-z0-9_]{3,48}', lote)),
           'Lote invalido: use somente letras ASCII, numeros e underscore (3 a 48).')
    exigir(base.ROOT.resolve() == ROOT.resolve(), 'Executor base aponta para outra raiz.')
    exigir(base.FRACAO == .03 and base.SEMENTE == 42,
           'Executor 1D nao corresponde a 3%, semente 42.')
    exigir(base.SAIDA_F10 == 'avaliacao_f10_3pct' and
           base.SAIDA_GALERIA == 'galeria_3pct',
           'Contrato do executor 1D alterado.')
    base.MARCADOR = 'TCC_EXPERIMENTAL_V3_3PCT_3D_REAL_' + lote


def conferir_baselines():
    """Checar identidade de registros frente aos resultados 2D preexistentes.

    Lemos somente IDs salvos, nunca as rotulacoes de F10 aqui.
    """
    oofs, f10s = {}, {}
    for metodo, pasta in BASELINES.items():
        arq_oof = pasta / 'vencedor' / 'oof' / 'predicoes.npz'
        arq_f10 = pasta / 'vencedor' / 'teste_final' / 'avaliacao_f10_3pct' / 'indices_f10_amostra.npy'
        exigir(arq_oof.is_file() and arq_f10.is_file(),
               f'Baseline 2D {metodo.upper()} nao encontrada. Nao executar sem verificar: {pasta}')
        with np.load(arq_oof, allow_pickle=False) as conteudo:
            ids_oof = np.asarray(conteudo['linha_original'], dtype=np.int64)
        ids_f10 = np.asarray(np.load(arq_f10, allow_pickle=False), dtype=np.int64)
        exigir(len(ids_oof) == N_OOF_ESPERADO and len(ids_f10) == N_F10_ESPERADO,
               f'Baseline 2D {metodo.upper()} com quantidade inesperada.')
        exigir(len(np.unique(ids_oof)) == len(ids_oof) and
               len(np.unique(ids_f10)) == len(ids_f10) and
               not np.intersect1d(ids_oof, ids_f10).size,
               f'Baseline 2D {metodo.upper()} tem IDs invalidos/sobrepostos.')
        oofs[metodo] = ids_oof
        f10s[metodo] = ids_f10
    exigir(np.array_equal(oofs['pca'], oofs['tsne']),
           'Baselines PCA2D/TSNE2D possuem OOF divergente.')
    exigir(np.array_equal(f10s['pca'], f10s['tsne']),
           'Baselines PCA2D/TSNE2D possuem F10 divergente.')
    print('REFERENCIA_2D_OOF=' + str(N_OOF_ESPERADO), flush=True)
    print('REFERENCIA_2D_F10=' + str(N_F10_ESPERADO), flush=True)
    print('REFERENCIA_2D_IDS_IGUAIS=SIM', flush=True)
    return oofs['pca'], f10s['pca']


def configurar_e_testar_metodo(metodo):
    cfg, grade = base.construir_config(metodo)
    exigir(cfg['experimento']['nome'] == base.MARCADOR + '_' + metodo.upper(),
           'Identificador do novo ensaio nao foi aplicado.')
    exigir(cfg['dados']['fracao_base'] == .03 and
           cfg['experimento']['semente'] == 42 and
           cfg['reducao']['metodo'] == metodo.upper() and
           cfg['divisao']['qual_fold_teste'] == 10,
           'Configuracao do ensaio real inesperada.')
    exigir(cfg['dados']['arquivo'].replace('\\', '/') == 'data/raw/dataset_raw.parquet',
           'Fonte divergente do Parquet real original.')
    exigir(int(cfg['pca']['n_components'][0]) == 3 and
           int(cfg['tsne']['n_components'][0]) == 3,
           'Dimensoes 3D nao aplicadas.')
    exigir(cfg['gmm']['covariance_type'].lower() == 'full' and
           int(cfg['gmm']['n_init']) == 5 and
           int(cfg['gmm']['n_components'][0]) == 4,
           'GMM diverge de FULL K4 n_init=5.')
    exigir(cfg['amostragem']['modo'] == 'RUS' and
           int(cfg['amostragem']['razao_negativos_positivos']) == 16,
           'RUS diverge de 16:1.')
    exigir(len(grade) == (1 if metodo == 'pca' else 2),
           'Grade 3D inesperada (PCA=1, TSNE=2).')
    if metodo == 'pca':
        # Módulos do catálogo geométrico PCA 3D precisam existir antes de rodar OOF.
        from src import (visualizacao_gmm01_3d, visualizacao_gmm04_3d,
                         visualizacao_gmm07_3d, visualizacao_gmm08_3d)
        import plotly
    if metodo == 'tsne':
        from src import (visualizacao_gmm01_3d, visualizacao_gmm04_3d,
                         visualizacao_gmm07_3d)
        from src import visualizacao_gmm07
        exigir(callable(visualizacao_gmm07._grafico),
               'GMM07 generico 3D nao disponivel.')
        from src.garantia_layout_tsne import verificar_layout_tsne
        verificar_layout_tsne(cfg, fase='reteste_3d_real_3pct')
    pre01_src = PRE01_REFERENCIAS_1D[metodo] / 'vencedor' / 'visualizacao' / 'galeria' / 'oof' / 'PRE01_diagnostico_padronizacao_completo.html'
    exigir(pre01_src.is_file() and pre01_src.stat().st_size > 1000,
           'Diagnostico PRE01 do 1D nao encontrado ou invalido: ' + str(pre01_src))
    pre01_conteudo=pre01_src.read_text(encoding='utf-8-sig')
    exigir('PRE01_DATA' in pre01_conteudo and 'plot-pca' in pre01_conteudo,
           'PRE01 de referencia 1D nao possui diagnosticos completos.')
    concluido, parcial = base.localizar_experimento(cfg, metodo)
    return cfg, concluido, parcial


def ler_ids(metodo, pasta):
    caminho_oof = pasta / 'vencedor' / 'oof' / 'predicoes.npz'
    with np.load(caminho_oof, allow_pickle=False) as a:
        oof = np.asarray(a['linha_original'], dtype=np.int64)
        folds = np.asarray(a['fold'], dtype=np.int64)
    f10_dir = pasta / 'vencedor' / 'teste_final' / 'avaliacao_f10_3pct'
    f10 = np.asarray(np.load(f10_dir / 'indices_f10_amostra.npy', allow_pickle=False), dtype=np.int64)
    df = pd.read_csv(f10_dir / 'predicoes_f10.csv')
    exigir(df['linha_original'].to_numpy(dtype=np.int64).tolist() == f10.tolist(),
           f'Predicoes F10 desalinhadas para {metodo}.')
    exigir(len(oof) == N_OOF_ESPERADO and len(f10) == N_F10_ESPERADO and
           len(df) == N_F10_ESPERADO and int(df['y_true'].sum()) == N_FRAUDES_F10,
           f'Amostragem real 3% inesperada para {metodo}.')
    exigir(set(np.unique(folds)) == set(range(1, 10)) and
           np.array_equal(df['fold'].to_numpy(dtype=np.int64), np.full(len(df), 10)) and
           not np.intersect1d(oof, f10).size,
           f'Fold F10 ou F1-F9 invalido para {metodo}.')
    exigir(np.all(np.diff(oof)>0) and np.all(np.diff(f10)>0),
           f'IDs de {metodo} nao estao em ordem crescente.')
    return oof, f10


from scripts.execucao.reteste_real_3d_pre01 import conferir_pre01, incluir_pre01_nos_resultados

def validar_metodo(metodo, referencia_oof, referencia_f10):
    cfg, pasta, parcial = configurar_e_testar_metodo(metodo)
    exigir(pasta is not None and parcial is None,
           f'{metodo.upper()} 3D do novo lote ainda nao concluido.')
    # Inclui hashes e proveniencia dos resultados produzidos pelo executor base.
    base.validar_concluido(metodo)
    oof, f10 = ler_ids(metodo, pasta)
    exigir(np.array_equal(oof, referencia_oof),
           f'OOF 3D de {metodo.upper()} diverge dos 50.013 IDs dos ensaios 2D.')
    exigir(np.array_equal(f10, referencia_f10),
           f'F10 3D de {metodo.upper()} diverge dos 5.557 IDs dos ensaios 2D.')
    vis = pasta / 'vencedor' / 'visualizacao' / 'galeria' / 'index.html'
    gal = pasta / 'vencedor' / 'teste_final' / 'galeria_3pct'
    html = gal / 'index.html'
    man_galeria = json.loads((gal / 'manifesto_galeria_3pct.json').read_text(encoding='utf-8'))
    exigir(vis.is_file() and vis.stat().st_size>100 and html.is_file() and
           sha256_arquivo(html) == man_galeria['sha256_html'],
           f'Galeria OOF/F10 de {metodo.upper()} ausente ou com hash divergente.')
    exigir(len(man_galeria['hashes_png']) >= 13 and
           all((gal / nome).is_file() and sha256_arquivo(gal / nome) == digest
               for nome, digest in man_galeria['hashes_png'].items()),
           f'Imagens F10 de {metodo.upper()} ausentes/alteradas.')
    conferir_pre01(metodo,pasta,PRE01_REFERENCIAS_1D)
    print(f'{metodo.upper()}_STATUS=OOF_E_F10_REAIS_3PCT_APROVADOS', flush=True)
    print(f'{metodo.upper()}_EXPERIMENTO={pasta}', flush=True)
    print(f'{metodo.upper()}_N_OOF={len(oof)}', flush=True)
    print(f'{metodo.upper()}_N_F10={len(f10)}', flush=True)
    print(f'{metodo.upper()}_FRAUDES_F10={N_FRAUDES_F10}', flush=True)
    print(f'{metodo.upper()}_GALERIA_OOF={vis}', flush=True)
    print(f'{metodo.upper()}_GALERIA_F10={html}', flush=True)
    escolha = json.loads((pasta / 'vencedor' / 'metricas' / 'melhor_configuracao.json').read_text(encoding='utf-8'))
    met_f10 = json.loads((pasta / 'vencedor' / 'teste_final' / 'avaliacao_f10_3pct' / 'metricas_f10.json').read_text(encoding='utf-8'))
    print(f'{metodo.upper()}_VENCEDOR_OOF={escolha["id_candidato"]}', flush=True)
    print(f'{metodo.upper()}_TAU_OOF={float(escolha["limiar_mcc_oof"]):.7f}', flush=True)
    for nome, campo in (('AP_F10', 'ap'), ('MCC_F10', 'mcc_limiar_oof'),
                        ('LOGLOSS_F10', 'logloss'), ('SCORE_F10', 'score_composto_limiar_oof')):
        print(f'{metodo.upper()}_{nome}={float(met_f10[campo]):.7f}', flush=True)
    print(f'{metodo.upper()}_IDS_IGUAIS_AO_2D=SIM', flush=True)
    return pasta, oof, f10, vis, html


def validar_tudo(ref_oof, ref_f10):
    a = validar_metodo('pca', ref_oof, ref_f10)
    b = validar_metodo('tsne', ref_oof, ref_f10)
    exigir(np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]),
           'Amostras PCA1D e TSNE1D nao coincidem.')
    print('RETESTE_3D_IDS_OOF_IGUAIS=SIM', flush=True)
    print('RETESTE_3D_IDS_F10_IGUAIS=SIM', flush=True)
    print('RETESTE_3D_HTMLS=4/4', flush=True)
    print('RETESTE_3D_PRE01=4_LINKS',flush=True)
    print(f'RETESTE_3D_PNGS_F10={len(json.loads((a[0]/"vencedor"/"teste_final"/"galeria_3pct"/"manifesto_galeria_3pct.json").read_text(encoding="utf-8"))["hashes_png"])+len(json.loads((b[0]/"vencedor"/"teste_final"/"galeria_3pct"/"manifesto_galeria_3pct.json").read_text(encoding="utf-8"))["hashes_png"])}',flush=True)
    print('RETESTE_3D_STATUS=APROVADO_PCA_TSNE_OOF_F10_REAIS_3PCT', flush=True)
    return a, b


def abrir_resultados(ref_oof, ref_f10):
    a, b = validar_tudo(ref_oof, ref_f10)
    exigir(os.name == 'nt', 'Abertura automatica de pastas requer Windows.')
    print('ABRINDO_2_PASTAS_DOS_EXPERIMENTOS_E_4_HTMLS=SIM', flush=True)
    for pasta in (a[0], b[0]):
        os.startfile(str(pasta))
        print(f'PASTA_ABERTA={pasta}', flush=True)
    for html in (a[3], a[4], b[3], b[4]):
        os.startfile(str(html))
        print(f'HTML_ABERTO={html}', flush=True)
    print('RETESTE_3D_ABERTURAS_SOLICITADAS=6', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lote', default=LOTE_PADRAO)
    group = parser.add_mutually_exclusive_group(required=True)
    for modo in ('planejar','pca','tsne','pre01','validar','abrir'):
        group.add_argument('--'+modo, action='store_true')
    args = parser.parse_args(argv)
    try:
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        configurar(args.lote)
        yaml_original = base.YAML_ORIGINAL
        exigir(yaml_original.is_file(), 'YAML original nao encontrado.')
        hash_yaml_antes = sha256_arquivo(yaml_original)
        # Baselines tecnicamente auditadas dos testes 2D anteriores.
        oof_ref, f10_ref = conferir_baselines()
        print('RETESTE_3D_LOTE=' + args.lote, flush=True)
        if args.planejar:
            for metodo in ('pca','tsne'):
                cfg, anterior, parcial = configurar_e_testar_metodo(metodo)
                print(f'{metodo.upper()}_EXPERIMENTO_NOVO=' + cfg['experimento']['nome'], flush=True)
                print(f'{metodo.upper()}_JA_FINALIZADO=' + str(anterior or ''), flush=True)
                print(f'{metodo.upper()}_PARCIAL=' + str(parcial or ''), flush=True)
                print(f'{metodo.upper()}_CANDIDATOS=' + str(len(base.construir_config(metodo)[1])), flush=True)
            print('RETESTE_3D_PLANEJAMENTO_SEM_TREINAR=SIM', flush=True)
        elif args.pca or args.tsne:
            metodo = 'pca' if args.pca else 'tsne'
            configurar_e_testar_metodo(metodo)
            ret = base.main(['--'+metodo])
            exigir(ret == 0, f'Executor base retornou erro ao executar {metodo}.')
            cfg, pronto, parcial = configurar_e_testar_metodo(metodo)
            exigir(pronto is not None and parcial is None,'Ensaio 3D nao concluido.')
            incluir_pre01_nos_resultados(metodo,pronto,oof_ref,PRE01_REFERENCIAS_1D)
            validar_metodo(metodo, oof_ref, f10_ref)
        elif args.pre01:
            for metodo in ('pca','tsne'):
                cfg,pronto,parcial=configurar_e_testar_metodo(metodo)
                exigir(pronto is not None and parcial is None,metodo+' 3D nao concluido.')
                incluir_pre01_nos_resultados(metodo,pronto,oof_ref,PRE01_REFERENCIAS_1D)
        elif args.validar:
            validar_tudo(oof_ref, f10_ref)
        else:
            abrir_resultados(oof_ref, f10_ref)
        exigir(sha256_arquivo(yaml_original) == hash_yaml_antes,
               'YAML original foi alterado durante a execucao.')
        print('YAML_ORIGINAL_INALTERADO=SIM', flush=True)
        print('EXIT_CODE_RETESTE_3D=0', flush=True)
        return 0
    except Exception as exc:
        print('RETESTE_3D_STATUS=BLOQUEADO_SEM_SOBRESCRITA', file=sys.stderr, flush=True)
        print('RETESTE_3D_MOTIVO=' + str(exc), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
