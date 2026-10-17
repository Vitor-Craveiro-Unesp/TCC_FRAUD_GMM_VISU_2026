"""Passo 06.03: ensaio verdadeiro dos 9 folds usando 300 linhas por fold.

O ensaio nao e o experimento OOF oficial. Salva resultados amostrais em uma
pasta identificada como 'ensaio', sem ajustar, rotular ou avaliar F10.
Executar: python .\\tests\\verificar_orquestracao_base_real.py
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.orquestracao_oof import main, _identificar_folds


def main_verificar():
    caminho_cfg=ROOT/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    anterior=caminho_cfg.read_bytes()
    config=yaml.safe_load(anterior.decode('utf-8-sig'))
    if config['reducao']['metodo'].upper()!='PCA':
        raise RuntimeError('Este smoke test cobre o modo PCA atualmente configurado.')
    if config['dados']['fracao_base']!=1.0:
        raise RuntimeError('Para o ensaio use fracao_base: 1.0 (amostragem pelo --por-fold).')
    print('--- Grade configurada ---',flush=True)
    main(['--planejar'],raiz=ROOT)
    print('--- Ensaio de 9 folds com base real ---',flush=True)
    pasta=main(['--ensaio','--por-fold','300'],raiz=ROOT)
    manifest=json.loads((pasta/'manifesto.json').read_text(encoding='utf-8'))
    if manifest['tipo']!='ensaio' or manifest['status']!='concluido_06_03':
        raise RuntimeError('Ensaio nao registrado como concluido e amostral.')
    pasta_oof=pasta/config['saidas']['pastas']['oof']
    tabela=pd.read_csv(pasta_oof/'resultados_grade.csv')
    if len(tabela)!=4 or not tabela['score'].between(0,1).all():
        raise RuntimeError('Resultado da grade PCA do YAML contem erros.')
    parquet=ROOT/config['dados']['arquivo']
    folds_orig, _, _=_identificar_folds(ROOT,config,parquet)
    for arq in (pasta_oof/'predicoes').glob('*.npz'):
        with np.load(arq,allow_pickle=False) as a:
            idx=a['linha_original']
            folds=a['fold']
            if len(idx)!=2700 or len(np.unique(idx))!=2700:
                raise RuntimeError('Ensaio nao contem um score por observacao amostrada.')
            if not np.array_equal(folds,folds_orig[idx]) or (folds==10).any():
                raise RuntimeError('F10 foi usado no OOF ou os folds nao batem.')
        metricas=json.loads((pasta_oof/'metricas'/(arq.stem+'.json')).read_text(encoding='utf-8'))
        if metricas['tipo_avaliacao']!='OOF_AMOSTRAL_NAO_OFICIAL':
            raise RuntimeError('Ensaio foi identificado incorretamente como oficial.')
        por_fold=pd.read_csv(pasta_oof/'metricas'/(arq.stem+'_por_fold.csv'))
        if len(por_fold)!=9 or not set(por_fold['fold'])==set(range(1,10)):
            raise RuntimeError('Metricas de algum fold ausentes.')
    if list((pasta/'teste_final').rglob('*.npz')):
        raise RuntimeError('Teste F10 nao deveria ter predições neste passo.')
    if caminho_cfg.read_bytes()!=anterior:
        raise RuntimeError('YAML oficial foi modificado.')
    print('[OK] Grade completa do YAML percorrida no ensaio: 4 candidatos x 9 folds.')
    print('[OK] Cache, resultados por candidato, AP, MCC e LogLoss persistidos.')
    print('[OK] Resultados amostrais identificados corretamente.')
    print('[OK] F10 preservado, YAML original nao alterado.')
    print('PASSO 06.03 - ENSAIO COM BASE REAL APROVADO')


if __name__=='__main__':
    main_verificar()
