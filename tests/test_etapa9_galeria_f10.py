# -*- coding: utf-8 -*-
"""Testes sintéticos: NÃO usa F10 real, arquivos reais, Parquet ou modelos."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

CODIGO = Path(__file__).resolve().parents[1] / 'scripts' / 'execucao' / 'ETAPA9_GERAR_GALERIA_F10_PCA.py'


def carregar():
    spec = importlib.util.spec_from_file_location('etapa9_isolada', CODIGO)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def salvar_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def conjunto_artificial(tmp_path, mod):
    """Três CSVs sobre a MESMA população FICTÍCIA (60 transações)."""
    rng = np.random.default_rng(37)
    root = tmp_path
    pasta = root / 'outputs' / 'avaliacao_final_f10_pca_10pct_SINTETICA'
    pasta.mkdir(parents=True)
    pre = root / 'outputs' / 'protocolo_pre_f10_sintetico' / 'manifesto_pre_f10.json'
    pre.parent.mkdir()
    dims = (1,2,3)
    limiares = {1:.04, 2:.02, 3:.06}
    pre_modelos = [{'pca_dimensao': d, 'limiar_mcc_oof_congelado': limiares[d],
                    'candidato_oof': f'candidato-d{d}'} for d in dims]
    salvar_json(pre, {'situacao':'PREFLIGHT_CONGELADO_SEM_AVALIACAO',
                     'modelos':pre_modelos, 'protocolo_teste_pre_registrado':{'fracao_teste':.1}})
    y = np.r_[np.zeros(51, dtype=int), np.ones(9, dtype=int)]
    indices = np.arange(500,560, dtype=int)
    entradas = []
    for d in dims:
        # Alguns exemplos intencionalmente acima e abaixo dos cortes.
        scores = rng.beta(1.3, 20-d, size=60)
        scores[-9:] = rng.beta(2, 10, size=9)
        tau = limiares[d]
        df = pd.DataFrame({
            'linha_original': indices,
            'fold': np.full(60,10,dtype=int),
            'y_true': y,
            'prob_fraude': scores,
            'predicao_tau_0_50': (scores >= 0.5).astype(int),
            'predicao_tau_mcc_oof': (scores >= tau).astype(int),
        })
        from sklearn.metrics import average_precision_score, log_loss, matthews_corrcoef, precision_score, recall_score
        def corte(t):
            p = (scores >= t).astype(int)
            tn = int(((y==0)&(p==0)).sum()); fp=int(((y==0)&(p==1)).sum())
            fn = int(((y==1)&(p==0)).sum()); tp=int(((y==1)&(p==1)).sum())
            return {'limiar':t, 'mcc':float(matthews_corrcoef(y,p)), 'precision':float(precision_score(y,p,zero_division=0)),
                    'recall_sensibilidade':float(recall_score(y,p,zero_division=0)),
                    'positivos_preditos': int(p.sum()),
                    'matriz_confusao':{'tn':tn,'fp':fp,'fn':fn,'tp':tp,
                         'contagens':[[tn,fp],[fn,tp]],
                         'percentuais_por_classe_real':[[100*tn/51,100*fp/51],[100*fn/9,100*tp/9]]}}
        c05=corte(.5); ct=corte(tau)
        AP=float(average_precision_score(y,scores))
        loss=float(log_loss(y,scores, labels=[0,1]))
        import math
        score=float((AP+(ct['mcc']+1)/2+math.exp(-loss))/3)
        m={'dimensao_pca': d, 'id_candidato_oof':f'candidato-d{d}', 'n':60,'n_legitimas':51,
           'n_fraudes':9,'prevalencia':.15, 'ap':AP,'logloss':loss,
           'mcc_limiar_oof':ct['mcc'],'score_composto_limiar_oof':score,
           'limiar_mcc_oof_congelado':tau,'limiar_otimizado_no_f10':False,
           'comparacao_limiares':{'tau_0_50':c05,'tau_mcc_oof':ct}}
        sub=pasta / f'PCA_{d}D';sub.mkdir()
        df.to_csv(sub/'predicoes_f10.csv', index=False, float_format='%.17g')
        salvar_json(sub/'metricas_f10.json', m)
        entradas.append({'pca_dimensao':d, 'candidato_oof':f'candidato-d{d}',
             'limiar_mcc_oof_congelado':tau,
             'predicoes_f10':str((sub/'predicoes_f10.csv').relative_to(root)),
             'sha256_predicoes_f10':sha(sub/'predicoes_f10.csv'),
             'metricas_f10':str((sub/'metricas_f10.json').relative_to(root)),
             'sha256_metricas_f10':sha(sub/'metricas_f10.json'),
             'ap':AP,'mcc_limiar_oof':ct['mcc'],'logloss':loss,'score_composto_limiar_oof':score})
    manifesto={'versao':'PCA_F10_10PCT_V1','carater':'AVALIACAO_EXPLORATORIA_AMOSTRAL',
        'f10_avaliado':True,'otimizacao_limiar_em_f10':False,'selecionar_vencedor_em_f10':False,
        'amostra_f10':{'tipo':'AMOSTRA_F10_10PCT_ESTRATIFICADA_PRE_REGISTRADA',
                      'n_linhas':60,'n_legitimas':51,'n_fraudes':9,'semente':42,'fracao':.1},
        'referencia_protocolo':str(pre.relative_to(root)),
        'sha256_protocolo_pre_f10':sha(pre), 'resultados_por_dimensao':entradas}
    salvar_json(pasta/'manifesto_avaliacao_f10.json',manifesto)
    mod.RAIZ = root
    return pasta, manifesto


def test_verificacao_sintetica_sem_escrever(tmp_path):
    m=carregar()
    pasta,_=conjunto_artificial(tmp_path,m)
    antes={str(f):sha(f) for f in tmp_path.rglob('*') if f.is_file()}
    manifesto, resultados=m.conferir_fonte(pasta)
    depois={str(f):sha(f) for f in tmp_path.rglob('*') if f.is_file()}
    assert set(resultados)=={1,2,3}
    assert antes==depois
    assert manifesto['f10_avaliado']


def test_detecta_predicoes_adulteradas(tmp_path):
    m=carregar(); pasta,_=conjunto_artificial(tmp_path,m)
    csv=pasta/'PCA_2D'/'predicoes_f10.csv'
    csv.write_bytes(csv.read_bytes()+b'\n')
    with pytest.raises(RuntimeError, match='CSV PCA 2D alterado'):
        m.conferir_fonte(pasta)


def test_detecta_limiar_alterado_no_protocolo(tmp_path):
    m=carregar(); pasta,meta=conjunto_artificial(tmp_path,m)
    met=pasta/'PCA_3D'/'metricas_f10.json'
    conteudo=json.loads(met.read_text(encoding='utf8'))
    conteudo['limiar_mcc_oof_congelado']=.9
    salvar_json(met, conteudo)
    meta['resultados_por_dimensao'][2]['sha256_metricas_f10']=sha(met)
    salvar_json(pasta/'manifesto_avaliacao_f10.json', meta)
    with pytest.raises(RuntimeError, match='Tau manifesto divergente'):
        m.conferir_fonte(pasta)


def test_detecta_troca_do_modelo_pre_registrado(tmp_path):
    m=carregar(); pasta,meta=conjunto_artificial(tmp_path,m)
    meta['resultados_por_dimensao'][0]['limiar_mcc_oof_congelado'] = .2222
    salvar_json(pasta/'manifesto_avaliacao_f10.json',meta)
    with pytest.raises(RuntimeError, match='diferente do PRE-F10'):
        m.conferir_fonte(pasta)


def test_detecta_amostras_diferentes(tmp_path):
    m=carregar(); pasta,meta=conjunto_artificial(tmp_path,m)
    csv=pasta/'PCA_3D'/'predicoes_f10.csv'
    df=pd.read_csv(csv); df.loc[0,'linha_original']=400
    df.to_csv(csv,index=False,float_format='%.17g')
    meta['resultados_por_dimensao'][2]['sha256_predicoes_f10']=sha(csv)
    salvar_json(pasta/'manifesto_avaliacao_f10.json',meta)
    with pytest.raises(RuntimeError, match='Diferenca na amostra'):
        m.conferir_fonte(pasta)


def test_gera_galeria_sintetica_em_nova_pasta(tmp_path):
    m=carregar(); pasta,_=conjunto_artificial(tmp_path,m)
    original={str(f):sha(f) for f in pasta.rglob('*') if f.is_file()}
    manifest, dados=m.conferir_fonte(pasta)
    destino=m.gerar(pasta,manifest,dados)
    assert (destino/'index.html').exists()
    assert len(list(destino.rglob('*.png')))==13
    assert (destino/'manifesto_galeria_f10.json').exists()
    assert original=={str(f):sha(f) for f in pasta.rglob('*') if f.is_file()}
    assert 'AVALIACAO_EXPLORATORIA_AMOSTRAL' in json.loads((pasta/'manifesto_avaliacao_f10.json').read_text())['carater']
