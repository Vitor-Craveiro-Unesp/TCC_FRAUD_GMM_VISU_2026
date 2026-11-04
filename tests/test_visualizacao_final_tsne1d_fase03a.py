"""Fase 3A: visualizacao preditiva F10 congelado, sem tocar base real."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.folds import hash_arquivo
from src import visualizacao_final_tsne1d as vis


def preparar(tmp_path):
    p=Path(__file__).resolve().parents[1]/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    cfg=yaml.safe_load(p.read_text(encoding='utf-8-sig'))
    cfg['reducao']['metodo']='TSNE'
    cfg['tsne']['n_components']=[1]
    cfg['dados']['fracao_base']=.01
    exp=tmp_path/'outputs'/'ensaio_tsne_gmm_teste'
    t=exp/'vencedor'/'teste_final'
    model=t/'modelo_final_f1_f9'
    model.mkdir(parents=True)
    (exp/'configuracao.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    (model/'modelo_final_f1_f9.joblib').write_bytes(b'UNIDADE SINTETICA')
    rng=np.random.default_rng(42)
    y=np.r_[np.zeros(90,dtype=np.uint8),np.ones(10,dtype=np.uint8)]
    scores=rng.uniform(.02,.95,len(y))
    ids=np.arange(1000,1100,dtype=np.int64)
    tau=.3
    tabela=pd.DataFrame({'linha_original':ids, 'fold':10, 'y_true':y,
                         'prob_fraude':scores, 'predicao_tau_0_50': (scores>=.5).astype(np.uint8),
                         'predicao_tau_mcc_oof':(scores>=tau).astype(np.uint8)})
    tabela.to_csv(t/'predicoes_f10.csv',index=False,float_format='%.17g')
    np.save(t/'indices_f10_amostra.npy',ids,allow_pickle=False)
    from src.avaliacao_final_tsne import avaliar_scores_com_limiar_oof_congelado
    met=avaliar_scores_com_limiar_oof_congelado(y,scores,limiar_mcc_oof=tau)
    met.update(id_candidato_oof='fakeseguro',limiar_otimizado_no_f10=False,
               fracao_f10_dentro_do_fold=.1)
    (t/'metricas_f10.json').write_text(json.dumps(met),encoding='utf-8')
    marker={'etapa':'F10_AVALIADO_GALERIA_PENDENTE', 'metodo':'TSNE', 'dimensao':1,
            'id_candidato_oof':'fakeseguro', 'fracao_desenvolvimento':.01,
            'fracao_f10_dentro_do_fold':.1,
            'sha256_predicoes':hash_arquivo(t/'predicoes_f10.csv'),
            'sha256_metricas':hash_arquivo(t/'metricas_f10.json'),
            'sha256_indices_arquivo':hash_arquivo(t/'indices_f10_amostra.npy'),
            'sha256_modelo':hash_arquivo(model/'modelo_final_f1_f9.joblib')}
    (t/'manifesto_finalizacao_tsne.json').write_text(json.dumps(marker),encoding='utf-8')
    return exp,t


def test_preflight_sem_modelo_ou_f10_refit(tmp_path):
    exp,t=preparar(tmp_path)
    tpath,cfg,met,m,y,p,tau=vis._entradas(exp)
    assert len(y)==100 and tau==.3
    assert tpath==t
    assert cfg['dados']['fracao_base']==.01


def test_alteracao_scores_protegida(tmp_path):
    exp,t=preparar(tmp_path)
    (t/'predicoes_f10.csv').write_text('fraude',encoding='utf-8')
    with pytest.raises(RuntimeError,match='alterado'):
        vis._entradas(exp)


def test_alteracao_modelo_protegida(tmp_path):
    exp,t=preparar(tmp_path)
    (t/'modelo_final_f1_f9'/'modelo_final_f1_f9.joblib').write_bytes(b'TROCADO')
    with pytest.raises(RuntimeError,match='alterado'):
        vis._entradas(exp)


def test_amostra_nao_fixada_em_18523(tmp_path):
    exp,_=preparar(tmp_path)
    assert len(vis._entradas(exp)[4]) == 100


def test_visualizacao_completa_seis_pngs_sem_fit(tmp_path):
    exp,t=preparar(tmp_path)
    origem_hash=hash_arquivo(t/'predicoes_f10.csv')
    dest=vis.gerar(exp)
    assert dest.name=='visualizacao_preditiva'
    assert set(vis.PNG_FILES)=={p.name for p in dest.glob('*.png')}
    assert all((dest/f).read_bytes().startswith(b'\x89PNG\r\n\x1a\n') for f in vis.PNG_FILES)
    assert 'PRÉVIA PARCIAL' in (dest/'index.html').read_text(encoding='utf-8')
    assert hash_arquivo(t/'predicoes_f10.csv')==origem_hash
    assert vis.gerar(exp)==dest
    man=json.loads((dest/'manifesto_visualizacao_preditiva.json').read_text(encoding='utf-8'))
    assert man['galeria_f10_completa'] is False
    assert man['sem_novas_predicoes'] is True


def test_retomada_bloqueia_edicao_figura(tmp_path):
    exp,_=preparar(tmp_path)
    dest=vis.gerar(exp)
    (dest/vis.PNG_FILES[0]).write_bytes(b'MODIFICADO')
    with pytest.raises(RuntimeError,match='alterado'):
        vis.gerar(exp)
