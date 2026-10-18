"""Testes isolados do Passo 07.01; nunca usam F10 nem o dataset real."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import average_precision_score, log_loss, matthews_corrcoef

from src.visualizacao_oof import gerar_graficos_oof, _mcc_curva_exata


@pytest.fixture
def experimento(tmp_path):
    cfg_path = Path(__file__).resolve().parents[1] / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    if cfg_path.is_file():
        config=yaml.safe_load(cfg_path.read_text(encoding='utf-8-sig'))
    else:
        config={
            'saidas':{'pastas':{'oof':'oof','graficos':'graficos_gerados'},'salvar_graficos':True},
            'visualizacao':{'habilitar':True,'gerar_em':['oof','teste_final'],
                'padrao':{'figsize':[9,6],'formato':'png','dpi':100,'salvar_dados_grafico':True},
                'graficos':{name:{'habilitar':True,'gerar_em':['oof']} for name in (
                  'curva_pr','curva_mcc','matriz_confusao_05','matriz_confusao_mcc',
                  'distribuicao_probabilidades','comparacao_hiperparametros')},
                'estilo_matrizes':{}}}
    config['visualizacao']['padrao']['dpi']=70
    config['visualizacao']['padrao']['figsize']=[6,4]
    config['visualizacao']['graficos']['curva_mcc']['pontos_visualizacao']=20
    config['visualizacao']['graficos']['distribuicao_probabilidades']['bins']=10
    config['visualizacao']['graficos']['comparacao_hiperparametros']['maximo_configuracoes']=3
    root=tmp_path/'ensaio_pca_gmm_20261017_195252'
    pasta=root/'oof';(pasta/'predicoes').mkdir(parents=True)
    (root/'configuracao.yaml').write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf-8')
    rng=np.random.default_rng(42)
    n=360
    y=rng.choice([0,1],n,p=[.92,.08]).astype(np.uint8)
    p=np.clip(rng.beta(2,18,n) + y*.07,1e-5,1-1e-5)
    limiar=.13
    ap=average_precision_score(y,p)
    mcc=matthews_corrcoef(y,p>=limiar)
    ll=log_loss(y,p,labels=[0,1]);score=(ap+(mcc+1)/2+np.exp(-ll))/3
    cid='abc012'
    cand={'metodo':'PCA','modo':'NORMAL','razao':None,'reducao':{'n_components':2},'gmm':{'n_components':4},'suavizacao':0.0}
    grade=pd.DataFrame([{'id_candidato':cid,'ap':ap,'mcc':mcc,'logloss':ll,'score':score,
        'limiar_mcc':limiar,'n':n,'prevalencia':y.mean(),
        'configuracao':json.dumps(cand)},
        {'id_candidato':'xyz999','ap':ap/2,'mcc':mcc/2,'logloss':ll,'score':score-.1,
         'limiar_mcc':limiar,'n':n,'prevalencia':y.mean(),
         'configuracao':json.dumps({**cand,'gmm':{'n_components':2}})}])
    grade.to_csv(pasta/'resultados_grade.csv',index=False)
    escolha={'id_candidato':cid,'configuracao':cand,'limiar_mcc_oof':limiar,
             'score_oof':score,'status':'selecionado_em_ensaio','teste_final_usado':False}
    (pasta/'melhor_configuracao.json').write_text(json.dumps(escolha),encoding='utf-8')
    np.savez_compressed(pasta/'predicoes'/f'{cid}.npz',
        y_true=y,prob_fraude=p,fold=np.tile(np.arange(1,10,dtype=np.uint8),n//9),
        linha_original=np.arange(n,dtype=np.int64),id_candidato=np.array(cid),
        fold_esperado=np.array(0,dtype=np.uint8))
    return root


def test_gera_seis_tipos_arquivos_e_csv(experimento):
    resumo=gerar_graficos_oof(experimento)
    pasta=experimento/'oof'/'graficos_gerados'
    assert len(resumo['tipos_gerados'])==6
    assert len(list(pasta.glob('*.png')))==6
    assert len(list(pasta.glob('*.csv')))==6
    assert resumo['usa_f10'] is False and resumo['reajustou_modelo'] is False
    assert resumo['id_vencedor']=='abc012'
    assert (pasta/'manifesto_graficos_oof_07_01.json').is_file()


def test_idempotente_e_preserva_fontes(experimento):
    fontes=[experimento/'configuracao.yaml',experimento/'oof'/'resultados_grade.csv',
            experimento/'oof'/'melhor_configuracao.json',
            experimento/'oof'/'predicoes'/'abc012.npz']
    antes={str(f):f.read_bytes() for f in fontes}
    gerar_graficos_oof(experimento)
    png=experimento/'oof'/'graficos_gerados'/'01_curva_precision_recall.png'
    bytes_png=png.read_bytes()
    gerar_graficos_oof(experimento)
    assert png.read_bytes()==bytes_png
    assert all(f.read_bytes()==antes[str(f)] for f in fontes)


def test_respeita_flag_yaml_desativada(experimento):
    caminho=experimento/'configuracao.yaml'
    c=yaml.safe_load(caminho.read_text(encoding='utf-8'))
    c['visualizacao']['graficos']['curva_mcc']['habilitar']=False
    caminho.write_text(yaml.safe_dump(c),encoding='utf-8')
    resumo=gerar_graficos_oof(experimento)
    assert len(resumo['tipos_gerados'])==5
    assert 'curva_mcc' in resumo['tipos_omitidos_por_yaml']
    assert not (experimento/'oof'/'graficos_gerados'/'02_mcc_por_limiar.png').exists()


def test_bloqueia_f10(experimento):
    arq=experimento/'oof'/'predicoes'/'abc012.npz'
    with np.load(arq) as z:
        data={key:z[key] for key in z.files}
    data['fold'][0]=10
    np.savez_compressed(arq,**data)
    with pytest.raises(ValueError,match='F1..F9'):
        gerar_graficos_oof(experimento)


def test_bloqueia_vencedor_inconsistente(experimento):
    arquivo=experimento/'oof'/'melhor_configuracao.json'
    escolha=json.loads(arquivo.read_text())
    escolha['id_candidato']='xyz999'
    arquivo.write_text(json.dumps(escolha))
    with pytest.raises(ValueError,match='Vencedor divergente'):
        gerar_graficos_oof(experimento)


def test_mcc_vetorizado_inclui_empates_sem_quebrar():
    y=np.array([1,0,0,1,1,0,0,0],dtype=np.uint8)
    p=np.array([.8,.8,.4,.4,.4,.1,.1,.1])
    thresholds, valores=_mcc_curva_exata(y,p)
    assert np.array_equal(thresholds,np.array([.8,.4,.1]))
    assert np.allclose(valores,[matthews_corrcoef(y,p>=t) for t in thresholds])
