import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from src import visualizacao_geometrica as g
from src import visualizacao_integrada as integ

CFG_PATH=Path(__file__).resolve().parents[1]/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'


def dados_exemplo(d=3,k=4):
    rng=np.random.default_rng(56)
    weights=np.array([.1,.2,.3,.4] if k==4 else np.ones(k)/k)
    risk=np.array([.002,.016,.003,.00225] if k==4 else np.linspace(.001,.012,k))
    names=[f'G{i+1}' for i in range(k)]
    cov=np.array([np.eye(d)*(i+1)+.1*(np.ones((d,d))-np.eye(d)) for i in range(k)])
    gmm=SimpleNamespace(saidas_=names,modelo_=SimpleNamespace(
        weights_=weights, means_=rng.normal(size=(k,d)),covariances_=cov,
        covariance_type='full'))
    rh=SimpleNamespace(riscos_=pd.Series(risk,index=names),
      massas_=pd.Series(weights*10000,index=names),
      fraudes_ponderadas_=pd.Series(weights*10000*risk,index=names),
      prevalencia_treino_=.0052,n_treino_=10000)
    return {'gmm':gmm,'rh':rh,'coord':pd.DataFrame(np.zeros((10,d)),columns=[f'PC{i+1}' for i in range(d)]),
            'n_treino_gmm':10, 'y':np.zeros(10,dtype=np.uint8)}

@pytest.fixture
def c():return yaml.safe_load(CFG_PATH.read_text(encoding='utf-8'))

def test_2x_regra_exclusiva(c):
    d=dados_exemplo()
    rows,reg=g._linhas_gaussianas(c,d)
    assert reg['prevalencia_treino']==pytest.approx(.0052)
    assert reg['limiar_alto_risco_gaussiana']==pytest.approx(.0104)
    assert [x['rotulo_descritivo'] for x in rows]==[
        'baixo_risco_fraude','alto_risco_fraude','baixo_risco_fraude','baixo_risco_fraude']
    assert c['gmm']['covariance_type']=='full'
    assert 'limiar_absoluto' not in c['visualizacao']['relatorio_gaussianas']['rotulo_descritivo']
    assert g._rotulo_gaussiana(.0104,reg)[0]=='alto_risco_fraude'
    assert g._rotulo_gaussiana(.010399,reg)[0]=='baixo_risco_fraude'

@pytest.mark.parametrize('d',[1,2,3])
def test_panel_1d_2d_3d_e_relatorios(c,d,tmp_path):
    data=dados_exemplo(d)
    g._painel_parametros_gaussianos(c,data,tmp_path,'png',80,True)
    g._relatorio(c,data,tmp_path)
    assert (tmp_path/'09_painel_parametros_gaussianos.png').is_file()
    assert len(pd.read_csv(tmp_path/'09_painel_parametros_gaussianos.csv'))==4
    j=json.loads((tmp_path/'relatorio_gaussianas_desenvolvimento.json').read_text())
    assert j['regra_rotulagem_descritiva']['limiar_alto_risco_gaussiana']==pytest.approx(.0104)
    assert len(j['gaussianas'][1]['covariancia_completa'])==d
    assert len(j['gaussianas'][1]['covariancia_completa'][0])==d
    assert pd.read_csv(tmp_path/'relatorio_gaussianas_desenvolvimento.csv')['rotulo_descritivo'].tolist()==[x['rotulo_descritivo'] for x in j['gaussianas']]
    m=(tmp_path/'relatorio_gaussianas_desenvolvimento.md').read_text()
    assert '0.52000000%' in m and '1.04000000%' in m
    assert 'F10' in m


def test_qtd_gaussianas_paginas(c,tmp_path):
    data=dados_exemplo(2,6)
    g._painel_parametros_gaussianos(c,data,tmp_path,'png',65,False)
    assert (tmp_path/'09_painel_parametros_gaussianos_parte_01.png').exists()
    assert (tmp_path/'09_painel_parametros_gaussianos_parte_02.png').exists()


def test_old_plots_disabled(c):
    assert g.TIPOS[0]=='painel_parametros_gaussianos'
    assert g.NUM['painel_parametros_gaussianos']==9
    assert integ.GEO_PREFIXOS['painel_parametros_gaussianos'].startswith('09_')
    for nome in ('risco_por_gaussiana','pesos_responsabilidades','medias_gaussianas','covariancias_gaussianas'):
        assert nome not in g.TIPOS
        assert nome not in c['visualizacao']['graficos']


def test_estrategia_incorreta_falha(c):
    c['visualizacao']['relatorio_gaussianas']['rotulo_descritivo']['estrategia']='limiar_absoluto'
    with pytest.raises(ValueError,match='relativo_prevalencia'):
        g._linhas_gaussianas(c,dados_exemplo())
