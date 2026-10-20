"""GMM 04: exemplos sintéticos, sem dados reais e sem F10."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.mixture import GaussianMixture

from src import visualizacao_gmm04 as g04
from src import visualizacao_integrada as galeria
from src import visualizacao_geometrica as geo


@pytest.fixture
def cfg():
    caminho=Path(__file__).resolve().parents[1]/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    cfg=yaml.safe_load(caminho.read_text(encoding='utf-8-sig'))
    cfg['visualizacao']['padrao']['dpi']=65
    cfg['visualizacao']['graficos']['regioes_dominancia'].update(
        resolucao_1d=200,resolucao_2d=60,pontos_scatter=100)
    return cfg


def sint(d):
    rng=np.random.default_rng(42)
    centros=[np.array([-2.,0.4,0.]),np.array([1.,-1.,2.]),np.array([2.,2.,-1.])]
    X=np.concatenate([rng.normal(size=(110,d))*.65+mu[:d] for mu in centros])
    modelo=GaussianMixture(n_components=3, covariance_type='full',n_init=1,
                           random_state=42).fit(X)
    nomes=[f'PC{i+1}' for i in range(d)]
    gmm=SimpleNamespace(modelo_=modelo,saidas_=[f'G{k+1}' for k in range(3)],colunas_=nomes)
    return {'gmm':gmm,'coord':pd.DataFrame(X,columns=nomes),
            'y':np.zeros(len(X),dtype=np.uint8)}


def test_argmax_responsabilidades_verdadeiras():
    modelo=sint(2)['gmm']
    pontos=np.array([[-2,0.5],[1,-1.],[2,2.],[0,0.]])
    ind,gamma=g04.calcular_dominancia(modelo,pontos)
    assert np.array_equal(ind,np.argmax(modelo.modelo_.predict_proba(pontos),axis=1))
    assert np.allclose(gamma.sum(axis=1),1)
    with pytest.raises(ValueError,match='Dimensoes'):
        g04.calcular_dominancia(modelo,np.ones((4,3)))


@pytest.mark.parametrize('d', [1,2,3])
def test_regioes_por_dimensao_sem_f10(cfg,tmp_path,d):
    dados=sint(d)
    g04.gerar(cfg,dados,tmp_path,'png',65,True)
    png=tmp_path/f'GMM04_dominancia_{d}d.png'
    assert png.is_file() and png.stat().st_size>10000
    tabelas=list(tmp_path.glob('GMM04_*.csv'))
    assert len(tabelas)==(1 if d in (1,2) else 3)
    for arq in tabelas:
        df=pd.read_csv(arq)
        assert df['gaussiana_dominante'].isin(['G1','G2','G3']).all()
        gs=df[[f'responsabilidade_G{i}' for i in range(1,4)]]
        assert np.allclose(gs.sum(axis=1),1,atol=1e-6)
        assert np.array_equal(gs.to_numpy().argmax(axis=1)+1,
                              df['gaussiana_dominante'].str[1:].astype(int).to_numpy())
    assert len(list(tmp_path.glob('GMM04_dominancia_*.png')))==1


def test_nao_usa_labels_para_dominancia(cfg,tmp_path):
    dados=sint(2)
    g04.gerar(cfg,dados,tmp_path,'png',65,True)
    original=pd.read_csv(tmp_path/'GMM04_malha_PC1_PC2.csv')['gaussiana_dominante'].to_numpy()
    novo=tmp_path/'rotulos_modificados'
    novo.mkdir()
    dados['y']=np.ones(len(dados['y']),dtype=np.uint8)
    g04.gerar(cfg,dados,novo,'png',65,True)
    segunda=pd.read_csv(novo/'GMM04_malha_PC1_PC2.csv')['gaussiana_dominante'].to_numpy()
    assert np.array_equal(original,segunda)


@pytest.mark.parametrize('d',[1,2,3])
def test_galeria_exige_gmm04_com_dimensao_correta(cfg,tmp_path,d):
    # Provar que nao aceita um grafico 1D historico para um vencedor 2D/3D.
    for nome in list(galeria.oof.ARQUIVOS):
        cfg['visualizacao']['graficos'][nome]['habilitar']=False
    for nome in list(geo.TIPOS):
        cfg['visualizacao']['graficos'][nome]['habilitar']=(nome=='regioes_dominancia')
    exp=tmp_path/'ensaio'; (exp/'teste_final'/'graficos_gerados').mkdir(parents=True)
    grade=[]
    esc={'id_candidato':'teste','configuracao':{'metodo':'PCA','reducao':{'n_components':d}}}
    pasta=exp/'teste_final'/'graficos_gerados'
    (pasta/'GMM04_dominancia_1d.png').write_bytes(b'png')
    _,_,_,_,_,pronto=galeria._verificar_geracoes(exp,cfg,esc,executar=False)
    assert pronto==(d==1)
    (pasta/f'GMM04_dominancia_{d}d.png').write_bytes(b'png')
    _,_,_,_,_,pronto=galeria._verificar_geracoes(exp,cfg,esc,executar=False)
    assert pronto
