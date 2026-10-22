"""GMM 01: densidades marginais ponderadas; apenas dados sinteticos."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import norm

from src import visualizacao_gmm01 as m
from src import visualizacao_geometrica as geo
from src import visualizacao_integrada as galeria


def _simular(d):
    rng=np.random.default_rng(814)
    pesos=np.array([.2,.3,.5])
    medias=np.array([[-2, .2, .6], [0, -1, 1.2], [2.5, 1.5, -.9]])[:,:d]
    matriz_base=np.array([[.8,.25,.14],[.25,1.2,-.12],[.14,-.12,.6]])
    covs=np.stack([(1.0+.15*k)*matriz_base[:d,:d] for k in range(3)])
    modelo=SimpleNamespace(weights_=pesos,means_=medias,covariance_type='full',covariances_=covs)
    gm=SimpleNamespace(modelo_=modelo,saidas_=['G1','G2','G3'])
    obs=np.concatenate([rng.multivariate_normal(medias[k],covs[k],size=95) for k in range(3)])
    df=pd.DataFrame(obs,columns=[f'PC{k+1}' for k in range(d)])
    return {'coord':df,'gmm':gm}


def _config():
    path=Path(__file__).resolve().parents[1]/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    return yaml.safe_load(path.read_text(encoding='utf-8-sig'))


@pytest.mark.parametrize('d',[1,2,3])
def test_figuras_separadas_por_dimensao(d,tmp_path,monkeypatch):
    c=_config();c['visualizacao']['graficos']['densidades_marginais']['pontos_curva']=800
    capturas=[]
    anterior=geo._guardar
    def registrar(fig,dest,stem,ext,dpi):
        linhas=fig.axes[0].get_lines()
        capturas.append((stem,[l.get_label() for l in linhas],[l.get_color() for l in linhas]))
        return anterior(fig,dest,stem,ext,dpi)
    monkeypatch.setattr(geo,'_guardar',registrar)
    m.gerar(c,_simular(d),tmp_path,'png',75,True)
    assert len(capturas)==d
    for j in range(d):
        stem=f'GMM01_densidade_marginal_{d}d_PC{j+1}'
        assert (tmp_path/f'{stem}.png').is_file()
        tabela=pd.read_csv(tmp_path/f'{stem}.csv')
        assert len(tabela)==800
        assert np.allclose(tabela['densidade_mistura'],tabela[[f'contribuicao_G{k}' for k in range(1,4)]].sum(axis=1))
        assert set(tabela.columns)=={'coordenada','densidade_mistura','contribuicao_G1','contribuicao_G2','contribuicao_G3'}
        assert capturas[j][1][-1]=='Mistura marginal total'
        assert all(f'G{k} ' in cap for k,cap in enumerate(capturas[j][1][:3],start=1))


def test_formula_peso_e_covariancia_full():
    data=_simular(2)
    t=np.linspace(-4,4,301)
    comps,total=m.calcular_marginal(data['gmm'],0,t)
    model=data['gmm'].modelo_
    esperado=model.weights_[1]*norm.pdf(t,loc=model.means_[1,0],scale=np.sqrt(model.covariances_[1,0,0]))
    assert np.allclose(comps[1],esperado)
    assert np.allclose(total,np.sum(comps,axis=0))
    assert all(np.trapezoid(comps[k],t) <= model.weights_[k]+1e-8 for k in range(3))
    assert (np.asarray(model.covariances_)[:,0,1]!=0).all()


def test_bloqueio_dimensao_e_pesos(tmp_path):
    cfg=_config()
    data=_simular(3)
    data['gmm'].modelo_.weights_=np.array([.2,.2,.2])
    with pytest.raises(ValueError,match='Pesos'):
        m.gerar(cfg,data,tmp_path,'png',60,False)
    data=_simular(3)
    data['coord']['PC4']=0.0
    with pytest.raises(ValueError,match='1D, 2D ou 3D'):
        m.gerar(cfg,data,tmp_path,'png',60,False)


def test_yaml_e_congelamento_compatibilidade():
    c=_config()
    assert geo._ativo(c,'densidades_marginais')
    antigo=copy.deepcopy(c)
    del antigo['visualizacao']['graficos']['densidades_marginais']
    assert not geo._ativo(antigo,'densidades_marginais')


def test_galeria_exige_numero_exato_de_marginais(tmp_path,monkeypatch):
    cfg=_config()
    for nome in cfg['visualizacao']['graficos']:
        cfg['visualizacao']['graficos'][nome]['habilitar']=(nome=='densidades_marginais')
    exp=tmp_path/'experimento'
    (exp/'vencedor'/'visualizacao'/'geometria_desenvolvimento').mkdir(parents=True)
    dest=exp/'vencedor'/'visualizacao'/'geometria_desenvolvimento'
    choice={'configuracao':{'reducao':{'n_components':2},'metodo':'PCA'},'id_candidato':'sint'}
    # Planejar apenas verifica, sem treinar nem ler F10.
    monkeypatch.setattr(galeria.oof,'_ler_resultados',lambda _: (cfg,None,choice,None,None))
    *_,tem_oof,tem_geo=galeria._verificar_geracoes(exp,cfg,choice,executar=False)
    assert not tem_geo
    (dest/'GMM01_densidade_marginal_2d_PC1.png').write_bytes(b'png')
    *_,_,tem_geo=galeria._verificar_geracoes(exp,cfg,choice,executar=False)
    assert not tem_geo
    (dest/'GMM01_densidade_marginal_2d_PC2.png').write_bytes(b'png')
    *_,_,tem_geo=galeria._verificar_geracoes(exp,cfg,choice,executar=False)
    assert tem_geo
