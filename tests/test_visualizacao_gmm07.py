"""Testes sintéticos do catálogo GMM07 (não acessam F10 nem base real)."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import matthews_corrcoef
from sklearn.mixture import GaussianMixture

from src import visualizacao_gmm07 as s
from src import visualizacao_geometrica as geo


@pytest.fixture
def configuracao():
    path=Path(__file__).resolve().parents[1]/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    cfg=yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    cfg['visualizacao']['graficos']['sensibilidade_fronteira'].update(
        resolucao_1d=75, resolucao_2d=31, n_pontos_scatter=150)
    cfg['visualizacao']['geometria']['pontos_maximos_amostra']=180
    cfg['visualizacao']['padrao']['dpi']=58
    return cfg


def simular(d):
    rng=np.random.default_rng(710+d)
    X=rng.normal(0,1,size=(340,d))
    X[:100] += 1.0
    gm=GaussianMixture(n_components=4,covariance_type='full',random_state=7,n_init=2,
                       reg_covar=1e-4).fit(X)
    nomes=[f'PC{i+1}' for i in range(d)]
    saidas=[f'G{k+1}' for k in range(4)]
    gmm=SimpleNamespace(modelo_=gm,colunas_=nomes,saidas_=saidas)
    rh=SimpleNamespace(riscos_=pd.Series([.01,.08,.20,.39],index=saidas))
    coords=pd.DataFrame(X,columns=nomes)
    p=geo._pontuar_gmm_array(gmm,rh,X)
    y=(rng.random(len(X)) < np.clip(p+.12,0,1)).astype(np.int8)
    data={'coord':coords,'gmm':gmm,'rh':rh,'y':y,'scores':p}
    escolha={'limiar_mcc_oof':.12,'teste_final_usado':False}
    return data,escolha,{'y':y,'p':p,'origem':'simulado'}


@pytest.mark.parametrize('d', [1,2,3])
def test_comparativo_por_dimensao(configuracao, tmp_path, d):
    data, escolha, avaliacao=simular(d)
    resposta=s.gerar(configuracao,data,escolha,tmp_path,'png',58,True,avaliacao=avaliacao)
    nome=tmp_path/f'GMM07_sensibilidade_{d}d_comparativo.png'
    assert nome.is_file() and nome.stat().st_size>3000
    assert resposta['n_planos'] == (3 if d==3 else 1)
    assert resposta['origem_mcc']=='simulado'
    base=avaliacao
    assert resposta['mcc_fixo']==pytest.approx(matthews_corrcoef(base['y'],base['p']>=.5))
    assert resposta['mcc_otimo']==pytest.approx(matthews_corrcoef(base['y'],base['p']>=.12))
    planilhas=list(tmp_path.glob('GMM07_malha_*.csv'))
    assert len(planilhas)==(3 if d==3 else 1)
    for p in planilhas:
        tab=pd.read_csv(p)
        assert {'score_risco','pred_fixo','pred_otimo','mudou'}.issubset(tab.columns)
        assert np.array_equal(tab['mudou'],tab['pred_fixo'] ^ tab['pred_otimo'])
        assert not (tab['pred_fixo'] & ~tab['pred_otimo']).any()
    resumo=pd.read_csv(tmp_path/'GMM07_metricas_comparativas.csv')
    assert len(resumo)==2 and set(resumo['origem_mcc'])=={'simulado'}


def test_1d_duas_curvas_com_mesmos_pontos_mcc_simulado(configuracao,tmp_path,monkeypatch):
    data,escolha,avaliacao=simular(1)
    figura=[]
    real=geo._guardar
    def registrar(fig,*args):
        figura.append(fig)
        return real(fig,*args)
    monkeypatch.setattr(geo,'_guardar',registrar)
    s.gerar(configuracao,data,escolha,tmp_path,'png',58,False,avaliacao=avaliacao)
    assert len(figura)==1
    fig=figura[0]
    # Apenas os dois eixos do comparativo 1D.
    eixos=fig.axes
    assert len(eixos)==2
    assert np.allclose(eixos[0].lines[0].get_ydata(),eixos[1].lines[0].get_ydata())
    assert any('MCC da simulação' in a.get_title() for a in eixos)
    assert (tmp_path/'GMM07_sensibilidade_1d_comparativo.png').is_file()


def test_3d_malhas_iguais_entre_cortes(configuracao,tmp_path,monkeypatch):
    """O modulo dedicado 3D valida a mesma malha para tau fixo e tau OOF."""
    data,escolha,avaliacao=simular(3)
    s.gerar(configuracao,data,escolha,tmp_path,'png',58,True,avaliacao=avaliacao)
    png=tmp_path/'GMM07_sensibilidade_3d_comparativo.png'
    assert png.is_file() and png.stat().st_size>3000
    malhas=sorted(tmp_path.glob('GMM07_malha_*.csv'))
    assert len(malhas)==3
    for arquivo in malhas:
        df=pd.read_csv(arquivo)
        assert {'score_risco','pred_fixo','pred_otimo','mudou'}.issubset(df.columns)
        assert np.array_equal(df['mudou'].to_numpy(dtype=bool),
                              df['pred_fixo'].to_numpy(dtype=bool) ^
                              df['pred_otimo'].to_numpy(dtype=bool))
        assert df['score_risco'].between(0,1).all()
    resumo=pd.read_csv(tmp_path/'GMM07_metricas_comparativas.csv')
    assert set(resumo['origem_mcc'])=={'simulado'}


def test_limiares_iguais_e_maior_que_meio(configuracao,tmp_path):
    data,escolha,avaliacao=simular(2)
    for index,tau in enumerate((.5,.65)):
        pasta=tmp_path/str(index)
        pasta.mkdir()
        e=deepcopy(escolha)
        e['limiar_mcc_oof']=tau
        s.gerar(configuracao,data,e,pasta,'png',58,True,avaliacao=avaliacao)
        t=pd.read_csv(pasta/'GMM07_malha_PC1_PC2.csv')
        if tau==.5:
            assert not t['mudou'].any()
            assert np.array_equal(t['pred_fixo'],t['pred_otimo'])
        else:
            assert not (t['pred_otimo'] & ~t['pred_fixo']).any()


def test_bloqueia_covariancia_nao_full(configuracao,tmp_path):
    data,escolha,avaliacao=simular(1)
    data['gmm'].modelo_.covariance_type='diag'
    with pytest.raises(ValueError,match='FULL'):
        s.gerar(configuracao,data,escolha,tmp_path,'png',58,False,avaliacao=avaliacao)


def test_origem_desconhecida_nao_inventa_oof(configuracao,tmp_path):
    data,escolha,avaliacao=simular(1)
    del avaliacao['origem']
    with pytest.raises(ValueError,match='origem'):
        s.gerar(configuracao,data,escolha,tmp_path,'png',58,False,avaliacao=avaliacao)


def test_galeria_exige_gmm07_da_dimensao_correta(configuracao,tmp_path):
    from src import visualizacao_integrada as integrada
    config=deepcopy(configuracao)
    for nome in geo.TIPOS:
        config['visualizacao']['graficos'][nome]['habilitar']=(nome=='sensibilidade_fronteira')
    for nome in integrada.oof.ARQUIVOS:
        config['visualizacao']['graficos'][nome]['habilitar']=False
    e={'configuracao':{'reducao':{'n_components':1},'metodo':'PCA'},
       'id_candidato':'simulado','limiar_mcc_oof':.12}
    gd=tmp_path/'vencedor'/'visualizacao'/'geometria_desenvolvimento'
    gd.mkdir(parents=True)
    (gd/'GMM07_sensibilidade_3d_comparativo.png').write_bytes(b'imagem-errada')
    estado=integrada._verificar_geracoes(tmp_path,config,e,executar=False)
    assert not estado[5], 'Nao aceitar 3D como substituto de 1D.'
    (gd/'GMM07_sensibilidade_1d_comparativo.png').write_bytes(b'imagem-certa')
    estado=integrada._verificar_geracoes(tmp_path,config,e,executar=False)
    assert estado[5]


def test_yaml_historico_sem_gmm07_nao_e_quebrado(configuracao):
    antigo=deepcopy(configuracao)
    del antigo['visualizacao']['graficos']['sensibilidade_fronteira']
    assert not geo._ativo(antigo, 'sensibilidade_fronteira')
