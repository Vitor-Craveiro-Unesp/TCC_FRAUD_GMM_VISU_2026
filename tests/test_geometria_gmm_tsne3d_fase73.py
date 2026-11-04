# -*- coding: utf-8 -*-
"""Fase 7.3: GMM FULL t-SNE 3D isolado; nao usa dados reais/F10."""
from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture

from src.visualizacao_geometria_tsne3d import (
    EIXOS, MANIFESTO, NOMES, _elipsoide, _escores,
    gerar_geometria_tsne3d,
)


@pytest.fixture
def exemplo():
    rng = np.random.default_rng(123)
    X = np.vstack([
        rng.normal(loc=(0,0,0),scale=.8,size=(45,3)),
        rng.normal(loc=(2,2,-1),scale=.5,size=(30,3)),
        rng.normal(loc=(-1,1,2),scale=.35,size=(25,3)),
    ])
    df = pd.DataFrame(X, columns=EIXOS)
    y = np.zeros(len(X), dtype=int)
    y[::11]=1
    model = GaussianMixture(n_components=3, covariance_type='full', n_init=1, random_state=42).fit(X)
    risco = np.array([.08,.4,.7])
    scores = model.predict_proba(X) @ risco
    return df, y, model, risco, scores


def _chamar(exemplo, destino, **kwargs):
    df,y,model,risco,scores=exemplo
    return gerar_geometria_tsne3d(df,y,model,risco,destino,limiar_oof=.18,
        y_oof=y,scores_oof=scores,origem='simulado',dpi=55,
        resolucao_cortes=45,resolucao_volume=15,**kwargs)


def test_oito_graficos_offline_manifesto_csv(tmp_path, exemplo):
    out = _chamar(exemplo,tmp_path)
    assert set(out) == {n for par in NOMES.values() for n in par} | {MANIFESTO,'GMM_TSNE3D_malha_risco.csv'}
    for k, (png, html) in NOMES.items():
        assert out[png].read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        pg = out[html].read_text(encoding='utf-8')
        assert 'Plotly.newPlot' in pg and 'TSNE1' in pg and 'TSNE3' in pg
        assert not re.search(r'<script[^>]+\bsrc\s*=', pg, flags=re.IGNORECASE)
        assert 'PC1' not in pg and 'PC2' not in pg and 'PC3' not in pg
    for k in ('GMM07','GMM08'):
        txt=out[NOMES[k][1]].read_text(encoding='utf-8')
        assert 'scene2' in txt and '0.5000000' in txt and '0.1800000' in txt
        assert 'PREVIA SINTETICA' in txt
    ma=json.loads(out[MANIFESTO].read_text(encoding='utf-8'))
    assert ma['n_observacoes']==100
    assert ma['n_gaussianas']==3
    assert ma['nenhum_modelo_ajustado_durante_geracao'] and ma['nenhum_limiar_otimizado']
    assert ma['origem_mcc']=='simulado'
    for nome,hash in ma['hashes_sha256'].items():
        assert sha256(out[nome].read_bytes()).hexdigest()==hash
    malha=pd.read_csv(out['GMM_TSNE3D_malha_risco.csv'])
    assert {'TSNE1','TSNE2','TSNE3','score_risco','classe_tau_050','classe_tau_oof','regiao_mudanca'}<=set(malha.columns)
    assert np.array_equal(malha.regiao_mudanca.to_numpy(),
             (malha.classe_tau_050!=malha.classe_tau_oof).to_numpy())
    assert len(malha)==15**3


def test_scores_a_mesma_mistura_e_responsabilidades(exemplo):
    df,y,model,riscos,p=exemplo
    x=df.to_numpy()
    score,resp=_escores(model,riscos,x)
    np.testing.assert_allclose(score,model.predict_proba(x)@riscos,rtol=1e-12)
    np.testing.assert_allclose(resp.sum(axis=1),np.ones(len(x)),atol=1e-12)


def test_elipsoide_mahalanobis_full_95(exemplo):
    df,y,model,riscos,p=exemplo
    from scipy.stats import chi2
    for mu,cov in zip(model.means_,model.covariances_):
        x,y,z=_elipsoide(mu,cov)
        pts=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
        delta=pts-mu
        d2=np.einsum('ni,ij,nj->n',delta,np.linalg.inv(cov),delta)
        np.testing.assert_allclose(d2,chi2.ppf(.95,df=3),atol=1e-9)


def test_nao_sobrescreve(tmp_path, exemplo):
    out=_chamar(exemplo,tmp_path)
    b=out[NOMES['GMM08'][1]].read_bytes()
    with pytest.raises(FileExistsError,match='sobrescrever'):
        _chamar(exemplo,tmp_path)
    assert out[NOMES['GMM08'][1]].read_bytes()==b


def test_f10_requer_fracao_e_registra_origem(tmp_path, exemplo):
    df,y,model,riscos,scores=exemplo
    kwargs=dict(limiar_oof=.18,y_oof=y,scores_oof=scores,
                origem='f10',dpi=55,resolucao_cortes=45,resolucao_volume=15)
    with pytest.raises(ValueError,match='fracao_f10'):
        gerar_geometria_tsne3d(df,y,model,riscos,tmp_path,**kwargs)
    out=gerar_geometria_tsne3d(df,y,model,riscos,tmp_path,fracao_f10=.1,**kwargs)
    m=json.loads(out[MANIFESTO].read_text(encoding='utf-8'))
    assert m['fracao_f10_dentro_do_fold']==.1
    assert m['origem_mcc']=='oof_f1_a_f9'
    assert 'F10 e exploratorio' in m['avisos'][0]


@pytest.mark.parametrize('erro',[
    'coord_dim','coord_nome','nan','classe','risco','tau','pred_oof',
    'modelo_diag','peso','cov_indevida','origem','fracao',
])
def test_rejeita_entradas_invalidas(tmp_path,exemplo,erro):
    df,y,model,riscos,scores=exemplo
    df=df.copy();y=y.copy();riscos=riscos.copy();scores=scores.copy()
    tau=.18
    origem='simulado';frac=None
    if erro=='coord_dim':df=df.drop(columns=['TSNE3'])
    elif erro=='coord_nome':df.columns=['PC1','PC2','PC3']
    elif erro=='nan':df.loc[0,'TSNE1']=np.nan
    elif erro=='classe':y[0]=3
    elif erro=='risco':riscos[0]=1.2
    elif erro=='tau':tau=1.3
    elif erro=='pred_oof':scores[0]=np.nan
    elif erro=='modelo_diag':model.covariance_type='diag'
    elif erro=='peso':model.weights_[0]=-1
    elif erro=='cov_indevida':model.covariances_[0,0,0]=-1
    elif erro=='origem':origem='desconhecido'
    elif erro=='fracao':origem='f10';frac=0
    with pytest.raises(ValueError):
        gerar_geometria_tsne3d(df,y,model,riscos,tmp_path,
            limiar_oof=tau,y_oof=y,scores_oof=scores,origem=origem,
            fracao_f10=frac,dpi=55,resolucao_cortes=45,resolucao_volume=15)


def test_sem_modificar_objetos_de_entrada(tmp_path,exemplo):
    df,y,model,riscos,scores=exemplo
    antes=(df.copy(deep=True),y.copy(),model.means_.copy(),model.covariances_.copy(),
           riscos.copy(),scores.copy())
    _chamar(exemplo,tmp_path)
    pd.testing.assert_frame_equal(df,antes[0])
    for cur,ant in zip((y,model.means_,model.covariances_,riscos,scores),antes[1:]):
        assert np.array_equal(cur,ant)


def test_bloqueio_cli_sem_demo(tmp_path):
    import subprocess,sys
    proc=subprocess.run([sys.executable,'-m','src.visualizacao_geometria_tsne3d',
                         '--saida',str(tmp_path)],capture_output=True,text=True)
    assert proc.returncode!=0
    assert 'Somente --demo' in proc.stderr


def test_k4_full_e_4_elipsoides(tmp_path):
    rng=np.random.default_rng(44)
    pontos=np.vstack([rng.normal(loc=k*2,scale=.5,size=(23,3)) for k in range(4)])
    df=pd.DataFrame(pontos,columns=EIXOS)
    y=(np.arange(len(df))%15==0).astype(int)
    gm=GaussianMixture(n_components=4,covariance_type='full',random_state=42,n_init=1).fit(pontos)
    riscos=np.array([.02,.11,.41,.62])
    s=gm.predict_proba(pontos)@riscos
    out=gerar_geometria_tsne3d(df,y,gm,riscos,tmp_path,limiar_oof=.2,
        y_oof=y,scores_oof=s,origem='simulado',dpi=55,
        resolucao_cortes=45,resolucao_volume=15)
    ma=json.loads(out[MANIFESTO].read_text(encoding='utf-8'))
    assert ma['n_gaussianas']==4
    pagina=out[NOMES['GMM04'][1]].read_text(encoding='utf-8')
    assert all(f'G{k}' in pagina for k in range(1,5))
