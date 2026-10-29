# -*- coding: utf-8 -*-
"""Regressao sintética: GMM08 PCA1D/2D/3D e galeria HTML.
Nao altera YAML e nao executa o pipeline de OOF/F10.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import hashlib
import html
import re
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import matthews_corrcoef
from src import visualizacao_geometrica as geo
from src import visualizacao_integrada as galeria


def main():
    raiz=Path.cwd()
    f=raiz/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    assert f.is_file() and (raiz/'src'/'visualizacao_gmm08_3d.py').is_file(), ('Execute na raiz do projeto apos instalar.')
    orig=f.read_bytes()
    cfg=yaml.safe_load(orig.decode('utf-8-sig'))
    cfg['visualizacao']['geometria']['pares_2d']=[[1,2],[1,3],[2,3]]
    op=cfg['visualizacao']['graficos']['mapa_risco_fronteira']
    op.update({'resolucao_1d':160,'resolucao_2d':55,'resolucao_volume_html_3d':25,
               'n_pontos_scatter':450,'n_pontos_scatter_html_3d':550,
               'margem':.16,'mostrar_gaussianas':True,'cobertura_gaussiana':.95})
    rng=np.random.default_rng(20261027)
    medios=np.array([[-2.5,-1.4,-.5],[2.5,-1,1.3],[-.3,2.6,-1.6]])
    covs=np.array([[[.6,.2,.05],[.2,.8,-.1],[.05,-.1,.65]],
                   [[.8,-.2,.13],[-.2,.6,.02],[.13,.02,.70]],
                   [[.75,.15,-.1],[.15,.9,.2],[-.1,.2,.5]]])
    W=rng.normal(size=(3,39))
    def amostrar(n):
        L=np.vstack([rng.multivariate_normal(medios[k],covs[k],n) for k in range(3)])
        return L@W+rng.normal(scale=.53,size=(len(L),39))
    X=amostrar(125); V=amostrar(65)
    scaler=StandardScaler().fit(X)
    xt=scaler.transform(X);xv=scaler.transform(V)
    saida=raiz/'outputs'/('validacao_integracao_gmm08_pca3d_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    saida.mkdir(parents=True,exist_ok=False)
    for dim in (1,2,3):
        reducao=PCA(n_components=dim,svd_solver='full').fit(xt)
        Z=reducao.transform(xt); Zv=reducao.transform(xv)
        modelo=GaussianMixture(n_components=3,covariance_type='full',n_init=2,
                random_state=20261027,max_iter=300).fit(Z)
        assert modelo.converged_
        nomes=[f'G{k+1}' for k in range(3)]
        gm=SimpleNamespace(modelo_=modelo,saidas_=nomes,colunas_=[f'PC{k+1}' for k in range(dim)])
        riscos=np.array([.03,.32,.62]); rh=SimpleNamespace(riscos_=pd.Series(riscos,index=nomes))
        score=modelo.predict_proba(Z)@riscos
        pv=modelo.predict_proba(Zv)@riscos
        y=rng.binomial(1,score).astype(np.uint8);yv=rng.binomial(1,pv).astype(np.uint8)
        ts=np.linspace(.05,.95,451)
        mcc=np.array([matthews_corrcoef(yv,pv>=t) for t in ts])
        tau=float(ts[mcc.argmax()])
        dados={'coord':pd.DataFrame(Z,columns=gm.colunas_), 'gmm':gm,
               'rh':rh,'y':y,'scores':score}
        esc={'limiar_mcc_oof':tau}; av={'y':yv,'p':pv,'origem':'simulado'}
        pasta=saida/f'PCA_{dim}D';pasta.mkdir()
        geo._fronteira(cfg,dados,esc,pasta,'png',125,True,avaliacao=av)
        png=pasta/f'GMM08_mapa_risco_{dim}d_comparativo.png'
        assert png.is_file() and png.stat().st_size>10000, f'PNG ausente: {png}'
        inter=pasta/'GMM08_mapa_risco_3d_comparativo_INTERATIVA.html'
        if dim!=3:
            assert not inter.exists(), 'HTML 3D indevido em PCA1D/2D.'
        else:
            assert inter.is_file() and inter.stat().st_size>500000,'HTML offline ausente.'
            raw=inter.read_text(encoding='utf-8')
            normal=html.unescape(raw)
            normal=re.sub(r'\\u([0-9A-Fa-f]{4})',lambda m:chr(int(m.group(1),16)),normal)
            for el in ('GMM FULL','MCC da simulação','data-layer="g1"',
                       'data-layer="g2"','data-layer="g3"','data-layer="fraudes"',
                       'data-layer="legitimas"','data-layer="mudanca"',
                       'data-layer="volume"','Girar','Restaurar câmera'):
                assert el in normal, f'Elemento HTML nao encontrado: {el}'
            for i,j in ((1,2),(1,3),(2,3)):
                arq=pasta/f'GMM08_malha_PC{i}_PC{j}.csv'
                assert arq.is_file(), f'CSV ausente: {arq}'
                t=pd.read_csv(arq)
                assert len(t)==55**2
                assert np.array_equal(t.mudou_classe.astype(int),
                    (t.classe_corte_fixo!=t.classe_corte_mcc).astype(int))
            vol=pasta/'GMM08_malha_volumetrica_3d.csv'
            assert vol.is_file() and len(pd.read_csv(vol))==25**3
            destino=saida/'galeria_teste'/'geometria_desenvolvimento'/inter.name
            galeria._arquivo_seguro_copiar(inter,destino)
            assert destino.read_bytes()==inter.read_bytes()
        print(f'PCA {dim}D: VALIDADO')
    assert f.read_bytes()==orig,'YAML foi alterado.'
    print('GMM08 PCA 3D: validacao sintética concluida, F10 não utilizado.')
    print('PASTA='+str(saida.resolve()))
    print('PNG='+str((saida/'PCA_3D'/'GMM08_mapa_risco_3d_comparativo.png').resolve()))
    print('HTML='+str((saida/'PCA_3D'/'GMM08_mapa_risco_3d_comparativo_INTERATIVA.html').resolve()))

if __name__=='__main__':main()
