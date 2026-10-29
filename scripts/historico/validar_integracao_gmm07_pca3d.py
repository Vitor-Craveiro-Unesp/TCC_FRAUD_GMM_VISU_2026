# -*- coding: utf-8 -*-
"""Teste isolado e regressao GMM07 PCA 1D/2D/3D com dados sinteticos.
Nao altera YAML, nao treina OOF, nao utiliza F10 nem toca nos ensaios.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import html
import re

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src import visualizacao_gmm07 as gmm07
from src import visualizacao_geometrica as geo
from src import visualizacao_integrada as galeria


def verificar_html(path, ncomp):
    s=path.read_text(encoding='utf-8')
    s=html.unescape(s)
    s=re.sub(r'\\u([0-9A-Fa-f]{4})',lambda m:chr(int(m.group(1),16)),s)
    for trecho in ('SENSIBILIDADE DA FRONTEIRA AO LIMIAR',
                   'data-layer="fixo"','data-layer="otimo"',
                   'data-layer="mudanca"','data-layer="legitimas"',
                   'data-layer="fraudes"','data-layer="centros"',
                   'Restaurar câmera','Limpar camadas','Girar',
                   'MCC da simulação'):
        assert trecho in s, f'Elemento ausente do HTML: {trecho}'
    for k in range(1,ncomp+1):
        assert f'data-layer="g{k}"' in s
        assert f'Centro G{k}' in s
    assert len(s)>500_000, 'HTML nao contem a biblioteca Plotly offline.'


def main():
    raiz=Path.cwd()
    arq=raiz/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    assert arq.is_file() and (raiz/'src'/'visualizacao_gmm07.py').is_file(), (
        'Execute na raiz de TCC_EXPERIMENTAL_V3.')
    raw=arq.read_bytes()
    cfg=yaml.safe_load(raw.decode('utf-8-sig'))
    op=cfg['visualizacao']['graficos']['sensibilidade_fronteira']
    op.update({'resolucao_1d':220,'resolucao_2d':55,
               'resolucao_volume_html_3d':28,
               'n_pontos_scatter':450,'n_pontos_scatter_html_3d':800,
               'margem':.16})
    rng=np.random.default_rng(20261027)
    medias=np.array([[-2.5,-1.4,-.5],[2.5,-1.,1.3],[-.3,2.6,-1.6]])
    cov=np.array([[[.6,.2,.05],[.2,.8,-.1],[.05,-.1,.65]],
                  [[.8,-.2,.13],[-.2,.6,.02],[.13,.02,.70]],
                  [[.75,.15,-.10],[.15,.9,.2],[-.10,.2,.5]]])
    W=rng.normal(size=(3,39))
    def gerar_base(n):
        latente=np.vstack([rng.multivariate_normal(medias[i],cov[i],n)
                           for i in range(3)])
        return latente@W+rng.normal(scale=.53,size=(len(latente),39))
    X=gerar_base(180)
    X_val=gerar_base(90)
    normalizador=StandardScaler().fit(X)
    xt=normalizador.transform(X)
    xv=normalizador.transform(X_val)
    raiz_saida=raiz/'outputs'/('validacao_integracao_gmm07_v2_pca3d_'+
                  datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    raiz_saida.mkdir(parents=True,exist_ok=False)
    for d in (1,2,3):
        pca=PCA(n_components=d,svd_solver='full').fit(xt)
        Z=pca.transform(xt)
        Zv=pca.transform(xv)
        modelo=GaussianMixture(n_components=3,covariance_type='full',
                   n_init=2,max_iter=300,random_state=20261027).fit(Z)
        assert modelo.converged_
        nomes=['G1','G2','G3']
        gm=SimpleNamespace(modelo_=modelo,saidas_=nomes,
                           colunas_=[f'PC{i+1}' for i in range(d)])
        riscos=np.array([.03,.32,.62])
        rh=SimpleNamespace(riscos_=pd.Series(riscos,index=nomes))
        p=modelo.predict_proba(Z)@riscos
        pv=modelo.predict_proba(Zv)@riscos
        y=rng.binomial(1,p).astype(np.int8)
        yv=rng.binomial(1,pv).astype(np.int8)
        limiares=np.linspace(.05,.95,451)
        mcc=np.array([matthews_corrcoef(yv,pv>=limiar) for limiar in limiares])
        tau=float(limiares[mcc.argmax()])
        dados={'coord':pd.DataFrame(Z,columns=gm.colunas_),
               'gmm':gm,'rh':rh,'y':y,'scores':p}
        escolha={'limiar_mcc_oof':tau}
        avaliacao={'origem':'simulado','y':yv,'p':pv}
        pasta=raiz_saida/f'PCA_{d}D'
        pasta.mkdir()
        retorno=gmm07.gerar(cfg,dados,escolha,pasta,'png',125,True,avaliacao=avaliacao)
        png=pasta/f'GMM07_sensibilidade_{d}d_comparativo.png'
        html_path=pasta/'GMM07_sensibilidade_3d_comparativo_INTERATIVA.html'
        assert png.is_file() and png.stat().st_size>12000, f'PNG nao gerado: {png}'
        assert retorno['arquivo']==png.name
        assert retorno['origem_mcc']=='simulado'
        assert retorno['n_planos']==(3 if d==3 else 1)
        if d!=3:
            assert not html_path.exists(), 'HTML 3D indevido no PCA 1D/2D.'
        else:
            assert html_path.is_file(), 'HTML 3D nao gerado.'
            verificar_html(html_path,modelo.n_components)
            for a,b in (('PC1','PC2'),('PC1','PC3'),('PC2','PC3')):
                csv=pasta/f'GMM07_malha_{a}_{b}.csv'
                assert csv.is_file(), f'CSV do plano ausente: {csv}'
                tab=pd.read_csv(csv)
                assert len(tab)==55**2
                assert np.array_equal(
                    (tab['pred_fixo']!=tab['pred_otimo']).astype('uint8'),
                    tab['mudou'].astype('uint8'))
            dest=raiz_saida/'galeria_teste'/'geometria_desenvolvimento'/html_path.name
            galeria._arquivo_seguro_copiar(html_path,dest)
            assert dest.read_bytes()==html_path.read_bytes()
        print(f'PCA {d}D: VALIDADO')
    assert arq.read_bytes()==raw,'YAML foi alterado.'
    print('GMM07 V2 oficial: PNGs, HTML, CSVs e copia segura validados.')
    print('YAML preservado. Nenhum F10 ou OOF real utilizado.')
    print('PASTA='+str(raiz_saida.resolve()))
    print('PNG='+str((raiz_saida/'PCA_3D'/'GMM07_sensibilidade_3d_comparativo.png').resolve()))
    print('HTML='+str((raiz_saida/'PCA_3D'/'GMM07_sensibilidade_3d_comparativo_INTERATIVA.html').resolve()))


if __name__=='__main__':
    main()
