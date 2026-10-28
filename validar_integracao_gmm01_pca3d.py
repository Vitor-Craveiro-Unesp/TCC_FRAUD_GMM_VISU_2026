# -*- coding: utf-8 -*-
"""Auditoria da integracao oficial GMM01 em PCA 1D/2D/3D (sintetica).
Nao ajusta o pipeline OOF nem toca em F10. YAML somente leitura.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yaml
from sklearn.mixture import GaussianMixture
from src import visualizacao_gmm01 as gmm01

raiz=Path.cwd()
yaml_file=raiz/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
assert yaml_file.is_file(),'Execute na raiz do TCC_EXPERIMENTAL_V3.'
before=yaml_file.read_bytes()
config=yaml.safe_load(before.decode('utf-8-sig'))
assert (raiz/'src'/'visualizacao_gmm01_3d.py').is_file(), 'Instale primeiro o modulo 3D.'
assert '# PCA3D_GMM01_EXTRAS_OFICIAIS_V1' in (
    raiz/'src'/'visualizacao_gmm01.py').read_text(encoding='utf-8-sig')

rng=np.random.default_rng(20261027)
sup_dir=raiz/'outputs'/('validacao_integracao_gmm01_3d_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
sup_dir.mkdir(parents=True,exist_ok=False)

for dim in (1,2,3):
    n=60
    centros=np.array([[-2.3, .3, -.4],[2.5,-.7,1.3],[.2,2.5,-1.2]])[:,:dim]
    dados=np.vstack([rng.normal(loc=centros[k],scale=.85,size=(n,dim)) for k in range(3)])
    colunas=[f'PC{j+1}' for j in range(dim)]
    df=pd.DataFrame(dados,columns=colunas)
    gm=GaussianMixture(n_components=3,covariance_type='full',n_init=2,max_iter=200,
                       random_state=20261027).fit(df)
    assert gm.covariance_type=='full' and gm.converged_
    obj=SimpleNamespace(modelo_=gm,saidas_=['G1','G2','G3'],colunas_=colunas)
    output=sup_dir/f'PCA_{dim}D'
    output.mkdir()
    gmm01.gerar(config,{'coord':df,'gmm':obj},output,'png',90,True)
    for col in colunas:
        expected=output/f'GMM01_densidade_marginal_{dim}d_{col}.png'
        assert expected.is_file(),f'Ausente: {expected}'
    if dim==3:
        nomes=[
            'GMM01_conjunta_corte_PC1_PC2.png',
            'GMM01_conjunta_corte_PC1_PC3.png',
            'GMM01_conjunta_corte_PC2_PC3.png',
            'GMM01_geometria_gaussianas_FULL_3d_95.png',
            'GMM01_densidade_conjunta_PCA3D_INTERATIVA.html',
        ]
        for nome in nomes:
            p=output/nome
            assert p.is_file() and p.stat().st_size>5000,f'Ausente/vazio: {p}'
        assert len(list(output.glob('GMM01_conjunta_corte_*_plano_G*.csv')))==9
    else:
        assert not list(output.glob('GMM01_conjunta_corte_*.png')), 'PCA 1D/2D alterado.'
    print(f'PCA {dim}D: VALIDADO ({len(list(output.glob("*.png")))} PNGs)')

assert before==yaml_file.read_bytes(),'YAML foi alterado inesperadamente.'
print('YAML preservado. Nenhum F10 usado. Sem OOF completo.')
print('PASTA='+str(sup_dir.resolve()))
print('HTML='+str((sup_dir/'PCA_3D'/'GMM01_densidade_conjunta_PCA3D_INTERATIVA.html').resolve()))
print('PNG='+str((sup_dir/'PCA_3D'/'GMM01_conjunta_corte_PC1_PC3.png').resolve()))
