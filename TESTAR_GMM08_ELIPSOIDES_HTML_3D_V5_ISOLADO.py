# -*- coding: utf-8 -*-
"""Renderiza um GMM08 3D sintético sem tocar nos experimentos reais."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import importlib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def main():
    raiz=Path(__file__).resolve().parent
    from src import gmm_contornos_quatro_niveis as vis
    marca='GMM08_HTML_3D_ELIPSOIDES_FULL_95_V5'
    if marca not in Path(vis.__file__).read_text(encoding='utf-8-sig'):
        raise RuntimeError('Instale V5 primeiro: versao nao encontrada no fonte.')
    rng=np.random.default_rng(88)
    X=np.vstack([
        rng.multivariate_normal([-2,0,0],np.diag([1.2,.8,1.1]),size=170),
        rng.multivariate_normal([2,0,1],np.diag([1.,1.,.8]),size=180),
        rng.multivariate_normal([0,2,-2],np.diag([1.4,.7,1.]),size=170),
    ])
    # Modelo SOMENTE sintético, sem usar dados, treinamento ou F10 reais.
    gm=GaussianMixture(n_components=3,covariance_type='full',n_init=1,random_state=88).fit(X)
    riscos=np.array([.013,.007,.002],dtype=float)
    ys=np.zeros(len(X),dtype=np.uint8)
    ys[:25]=1
    data={'coord':pd.DataFrame(X,columns=['PC1','PC2','PC3']),
          'y':ys,'scores':gm.predict_proba(X)@riscos,
          'gmm':SimpleNamespace(modelo_=gm),
          'rh':SimpleNamespace(riscos_=pd.Series(riscos,index=['G1','G2','G3']))}
    cfg={'experimento':{'semente':42},'visualizacao':{'graficos':{
        'mapa_risco_fronteira':{'resolucao_volume_html_3d':22,'n_pontos_scatter_html_3d':400}}}}
    pasta=raiz/'outputs'/('validacao_GMM08_HTML_3D_95_V5_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    pasta.mkdir(parents=True,exist_ok=False)
    arquivo=vis._html3d(cfg,data,{'limiar_mcc_oof':.008},pasta,'GMM08','F10',
                         (0.,0.),[],.002,.013)
    txt=arquivo.read_text(encoding='utf-8')
    for k in (1,2,3):
        assert f'G{k} — elipsoide FULL 95%' in txt, f'Gaussiana G{k} ausente!'
        assert f'Centro G{k}' in txt, f'Centro G{k} ausente!'
    assert txt.count('"showlegend":true')>=3
    assert '"visible":"legendonly"' not in txt
    assert 'τ* OOF' in txt
    print('[APROVADO] 3 elipsoides FULL 95%, 3 centros, limiar OOF congelado.')
    print('[APROVADO] Elipsoides visíveis por padrão no HTML, legendas individuais.')
    print('[HTML SINTETICO]',arquivo)
    print('Nenhuma observacao F10 real consultada. Nenhum resultado anterior modificado.')


if __name__=='__main__':
    main()
