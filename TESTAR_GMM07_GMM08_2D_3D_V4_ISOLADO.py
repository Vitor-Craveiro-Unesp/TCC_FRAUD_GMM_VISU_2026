# -*- coding: utf-8 -*-
"""Teste de renderização PNG GMM07/GMM08 (PCA/t-SNE, 2D/3D).

Uso na raiz do TCC_EXPERIMENTAL_V3, APÓS V4:
  python .\\TESTAR_GMM07_GMM08_2D_3D_V4_ISOLADO.py

Não carrega parquet, F10 real nem modelos serializados; não treina nem sobrescreve.
Gera 8 PNGs + 4 HTMLs offline sintéticos em outputs/validacao_... .
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def _gmm(d):
    modelo = GaussianMixture(n_components=2, covariance_type='full', random_state=2026)
    modelo.weights_ = np.array([.5,.5])
    modelo.means_ = np.stack([np.full(d,-1.6),np.full(d,1.6)])
    modelo.covariances_ = np.stack([np.eye(d)*.7,np.eye(d)*.7])
    modelo.precisions_cholesky_ = np.stack([
        np.linalg.cholesky(np.linalg.inv(cov)).T for cov in modelo.covariances_])
    modelo.n_features_in_=d
    modelo.converged_=True
    return modelo


def main():
    from src.gmm_contornos_quatro_niveis import gerar
    from src.gmm_fronteira_f10 import diagnosticar

    raiz = Path(__file__).resolve().parent
    pasta = raiz/'outputs'/('validacao_GMM07_GMM08_2D3D_V4_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    pasta.mkdir(parents=True,exist_ok=False)
    rng = np.random.default_rng(20261106)
    vistos = []
    original_savefig=Figure.savefig

    def conferir(fig, destino, **kwargs):
        nome=Path(destino).name
        textos=[x.get_text() for x in fig.texts]
        assert not any('F10 |' in t or 'F10: modelo' in t for t in textos), (nome,textos)
        assert not any('Positivos na MALHA' in t or 'Predições positivas F10' in t for t in textos), (nome,textos)
        assert any('Malha s(x)' in t for t in textos), (nome,textos)
        assert any('SENSIBILIDADE' in t or 'MAPA DE RISCO' in t for t in textos), nome
        assert fig.legends, 'Sem legenda superior'
        labels=' '.join(t.get_text() for l in fig.legends for t in l.get_texts())
        assert 'τ* OOF' in labels, labels
        assert 'ponto médio' in labels, labels
        eixos=[e for e in fig.axes if len(e.get_xlabel())]
        assert len(eixos) == (3 if '3d_' in nome else 1), (nome,len(eixos))
        if nome.startswith('GMM08'):
            assert 'Gaussianas FULL 95%' in labels, labels
            assert all(any(isinstance(p,Ellipse) for p in e.patches) for e in eixos), \
                'Faltou elipse 95% no(s) corte(s) do GMM08'
        else:
            assert not any(any(isinstance(p,Ellipse) for p in e.patches) for e in eixos), \
                'GMM07 nao deveria adicionar elipses 95%'
        vistos.append(nome)
        return original_savefig(fig,destino,**kwargs)

    for metodo in ('PCA','TSNE'):
        for d in (2,3):
            gm=_gmm(d)
            X=np.vstack([rng.normal(-1.6,.45,(90,d)),rng.normal(1.6,.45,(20,d))])
            y=np.r_[np.zeros(90,dtype=int),np.ones(20,dtype=int)]
            riscos=np.array([.002,.028])
            scores=gm.predict_proba(X)@riscos
            tau=.015
            cert=diagnosticar(gm,riscos,X,scores,tau)
            assert cert['cruzamento_confirmado'],(metodo,d,cert)
            names=[('PC' if metodo=='PCA' else 'TSNE')+str(i+1) for i in range(d)]
            data={'coord':pd.DataFrame(X,columns=names),'y':y,'scores':scores,
                  'gmm':SimpleNamespace(modelo_=gm),
                  'rh':SimpleNamespace(riscos_=pd.Series(riscos,index=['G1','G2']))}
            cfg={'experimento':{'semente':2026},'visualizacao':{'graficos':{
                'sensibilidade_fronteira':{'resolucao_2d':70,'resolucao_volume_html_3d':20,'n_pontos_scatter':100},
                'mapa_risco_fronteira':{'resolucao_2d':70,'resolucao_volume_html_3d':20,'n_pontos_scatter':100},
            }}}
            destino=pasta/metodo/f'{d}D'/'vencedor'/'teste_final'/'graficos_gerados'
            destino.mkdir(parents=True,exist_ok=False)
            for tipo in ('GMM07','GMM08'):
                with patch.object(Figure,'savefig',conferir):
                    gerar(cfg,data,{'limiar_mcc_oof':tau},destino,'png',90,False,
                          tipo=tipo,avaliacao={'origem':'f10_amostral_congelado'})
                stem=(f'{tipo}_sensibilidade_{d}d_comparativo' if tipo=='GMM07'
                      else f'{tipo}_mapa_risco_{d}d_comparativo')
                png=destino/(stem+'.png')
                assert png.is_file() and png.stat().st_size>5000,png
                if d==3:
                    html=destino/(stem+'_INTERATIVA.html')
                    assert html.is_file() and html.stat().st_size>1000,html
            print(f'[APROVADO] {metodo} {d}D: subtitulo e rodape removidos; GMM08 FULL 95% preservado.')
    assert len(vistos)==8,vistos
    plt.close('all')
    print('\n[TESTE VISUAL V4 APROVADO] 8 PNGs + 4 HTMLs sintéticos; nenhuma execução OOF/F10 real.')
    print('[PASTA]',pasta)


if __name__=='__main__':
    main()
