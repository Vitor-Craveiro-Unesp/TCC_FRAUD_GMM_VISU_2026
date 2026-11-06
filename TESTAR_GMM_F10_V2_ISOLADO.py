# -*- coding: utf-8 -*-
"""Teste de renderizacao sintetica dos GMM07/GMM08 EXCLUSIVOS do F10.

Na raiz do projeto, depois de aplicar o instalador:
  python TESTAR_GMM_F10_V2_ISOLADO.py

Nao usa o dataset real, nao le F10 real, nao faz fit, nao altera YAML nem
sobrescreve PNGs/HTMLs existentes. Gera nova pasta outputs/validacao_... .
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def construir_gmm_sem_fit(d):
    # Duas gaussianas artificiais suficientes para score variar ao longo de tau.
    g = GaussianMixture(n_components=2, covariance_type='full', random_state=123)
    g.weights_ = np.array([.5, .5])
    g.means_ = np.vstack((np.full(d,-1.6), np.full(d,1.6)))
    g.covariances_ = np.array([np.eye(d)*.7, np.eye(d)*.7])
    g.precisions_cholesky_ = np.stack([np.linalg.cholesky(np.linalg.inv(c)).T
                                      for c in g.covariances_])
    g.n_features_in_ = d
    g.converged_ = True
    return g


def main():
    from src.gmm_fronteira_f10 import diagnosticar
    from src.gmm_contornos_quatro_niveis import gerar, _planos, _faixa
    from src.visualizacao_geometrica import _ativo

    raiz=Path(__file__).resolve().parent
    pasta=raiz/'outputs'/('validacao_SO_F10_GMM07_GMM08_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    pasta.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(2031)
    resultados=[]

    for metodo in ('PCA','TSNE'):
        for d in (1,2,3):
            # 90 legitimas e 20 fraudes artificiais, so para representacao.
            X=np.vstack((rng.normal(-1.6,.45,size=(90,d)),
                         rng.normal(+1.6,.45,size=(20,d))))
            y=np.r_[np.zeros(90,dtype=int),np.ones(20,dtype=int)]
            gm=construir_gmm_sem_fit(d)
            riscos=np.array([.002,.028])
            scores=gm.predict_proba(X) @ riscos
            tau=.015
            cert=diagnosticar(gm,riscos,X,scores,tau)
            assert cert['cruzamento_confirmado'], (metodo,d,cert)
            assert abs((gm.predict_proba(cert['ponto_fronteira'][None,:])@riscos)[0]-tau)<1e-9
            # O ajuste deve usar todos os extremos reais, nao somente quantis.
            for i in range(d):
                low,high=_faixa(X[:,i],.15)
                assert low < X[:,i].min() and high > X[:,i].max()
            planos,fixos=_planos(X,gm,riscos,d,80,.15,ancora=cert['ponto_fronteira'])
            assert len(planos)==(3 if d==3 else 1)
            cfg={'experimento':{'semente':2031},'visualizacao':{
                'habilitar':True, 'gerar_em':['oof','teste_final'],
                'graficos':{'sensibilidade_fronteira':{'habilitar':True,'gerar_em':['teste_final'],
                            'resolucao_1d':120,'resolucao_2d':65,
                            'resolucao_volume_html_3d':20,'n_pontos_scatter':100},
                            'mapa_risco_fronteira':{'habilitar':True,'gerar_em':['teste_final'],
                            'resolucao_1d':120,'resolucao_2d':65,
                            'resolucao_volume_html_3d':20,'n_pontos_scatter':100}}}}
            assert not _ativo(cfg,'sensibilidade_fronteira')
            assert not _ativo(cfg,'mapa_risco_fronteira')
            nomes=[f'{"PC" if metodo=="PCA" else "TSNE"}{i+1}' for i in range(d)]
            dados={'coord':pd.DataFrame(X,columns=nomes),
                   'y':y, 'scores':scores, 'gmm':SimpleNamespace(modelo_=gm),
                   'rh':SimpleNamespace(riscos_=pd.Series(riscos,index=['G1','G2']))}
            escolha={'limiar_mcc_oof':tau}
            destino=pasta/f'{metodo}_{d}D'/'vencedor'/'teste_final'/'graficos_gerados'
            for tipo in ('GMM07','GMM08'):
                gerar(cfg,dados,escolha,destino,'png',85,False,tipo=tipo,
                      avaliacao={'origem':'f10_amostral_congelado'})
                png=destino/f'{tipo}_{"sensibilidade" if tipo=="GMM07" else "mapa_risco"}_{d}d_comparativo.png'
                assert png.exists() and png.stat().st_size>4000, png
                resultados.append(png)
                if d==3:
                    html=png.with_name(png.stem+'_INTERATIVA.html')
                    assert html.exists() and 'Cruzamento real' in html.read_text(encoding='utf-8'), html
                    resultados.append(html)
            print(f'[APROVADO] {metodo} {d}D: GMM07 + GMM08, F10 sintetico, contorno certificado.')

    assert len(resultados)==16, len(resultados)
    print('\n[TESTE ISOLADO APROVADO] 12 PNGs + 4 HTMLs gerados.')
    print('[PASTA]',pasta)
    print('Nenhuma amostra F10 real acessada. Nenhum modelo treinado.')

if __name__=='__main__':
    main()
