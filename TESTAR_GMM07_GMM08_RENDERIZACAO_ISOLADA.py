# -*- coding: utf-8 -*-
"""Renderiza GMM07/GMM08 com modelo SIMULADO (nenhum fit ou acesso a base real).

Uso: python TESTAR_GMM07_GMM08_RENDERIZACAO_ISOLADA.py
Saida: outputs/validacao_gmm07_gmm08_4niveis_<timestamp>/
Nao reexecuta OOF, nao le F10 e nao modifica resultados anteriores.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import traceback
import sys

import numpy as np
import pandas as pd


class GaussianasSimuladas:
    """GMM de duas componentes com responsabilidades analiticas, SEM treinar."""
    covariance_type = 'full'
    n_components = 2

    def __init__(self, dim: int):
        self.dim = dim
        self.means_ = np.array([[-1.3] * dim, [1.3] * dim], dtype=float)
        self.covariances_ = np.array([np.eye(dim) * .95, np.eye(dim) * .95])
        self.weights_ = np.array([.5, .5])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError('Dimensionalidade incorreta do GMM sintético')
        # Posteriores para gaussianas isótropas de mesma variância/peso.
        distancia = np.sum((X[:, None, :] - self.means_[None, :, :]) ** 2,
                           axis=-1) / (2.0 * .95)
        logits = -distancia
        logits -= np.max(logits, axis=1, keepdims=True)
        pesos = np.exp(logits)
        return pesos / pesos.sum(axis=1, keepdims=True)


def dados_sinteticos(nome, dim, *, f10=False):
    rng = np.random.default_rng(20261104 + dim + (100 if f10 else 0))
    n_leg, n_fra = 180, 65
    centro_leg = -1.20 if not f10 else -1.05
    centro_fra = 1.2 if not f10 else 1.1
    X = np.vstack((rng.normal(centro_leg, 1.0, size=(n_leg, dim)),
                   rng.normal(centro_fra, 1.0, size=(n_fra, dim))))
    y = np.concatenate((np.zeros(n_leg, dtype=np.uint8),
                        np.ones(n_fra, dtype=np.uint8)))
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    modelo = GaussianasSimuladas(dim)
    riscos = np.array([0.005, 0.85], dtype=float)
    scores = modelo.predict_proba(X) @ riscos
    prefixo = 'PC' if nome.upper() == 'PCA' else 'TSNE'
    coord = pd.DataFrame(X, columns=[f'{prefixo}{k+1}' for k in range(dim)])
    data = {
        'coord': coord,
        'y': y,
        'scores': scores,
        'gmm': SimpleNamespace(modelo_=modelo),
        'rh': SimpleNamespace(riscos_=pd.Series(riscos, index=['G1','G2']))
    }
    return data


def principal():
    raiz = Path(__file__).resolve().parent
    if not (raiz/'src'/'gmm_contornos_quatro_niveis.py').is_file():
        raise FileNotFoundError('Execute o script NA RAIZ do projeto, onde esta src/gmm_contornos_quatro_niveis.py')
    from src import visualizacao_gmm07 as g07
    from src import visualizacao_geometrica as geo
    from src import gmm_contornos_quatro_niveis as quatro
    from src import visualizacao_geometria_tsne3d as tsne3d
    import inspect
    for objeto, rotulo in ((g07.gerar, 'GMM07'), (geo._fronteira,'GMM08')):
        if 'GMM_QUATRO_NIVEIS_F1F9_F10_V1' not in inspect.getsource(objeto):
            raise RuntimeError(f'{rotulo}: nao esta ligado ao novo modulo de quatro niveis.')
    if 'substituir_png_html_tsne3d' not in inspect.getsource(tsne3d.gerar_geometria_tsne3d):
        raise RuntimeError('Hook de t-SNE 3D F10 nao instalado.')

    destino = raiz/'outputs'/('validacao_gmm_4_niveis_ISOLADA_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destino.mkdir(parents=True, exist_ok=False)
    (destino/'LEIA_PRIMEIRO.txt').write_text(
        'APENAS DADOS SINTETICOS. Nenhum modelo do TCC foi treinado nem lido. '
        'OOF e F10 sao rotulos de caminhos de TESTE, nao avaliacao real.\n', encoding='utf-8')
    cenarios = (
        ('PCA_1D_OOF_SIMULADO','PCA',1,False),
        ('TSNE_2D_OOF_SIMULADO','TSNE',2,False),
        ('PCA_3D_OOF_SIMULADO','PCA',3,False),
        ('TSNE_3D_F10_SIMULADO','TSNE',3,True),
    )
    feitos=[]
    config = {
        'experimento': {'semente':42},
        'visualizacao': {'graficos': {
            'sensibilidade_fronteira': {'resolucao_1d':160,'resolucao_2d':45,
                'resolucao_volume_html_3d':20,'margem':.15,'n_pontos_scatter':160},
            'mapa_risco_fronteira': {'resolucao_1d':160,'resolucao_2d':45,
                'resolucao_volume_html_3d':20,'margem':.15,'n_pontos_scatter':160,
                'mostrar_gaussianas':True}
        }}
    }
    escolha = {'limiar_mcc_oof': 0.16}
    print('=== TESTE ISOLADO DE RENDERIZACAO GMM07/GMM08 ===',flush=True)
    print('Destino:',destino,flush=True)
    try:
        for nome,metodo,dim,f10 in cenarios:
            data=dados_sinteticos(metodo,dim,f10=f10)
            if f10:
                pasta=destino/nome/'teste_final'/'graficos_gerados'
                avaliacao={'origem':'f10_amostral_congelado'}
            else:
                pasta=destino/nome/'OOF_SIMULADO'
                avaliacao={'origem':'simulado','y':data['y'],'p':data['scores']}
            pasta.mkdir(parents=True,exist_ok=False)
            print(f'[RENDER] {nome}...',flush=True)
            # Chamadas REAIS dos renderizadores instalados; sem ajuste de modelo.
            g07.gerar(config,data,escolha,pasta,'png',90,True,avaliacao=avaliacao)
            geo._fronteira(config,data,escolha,pasta,'png',90,True,avaliacao=avaliacao)
            p1=pasta/f'GMM07_sensibilidade_{dim}d_comparativo.png'
            p2=pasta/f'GMM08_mapa_risco_{dim}d_comparativo.png'
            for p in (p1,p2):
                if not p.is_file() or p.stat().st_size<3000:
                    raise AssertionError(f'PNG invalido ou ausente: {p}')
                feitos.append(str(p.relative_to(destino)))
                print('[OK] PNG:',p.name,round(p.stat().st_size/1024),'KB',flush=True)
            if dim==3:
                for base in (p1.stem,p2.stem):
                    h=pasta/(base+'_INTERATIVA.html')
                    if not h.is_file() or h.stat().st_size<10000:
                        raise AssertionError('HTML offline invalido: '+str(h))
                    feitos.append(str(h.relative_to(destino)))
                    print('[OK] HTML:',h.name,flush=True)
            # Testa calculos basicos do intervalo e da decisao sem alterar tau.
            modelo=data['gmm'].modelo_
            campos, _ = quatro._planos(data['coord'].to_numpy(),modelo,
                           np.array([.005,.85]),dim,45,.15)
            scores_malha=np.concatenate([p[2].ravel() for p in campos])
            lo,hi=float(scores_malha.min()),float(scores_malha.max())
            mid=(hi+lo)/2
            assert lo <= mid <= hi and lo <= .16 <= hi, (nome,lo,hi)
        # Rota dedicada ao t-SNE 3D usada na geracao F10.
        print('[RENDER] HOOK dedicado TSNE3D F10...',flush=True)
        data=dados_sinteticos('TSNE',3,f10=True)
        pasta=destino/'TSNE_3D_HOOK_F10_SIMULADO'/'teste_final'/'graficos_gerados'
        pasta.mkdir(parents=True,exist_ok=False)
        quatro.substituir_png_html_tsne3d(data['coord'],data['y'],data['gmm'].modelo_,
            np.array([.005,.85]),pasta,tau=.16,origem='f10',resolucao_cortes=45,dpi=90)
        for g in ('GMM07_sensibilidade_3d_comparativo', 'GMM08_mapa_risco_3d_comparativo'):
            for suffix in ('.png','_INTERATIVA.html'):
                p=pasta/(g+suffix)
                if not p.exists() or p.stat().st_size < 3000:
                    raise AssertionError('Hook nao gerou: '+str(p))
                feitos.append(str(p.relative_to(destino)))
                print('[OK] HOOK:',p.name,flush=True)
        registro={'teste':'RENDERIZACAO_SINTETICA_ISOLADA','aprovado':True,
                  'n_arquivos':len(feitos),'arquivos':feitos,
                  'fit_executado':False,'dados_reais_lidos':False,
                  'versao':quatro.VERSION}
        (destino/'RESULTADO.json').write_text(json.dumps(registro,indent=2,ensure_ascii=False),encoding='utf-8')
        print('\n[APROVADO] Renderizacao isolada: '+str(len(feitos))+' PNGs/HTMLs.',flush=True)
        print('[PASTA]',destino,flush=True)
        print('Abra os PNGs e confira manualmente as 4 referencias e as estrelas vermelhas.',flush=True)
        return 0
    except Exception as exc:
        (destino/'FALHA.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('[REPROVADO] '+str(exc),file=sys.stderr,flush=True)
        print('[PASTA]',destino,flush=True)
        return 1


if __name__=='__main__':
    try: raise SystemExit(principal())
    except Exception as e:
        traceback.print_exc()
        raise SystemExit(1)
