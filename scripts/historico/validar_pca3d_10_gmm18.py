# -*- coding: utf-8 -*-
r"""Auditoria isolada do GMM18 — Convergencia do EM em PCA 3D, K=3.

Coloque este arquivo na raiz de TCC_EXPERIMENTAL_V3 e execute:
    .\.venv\Scripts\python.exe .\validar_pca3d_10_gmm18.py

Usa somente dados sinteticos. O GMM e ajustado somente para a PREVIA.
O PNG e o CSV sao produzidos pelo modulo OFICIAL src.visualizacao_gmm18.
Nao altera src/, YAML, ensaios anteriores ou dados reais. Nao acessa F10.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.gmm import GMMNaoSupervisionado
from src import visualizacao_gmm18 as gmm18

SEMENTE = 20261027
K = 3
N_INITS = 3
N_VARIAVEIS = 39
N_POR_GAUSSIANA = 390


def dados_sinteticos_pca3d() -> pd.DataFrame:
    rng = np.random.default_rng(SEMENTE)
    medias = np.array([
        [-2.6, -1.4, -.6],
        [2.1, -1.0, 1.6],
        [-.2, 2.4, -1.4],
    ])
    covariancias = np.array([
        [[.85, .30, .07], [.30, .70, -.12], [.07, -.12, .65]],
        [[.78, -.20, .14], [-.20, .90, .04], [.14, .04, .71]],
        [[.83, .12, -.11], [.12, .81, .19], [-.11, .19, .62]],
    ])
    z = np.vstack([
        rng.multivariate_normal(medias[k], covariancias[k], N_POR_GAUSSIANA)
        for k in range(K)
    ])
    pesos = rng.normal(size=(3, N_VARIAVEIS))
    x = z @ pesos + rng.normal(scale=.55, size=(len(z), N_VARIAVEIS))
    x_escalado = StandardScaler().fit_transform(x)
    pca = PCA(n_components=3, svd_solver='full', random_state=SEMENTE)
    coordenadas = pca.fit_transform(x_escalado)
    assert coordenadas.shape == (K * N_POR_GAUSSIANA, 3)
    assert np.isfinite(coordenadas).all()
    return pd.DataFrame(coordenadas, columns=['PC1', 'PC2', 'PC3'])


def main():
    raiz = Path(__file__).resolve().parent
    yaml_path = raiz / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    if not yaml_path.is_file():
        raise FileNotFoundError('Coloque este script na raiz do TCC_EXPERIMENTAL_V3.')
    if not (raiz / 'src' / 'visualizacao_gmm18.py').is_file():
        raise FileNotFoundError('Modulo src/visualizacao_gmm18.py nao encontrado.')

    yaml_antes = yaml_path.read_bytes()
    cfg = yaml.safe_load(yaml_antes.decode('utf-8-sig'))
    assert cfg['reducao']['metodo'].upper() == 'PCA'
    assert cfg['dados']['coluna_alvo'] not in {'PC1', 'PC2', 'PC3'}

    # Os parametros abaixo afetam APENAS este teste em memoria.
    # O YAML do ensaio oficial nao e modificado.
    parametros = {
        'n_components': K,
        'covariance_type': 'full',
        'reg_covar': 1e-6,
        'max_iter': 300,
        'n_init': N_INITS,
        'tol': 1e-5,
        'init_params': 'kmeans',
    }

    coord = dados_sinteticos_pca3d()
    gmm = GMMNaoSupervisionado(cfg, parametros).fit(
        coord, registrar_historico_em=True
    )
    modelo = gmm.modelo_
    melhor = modelo.validar_historicos_em()
    assert modelo.converged_
    assert modelo.n_components == K
    assert modelo.covariance_type == 'full'
    assert modelo.covariances_.shape == (K, 3, 3)
    assert len(gmm.colunas_) == 3
    assert len(modelo.historicos_em_) == N_INITS
    assert modelo.historicos_em_[melhor - 1]['convergiu']

    destino = raiz / 'outputs' / (
        'validacao_pca3d_10_gmm18_'
        + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    destino.mkdir(parents=True, exist_ok=False)
    tabela = gmm18.gerar(cfg, {'gmm': gmm}, destino, 'png', 165, True)

    png = destino / (gmm18.NOME + '.png')
    csv = destino / (gmm18.NOME + '.csv')
    assert png.is_file() and png.stat().st_size > 10000, 'PNG nao gerado.'
    assert csv.is_file() and csv.stat().st_size > 100, 'CSV nao gerado.'
    esperado = {
        'inicializacao', 'iteracao', 'lower_bound_medio_por_amostra',
        'log_verossimilhanca_total_equivalente',
        'variacao_vs_iteracao_anterior', 'inicializacao_selecionada',
        'convergiu', 'n_treino_gmm',
    }
    assert esperado.issubset(tabela.columns)
    assert set(tabela['inicializacao']) == set(range(1, N_INITS + 1))
    assert int(tabela['inicializacao_selecionada'].sum()) == len(
        modelo.historicos_em_[melhor - 1]['lower_bounds']
    )
    assert yaml_path.read_bytes() == yaml_antes, 'O YAML foi modificado.'

    resumo = {
        'auditoria': 'GMM18 PCA 3D — sintético',
        'covariancia': 'FULL',
        'K': K,
        'n_init': N_INITS,
        'melhor_init': int(melhor),
        'n_treino_sintetico': int(len(coord)),
        'iteracoes_melhor': int(modelo.n_iter_),
        'lower_bound_melhor': float(modelo.lower_bound_),
        'f10_utilizado': False,
        'yaml_modificado': False,
    }
    (destino / 'validacao_GMM18_PCA3D.json').write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('GMM18 PCA 3D VALIDADO')
    print('K = 3 | FULL | 3 inicializacoes EM')
    print(f'Melhor inicializacao: {melhor}')
    print('YAML preservado | F10 nao utilizado')
    print('PNG=' + str(png.resolve()))
    print('CSV=' + str(csv.resolve()))


if __name__ == '__main__':
    main()
