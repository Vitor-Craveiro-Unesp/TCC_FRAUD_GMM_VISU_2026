# -*- coding: utf-8 -*-
"""Auditoria isolada GMM09: PCA 3D + GMM FULL com K=3.

Coloque este arquivo na raiz do TCC_EXPERIMENTAL_V3 e execute:
    .\\.venv\\Scripts\\python.exe .\\validar_pca3d_09_gmm09.py

Apenas dados sinteticos. Usa a funcao real do painel matematico GMM09.
Nao modifica codigo-fonte, YAML, ensaios existentes, nem usa F10.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src import visualizacao_geometrica as geo

SEMENTE = 20261027
ARQUIVO_CONFIG = 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
BASE_PNG = '09_painel_parametros_gaussianos.png'
BASE_CSV = '09_painel_parametros_gaussianos.csv'


def main() -> None:
    raiz = Path.cwd()
    assert (raiz / 'src' / 'visualizacao_geometrica.py').is_file(), (
        'Execute a partir da raiz de TCC_EXPERIMENTAL_V3.'
    )
    caminho_yaml = raiz / ARQUIVO_CONFIG
    original = caminho_yaml.read_bytes()
    config = yaml.safe_load(original.decode('utf-8-sig'))

    # Nada e escrito no YAML. Ajuste apenas a configuracao em memoria.
    config['visualizacao']['graficos']['painel_parametros_gaussianos'][
        'gaussianas_por_pagina'] = 4
    rng = np.random.default_rng(SEMENTE)

    centros = np.array([
        [-2.5, -1.4, -.5],
        [2.5, -1., 1.3],
        [-.3, 2.6, -1.6],
    ])
    covs = np.array([
        [[.6, .2, .05], [.2, .8, -.1], [.05, -.1, .65]],
        [[.8, -.2, .13], [-.2, .6, .02], [.13, .02, .70]],
        [[.75, .15, -.10], [.15, .9, .2], [-.10, .2, .5]],
    ])
    latentes = np.vstack([
        rng.multivariate_normal(centros[k], covs[k], 420)
        for k in range(3)
    ])
    pesos = rng.normal(size=(3, 39))
    X = latentes @ pesos + rng.normal(scale=.53, size=(len(latentes), 39))
    Z = PCA(n_components=3, svd_solver='full').fit_transform(
        StandardScaler().fit_transform(X)
    )

    modelo = GaussianMixture(
        n_components=3,
        covariance_type='full',
        n_init=3,
        max_iter=300,
        random_state=SEMENTE,
    ).fit(Z)
    assert modelo.converged_, 'GMM de teste nao convergiu.'
    assert modelo.means_.shape == (3, 3)
    assert modelo.covariances_.shape == (3, 3, 3)
    assert abs(modelo.weights_.sum() - 1) < 1e-12

    nomes = ['G1', 'G2', 'G3']
    responsabilidades = modelo.predict_proba(Z)
    # Probabilidades sinteticas APENAS para simular RH no desenvolvimento.
    referencia = np.array([.06, .22, .55])
    y = rng.binomial(1, responsabilidades @ referencia)
    massa = responsabilidades.sum(axis=0)
    fraudes = responsabilidades.T @ y
    riscos = fraudes / massa

    # Estrutura real exigida pela implementacao oficial do GMM09.
    dados = {
        'gmm': SimpleNamespace(modelo_=modelo, saidas_=nomes),
        'rh': SimpleNamespace(
            riscos_=pd.Series(riscos, index=nomes),
            massas_=pd.Series(massa, index=nomes),
            fraudes_ponderadas_=pd.Series(fraudes, index=nomes),
            prevalencia_treino_=float(y.mean()),
        ),
        'y': y,
        'n_treino_gmm': len(Z),
    }

    destino = raiz / 'outputs' / (
        'validacao_pca3d_09_gmm09_' +
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    destino.mkdir(parents=True, exist_ok=False)

    geo._painel_parametros_gaussianos(
        config, dados, destino, 'png', 175, True
    )

    png = destino / BASE_PNG
    csv = destino / BASE_CSV
    assert png.is_file(), f'PNG nao encontrado: {png}'
    assert csv.is_file(), f'CSV nao encontrado: {csv}'
    tabela = pd.read_csv(csv)
    assert len(tabela) == 3, 'CSV nao contem exatamente 3 gaussianas.'
    assert set(tabela['gaussiana']) == set(nomes)
    assert all(tabela['covariance_type'] == 'full')
    assert abs(tabela['peso_mistura'].sum() - 1) < 1e-10

    for k in range(3):
        row = tabela.iloc[k]
        assert len(json.loads(row['media_vetor_completo'])) == 3
        cov = np.array(json.loads(row['covariancia_completa']), dtype=float)
        assert cov.shape == (3, 3)
        assert np.allclose(cov, cov.T, atol=1e-12)
        assert np.linalg.eigvalsh(cov).min() > 0

    assert caminho_yaml.read_bytes() == original, 'YAML foi modificado.'
    print('GMM09 PCA 3D: VALIDACAO CONCLUIDA')
    print('GMM: K=3 | FULL | medias 3D | matrizes 3x3')
    print('CSV: parametros numericos completos')
    print('DADOS: sinteticos | F10: nao utilizado')
    print('PASTA:', destino.resolve())
    print('PNG:', png.resolve())
    print('CSV:', csv.resolve())


if __name__ == '__main__':
    main()
