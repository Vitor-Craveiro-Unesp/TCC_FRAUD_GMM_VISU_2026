r"""Auditoria PCA 3D - 04: Scree plot do codigo oficial (PCA11).

Execute na raiz do TCC_EXPERIMENTAL_V3:
    .\.venv\Scripts\python.exe .\validar_pca3d_04_cotovelo.py

Simulacao de 39 features; destaca 3 componentes; produz PNG + CSV.
Nao altera YAML, nao usa F10 nem executa o pipeline completo.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src import visualizacao_geometrica as geo


def dados_sinteticos_39_features():
    """Replica a geracao sintetica usada na auditoria dos coeficientes 3D."""
    rng = np.random.default_rng(20261027)
    centros = np.array([
        [-2.5, -1.4, -.5],
        [2.5, -1.0, 1.3],
        [-.3, 2.6, -1.6]
    ])
    covs = np.array([
        [[.6, .2, .05], [.2, .8, -.1], [.05, -.1, .65]],
        [[.8, -.2, .13], [-.2, .6, .02], [.13, .02, .70]],
        [[.75, .15, -.10], [.15, .9, .2], [-.10, .2, .5]]
    ])
    latente = np.vstack([
        rng.multivariate_normal(centros[k], covs[k], size=420)
        for k in range(3)
    ])
    pesos = rng.normal(size=(3, 39))
    x = latente @ pesos + rng.normal(scale=.53, size=(len(latente), 39))
    x_pad = StandardScaler().fit_transform(x)
    nomes = [f'FEATURE_{i+1:02d}' for i in range(39)]
    pre = pd.DataFrame(x_pad, columns=nomes)
    pca = PCA(n_components=3, svd_solver='full').fit(pre)
    coord = pd.DataFrame(pca.transform(pre), columns=['PC1', 'PC2', 'PC3'])
    return pre, coord, pca


def main():
    raiz = Path.cwd()
    fonte = raiz / 'src' / 'visualizacao_geometrica.py'
    yaml_path = raiz / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    if not fonte.is_file() or not yaml_path.is_file():
        raise SystemExit('Execute na raiz do TCC_EXPERIMENTAL_V3 (src e YAML devem existir).')

    yaml_antes = yaml_path.read_bytes()
    cfg = deepcopy(yaml.safe_load(yaml_antes.decode('utf-8-sig')))
    assert 'pca_cotovelo' in cfg['visualizacao']['graficos'], 'Grafico PCA11 ausente do YAML.'

    pre, coord, pca3 = dados_sinteticos_39_features()
    assert pre.shape == (1260, 39)
    assert coord.shape == (1260, 3)

    # A funcao oficial determina o PCA selecionado pelo numero de colunas
    # de coord, e calcula o espectro completo a partir das 39 features.
    dados = {'pre': pre, 'coord': coord}
    pasta = raiz / 'outputs' / (
        'validacao_pca3d_04_cotovelo_' +
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    pasta.mkdir(parents=True, exist_ok=False)

    # Sem ajustar GMM, sem treinar folds, sem acessar a base real.
    geo._scree(cfg, dados, pasta, 'png', 165, True, origem='simulado')

    png = pasta / '11_pca_cotovelo.png'
    csv = pasta / '11_pca_cotovelo.csv'
    assert png.is_file(), 'PNG do Scree plot nao foi gerado.'
    assert csv.is_file(), 'CSV do Scree plot nao foi gerado.'

    t = pd.read_csv(csv)
    assert len(t) == 39, 'O espectro completo deve conter 39 componentes.'
    assert {'componente', 'variancia_individual',
            'variancia_acumulada', 'variancia_absoluta'}.issubset(t.columns)
    assert np.array_equal(t['componente'].to_numpy(), np.arange(1, 40))
    assert np.isclose(float(t['variancia_individual'].sum()), 1., atol=1e-9)
    assert np.allclose(
        t['variancia_acumulada'].to_numpy(),
        np.cumsum(t['variancia_individual'].to_numpy()), atol=1e-9
    )
    assert np.isclose(
        float(t['variancia_acumulada'].iloc[2]),
        float(pca3.explained_variance_ratio_.sum()), atol=1e-9
    )
    assert yaml_path.read_bytes() == yaml_antes, 'YAML foi modificado indevidamente.'

    print('PCA 3D: SCREE PLOT GERADO COM SUCESSO')
    print('39 features sinteticas | linha vertical em PC3 | F10 nao utilizado')
    print('Variancia acumulada PC1+PC2+PC3: {:.7f}'.format(
        float(t['variancia_acumulada'].iloc[2])
    ))
    print('PNG=' + str(png.resolve()))
    print('CSV=' + str(csv.resolve()))
    print('PASTA=' + str(pasta.resolve()))


if __name__ == '__main__':
    main()
