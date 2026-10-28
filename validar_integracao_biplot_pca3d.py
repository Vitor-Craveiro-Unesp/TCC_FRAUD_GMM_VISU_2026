# -*- coding: utf-8 -*-
"""Smoke test PCA1D/2D/3D em dados sinteticos (sem OOF/F10)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src import visualizacao_pca_biplot as bip


def main():
    raiz = Path.cwd()
    yaml_path = raiz / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    if not yaml_path.is_file():
        raise RuntimeError('Execute na raiz TCC_EXPERIMENTAL_V3 (YAML ausente).')
    yaml_bytes = yaml_path.read_bytes()
    config = yaml.safe_load(yaml_bytes.decode('utf-8-sig'))
    config['visualizacao']['graficos']['pca_biplot']['max_setas'] = 6
    config['visualizacao']['geometria']['pontos_maximos_amostra'] = 600

    rng = np.random.default_rng(20261027)
    centros = np.array([[-2.5, -1.4, -.5], [2.5, -1.0, 1.3], [-.3, 2.6, -1.6]])
    covs = np.array([
        [[.6, .2, .05], [.2, .8, -.1], [.05, -.1, .65]],
        [[.8, -.2, .13], [-.2, .6, .02], [.13, .02, .70]],
        [[.75, .15, -.10], [.15, .9, .2], [-.10, .2, .5]],
    ])
    lat = np.vstack([rng.multivariate_normal(centros[k], covs[k], 140)
                     for k in range(3)])
    pesos = rng.normal(size=(3, 39))
    X = lat @ pesos + rng.normal(scale=.53, size=(len(lat), 39))
    nomes = [f'FEATURE_{j+1:02d}' for j in range(39)]
    x = pd.DataFrame(StandardScaler().fit_transform(X), columns=nomes)
    y = rng.binomial(1, np.repeat([.025, .15, .48], 140))

    saida = raiz / 'outputs' / (
        'validacao_integracao_biplot_pca3d_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    saida.mkdir(parents=True, exist_ok=False)

    for d in (1, 2, 3):
        modelo = PCA(n_components=d, random_state=20261027).fit(x)
        coord = pd.DataFrame(modelo.transform(x),
                             index=x.index, columns=[f'PC{i}' for i in range(1, d + 1)])
        dados = {'redutor': SimpleNamespace(modelo_=modelo, colunas_=nomes),
                 'pre': x, 'coord': coord, 'y': y}
        pasta = saida / f'PCA_{d}D'
        pasta.mkdir()
        retorno = bip.gerar(config, dados, pasta, 'png', 110, True)
        assert retorno['dimensoes'] == d
        assert (pasta / f'PCA_BIPLOT_{d}D.png').is_file()
        assert (pasta / f'PCA_BIPLOT_{d}D_correlacoes.csv').is_file()
        assert (pasta / f'PCA_BIPLOT_{d}D_scores_amostra.csv').is_file()
        if d == 3:
            coef = pasta / 'PCA_COEFICIENTES_PC1_PC2_PC3.csv'
            png = pasta / 'PCA_COEFICIENTES_PC1_PC2_PC3.png'
            html = pasta / 'PCA_BIPLOT_3D_INTERATIVO.html'
            assert all(f.is_file() for f in (coef, png, html)), 'Complementos PCA3D ausentes.'
            dados_coef = pd.read_csv(coef)
            for j in (1, 2, 3):
                coluna = f'coeficiente_PC{j}'
                assert coluna in dados_coef
                esperado = dict(zip(nomes, modelo.components_[j-1]))
                erro = max(abs(float(r[coluna]) - esperado[r['variavel_preprocessada']])
                           for _, r in dados_coef.iterrows())
                assert erro < 1e-14, (coluna, erro)
            assert len(dados_coef) == 39
            pagina = html.read_text(encoding='utf-8')
            assert '39 vetores (todos)' in pagina and 'Rotação automática' in pagina
            assert html.stat().st_size > 100_000, 'HTML offline incompleto.'
        else:
            assert not (pasta / 'PCA_COEFICIENTES_PC1_PC2_PC3.png').exists()
            assert not (pasta / 'PCA_BIPLOT_3D_INTERATIVO.html').exists()
        print(f'PCA {d}D: VALIDADO')

    assert yaml_path.read_bytes() == yaml_bytes, 'YAML alterado na validacao.'
    print('39 features em CSV | 6 vetores iniciais e 39 no HTML 3D.')
    print('YAML preservado. Nenhum F10 usado. Nenhum OOF completo.')
    print('PASTA=' + str(saida.resolve()))
    print('PNG=' + str((saida / 'PCA_3D' / 'PCA_COEFICIENTES_PC1_PC2_PC3.png').resolve()))
    print('HTML=' + str((saida / 'PCA_3D' / 'PCA_BIPLOT_3D_INTERATIVO.html').resolve()))


if __name__ == '__main__':
    main()
