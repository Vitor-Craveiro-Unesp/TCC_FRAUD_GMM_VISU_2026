# -*- coding: utf-8 -*-
"""Teste de regressao GMM04 para PCA 1D/2D/3D com dados sinteticos.

Executar na raiz TCC_EXPERIMENTAL_V3 apos instalar. Nao altera YAML,
modelos reais, pastas OOF, galeria oficial ou dados do usuario.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src import visualizacao_gmm04 as gmm04
from src import visualizacao_integrada as galeria


def main():
    raiz = Path.cwd()
    arquivo_yaml = raiz / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    if not (arquivo_yaml.is_file() and (raiz/'src'/'visualizacao_gmm04.py').is_file()):
        raise RuntimeError('Execute na raiz do TCC_EXPERIMENTAL_V3.')
    yaml_original = arquivo_yaml.read_bytes()
    cfg = yaml.safe_load(yaml_original.decode('utf-8-sig'))
    op = cfg['visualizacao']['graficos']['regioes_dominancia']
    op['resolucao_1d'] = 250
    op['resolucao_2d'] = 55
    op['resolucao_html_3d'] = 35
    op['resolucao_volume_html_3d'] = 15
    op['pontos_scatter'] = 200
    op['margem'] = .16

    rng = np.random.default_rng(20261027)
    mu = np.array([[-2.5,-1.4,-.5], [2.5,-1.,1.3], [-.3,2.6,-1.6]])
    cov = np.array([
        [[.6,.2,.05],[.2,.8,-.1],[.05,-.1,.65]],
        [[.8,-.2,.13],[-.2,.6,.02],[.13,.02,.7]],
        [[.75,.15,-.1],[.15,.9,.2],[-.1,.2,.5]],
    ])
    latente = np.vstack([rng.multivariate_normal(mu[k], cov[k], 180)
                         for k in range(3)])
    pesos = rng.normal(size=(3,39))
    X = StandardScaler().fit_transform(
        latente @ pesos + rng.normal(scale=.53, size=(len(latente),39)))
    labels = rng.binomial(1, np.repeat([.025,.15,.48],180))

    pasta = raiz / 'outputs' / (
        'validacao_integracao_gmm04_pca3d_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    pasta.mkdir(parents=True, exist_ok=False)
    for d in (1, 2, 3):
        Z = PCA(n_components=d, svd_solver='full').fit_transform(X)
        modelo = GaussianMixture(n_components=3, covariance_type='full',
                                 n_init=3, max_iter=300, random_state=20261027).fit(Z)
        assert modelo.converged_
        gm = SimpleNamespace(modelo_=modelo, saidas_=['G1','G2','G3'],
                             colunas_=[f'PC{i+1}' for i in range(d)])
        coord = pd.DataFrame(Z, columns=gm.colunas_)
        saida = pasta / f'PCA_{d}D'
        saida.mkdir()
        gmm04.gerar(cfg, {'coord':coord, 'gmm':gm, 'y':labels}, saida, 'png', 125, True)
        png = saida / f'GMM04_dominancia_{d}d.png'
        html = saida / 'GMM04_dominancia_3d_INTERATIVA.html'
        assert png.is_file() and png.stat().st_size > 15000, f'PNG {d}D ausente.'
        if d != 3:
            assert not html.exists(), f'HTML 3D criado indevidamente no PCA {d}D.'
        else:
            assert html.is_file() and html.stat().st_size > 100000
            pagina = html.read_text(encoding='utf-8')
            # Normalizar Unicode escapado e entidades HTML do Plotly.
            import html as html_lib
            import re
            import unicodedata

            pagina_normalizada = html_lib.unescape(pagina)
            pagina_normalizada = re.sub(
                r'\\u([0-9A-Fa-f]{4})',
                lambda m: chr(int(m.group(1), 16)),
                pagina_normalizada
            )
            pagina_normalizada = unicodedata.normalize(
                'NFKC', pagina_normalizada
            )
            for trecho in ('G1 — elipsoide FULL 95%', 'G2 — elipsoide FULL 95%',
                           'G3 — elipsoide FULL 95%', 'Girar automaticamente',
                           'Centro G1', 'Centro G2', 'Centro G3'):
                assert trecho in pagina_normalizada, f'Controle HTML ausente: {trecho}'
            for par in (('PC1','PC2'),('PC1','PC3'),('PC2','PC3')):
                csv = saida / f'GMM04_malha_{par[0]}_{par[1]}.csv'
                assert csv.is_file(), csv
                tab = pd.read_csv(csv)
                assert len(tab) == 55**2
                assert np.allclose(tab[['responsabilidade_G1','responsabilidade_G2',
                                        'responsabilidade_G3']].sum(axis=1), 1)
            volume = pd.read_csv(saida/'GMM04_dominancia_volume_3d.csv')
            assert len(volume) == 15**3
            assert np.allclose(volume[['responsabilidade_G1','responsabilidade_G2',
                                       'responsabilidade_G3']].sum(axis=1), 1)
            # Testar copia segura de HTML na galeria isolada (sem manifestos reais).
            destino = pasta / 'galeria_teste' / 'geometria_desenvolvimento' / html.name
            galeria._arquivo_seguro_copiar(html, destino)
            assert destino.read_bytes() == html.read_bytes()
        print(f'PCA {d}D: VALIDADO')

    assert arquivo_yaml.read_bytes() == yaml_original, 'YAML foi alterado.'
    print('PNG oficial preservado; HTML exclusivo do 3D e copia segura validados.')
    print('Dados sinteticos; sem OOF real e sem F10.')
    print('PASTA=' + str(pasta.resolve()))
    print('PNG=' + str((pasta/'PCA_3D'/'GMM04_dominancia_3d.png').resolve()))
    print('HTML=' + str((pasta/'PCA_3D'/'GMM04_dominancia_3d_INTERATIVA.html').resolve()))


if __name__ == '__main__':
    main()
