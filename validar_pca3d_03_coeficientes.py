"""Auditoria PCA 3D - coeficientes originais PC1/PC2/PC3 (39 features).

Na raiz do projeto TCC_EXPERIMENTAL_V3:
    .\\.venv\\Scripts\\python.exe .\\validar_pca3d_03_coeficientes.py

Gera PNG e CSV em outputs/validacao_pca3d_03_coeficientes_*/.
Usa SOMENTE dados sinteticos. Nao altera YAML, nao acessa F10,
nao executa o pipeline do TCC e nao modifica src/.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def cores_consistentes(variaveis):
    """Mesma regra de paleta de _mapa_cores_variaveis_pca no projeto."""
    nomes = sorted(set(str(v) for v in variaveis))
    paleta = []
    for pal in ('tab20', 'tab20b', 'tab20c'):
        cmap = plt.get_cmap(pal)
        for cor in cmap.colors:
            paleta.append(to_hex(tuple(.82 * canal for canal in cor[:3])))
    if len(nomes) > len(paleta):
        raise ValueError('A paleta permite no maximo 60 variaveis.')
    return dict(zip(nomes, paleta))


def gerar_dados_sinteticos():
    """Mesmos dados do teste isolado do Biplot PCA3D (semente 20261027)."""
    rng = np.random.default_rng(20261027)
    centros = np.array([
        [-2.5, -1.4, -.5],
        [2.5, -1.0, 1.3],
        [-.3, 2.6, -1.6]
    ])
    covariancias = np.array([
        [[.6, .2, .05], [.2, .8, -.1], [.05, -.1, .65]],
        [[.8, -.2, .13], [-.2, .6, .02], [.13, .02, .70]],
        [[.75, .15, -.10], [.15, .9, .2], [-.10, .2, .5]],
    ])
    latente = np.vstack([
        rng.multivariate_normal(centros[k], covariancias[k], size=420)
        for k in range(3)
    ])
    pesos = rng.normal(size=(3, 39))
    X = latente @ pesos + rng.normal(scale=.53, size=(len(latente), 39))
    nomes = [f'FEATURE_{i+1:02d}' for i in range(39)]
    X_pre = StandardScaler().fit_transform(X)
    modelo = PCA(n_components=3, random_state=20261027).fit(X_pre)
    assert modelo.components_.shape == (3, 39)
    return modelo, nomes


def gerar_grafico(tabela: pd.DataFrame, variancia: np.ndarray, caminho: Path):
    nomes = tabela['variavel_preprocessada'].tolist()
    n = len(nomes)
    y = np.arange(n)
    valor_maximo = float(np.max(np.abs(tabela[[
        'coeficiente_PC1', 'coeficiente_PC2', 'coeficiente_PC3'
    ]].to_numpy())))
    limite = max(.08, valor_maximo * 1.52)

    # Espaco confortavel para 39 linhas + tres eixos e sete casas decimais.
    altura = max(12.5, 2.6 + .405 * n)
    fig, axes = plt.subplots(1, 3, figsize=(25.5, altura), sharey=True)
    fig.patch.set_facecolor('white')
    cores = tabela['cor_hex'].tolist()

    for j, ax in enumerate(axes):
        coluna = f'coeficiente_PC{j+1}'
        valores = tabela[coluna].to_numpy(dtype=float)
        ax.axvline(0, color='#526174', lw=1.25, zorder=1)
        ax.set_axisbelow(True)
        ax.grid(axis='x', linestyle=':', alpha=.27)
        for i, (valor, cor) in enumerate(zip(valores, cores)):
            ax.annotate('', xy=(valor, i), xytext=(0, i),
                arrowprops={'arrowstyle': '-|>', 'lw': 1.75,
                            'color': cor, 'mutation_scale': 12}, zorder=3)
            zero_seguro = 0.0 if abs(valor) < 5e-8 else float(valor)
            deslocamento = limite * .022
            ax.text(valor + (deslocamento if valor >= 0 else -deslocamento), i,
                f'{zero_seguro:+.7f}',
                ha='left' if valor >= 0 else 'right',
                va='center', fontsize=7.0, color=cor, fontweight='semibold')

        ax.set_xlim(-limite, limite)
        ax.set_ylim(n - .7, -.8)
        ax.set_title(f'PC{j+1}  |  Variancia explicada: {variancia[j]:.7%}',
                     fontweight='bold', fontsize=12.5, pad=15)
        ax.set_xlabel('Coeficiente da combinacao linear', fontsize=10.5, labelpad=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{0. if abs(x)<5e-8 else x:.7f}'))
        ax.tick_params(axis='x', labelsize=8.3, rotation=12)
        ax.tick_params(axis='y', length=0, pad=8)
        for s in ['top','right']:
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color('#b3bece')
        ax.spines['left'].set_visible(False)

    # Visibilizar o nome completo de cada uma das 39 features, uma so vez.
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(nomes, fontsize=9.2)
    axes[0].tick_params(axis='y', labelleft=True)
    for ax in axes[1:]:
        ax.tick_params(axis='y', labelleft=False)

    fig.suptitle('COEFICIENTES DAS COMBINACOES LINEARES DO PCA 3D',
                 fontsize=20, fontweight='bold', y=.995)
    fig.text(.5, .951,
        f'{n} variaveis preprocessadas  |  Mesma ordem, cor e escala nos tres paineis',
        ha='center', fontsize=11, color='#506278')
    fig.subplots_adjust(left=.115, right=.986, top=.908, bottom=.072, wspace=.115)
    fig.savefig(caminho, dpi=145, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    raiz = Path.cwd()
    # Uma auditoria isolada na raiz, sem alterar configuracoes de treinamento.
    if not (raiz / 'src' / 'visualizacao_pca_biplot.py').is_file():
        raise SystemExit('Execute na raiz do TCC_EXPERIMENTAL_V3 (pasta src ausente).')
    yaml_file = raiz / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    yaml_antes = yaml_file.read_bytes() if yaml_file.is_file() else None

    pca, nomes = gerar_dados_sinteticos()
    tabela = pd.DataFrame(pca.components_.T,
        columns=['coeficiente_PC1','coeficiente_PC2','coeficiente_PC3'])
    tabela.insert(0, 'variavel_preprocessada', nomes)
    tabela['magnitude_conjunta'] = np.linalg.norm(
        tabela[['coeficiente_PC1','coeficiente_PC2','coeficiente_PC3']].to_numpy(), axis=1)
    tabela = tabela.sort_values(
        ['magnitude_conjunta','variavel_preprocessada'],
        ascending=[False, True], kind='stable').reset_index(drop=True)
    mapa = cores_consistentes(nomes)
    tabela['cor_hex'] = tabela['variavel_preprocessada'].map(mapa)

    pasta = raiz / 'outputs' / ('validacao_pca3d_03_coeficientes_' +
                               datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    pasta.mkdir(parents=True, exist_ok=False)
    png = pasta / 'PCA_COEFICIENTES_PC1_PC2_PC3.png'
    csv = pasta / 'PCA_COEFICIENTES_PC1_PC2_PC3.csv'

    # O CSV conserva toda a precisao numerica do PCA; o PNG exibe 7 decimais.
    tabela.to_csv(csv, index=False, float_format='%.17g', encoding='utf-8')
    gerar_grafico(tabela, pca.explained_variance_ratio_, png)

    assert png.is_file() and csv.is_file()
    assert len(pd.read_csv(csv)) == 39
    assert set(tabela['variavel_preprocessada']) == set(nomes)
    if yaml_antes is not None:
        assert yaml_file.read_bytes() == yaml_antes, 'ERRO: YAML foi alterado'

    print('PCA 3D: COEFICIENTES PC1 PC2 PC3 GERADOS')
    print('Features: 39 | tres paineis | dados sinteticos | F10 nao utilizado')
    print('PNG=' + str(png.resolve()))
    print('CSV=' + str(csv.resolve()))
    print('PASTA=' + str(pasta.resolve()))


if __name__ == '__main__':
    main()
