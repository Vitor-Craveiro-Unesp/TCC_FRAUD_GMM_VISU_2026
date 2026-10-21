"""P08 — Protótipo de Biplot PCA 1D adaptado, totalmente sintético.

Não importa módulos do projeto nem lê bases, folds, OOF ou F10.
PCA ajustado somente nas features. Classes simuladas somente para coloração.
Execute: python P08_BIPLOT_PCA_1D_PREVIA_SIMULADA.py
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D

SEED = 20261020
LEGITIMA = '#1f77b4'  # YAML: visualizacao.classes.legitima.cor
FRAUDE = '#d62728'    # YAML: visualizacao.classes.fraude.cor


def dados_sinteticos():
    rng = np.random.default_rng(SEED)
    n_leg, n_fraude = 550, 50
    rotulos = np.r_[np.zeros(n_leg, dtype=int), np.ones(n_fraude, dtype=int)]
    fator = np.r_[rng.normal(0, 1.0, n_leg), rng.normal(0.95, 1.15, n_fraude)]
    eps = rng.normal(size=(n_leg+n_fraude, 6))
    matriz = np.column_stack([
        1.00*fator + 0.45*eps[:, 0],
        0.82*fator + 0.65*eps[:, 1],
       -0.92*fator + 0.55*eps[:, 2],
       -0.72*fator + 0.78*eps[:, 3],
        0.12*fator + 1.08*eps[:, 4],
        0.46*fator + 0.88*eps[:, 5],
    ])
    nomes = [f'Característica {t}' for t in 'ABCDEF']
    return pd.DataFrame(matriz, columns=nomes), rotulos


def prototipo(output_dir):
    x_raw, y = dados_sinteticos()
    x = StandardScaler().fit_transform(x_raw)
    pca = PCA(n_components=1, svd_solver='full', random_state=SEED)
    z = pca.fit_transform(x)[:, 0]
    # Fórmula do coeficiente de correlação com PC1 em dados de treino:
    # corr(X_j, PC1) = v_{1j} * sqrt(lambda_1) / s_j
    # s_j usa ddof=1, assim como explained_variance_ do sklearn.
    desvio = np.std(x, axis=0, ddof=1)
    corr = pca.components_[0] * np.sqrt(pca.explained_variance_[0]) / desvio
    corr_direta = np.array([np.corrcoef(x[:, i], z)[0, 1] for i in range(x.shape[1])])
    np.testing.assert_allclose(corr, corr_direta, atol=1e-10, rtol=1e-10)
    assert z.ndim == 1 and pca.n_components_ == 1
    assert np.max(np.abs(corr)) <= 1.0 + 1e-12
    assert np.all(np.isfinite(z))

    fig = plt.figure(figsize=(13.4, 8.6), facecolor='#f8fafc')
    fig.suptitle('P08  |  PCA 1D — Biplot adaptado',
                 x=0.065, y=.974, ha='left', va='top', fontsize=20,
                 color='#132238', fontweight='bold')
    fig.text(.065, .924,
             f'PRÉVIA SINTÉTICA · {len(z)} observações · {x.shape[1]} características · '
             f'PC1 explica {pca.explained_variance_ratio_[0]:.1%} da variância',
             ha='left', va='top', fontsize=11, color='#546477')

    gs = fig.add_gridspec(2, 1, left=.095, right=.94, top=.865, bottom=.135,
                          height_ratios=[1.08, 1.22], hspace=.42)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    for ax in (ax1, ax2):
        ax.set_facecolor('white')
        for side in ('right', 'top'):
            ax.spines[side].set_visible(False)
        for side in ('bottom', 'left'):
            ax.spines[side].set_color('#d9e2ec')

    # Todos os pontos têm somente coordenada PC1. O deslocamento vertical
    # abaixo é ruído visual de apresentação, não um eixo PCA adicional.
    j_rng = np.random.default_rng(SEED + 1)
    jitter = j_rng.uniform(-.31, .31, size=len(z))
    ax1.scatter(z[y == 0], jitter[y == 0], s=20, c=LEGITIMA,
                alpha=.37, marker='o', linewidths=0,
                label=f'Legítima (n={int(np.sum(y==0))})', rasterized=True)
    ax1.scatter(z[y == 1], jitter[y == 1], s=80, c=FRAUDE,
                alpha=.92, marker='*', edgecolors='white', linewidths=.4,
                label=f'Fraude (n={int(np.sum(y==1))})', rasterized=True)
    ax1.axvline(0, ls='--', lw=1.2, color='#75859a')
    ax1.axhline(0, color='#e2e8f0', lw=.9)
    ax1.set_ylim(-.48, .52)
    ax1.set_yticks([])
    ax1.set_xlabel('Score PCA — PC1 (unidades da projeção)', fontsize=11)
    ax1.set_title('A · Observações projetadas sobre PC1', loc='left', pad=14,
                  color='#233650', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper right', ncol=2, frameon=False, fontsize=10)
    ax1.grid(axis='x', color='#eaf0f5', lw=1)
    ax1.text(.008, .08, 'Altura = deslocamento visual (jitter), NÃO é PC2',
             transform=ax1.transAxes, fontsize=9, color='#66758a',
             va='bottom', bbox=dict(facecolor='white', edgecolor='none', alpha=.86, pad=3))

    # Painel de vetores 1D: origens em zero, coordenada horizontal = correlação.
    # Não sobrepomos escalas incompatíveis dos scores e das correlações.
    nomes = list(x_raw.columns)
    positions = np.arange(len(nomes))[::-1]
    for j, yy in enumerate(positions):
        val = float(corr[j])
        color = '#177e73' if val >= 0 else '#7659a6'
        ax2.annotate('', xy=(val, yy), xytext=(0, yy),
                     arrowprops=dict(arrowstyle='-|>', mutation_scale=16,
                                     linewidth=2.6, color=color,
                                     shrinkA=0, shrinkB=0))
        ax2.plot([val, val], [yy-.08, yy+.08], color=color, lw=1.3)
        ax2.text(val + (.035 if val >= 0 else -.035), yy,
                 f'{val:+.2f}', ha='left' if val >= 0 else 'right', va='center',
                 color=color, fontsize=10, fontweight='bold')
    ax2.axvline(0, ls='--', lw=1.2, color='#738197')
    ax2.set_yticks(positions, nomes)
    ax2.tick_params(axis='y', length=0, pad=12, labelsize=10)
    ax2.set_xlim(-1.10, 1.10)
    ax2.set_xticks(np.linspace(-1, 1, 9))
    ax2.set_xlabel('Correlação da característica com PC1 (escala −1 a +1)', fontsize=11)
    ax2.set_title('B · Direção e intensidade das características em PC1',
                  loc='left', pad=14, color='#233650', fontweight='bold', fontsize=13)
    ax2.grid(axis='x', color='#eaf0f5', lw=1)
    ax2.set_axisbelow(True)
    ax2.set_ylim(-.7, len(nomes)-.25)
    fig.text(.095, .070,
             'PCA ajustado sem rótulos; classes aparecem apenas para leitura descritiva. '
             'O sinal de PC1 pode inverter sem alterar o PCA.',
             ha='left', fontsize=9.3, color='#465870')
    fig.text(.095, .045,
             'Escalas distintas: painel A = scores; painel B = correlações. '
             'Não é um biplot 2D nem um resultado real do TCC.',
             ha='left', fontsize=9.3, color='#465870')
    fig.text(.94, .974, 'PROTÓTIPO · NÃO INTEGRADO', ha='right', va='top',
             fontsize=9.5, fontweight='bold', color='#9a5c21')

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'P08_BIPLOT_PCA_1D_PREVIA_SIMULADA.png'
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    tabela = pd.DataFrame({'caracteristica': nomes, 'correlacao_com_PC1': corr,
                           'componente_PCA_1': pca.components_[0]})
    tabela.to_csv(output_dir / 'P08_BIPLOT_PCA_1D_VARIAVEIS_SIMULADAS.csv', index=False)
    print(f'PREVIA GERADA: {path}')
    print(f'n={len(z)} | PCA 1D | variancia_explicada={pca.explained_variance_ratio_[0]:.6f}')
    print('TESTE MATEMATICO OK: correlacao = loading * sqrt(autovalor) / desvio(ddof=1)')
    print(tabela.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print('NENHUM dado real, modulo da pipeline ou F10 foi acessado.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saida', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    prototipo(args.saida)
