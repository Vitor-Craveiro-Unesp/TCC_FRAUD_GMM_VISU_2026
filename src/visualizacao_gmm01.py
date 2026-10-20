"""GMM 01 — densidades marginais de uma unica mistura gaussiana.

Para cada dimensao j, calcula f_j(t) = somatorio_k pi_k * N(t | mu_kj, Sigma_kjj).
Nao utiliza RH_k, rotulos, limiares ou teste F10. A densidade NAO e risco de fraude.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm


def calcular_marginal(gmm, j: int, eixo: np.ndarray):
    """Retorna contribuicoes KxN ponderadas e densidade marginal total Nx1."""
    from src import visualizacao_geometrica as geo
    modelo = gmm.modelo_
    mu = np.asarray(modelo.means_, dtype=float)
    pesos = np.asarray(modelo.weights_, dtype=float)
    t = np.asarray(eixo, dtype=float)
    geo._exigir(mu.ndim == 2 and j in range(mu.shape[1]), 'Indice de dimensao invalido.')
    geo._exigir(t.ndim == 1 and len(t) >= 2 and np.isfinite(t).all(),
                'Eixo da marginal invalido.')
    geo._exigir(len(pesos) == mu.shape[0] and np.isfinite(pesos).all()
                and np.all(pesos > 0) and np.isclose(pesos.sum(), 1.0, atol=1e-5),
                'Pesos da mistura invalidos.')
    variancias = np.array([geo._covariancia(gmm, k)[j, j]
                           for k in range(len(pesos))], dtype=float)
    geo._exigir(np.isfinite(variancias).all() and np.all(variancias > 0),
                'Variancias marginais invalidas.')
    contribuicoes = np.array([
        pesos[k] * norm.pdf(t, loc=mu[k, j], scale=np.sqrt(variancias[k]))
        for k in range(len(pesos))
    ])
    total = contribuicoes.sum(axis=0)
    geo._exigir(np.isfinite(total).all() and np.all(total >= 0),
                'Densidades marginais invalidas.')
    return contribuicoes, total


def gerar(config, data, destino, ext, dpi, csv):
    """Gera exatamente uma figura por dimensao do GMM (1D, 2D ou 3D)."""
    from src import visualizacao_geometrica as geo
    op = config['visualizacao']['graficos']['densidades_marginais']
    gm = data['gmm']
    x = data['coord']
    d = int(x.shape[1])
    geo._exigir(d in (1, 2, 3), 'Catalogo GMM01 definido para 1D, 2D ou 3D.')
    geo._exigir(np.asarray(gm.modelo_.means_).shape[1] == d,
                'Dimensoes dos dados e do GMM nao coincidem.')
    pontos = int(op.get('pontos_curva', 600))
    geo._exigir(100 <= pontos <= 10000, 'Quantidade de pontos da marginal invalida.')
    sigmas = float(op.get('margem_sigmas', 4.0))
    geo._exigir(np.isfinite(sigmas) and 2 <= sigmas <= 10,
                'Margem de sigmas invalida.')
    nomes = list(x.columns)
    pesos = np.asarray(gm.modelo_.weights_, dtype=float)

    for j, nome in enumerate(nomes):
        # A faixa combina quantis do desenvolvimento e suporte das gaussianas;
        # evitar corte dos componentes de peso pequeno.
        valores = x.iloc[:, j].to_numpy(dtype=float)
        geo._exigir(np.isfinite(valores).all(), 'Dados marginais nao finitos.')
        medias = np.asarray(gm.modelo_.means_[:, j], dtype=float)
        desvios = np.array([np.sqrt(geo._covariancia(gm, k)[j, j])
                            for k in range(len(pesos))])
        q0, q1 = np.quantile(valores, [.005, .995])
        xmin = min(q0, float(np.min(medias-sigmas*desvios)))
        xmax = max(q1, float(np.max(medias+sigmas*desvios)))
        geo._exigir(np.isfinite([xmin,xmax]).all() and xmin < xmax,
                    'Dominio da marginal invalido.')
        eixo = np.linspace(xmin, xmax, pontos)
        componentes, mistura = calcular_marginal(gm, j, eixo)
        fig, ax = plt.subplots(figsize=(10.8, 6.9))
        cores = plt.get_cmap('tab10')
        for k, nome_gaussiana in enumerate(gm.saidas_):
            ax.plot(eixo, componentes[k], ls='--', lw=1.9,
                    color=cores(k % 10),
                    label=f'{nome_gaussiana} — π={pesos[k]:.4g}', alpha=.94)
        ax.plot(eixo, mistura, color='#172b46', lw=2.9,
                label='Mistura marginal total', zorder=5)
        ax.set(xlabel=nome, ylabel='Densidade marginal', xlim=(xmin, xmax))
        ax.set_ylim(bottom=0)
        ax.set_title(f'GMM 01 — Densidade marginal | {nome} | '
                     f'{gm.modelo_.covariance_type.upper()} | {d}D',
                     fontsize=14, fontweight='bold', pad=14)
        ax.grid(alpha=.18, ls=':')
        ax.legend(loc='best', fontsize=9, frameon=True, ncol=2 if len(pesos)>4 else 1)
        fig.text(.5, .035,
                 'Curvas tracejadas = πₖ × N(t | μₖⱼ, Σₖⱼⱼ); linha escura = soma. '
                 'Densidade NÃO é probabilidade de fraude. Ajuste: F1–F9; sem F10.',
                 fontsize=8.6, color='#46556e', ha='center')
        fig.subplots_adjust(left=.09, right=.975, top=.875, bottom=.16)
        stem = f'GMM01_densidade_marginal_{d}d_{nome}'
        geo._guardar(fig, destino, stem, ext, dpi)
        if csv:
            tab = pd.DataFrame({'coordenada':eixo, 'densidade_mistura':mistura})
            for k, nome_gaussiana in enumerate(gm.saidas_):
                tab[f'contribuicao_{nome_gaussiana}'] = componentes[k]
            geo._salvar_dados(tab, destino, stem, True)
