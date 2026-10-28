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
from scipy.stats import norm, multivariate_normal


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


def calcular_conjunta_2d(gmm, eixo_x: np.ndarray, eixo_y: np.ndarray):
    """Retorna grade 2D, contribuicoes KxNyxNx e densidade conjunta total NyxNx."""
    from src import visualizacao_geometrica as geo

    modelo = gmm.modelo_
    mu = np.asarray(modelo.means_, dtype=float)
    pesos = np.asarray(modelo.weights_, dtype=float)

    x = np.asarray(eixo_x, dtype=float)
    y = np.asarray(eixo_y, dtype=float)

    geo._exigir(mu.ndim == 2 and mu.shape[1] == 2,
                'A densidade conjunta do GMM01 exige PCA 2D.')
    geo._exigir(x.ndim == 1 and len(x) >= 2 and np.isfinite(x).all(),
                'Eixo X da densidade conjunta invalido.')
    geo._exigir(y.ndim == 1 and len(y) >= 2 and np.isfinite(y).all(),
                'Eixo Y da densidade conjunta invalido.')
    geo._exigir(len(pesos) == mu.shape[0] and np.isfinite(pesos).all()
                and np.all(pesos > 0) and np.isclose(pesos.sum(), 1.0, atol=1e-5),
                'Pesos da mistura invalidos.')

    xx, yy = np.meshgrid(x, y)
    pos = np.dstack((xx, yy))

    contribuicoes = []
    for k in range(len(pesos)):
        cov = np.asarray(geo._covariancia(gmm, k)[:2, :2], dtype=float)
        geo._exigir(cov.shape == (2, 2) and np.isfinite(cov).all(),
                    'Covariancia 2D invalida na densidade conjunta.')
        dens = pesos[k] * multivariate_normal(
            mean=mu[k, :2],
            cov=cov,
            allow_singular=False
        ).pdf(pos)
        contribuicoes.append(dens)

    contribuicoes = np.asarray(contribuicoes, dtype=float)
    total = contribuicoes.sum(axis=0)

    geo._exigir(np.isfinite(total).all() and np.all(total >= 0),
                'Densidade conjunta invalida.')

    return xx, yy, contribuicoes, total


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

                # =====================================================
        # VISUALIZACAO DA DENSIDADE MARGINAL
        # =====================================================

        fig, ax = plt.subplots(figsize=(10.8, 6.9))

        cores = plt.get_cmap('tab10')

        # Componentes gaussianas
        for k, nome_gaussiana in enumerate(gm.saidas_):
            ax.plot(
                eixo,
                componentes[k],
                ls='--',
                lw=1.7,
                color=cores(k % 10),
                label=f'{nome_gaussiana} (π={pesos[k]:.7f})',
                alpha=.82
            )

        # Mistura gaussiana total
        ax.plot(
            eixo,
            mistura,
            color='#172b46',
            lw=3.2,
            label='Mistura total',
            zorder=5
        )

        # Eixos
        ax.set(
            xlabel=nome,
            ylabel='Densidade marginal',
            xlim=(xmin, xmax)
        )

        ax.set_ylim(bottom=0)

        # Titulo principal
        fig.suptitle(
            'DENSIDADE MARGINAL DA MISTURA GAUSSIANA',
            fontsize=16,
            fontweight='bold',
            y=.965
        )

        # Subtitulo
        fig.text(
            .5,
            .905,
            f'{nome} | {gm.modelo_.covariance_type.upper()} | {d}D',
            ha='center',
            fontsize=11,
            color='#46556e'
        )

        # Grade
        ax.grid(
            alpha=.18,
            ls=':'
        )

        # Legenda
        ax.legend(
            loc='best',
            fontsize=9,
            frameon=True,
            ncol=2 if len(pesos) > 4 else 1
        )



        # Sem rodape inferior




        # Margens
        fig.subplots_adjust(
            left=.09,
            right=.975,
            top=.89,
            bottom=.09
        )

        # Nome original do arquivo
        stem = f'GMM01_densidade_marginal_{d}d_{nome}'

        # Salvar figura
        geo._guardar(
            fig,
            destino,
            stem,
            ext,
            dpi
        )

        if csv:
            tab = pd.DataFrame({'coordenada':eixo, 'densidade_mistura':mistura})
            for k, nome_gaussiana in enumerate(gm.saidas_):
                tab[f'contribuicao_{nome_gaussiana}'] = componentes[k]
            geo._salvar_dados(tab, destino, stem, True)

    # =====================================================
    # NOVO GRAFICO: DENSIDADE CONJUNTA 2D DA MISTURA
    # =====================================================

    if d == 2:
        grade_2d = int(op.get('pontos_grade_2d', 220))
        geo._exigir(80 <= grade_2d <= 500,
                    'Quantidade de pontos da grade 2D invalida.')

        # Dominios de PC1 e PC2 com base nos dados e no suporte das gaussianas
        limites = []
        for j in range(2):
            valores = x.iloc[:, j].to_numpy(dtype=float)
            geo._exigir(np.isfinite(valores).all(),
                        'Dados 2D nao finitos na densidade conjunta.')

            medias = np.asarray(gm.modelo_.means_[:, j], dtype=float)
            desvios = np.array([
                np.sqrt(geo._covariancia(gm, k)[j, j])
                for k in range(len(pesos))
            ], dtype=float)

            q0, q1 = np.quantile(valores, [.005, .995])

            vmin = min(q0, float(np.min(medias - sigmas * desvios)))
            vmax = max(q1, float(np.max(medias + sigmas * desvios)))

            geo._exigir(np.isfinite([vmin, vmax]).all() and vmin < vmax,
                        'Dominio 2D invalido no GMM01.')

            limites.append((vmin, vmax))

        eixo_x = np.linspace(limites[0][0], limites[0][1], grade_2d)
        eixo_y = np.linspace(limites[1][0], limites[1][1], grade_2d)

        xx, yy, contribs_2d, mistura_2d = calcular_conjunta_2d(
            gm, eixo_x, eixo_y
        )

        fig, ax = plt.subplots(figsize=(10.6, 8.2))
        cores = plt.get_cmap('tab10')

        # Fundo: densidade total da mistura
        cf = ax.contourf(
            xx,
            yy,
            mistura_2d,
            levels=18,
            cmap='Blues'
        )

        # Curvas da mistura total
        ax.contour(
            xx,
            yy,
            mistura_2d,
            levels=8,
            colors='#172b46',
            linewidths=1.0,
            alpha=.75
        )

        # Curvas de cada componente
        for k, nome_gaussiana in enumerate(gm.saidas_):
            ax.contour(
                xx,
                yy,
                contribs_2d[k],
                levels=4,
                colors=[cores(k % 10)],
                linestyles='--',
                linewidths=1.35,
                alpha=.95
            )

            ax.scatter(
                gm.modelo_.means_[k, 0],
                gm.modelo_.means_[k, 1],
                color=cores(k % 10),
                marker='x',
                s=72,
                linewidths=2.0,
                label=f'{nome_gaussiana} (π={pesos[k]:.7f})',
                zorder=5
            )

        barra = fig.colorbar(cf, ax=ax, pad=.02)
        barra.set_label('Densidade conjunta', fontsize=10)

        ax.set(
            xlabel=nomes[0],
            ylabel=nomes[1],
            xlim=limites[0],
            ylim=limites[1]
        )

        fig.suptitle(
            'DENSIDADE CONJUNTA DA MISTURA GAUSSIANA',
            fontsize=16,
            fontweight='bold',
            y=.965
        )

        fig.text(
            .5,
            .905,
            f'{nomes[0]} × {nomes[1]} | {gm.modelo_.covariance_type.upper()} | {d}D',
            ha='center',
            fontsize=11,
            color='#46556e'
        )

        ax.grid(alpha=.14, ls=':')

        ax.legend(
            loc='best',
            fontsize=9,
            frameon=True,
            ncol=2 if len(pesos) > 4 else 1
        )

        fig.subplots_adjust(
            left=.10,
            right=.93,
            top=.89,
            bottom=.10
        )

        stem = f'GMM01_densidade_conjunta_{d}d_{nomes[0]}_{nomes[1]}'

        geo._guardar(
            fig,
            destino,
            stem,
            ext,
            dpi
        )

        if csv:
            tab2d = pd.DataFrame({
                nomes[0]: xx.ravel(),
                nomes[1]: yy.ravel(),
                'densidade_mistura': mistura_2d.ravel()
            })

            for k, nome_gaussiana in enumerate(gm.saidas_):
                tab2d[f'contribuicao_{nome_gaussiana}'] = contribs_2d[k].ravel()

            geo._salvar_dados(tab2d, destino, stem, True)

        # =====================================================
        # DENSIDADE CONJUNTA: SUPERFICIE 3D DO GMM PCA 2D
        # =====================================================

        from scipy.stats import chi2
        from matplotlib.lines import Line2D

        fig3d = plt.figure(figsize=(12.0, 8.5))
        ax3d = fig3d.add_subplot(111, projection='3d')

        # Superficie da densidade total
        passo = max(1, int(np.ceil(grade_2d / 125)))

        ax3d.plot_surface(
            xx[::passo, ::passo],
            yy[::passo, ::passo],
            mistura_2d[::passo, ::passo],
            cmap='viridis',
            linewidth=0,
            antialiased=True,
            alpha=.92,
            rstride=1,
            cstride=1
        )

        # Elipses gaussianas FULL de concentracao de 95%
        raio95 = np.sqrt(chi2.ppf(0.95, df=2))

        angulos = np.linspace(0, 2 * np.pi, 250)

        circunferencia = np.vstack([
            np.cos(angulos),
            np.sin(angulos)
        ])

        cores3d = plt.get_cmap('tab10')
        legenda3d = []

        for k, nome_gaussiana in enumerate(gm.saidas_):

            cov = np.asarray(
                geo._covariancia(gm, k)[:2, :2],
                dtype=float
            )

            autovalores, autovetores = np.linalg.eigh(cov)

            geo._exigir(
                np.all(autovalores > 0),
                'Covariancia FULL nao positiva definida.'
            )

            centro = np.asarray(
                gm.modelo_.means_[k, :2],
                dtype=float
            )

            elipse = centro + (
                autovetores
                @ np.diag(np.sqrt(autovalores))
                @ (raio95 * circunferencia)
            ).T

            cor = cores3d(k % 10)

            # Projecao da elipse no plano Z=0
            ax3d.plot(
                elipse[:, 0],
                elipse[:, 1],
                np.zeros(len(angulos)),
                color=cor,
                linestyle=(0, (5, 3)),
                linewidth=2.2
            )

            ax3d.scatter(
                centro[0],
                centro[1],
                0,
                color=cor,
                marker='x',
                s=60,
                depthshade=False
            )

            legenda3d.append(
                Line2D(
                    [0], [0],
                    color=cor,
                    linestyle=(0, (5, 3)),
                    linewidth=2,
                    label=f'{nome_gaussiana} (π={pesos[k]:.7f}) — 95%'
                )
            )

        # Eixos
        ax3d.set_xlim(*limites[0])
        ax3d.set_ylim(*limites[1])

        ax3d.set_zlim(
            0,
            float(np.max(mistura_2d)) * 1.08
        )

        ax3d.set_xlabel('PC1', labelpad=9)
        ax3d.set_ylabel('PC2', labelpad=9)
        ax3d.set_zlabel(
            'Densidade conjunta',
            labelpad=11
        )

        # Perspectiva tridimensional
        ax3d.view_init(
            elev=28,
            azim=-58
        )

        ax3d.set_box_aspect((1.35, 1.0, 0.65))

        ax3d.legend(
            handles=legenda3d,
            loc='upper left',
            bbox_to_anchor=(0, 1),
            fontsize=8.5,
            frameon=True
        )

        fig3d.suptitle(
            'DENSIDADE CONJUNTA 3D DA MISTURA GAUSSIANA',
            fontsize=16,
            fontweight='bold',
            y=.965
        )

        fig3d.text(
            .5,
            .916,
            f'{nomes[0]} × {nomes[1]} × Densidade | '
            f'{gm.modelo_.covariance_type.upper()} | '
            f'{len(pesos)} Gaussianas',
            ha='center',
            fontsize=11,
            color='#46556e'
        )

        fig3d.subplots_adjust(
            left=.02,
            right=.94,
            bottom=.035,
            top=.88
        )

        stem3d = (
            'GMM01_densidade_conjunta_superficie_PCA2D'
        )

        geo._guardar(
            fig3d,
            destino,
            stem3d,
            ext,
            dpi
        )
    # PCA3D_GMM01_EXTRAS_OFICIAIS_V1
    # A extensao e executada somente apos as marginais PCA 3D.
    if d == 3:
        from src.visualizacao_gmm01_3d import gerar_complementares
        gerar_complementares(config, data, destino, ext, dpi, csv)
