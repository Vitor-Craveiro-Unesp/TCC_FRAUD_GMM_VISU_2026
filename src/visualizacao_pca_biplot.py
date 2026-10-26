"""Biplot PCA 1D/2D/3D do desenvolvimento, sem novos fits ou acesso ao F10.

A sobreposicao de classe usa rotulos apenas na apresentacao. As correlacoes
usam os coeficientes do PCA ajustado no mesmo X_pre e nao usam a classe.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import visualizacao_oof as vis

PREFIXO = 'PCA_BIPLOT_'  # Nome PROVISORIO; numero oficial pendente dos catalogos.


def _entrada(data):
    """Extrai (X_pre, scores, PCA) do MESMO ajuste do desenvolvimento."""
    red = data.get('redutor')
    if red is None or not hasattr(red, 'modelo_') or not hasattr(red, 'colunas_'):
        raise ValueError('Biplot PCA exige RedutorPCA ja ajustado.')
    x = data.get('pre')
    z = data.get('coord')
    if not isinstance(x, pd.DataFrame) or not isinstance(z, pd.DataFrame):
        raise ValueError('X preprocessado e scores devem ser DataFrames.')
    if x.empty or not x.index.equals(z.index) or list(x.columns) != list(red.colunas_):
        raise ValueError('X, scores e colunas do PCA estao desalinhados.')
    modelo = red.modelo_
    d = int(modelo.n_components_)
    if d not in (1, 2, 3) or z.shape != (len(x), d):
        raise ValueError('Biplot aceita somente PCA 1D, 2D ou 3D efetivamente ajustado.')
    if list(z.columns) != [f'PC{i+1}' for i in range(d)]:
        raise ValueError('Eixos PC nao correspondem ao PCA treinado.')
    xv, zv = x.to_numpy(copy=False), z.to_numpy(copy=False)
    if not np.isfinite(xv).all() or not np.isfinite(zv).all():
        raise ValueError('Coordenadas ou variaveis nao finitas.')
    if len(x) < 2:
        raise ValueError('Biplot exige ao menos duas observacoes.')
    y = np.asarray(data.get('y'))
    if y.shape != (len(x),) or not np.isin(y, [0, 1]).all():
        raise ValueError('Rotulos descritivos desalinhados ou invalidos.')
    return x, z, modelo, y


def tabela_correlacoes(data):
    """Correlacoes Pearson X_j vs PC_i a partir do PCA ja ajustado.

    Formula vale tambem para whiten=True (escala positiva das PCs).
    Colunas constantes ou autovalores nulos recebem correlacao 0 e sao
    identificadas explicitamente por `constante`.
    """
    x, z, modelo, _ = _entrada(data)
    a = np.asarray(modelo.components_, dtype=float)
    ev = np.asarray(modelo.explained_variance_, dtype=float)
    d, p = a.shape
    if d != z.shape[1] or p != x.shape[1]:
        raise ValueError('Matriz de componentes incompatível com X e scores.')
    if not np.isfinite(a).all() or not np.isfinite(ev).all() or np.any(ev < 0):
        raise ValueError('Parametros PCA nao finitos ou variancias negativas.')
    std = np.std(x.to_numpy(copy=False), axis=0, ddof=1, dtype=np.float64)
    valido = np.isfinite(std) & (std > 1e-12)
    corr = np.zeros((p, d), dtype=float)
    np.divide((a * np.sqrt(ev)[:, None]).T, std[:, None], out=corr,
              where=valido[:, None])
    if not np.isfinite(corr).all() or np.any(np.abs(corr) > 1.0001):
        raise ValueError('Correlacoes PCA incoerentes: conferir ajuste e dados.')
    corr = np.clip(corr, -1.0, 1.0)
    colunas = [f'corr_PC{i+1}' for i in range(d)]
    tabela = pd.DataFrame(corr, columns=colunas)
    tabela.insert(0, 'variavel_preprocessada', list(x.columns))
    tabela['constante'] = ~valido
    tabela['relevancia_no_espaco'] = np.linalg.norm(corr, axis=1)
    return tabela


def _principais(tabela, limite):
    if not 1 <= limite <= 40:
        raise ValueError('max_setas deve estar entre 1 e 40.')
    return tabela.sort_values(['relevancia_no_espaco', 'variavel_preprocessada'],
                              ascending=[False, True], kind='stable').head(limite)


def _estilo(config, y, idx):
    estilos = config['visualizacao']['classes']
    for grupo, chave in ((0, 'legitima'), (1, 'fraude')):
        ids = idx[y[idx] == grupo]
        if len(ids):
            yield ids, estilos[chave]


def _scatter(ax, valores, y, idx, config, *, d):
    for ids, e in _estilo(config, y, idx):
        ax.scatter(*([valores[ids, j] for j in range(d)]),
                   color=e['cor'], alpha=e.get('alpha', .5), marker=e['marcador'],
                   s=e.get('tamanho', 16), label=e.get('nome', ''), rasterized=True)


def _setas_2d(ax, selecionadas, spans, *, labels=True):
    norm = .30 * float(min(spans))
    for i, (_, r) in enumerate(selecionadas.iterrows(), start=1):
        u, v = (float(r[f'corr_PC{k}']) for k in (1, 2))
        ax.annotate('', xy=(u * norm, v * norm), xytext=(0, 0),
                    arrowprops={'arrowstyle': '->', 'color': '#222222', 'lw': 1.4})
        if labels:
            ax.text(u * norm * 1.10, v * norm * 1.10,
                    f'V{i}', fontsize=8, ha='center', va='center')
    return norm


def _circulo(ax, selecionadas):
    t = np.linspace(0, 2 * np.pi, 240)
    ax.plot(np.cos(t), np.sin(t), '--', c='grey', lw=1)
    ax.axhline(0, color='grey', lw=.7); ax.axvline(0, color='grey', lw=.7)
    for i, (_, r) in enumerate(selecionadas.iterrows(), start=1):
        u, v = float(r['corr_PC1']), float(r['corr_PC2'])
        ax.annotate('', xy=(u, v), xytext=(0, 0),
                    arrowprops={'arrowstyle': '->', 'color': '#274d83', 'lw': 1.5})
        ax.text(u * 1.10, v * 1.10, f'V{i}', fontsize=8,
                ha='center', va='center')
    ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), xlabel='corr(X, PC1)', ylabel='corr(X, PC2)',
           title='Circulo de correlacoes (escala real)')
    ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.2)







def _fig_observacoes_pc1(pc1, classes, config):
    """Observacoes sobre PC1 com dispersao vertical apenas visual."""

    pc1 = np.asarray(pc1, dtype=float).ravel()
    classes = np.asarray(classes).ravel()

    if (
        len(pc1) == 0
        or len(pc1) != len(classes)
        or not np.isfinite(pc1).all()
        or not np.isin(classes, [0, 1]).all()
    ):
        raise ValueError('Dados invalidos para observacoes sobre PC1.')

    fig, ax = plt.subplots(figsize=(13.5, 5.2))

    semente = int(config['experimento']['semente'])
    rng = np.random.default_rng(semente)

    # Dispersao vertical ficticia para evitar sobreposicao.
    # Nao representa PC2.
    jitter = rng.uniform(-0.11, 0.11, len(pc1))

    estilos = config['visualizacao']['classes']

    for classe, chave in ((0, 'legitima'), (1, 'fraude')):

        mascara = classes == classe

        if not mascara.any():
            continue

        estilo = estilos[chave]

        ax.scatter(
            pc1[mascara],
            jitter[mascara],
            color=estilo['cor'],
            marker=estilo.get('marcador', 'o'),
            s=estilo.get('tamanho', 16),
            alpha=estilo.get('alpha', .5),
            label=f"{estilo.get('nome', chave)} (n={int(mascara.sum())})",
            zorder=3 if classe == 0 else 5,
            rasterized=True
        )

    # Eixo horizontal verdadeiro da componente principal.
    amplitude = max(float(np.ptp(pc1)), 0.001)
    margem = amplitude * 0.025

    ax.set_xlim(
        float(pc1.min()) - margem,
        float(pc1.max()) + margem
    )

    # O eixo vertical serve somente para espalhar os pontos.
    ax.set_ylim(-0.14, 0.14)
    ax.set_yticks([])
    ax.set_ylabel('')

    ax.set_xlabel(
        'PC1',
        fontsize=12,
        labelpad=10
    )

    ax.set_title(
        'OBSERVAÇÕES SOBRE PC1',
        fontsize=17,
        fontweight='bold',
        pad=18
    )

    ax.grid(
        axis='x',
        linestyle=':',
        alpha=.18
    )

    ax.legend(
        loc='lower left',
        fontsize=10,
        frameon=True
    )

    ax.tick_params(axis='x', labelsize=10)

    fig.subplots_adjust(
        left=.055,
        right=.98,
        top=.87,
        bottom=.17
    )

    return fig









def _fig_1d(z, y, idx, config, selecao, variancia):
    """Correlacoes de todas as variaveis preprocessadas com PC1."""

    from matplotlib.ticker import FuncFormatter

    labels = list(selecao['variavel_preprocessada'])
    valores = np.asarray(selecao['corr_PC1'], dtype=float)

    n = len(labels)

    if n == 0:
        raise ValueError('Nenhuma variavel disponivel para o grafico.')

    # Altura proporcional ao numero de variaveis
    altura = max(7.0, 1.8 + 0.40 * n)

    fig, av = plt.subplots(
        figsize=(14.0, altura)
    )

    yy = np.arange(n)

    # Linha central de correlacao zero
    av.axvline(
        0,
        color='#78818e',
        linewidth=1.0
    )

    # Setas de correlacao
    for pos, valor in zip(yy, valores):

        cor = '#138276' if valor >= 0 else '#8051a8'

        av.annotate(
            '',
            xy=(valor, pos),
            xytext=(0, pos),
            arrowprops={
                'arrowstyle': '-|>',
                'lw': 1.9,
                'color': cor,
                'mutation_scale': 12
            }
        )

        valor_exibido = (
            0.0 if abs(valor) < 0.00000005 else valor
        )

        av.text(
            valor + (.045 if valor >= 0 else -.045),
            pos,
            f'{valor_exibido:+.7f}',
            ha='left' if valor >= 0 else 'right',
            va='center',
            color=cor,
            fontsize=9
        )

    # Todas as variaveis aparecem no eixo vertical
    av.set_yticks(
        yy,
        labels=labels
    )

    # Margem acima da primeira e abaixo da ultima seta
    av.set_ylim(
        n - .85,
        -.85
    )

    # Espaco horizontal para os valores com 7 casas
    av.set_xlim(
        -1.40,
        1.40
    )

    av.set_xlabel(
        'Correlação de Pearson com PC1',
        fontsize=11,
        labelpad=12
    )

    av.set_xticks([-1, -.5, 0, .5, 1])

    def numero7(valor, _pos):
        valor = 0.0 if abs(valor) < 0.00000005 else valor
        return f'{valor:.7f}'

    av.xaxis.set_major_formatter(
        FuncFormatter(numero7)
    )

    av.tick_params(
        axis='x',
        labelsize=9
    )

    av.tick_params(
        axis='y',
        labelsize=9,
        length=0,
        pad=8
    )

    av.set_axisbelow(True)
    av.grid(axis='x', alpha=.20)

    fig.suptitle(
        'CORRELAÇÃO DAS VARIÁVEIS COM A PCA 1D',
        fontsize=17,
        fontweight='bold',
        y=.995
    )

    fig.subplots_adjust(
        left=.27,
        right=.96,
        top=1 - .66 / altura,
        bottom=min(.13, .98 / altura)
    )

    return fig





def _fig_2d(z, y, idx, config, selecionadas, variancia):
    """Biplot PCA 2D com todas as variaveis e tabela completa."""

    n = len(selecionadas)

    if n == 0:
        raise ValueError("Biplot PCA 2D sem variaveis.")

    # Distribuir todas as variaveis entre duas tabelas.
    linhas_por_coluna = (n + 1) // 2

    # Altura adaptativa ao numero de variaveis.
    altura = max(10.0, 7.0 + 0.28 * linhas_por_coluna)

    fig = plt.figure(figsize=(20.0, altura))

    grade = fig.add_gridspec(
        2, 2,
        height_ratios=[
            6.5,
            max(3.5, 0.27 * linhas_por_coluna)
        ],
        left=0.055,
        right=0.970,
        top=0.920,
        bottom=0.055,
        hspace=0.32,
        wspace=0.22
    )

    # =====================================================
    # PAINEL A — SCORES PCA E VETORES DAS VARIAVEIS
    # =====================================================

    ax = fig.add_subplot(grade[0, 0])

    _scatter(ax, z, y, idx, config, d=2)

    spans = [
        np.ptp(z[idx, j])
        for j in range(2)
    ]

    fator = _setas_2d(
        ax,
        selecionadas,
        spans
    )

    ax.set_xlabel(f"PC1 ({variancia[0]:.2%})")
    ax.set_ylabel(f"PC2 ({variancia[1]:.2%})")

    ax.set_title(
        "A — Projecao PCA e vetores das variaveis",
        fontsize=12,
        fontweight="bold",
        pad=12
    )

    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)

    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.18)

    # =====================================================
    # PAINEL B — CIRCULO DE CORRELACOES
    # =====================================================

    av = fig.add_subplot(grade[0, 1])

    _circulo(av, selecionadas)

    av.set_title(
        "B — Circulo de correlacoes PC1 x PC2",
        fontsize=12,
        fontweight="bold",
        pad=12
    )

    # =====================================================
    # TABELAS — TODAS AS VARIAVEIS
    # =====================================================

    for coluna in range(2):

        inicio = coluna * linhas_por_coluna
        fim = min(inicio + linhas_por_coluna, n)

        bloco = selecionadas.iloc[inicio:fim]

        ax_tab = fig.add_subplot(grade[1, coluna])
        ax_tab.axis("off")

        linhas = []

        for numero, (_, r) in enumerate(
            bloco.iterrows(),
            start=inicio + 1
        ):

            nome = str(r["variavel_preprocessada"])
            c1 = float(r["corr_PC1"])
            c2 = float(r["corr_PC2"])

            linhas.append([
                f"V{numero}",
                nome,
                f"{c1:+.7f}",
                f"{c2:+.7f}"
            ])

        tabela = ax_tab.table(
            cellText=linhas,
            colLabels=[
                "ID",
                "Variavel preprocessada",
                "Corr. PC1",
                "Corr. PC2"
            ],
            colWidths=[0.08, 0.52, 0.20, 0.20],
            cellLoc="left",
            colLoc="left",
            bbox=[0, 0, 1, 0.96]
        )

        tabela.auto_set_font_size(False)

        tabela.set_fontsize(
            8.0 if n <= 30 else 7.2
        )

        for (linha, coluna_celula), celula in tabela.get_celld().items():

            celula.set_edgecolor("#dce3eb")
            celula.set_linewidth(0.4)

            if linha == 0:
                celula.set_facecolor("#e7edf5")
                celula.get_text().set_weight("bold")

            elif linha % 2 == 0:
                celula.set_facecolor("#f7f9fc")

        ax_tab.set_title(
            f"Variaveis V{inicio + 1} a V{fim}",
            fontsize=10,
            fontweight="bold",
            pad=8
        )

    # =====================================================
    # TITULO GERAL
    # =====================================================

    fig.suptitle(
        "BIPLOT PCA 2D — TODAS AS VARIAVEIS",
        fontsize=16,
        fontweight="bold",
        y=0.975
    )

    return fig




def _fig_3d(z, y, idx, config, selecionadas, variancia):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(17, 8.5))
    ax = fig.add_subplot(121, projection='3d')
    av = fig.add_subplot(122, projection='3d')
    _scatter(ax, z, y, idx, config, d=3)
    factor = .27 * min(float(np.ptp(z[idx, j])) for j in range(3))
    for i, (_, r) in enumerate(selecionadas.iterrows(), start=1):
        u, v, w = [float(r[f'corr_PC{k}']) for k in (1, 2, 3)]
        ax.quiver(0, 0, 0, u * factor, v * factor, w * factor,
                  color='#262626', arrow_length_ratio=.12, lw=1.3)
        ax.text(u * factor * 1.1, v * factor * 1.1, w * factor * 1.1,
                f'V{i}', fontsize=8)
        av.quiver(0, 0, 0, u, v, w, color='#274d83', lw=1.5,
                  arrow_length_ratio=.10)
        av.text(u * 1.1, v * 1.1, w * 1.1, str(r['variavel_preprocessada']), fontsize=7)
    u = np.linspace(0, 2*np.pi, 35)
    v = np.linspace(0, np.pi, 18)
    av.plot_wireframe(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                      np.outer(np.ones_like(u), np.cos(v)), alpha=.12, linewidth=.4)
    ax.set(xlabel=f'PC1 ({variancia[0]:.2%})', ylabel=f'PC2 ({variancia[1]:.2%})',
           zlabel=f'PC3 ({variancia[2]:.2%})', title=f'Scores + setas (fator {factor:.3g})')
    ax.legend(loc='best')
    av.set(xlabel='corr(X, PC1)', ylabel='corr(X, PC2)', zlabel='corr(X, PC3)',
           xlim=(-1.05,1.05), ylim=(-1.05,1.05), zlim=(-1.05,1.05),
           title='Esfera de correlacoes (escala real)')
    itens = [f'V{i}={r["variavel_preprocessada"]}: ' +
             ','.join(f'{r[f"corr_PC{k}"]:+.2f}' for k in (1,2,3))
             for i, (_, r) in enumerate(selecionadas.iterrows(), start=1)]
    tabela = '\n'.join('   |   '.join(itens[j:j+3]) for j in range(0, len(itens), 3))
    fig.text(.50,.025,'Correlacoes (PC1, PC2, PC3): ' + tabela,
             ha='center', va='bottom', fontsize=7, wrap=True)
    fig.suptitle('Biplot PCA 3D — desenvolvimento F1–F9', fontweight='bold')
    fig.tight_layout(rect=[0,.07,1,.95]); return fig


def _mapa_cores_variaveis_pca(variaveis):
    """Define cores consistentes para as variaveis do PCA."""

    from matplotlib.colors import to_hex

    nomes = sorted(set(str(v) for v in variaveis))

    paleta = []

    for nome_paleta in ('tab20', 'tab20b', 'tab20c'):
        cmap = plt.get_cmap(nome_paleta)

        for cor in cmap.colors:
            rgb = tuple(0.82 * canal for canal in cor[:3])
            paleta.append(to_hex(rgb))

    if len(nomes) > len(paleta):
        raise ValueError(
            'Quantidade de variaveis maior que a paleta disponivel.'
        )

    return dict(zip(nomes, paleta))


def _fig_coeficientes_pc1(tabela):
    """Exibe todos os coeficientes originais da combinacao linear da PC1."""

    from matplotlib.ticker import FuncFormatter

    exigidas = {
        'variavel_preprocessada',
        'coeficiente_PC1'
    }

    if not exigidas.issubset(tabela.columns) or tabela.empty:
        raise ValueError(
            'Tabela de coeficientes PC1 incompleta ou vazia.'
        )

    nomes = tabela['variavel_preprocessada'].astype(str).tolist()
    valores = tabela['coeficiente_PC1'].to_numpy(dtype=float)

    if not np.isfinite(valores).all():
        raise ValueError('Coeficientes PC1 nao finitos.')

    n = len(valores)

    # Altura proporcional ao numero de variaveis
    altura = max(7.0, 1.8 + 0.40 * n)

    fig, ax = plt.subplots(
        figsize=(14.0, altura)
    )

    posicoes = np.arange(n)

    # Uma cor para cada variavel
    mapa_cores = _mapa_cores_variaveis_pca(nomes)

    # Referencia central
    ax.axvline(
        0,
        color='#707a89',
        linewidth=1.0
    )

    # Uma seta por variavel, com cor propria
    for posicao, (nome, valor) in enumerate(zip(nomes, valores)):

        cor = mapa_cores[nome]

        ax.annotate(
            '',
            xy=(valor, posicao),
            xytext=(0, posicao),
            arrowprops={
                'arrowstyle': '-|>',
                'color': cor,
                'lw': 2.0,
                'mutation_scale': 12
            }
        )

        valor_mostrado = (
            0.0 if abs(valor) < 0.00000005 else valor
        )

        ax.text(
            valor + (0.02 if valor >= 0 else -0.02),
            posicao,
            f'{valor_mostrado:+.7f}',
            ha='left' if valor >= 0 else 'right',
            va='center',
            fontsize=8.5,
            color=cor
        )

    # Limites proporcionais aos coeficientes
    max_abs = max(
        float(np.max(np.abs(valores))),
        0.05
    )

    limite = max_abs * 1.45

    ax.set_xlim(-limite, limite)

    # Margens verticais para nao cortar as setas
    ax.set_ylim(
        n - 0.85,
        -0.85
    )

    ax.set_yticks(
        posicoes,
        labels=nomes
    )

    ax.set_xlabel(
        'Coeficiente na combinação linear da PC1',
        fontsize=11,
        labelpad=12
    )

    ax.set_title(
        'COEFICIENTES DA COMBINAÇÃO LINEAR DA PC1',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    # Sete casas decimais no eixo horizontal
    def sete_casas(valor, _pos):
        valor = (
            0.0 if abs(valor) < 0.00000005
            else valor
        )
        return f'{valor:.7f}'

    ax.xaxis.set_major_formatter(
        FuncFormatter(sete_casas)
    )

    ax.tick_params(
        axis='x',
        labelsize=9
    )

    ax.tick_params(
        axis='y',
        labelsize=9,
        length=0,
        pad=7
    )

    ax.grid(
        axis='x',
        linestyle=':',
        alpha=.25
    )

    ax.set_axisbelow(True)

    fig.subplots_adjust(
        left=.30,
        right=.96,
        top=1 - .60 / altura,
        bottom=min(.13, .95 / altura)
    )

    return fig


def _fig_coeficientes_2d(tabela):
    """Coeficientes das combinacoes lineares PC1 e PC2."""

    from matplotlib.ticker import FuncFormatter

    exigidas = {
        "variavel_preprocessada",
        "coeficiente_PC1",
        "coeficiente_PC2"
    }

    if tabela.empty or not exigidas.issubset(tabela.columns):
        raise ValueError("Tabela de coeficientes PCA 2D invalida.")

    nomes = tabela["variavel_preprocessada"].astype(str).tolist()

    valores = tabela[
        ["coeficiente_PC1", "coeficiente_PC2"]
    ].to_numpy(dtype=float)

    if not np.isfinite(valores).all():
        raise ValueError("Coeficientes PCA 2D nao finitos.")

    n = len(nomes)
    altura = max(7.0, 2.4 + 0.43 * n)

    fig, eixos = plt.subplots(
        1, 2,
        figsize=(22, altura),
        sharey=True
    )

    posicoes = np.arange(n)
    mapa_cores = _mapa_cores_variaveis_pca(nomes)

    max_abs = max(float(np.max(np.abs(valores))), 0.05)
    limite = max_abs * 1.50
    deslocamento = max_abs * 0.06

    for j, ax in enumerate(eixos):

        componente = f"PC{j + 1}"

        ax.axvline(
            0,
            color="#707a89",
            linewidth=1.0
        )

        for i, nome in enumerate(nomes):

            valor = float(valores[i, j])
            cor = mapa_cores[nome]

            ax.annotate(
                "",
                xy=(valor, i),
                xytext=(0, i),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": cor,
                    "lw": 1.9,
                    "mutation_scale": 11
                }
            )

            valor_exibido = (
                0.0 if abs(valor) < 0.00000005 else valor
            )

            ax.text(
                valor + (
                    deslocamento if valor >= 0
                    else -deslocamento
                ),
                i,
                f"{valor_exibido:+.7f}",
                ha="left" if valor >= 0 else "right",
                va="center",
                fontsize=8,
                color=cor
            )

        ax.set_xlim(-limite, limite)
        ax.set_ylim(n - 0.85, -0.85)

        ax.set_xlabel(
            f"Coeficiente na combinacao linear da {componente}",
            fontsize=11,
            labelpad=11
        )

        ax.set_title(
            f"COEFICIENTES DA {componente}",
            fontsize=15,
            fontweight="bold",
            pad=16
        )

        ax.set_xticks(
            np.linspace(-max_abs, max_abs, 5)
        )

        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda valor, pos:
                f"{0.0 if abs(valor) < 0.00000005 else valor:.7f}"
            )
        )

        ax.grid(
            axis="x",
            linestyle=":",
            alpha=0.23
        )

        ax.set_axisbelow(True)

        ax.tick_params(
            axis="x",
            labelsize=8
        )

    # Os nomes aparecem uma vez, com linhas alinhadas
    # entre os dois paines.
    eixos[0].set_yticks(
        posicoes,
        labels=nomes
    )

    eixos[0].tick_params(
        axis="y",
        labelsize=9,
        length=0,
        pad=7
    )

    eixos[1].tick_params(
        axis="y",
        left=False,
        labelleft=False
    )

    fig.suptitle(
        "COEFICIENTES DAS COMBINACOES LINEARES / PCA 2D",
        fontsize=17,
        fontweight="bold",
        x=0.60,
        y=0.995,
        ha="center"
    )

    fig.subplots_adjust(
        left=0.23,
        right=0.97,
        top=1 - 0.75 / altura,
        bottom=min(0.12, 0.95 / altura),
        wspace=0.22
    )

    return fig








def gerar(config, data, destino, ext, dpi, csv):
    """Produz grafico correspondente a dimensionalidade PCA e CSV completo."""
    x, coord, modelo, y = _entrada(data)
    opts = config['visualizacao']['graficos']['pca_biplot']


    tab = tabela_correlacoes(data)







    if coord.shape[1] == 1:
        # PCA 1D: preservar exatamente a selecao aprovada.
        selecao = tab.sort_values(
            ['relevancia_no_espaco', 'variavel_preprocessada'],
            ascending=[False, True],
            kind='stable'
        )

    elif coord.shape[1] == 2:
        # PCA 2D: incluir todas as variaveis preprocessadas.
        selecao = tab.sort_values(
            ['relevancia_no_espaco', 'variavel_preprocessada'],
            ascending=[False, True],
            kind='stable'
        ).reset_index(drop=True)

    else:
        # PCA 3D: manter comportamento anterior.
        max_setas = int(opts.get('max_setas', 6))
        selecao = _principais(tab, max_setas)







    limite = int(config['visualizacao']['geometria'].get('pontos_maximos_amostra', 20000))
    # Amostragem apenas visual; indices das observacoes permanecem alinhados.
    from src.visualizacao_geometrica import _amostra_plot
    idx = _amostra_plot(len(y), y, config, limite)
    z = coord.to_numpy(dtype=float)
    dim = int(z.shape[1])
    variancia = modelo.explained_variance_ratio_
    figura = (_fig_1d(z, y, idx, config, selecao, variancia) if dim == 1 else
              _fig_2d(z, y, idx, config, selecao, variancia) if dim == 2 else
              _fig_3d(z, y, idx, config, selecao, variancia))
    base = f'{PREFIXO}{dim}D'
    vis._salvar(figura, Path(destino) / f'{base}.{ext}', dpi=dpi)

        # Grafico independente das observacoes sobre PC1.
    if dim == 1:

        figura_observacoes = _fig_observacoes_pc1(
            z[idx, 0],
            y[idx],
            config
        )

        nome_observacoes = 'PCA_OBSERVACOES_PC1'

        vis._salvar(
            figura_observacoes,
            Path(destino) / f'{nome_observacoes}.{ext}',
            dpi=dpi
        )

        if csv:
            tabela_observacoes = pd.DataFrame({
                'indice_linha_desenvolvimento': np.asarray(coord.index)[idx],
                'PC1': z[idx, 0],
                'classe_real': y[idx]
            })

            vis._csv_protegido(
                tabela_observacoes,
                Path(destino) / f'{nome_observacoes}.csv'
            )

        # =====================================================
    # NOVO GRAFICO: COEFICIENTES DA COMBINACAO LINEAR DA PC1
    # Exclusivo do PCA 1D
    # =====================================================

    if dim == 1:

        # Coeficientes reais do PCA ajustado
        coeficientes = np.asarray(
            modelo.components_[0],
            dtype=float
        )

        if (
            len(coeficientes) != len(x.columns)
            or not np.isfinite(coeficientes).all()
        ):
            raise ValueError(
                'Coeficientes PC1 invalidos ou desalinhados.'
            )

        # Preservar os nomes e valores originais
        tabela_coeficientes = pd.DataFrame({
            'variavel_preprocessada': list(x.columns),
            'coeficiente_PC1': coeficientes
        })

        # Magnitude absoluta apenas para ordenacao
        tabela_coeficientes['magnitude_absoluta'] = (
            tabela_coeficientes['coeficiente_PC1'].abs()
        )

        tabela_coeficientes = tabela_coeficientes.sort_values(
            ['magnitude_absoluta', 'variavel_preprocessada'],
            ascending=[False, True],
            kind='stable'
        ).reset_index(drop=True)

                # Registrar a cor de cada variavel para reutilizacao
        # nas visualizacoes PCA 2D e 3D.
        mapa_cores = _mapa_cores_variaveis_pca(x.columns)

        tabela_coeficientes['cor_hex'] = (
            tabela_coeficientes['variavel_preprocessada']
            .map(mapa_cores)
        )

        # Criar o novo grafico
        figura_coeficientes = _fig_coeficientes_pc1(
            tabela_coeficientes
        )

        nome_coeficientes = 'PCA_COEFICIENTES_PC1'

        # Salvar imagem independente
        vis._salvar(
            figura_coeficientes,
            Path(destino) / f'{nome_coeficientes}.{ext}',
            dpi=dpi
        )

        # Salvar coeficientes com precisao original
        if csv:
            vis._csv_protegido(
                tabela_coeficientes,
                Path(destino) / f'{nome_coeficientes}.csv'
            )

    # =====================================================
    # COEFICIENTES DAS COMBINACOES LINEARES — PCA 2D
    # =====================================================

    if dim == 2:

        coef = np.asarray(
            modelo.components_[:2, :],
            dtype=float
        ).T

        if (
            coef.shape != (len(x.columns), 2)
            or not np.isfinite(coef).all()
        ):
            raise ValueError(
                "Coeficientes PCA 2D invalidos ou desalinhados."
            )

        tabela_coef_2d = pd.DataFrame({
            "variavel_preprocessada": list(x.columns),
            "coeficiente_PC1": coef[:, 0],
            "coeficiente_PC2": coef[:, 1]
        })

        tabela_coef_2d["magnitude_conjunta"] = (
            np.linalg.norm(coef, axis=1)
        )

        tabela_coef_2d = tabela_coef_2d.sort_values(
            ["magnitude_conjunta", "variavel_preprocessada"],
            ascending=[False, True],
            kind="stable"
        ).reset_index(drop=True)

        mapa_cores = _mapa_cores_variaveis_pca(
            x.columns
        )

        tabela_coef_2d["cor_hex"] = (
            tabela_coef_2d["variavel_preprocessada"]
            .map(mapa_cores)
        )

        figura_coef_2d = _fig_coeficientes_2d(
            tabela_coef_2d
        )

        nome_coef_2d = "PCA_COEFICIENTES_PC1_PC2"

        vis._salvar(
            figura_coef_2d,
            Path(destino) / f"{nome_coef_2d}.{ext}",
            dpi=dpi
        )

        if csv:
            vis._csv_protegido(
                tabela_coef_2d,
                Path(destino) / f"{nome_coef_2d}.csv"
            )

    if csv:
        vis._csv_protegido(tab, Path(destino) / f'{base}_correlacoes.csv')

        coords = coord.iloc[idx].copy()
        coords['classe_real'] = y[idx]
        coords.insert(0, 'indice_linha_desenvolvimento', np.asarray(coord.index)[idx])
        vis._csv_protegido(coords, Path(destino) / f'{base}_scores_amostra.csv')
    return {'arquivo': f'{base}.{ext}', 'dimensoes': dim, 'n_obs_total': len(z),
            'n_obs_amostra': len(idx), 'n_variaveis': x.shape[1],
            'n_setas': len(selecao), 'origem': 'desenvolvimento_F1_F9', 'f10_utilizado': False}
