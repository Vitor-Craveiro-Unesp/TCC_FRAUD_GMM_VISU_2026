"""GMM18 - historico real do algoritmo EM para todas as inicializacoes.

Usa exclusivamente o GMM final ajustado no desenvolvimento (F1..F9).
Nao realiza novos ajustes, nao acessa F10 ou dados de validacao OOF.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from src import visualizacao_oof as vis


NOME = 'GMM18_convergencia_em_inicializacoes'


def gerar(config, data, destino, ext, dpi, csv):
    gmm = data['gmm']
    modelo = gmm.modelo_
    if not hasattr(modelo, 'validar_historicos_em'):
        raise RuntimeError('O vencedor nao foi ajustado com registro do historico EM.')
    melhor = modelo.validar_historicos_em()
    historicos = modelo.historicos_em_
    n_amostras = int(gmm.n_ajuste_)
    registros = []
    for h in historicos:
        anterior = None
        for i, lb in enumerate(h['lower_bounds'], start=1):
            registros.append({
                'inicializacao': int(h['init']),
                'iteracao': i,
                'lower_bound_medio_por_amostra': float(lb),
                'log_verossimilhanca_total_equivalente': float(lb) * n_amostras,
                'variacao_vs_iteracao_anterior': np.nan if anterior is None else float(lb-anterior),
                'inicializacao_selecionada': bool(h['init'] == melhor),
                'convergiu': bool(h['convergiu']),
                'n_treino_gmm': n_amostras,
            })
            anterior = lb
    tabela = pd.DataFrame(registros)
    if tabela.empty:
        raise RuntimeError('Historico EM vazio.')

    fig, ax = plt.subplots(figsize=(12.5, 7.2), facecolor='white')
    fig.subplots_adjust(left=.135, right=.96, top=.78, bottom=.175)
    cores = ['#B12A90', '#D55E00', '#0057B8', '#009E73', '#7F3C8D']
    for h in historicos:
        indice = int(h['init'])
        valores = np.asarray(h['lower_bounds'], dtype=float)
        xs = np.arange(1, len(valores) + 1)
        selecionada = indice == melhor
        cor = cores[(indice - 1) % len(cores)]
        rotulo = f'Init {indice}'
        if selecionada:
            rotulo += ' — selecionada'
        if not h['convergiu']:
            rotulo += ' (não convergiu)'
        ax.plot(xs, valores, color=cor, ls='-' if selecionada else ('-.' if indice % 2 else '--'),
                linewidth=3.2 if selecionada else 2.2,
                alpha=1.0, label=rotulo,
                zorder=4 if selecionada else 2)



        # Marcadores distintos para revelar curvas sobrepostas.
        if not selecionada:
            marcador = 'x' if indice % 2 == 0 else '+'

            ax.scatter(
                xs,
                valores,
                marker=marcador,
                s=42,
                color=cor,
                linewidths=1.5,
                zorder=10
            )

        # Destacar o ponto final de cada inicializacao.
        ax.scatter(
            [xs[-1]],
            [valores[-1]],
            s=70 if selecionada else 35,
            color=cor,
            edgecolors='white',
            linewidths=1.0,
            zorder=6 if selecionada else 3
        )

    ax.set_xlabel('Iteração do algoritmo EM', fontsize=11)
    ax.set_ylabel('Log-verossimilhança média (lower bound)', fontsize=10.8)
    ax.set_xlim(left=1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.7f}'))
    ax.grid(axis='both', color='#dbe3ed', alpha=.7, lw=.8)
    ax.set_axisbelow(True)
    for lateral in ('top','right'):
        ax.spines[lateral].set_visible(False)

    fig.suptitle('CONVERGÊNCIA DO ALGORITMO EM', fontsize=16,
                 fontweight='bold', y=.968, color='#20324a')
    dim = len(gmm.colunas_)



    fig.text(.5, .895,
             f'{dim}D | {modelo.covariance_type.upper()} | K={modelo.n_components} | '
             f'{len(historicos)} inicializações',
             ha='center', fontsize=10.5, color='#61728a')


    fig.legend(*ax.get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .864), ncol=min(3, len(historicos)),
               frameon=False, fontsize=9.5)



    fig.text(.135, .085,
             f'Melhor inicialização: {melhor}  |  '
             f'Lower bound final = {float(modelo.lower_bound_):.7f}  |  '
             f'Iterações = {int(modelo.n_iter_)}',
             fontsize=9.8, color='#26374d', weight='bold')



    destino = Path(destino)
    vis._salvar(fig, destino / (NOME + '.' + ext), dpi=dpi)
    if csv:
        vis._csv_protegido(tabela, destino / (NOME + '.csv'))
    print(f'[GMM18] {len(historicos)} inits; melhor={melhor}; '
          f'lower_bound={modelo.lower_bound_:.7f}; sem novos ajustes.')
    return tabela
