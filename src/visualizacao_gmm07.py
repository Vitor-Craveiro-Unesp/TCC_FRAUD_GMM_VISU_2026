"""GMM 07 — sensibilidade da fronteira aos cortes 0,50 e MCC OOF.

Diagnostico geométrico do GMM FULL reajustado exclusivamente em F1–F9.
MCC sempre calculado nas mesmas observações de avaliação (OOF na execução
real). Sem rótulos da malha e sem acesso ao F10.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Ellipse
import numpy as np
import pandas as pd

from src import visualizacao_geometrica as geo


def _rotulo_mcc(avaliacao):
    origem = avaliacao.get('origem')
    geo._exigir(origem in ('oof', 'simulado'),
                'GMM07 exige origem da avaliacao: oof ou simulado.')
    return ('MCC OOF' if origem == 'oof' else 'MCC da simulação',
            ('Métricas calculadas nas predições OOF, não na malha.'
             if origem == 'oof' else
             'PRÉVIA SINTÉTICA: MCC ilustrativo, NÃO é OOF real.'))


def _definir_planos(config, data, tau, *, resolucao_1d, resolucao_2d, margem):
    """Um único campo de risco por plano, reutilizado nos dois limiares."""
    X = data['coord'].to_numpy(dtype=float)
    gm, rh = data['gmm'], data['rh']
    dimensoes = X.shape[1]
    geo._exigir(dimensoes in (1, 2, 3), 'GMM07 permite 1D, 2D ou 3D.')
    geo._exigir(gm.modelo_.covariance_type == 'full',
                'GMM07 exige o GMM FULL do catalogo.')
    geo._exigir(20 <= resolucao_1d <= 20000, 'Resolucao 1D fora do intervalo.')
    geo._exigir(20 <= resolucao_2d <= 450, 'Resolucao 2D fora do intervalo.')
    geo._exigir(0 <= margem <= 1, 'Margem do GMM07 invalida.')
    pares = geo._pares(dimensoes, config)
    if dimensoes == 3:
        geo._exigir(set(pares) == {(0, 1), (0, 2), (1, 2)},
                    'GMM07 3D exige os três planos 1-2, 1-3 e 2-3.')
    medias_fixas = X.mean(axis=0)
    planos = []
    for par in pares:
        if len(par) == 1:
            eixo = np.linspace(*geo._faixa(X[:, par[0]], margem), resolucao_1d)
            pontos = np.tile(medias_fixas, (len(eixo), 1))
            pontos[:, par[0]] = eixo
            grade = (eixo,)
            campo = geo._pontuar_gmm_array(gm, rh, pontos)
        else:
            a = np.linspace(*geo._faixa(X[:, par[0]], margem), resolucao_2d)
            b = np.linspace(*geo._faixa(X[:, par[1]], margem), resolucao_2d)
            xx, yy = np.meshgrid(a, b)
            pontos = np.tile(medias_fixas, (xx.size, 1))
            pontos[:, par[0]] = xx.ravel()
            pontos[:, par[1]] = yy.ravel()
            grade = (xx, yy)
            campo = geo._pontuar_gmm_array(gm, rh, pontos).reshape(xx.shape)
        fixo, otimo, mudou = geo._mudanca_classes(campo, tau)
        planos.append((par, grade, campo, fixo, otimo, mudou))
    return planos, medias_fixas


def _linhas_fronteira_1d(ax, eixo, campo, tau):
    """No caso 1D, as fronteiras são coordenadas do eixo, não linhas y=τ."""
    if float(np.min(campo)) > tau or float(np.max(campo)) < tau:
        return False
    d = np.asarray(campo) - tau
    raizes = []
    for i in range(len(eixo) - 1):
        if d[i] == 0:
            raizes.append(float(eixo[i]))
        if d[i] * d[i + 1] < 0:
            raizes.append(float(eixo[i] + (eixo[i+1] - eixo[i]) *
                                  (-d[i]) / (d[i+1]-d[i])))
    if d[-1] == 0:
        raizes.append(float(eixo[-1]))
    for x in sorted(set(raizes)):
        ax.axvline(x, color='#181818', lw=2.1, zorder=8)
    return len(raizes) > 0


def _grafico(config, data, escolha, planos, medias_fixas, *, destino, ext, dpi,
             mcc_fixo, mcc_otimo, avaliacao):
    X = data['coord'].to_numpy(dtype=float)
    y = np.asarray(data['y'], dtype=np.int8)
    dimensoes = X.shape[1]
    nomes = list(data['coord'].columns)
    tau_star = float(escolha['limiar_mcc_oof'])
    nome_mcc, nota_mcc = _rotulo_mcc(avaliacao)
    idx = geo._amostra_plot(len(y), y, config,
                           int(config['visualizacao']['graficos']['sensibilidade_fronteira'].get('n_pontos_scatter', 1500)))
    minimo = min(float(np.min(plano[2])) for plano in planos)
    maximo = max(float(np.max(plano[2])) for plano in planos)
    maximo_cor = max(maximo, 1e-8)
    if dimensoes == 3:
        fig, eixos = plt.subplots(2, 3, figsize=(19.8, 12.5), squeeze=False)
    else:
        fig, eixos = plt.subplots(1, 2, figsize=(16.1, 7.7), squeeze=False,
                                 sharex=(dimensoes == 1), sharey=(dimensoes == 1))
    fig.patch.set_facecolor('white')



    titulo = 'SENSIBILIDADE DA FRONTEIRA AO LIMIAR'
    fig.suptitle(titulo, fontsize=17, fontweight='bold', y=.988)

    subtitulo = (
    f'{nomes[0]} | FULL | {dimensoes}D'
    if dimensoes == 1
    else f'FULL | {dimensoes}D'
    )

    fig.text(
    .5, .920,
    subtitulo,
    ha='center',
    fontsize=11,
    color='#5a6c84'
    )



    cores = config['visualizacao']['classes']
    hachura_amarela = '#edbe31'
    quad = None



    for indice_corte, (limiar, mcc) in enumerate(((.5, mcc_fixo), (tau_star, mcc_otimo))):

        simbolo_tau = 'τ' if indice_corte == 0 else 'τ*'

        for indice_plano, (par, grade, campo, pred_fixo, pred_otimo, mudou) in enumerate(planos):
            ax = eixos[indice_corte, indice_plano] if dimensoes == 3 else eixos[0, indice_corte]
            if dimensoes == 1:
                x = grade[0]
                # Mesma intensidade do campo nos dois painéis: fundo informativo.
                ax.imshow(np.repeat(campo[np.newaxis, :], 2, axis=0),
                          extent=(x[0], x[-1], 0, 1), origin='lower',
                          aspect='auto', cmap='viridis', vmin=0, vmax=maximo_cor,
                          alpha=.13, zorder=0, transform=ax.get_xaxis_transform())
                ax.plot(x, campo, lw=2.7, color='#2361a1', zorder=4)
                for classe in (0, 1):
                    amostras = idx[y[idx] == classe]
                    if len(amostras):
                        opcao = cores['fraude' if classe else 'legitima']
                        ax.scatter(X[amostras, par[0]], np.asarray(data['scores'])[amostras],
                                   s=8 if classe == 0 else 16, c=opcao['cor'],
                                   alpha=.18 if classe == 0 else .27, zorder=3,
                                   rasterized=True)
                if indice_corte == 1 and mudou.any():
                    limites = np.flatnonzero(np.diff(np.r_[False, mudou, False]))
                    for inicio, fim in limites.reshape(-1, 2):
                        direita = x[min(fim, len(x)-1)]
                        ax.axvspan(x[inicio], direita, facecolor=hachura_amarela,
                                   alpha=.23, hatch='////', edgecolor='#b58b04', zorder=2)
                cruzou = _linhas_fronteira_1d(ax, x, campo, limiar)
                ax.set(xlim=(x[0], x[-1]), ylim=(0, min(1, max(.002, maximo*1.22,
                                                             tau_star*1.12))), xlabel=nomes[par[0]])
                if indice_corte == 0:
                    ax.set_ylabel('Probabilidade estimada de fraude')
            else:
                xx, yy = grade
                quad = ax.pcolormesh(xx, yy, campo, cmap='viridis', shading='auto',
                                     vmin=0, vmax=maximo_cor, rasterized=True, zorder=0)
                for classe in (0,1):
                    amostras = idx[y[idx] == classe]
                    if len(amostras):
                        opcao = cores['fraude' if classe else 'legitima']
                        ax.scatter(X[amostras,par[0]],X[amostras,par[1]],
                                   color=opcao['cor'], marker='*' if classe == 1 else 'o',
                                   s=22 if classe == 1 else 8, alpha=.52 if classe == 1 else .24,
                                   rasterized=True, zorder=2)


                        
                if (dimensoes == 2 or indice_corte == 1) and mudou.any():
                    # Mesma zona de mudanca nos dois paineis 2D.
                    cs = ax.contourf(
                        xx, yy, mudou.astype(float),
                        levels=[.5, 1.5],
                        colors='none',
                        hatches=['////'],
                        zorder=3
                    )


                    
                    cs.set_edgecolor(hachura_amarela)
                    cs.set_linewidth(.15)
                cruzou = float(np.min(campo)) <= limiar <= float(np.max(campo))




                if cruzou:
                    con = ax.contour(
                        xx, yy, campo,
                        levels=[limiar],
                        colors=['#101010'],
                        linewidths=2,
                        zorder=5
                    )
                    cruzou = any(
                        len(segmento) for segmento in con.allsegs[0]
                    )

                # Elipses FULL de concentracao 95% somente no PCA 2D.
                if dimensoes == 2:
                    from scipy.stats import chi2

                    raio_95 = float(
                        np.sqrt(chi2.ppf(.95, df=2))
                    )

                    gm = data['gmm']

                    for k in range(len(gm.saidas_)):
                        cov = np.asarray(
                            geo._covariancia(gm, k)[:2, :2],
                            dtype=float
                        )

                        eig, V = np.linalg.eigh(cov)

                        geo._exigir(
                            np.isfinite(eig).all() and np.all(eig > 0),
                            'Covariancia invalida nas elipses GMM07.'
                        )

                        angulo = np.degrees(
                            np.arctan2(V[1, 1], V[0, 1])
                        )

                        ax.add_patch(
                            Ellipse(
                                xy=gm.modelo_.means_[k, :2],
                                width=2 * raio_95 * np.sqrt(eig[1]),
                                height=2 * raio_95 * np.sqrt(eig[0]),
                                angle=angulo,
                                fill=False,
                                linestyle=(0, (5, 3)),
                                linewidth=1.7,
                                edgecolor=plt.get_cmap('tab10')(k % 10),
                                alpha=.95,
                                zorder=6
                            )
                        )

                ax.set(
                    xlim=(xx[0, 0], xx[0, -1]),
                    ylim=(yy[0, 0], yy[-1, 0]),
                    xlabel=nomes[par[0]],
                    ylabel=nomes[par[1]]
                )



            if not cruzou:



                ax.text(
                 .02, .03,
                f'Sem cruzamento com {simbolo_tau} = {limiar:.7f}',
                ha='left',
                va='bottom',
                transform=ax.transAxes,
                fontsize=8.4,
                bbox={'facecolor':'white', 'edgecolor':'#ccd3dc', 'alpha':.91},
                zorder=9)


            if dimensoes == 3:
                restante = next(i for i in range(3) if i not in par)
                titulo_plano = (f'{nomes[par[0]]} × {nomes[par[1]]}\n'
                                f'{nomes[restante]} = {medias_fixas[restante]:.3g} '
                                '(média desenvolvimento)')
                ax.set_title(titulo_plano, fontsize=9.5, pad=8)
                ax.text(.02,.96, f'τ = {limiar:.6g}', transform=ax.transAxes, va='top',
                        fontsize=8.5, color='#202e3f', fontweight='bold',
                        bbox={'facecolor':'white','alpha':.78,'edgecolor':'none'})
            else:


                titulo_corte = (
                'CORTE FIXO'
                 if indice_corte == 0
                else 'LIMIAR ÓTIMO DO MCC'
                )

                simbolo_tau = 'τ' if indice_corte == 0 else 'τ*'

                ax.set_title(
                f'{titulo_corte} | {simbolo_tau} = {limiar:.7f}\n{nome_mcc} = {mcc:.7f}',
                fontsize=11.7,
                fontweight='medium',
                pad=10
                )

            ax.grid(alpha=.08)
    if dimensoes == 3:
        fig.text(.026,.77, f'A — Corte fixo 0,50\n{nome_mcc}={mcc_fixo:.5f}',
                 rotation=90, va='center', ha='left',fontsize=10.5,fontweight='bold')
        fig.text(.026,.35, f'B — Corte ótimo MCC (τ*={tau_star:.6g})\n{nome_mcc}={mcc_otimo:.5f}',
                 rotation=90, va='center', ha='left',fontsize=10.5,fontweight='bold')



        fig.subplots_adjust(
        left=.075,
        right=.88 if dimensoes == 2 else .985,
        top=.75,
        bottom=.18,
        wspace=.22
        )

        cax=fig.add_axes([.91,.22,.012,.59])
        fig.colorbar(quad,cax=cax,label='Score conjunto s(x) — mesma escala')



    else:
        fig.subplots_adjust(
            left=.075,
            right=.88 if dimensoes == 2 else .985,
            top=.76,
            bottom=.12 if dimensoes == 2 else .19,
            wspace=.22
        )

        if dimensoes == 2:
            cax = fig.add_axes([.905, .12, .012, .64])

            fig.colorbar(
                quad,
                cax=cax,
                label='Score conjunto s(x) — mesma escala'
            )



    legenda = [
        Line2D([0], [0], color='#101010', lw=2, label='Fronteira (s(x) = τ)'),
        Patch(facecolor='#edbe31', edgecolor='#b58b04', hatch='////',
             label='Região de mudança'),
        Line2D([0], [0], color='#2361a1', lw=2,
               label='Risco contínuo' if dimensoes == 1 else 'Score conjunto'),
        Line2D([0], [0], marker='o', linestyle='', color=cores['legitima']['cor'],
               markersize=5,
            label='Legítimas (projeção)' if dimensoes == 3 else 'Legítimas'),
        Line2D([0], [0], marker='*', linestyle='', color=cores['fraude']['cor'],
               markersize=7,
            label='Fraudes (projeção)' if dimensoes == 3 else 'Fraudes')]






    if dimensoes == 1:
        legenda = legenda[:3]
    else:
        legenda.pop(2)

    if dimensoes == 2:
        legenda.append(
            Line2D(
                [0], [0],
                color='#7b4fa8',
                lw=1.7,
                ls=(0, (5, 3)),
                label='Elipses gaussianas 95%'
            )
        )


    fig.legend(
    handles=legenda,
    loc='upper center',
    bbox_to_anchor=(.5, .875),
    ncol=len(legenda),
    frameon=False,
    fontsize=9.5
)




    # Preservar o rodape original apenas para PCA 1D e PCA 3D.
    # PCA 2D ja exibe o MCC acima de cada painel.

    if dimensoes != 2:

        direcao = (
            'legítima → fraude' if tau_star < .5 else
            'fraude → legítima' if tau_star > .5 else
            'nenhuma mudança'
        )

        fig.text(
            .5, .090,
            f'{nome_mcc}(τ=0.5000000) = {mcc_fixo:.7f}   |   '
            f'{nome_mcc}(τ*={tau_star:.7f}) = {mcc_otimo:.7f}   |   '
            f'Mudança: {direcao}',
            ha='center',
            fontsize=10.2,
            color='#1b3048',
            fontweight='bold'
        )






    nome = f'GMM07_sensibilidade_{dimensoes}d_comparativo'
    geo._guardar(fig,destino,nome,ext,dpi)
    return nome


def gerar(config, data, escolha, destino, ext, dpi, csv, *, avaliacao):
    """Renderiza exatamente uma figura comparativa com dois cortes por dimensão."""
    op = config['visualizacao']['graficos']['sensibilidade_fronteira']
    tau = float(escolha['limiar_mcc_oof'])
    _rotulo_mcc(avaliacao)  # Impedir prévias que afirmem falsamente ser OOF.
    mcc_fixo, mcc_otimo = geo._avaliar_cortes_oof(avaliacao,tau)
    planos, medias = _definir_planos(config,data,tau,
        resolucao_1d=int(op.get('resolucao_1d',500)),
        resolucao_2d=int(op.get('resolucao_2d',180)),
        margem=float(op.get('margem',.15)))
    destino=Path(destino)
    destino.mkdir(parents=True,exist_ok=True)
    nome = _grafico(config,data,escolha,planos,medias,destino=destino,ext=ext,dpi=dpi,
                    mcc_fixo=mcc_fixo,mcc_otimo=mcc_otimo,avaliacao=avaliacao)
    if csv:
        cols=list(data['coord'].columns)
        for par,grade,campo,fixo,otimo,mudou in planos:
            tabela={'score_risco':campo.ravel(),
                    'pred_fixo':fixo.ravel().astype(np.uint8),
                    'pred_otimo':otimo.ravel().astype(np.uint8),
                    'mudou':mudou.ravel().astype(np.uint8)}
            if len(par)==1:
                tabela[cols[par[0]]]=grade[0]
            else:
                for eixo,dim in enumerate(par):tabela[cols[dim]]=grade[eixo].ravel()
            geo._salvar_dados(pd.DataFrame(tabela),destino,
                f'GMM07_malha_{"_".join(cols[i] for i in par)}',True)
        geo._salvar_dados(pd.DataFrame([
            {'corte':'fixo_050','limiar':.5,'mcc':mcc_fixo,'origem_mcc':avaliacao['origem']},
            {'corte':'otimo_mcc','limiar':tau,'mcc':mcc_otimo,'origem_mcc':avaliacao['origem']}
        ]),destino,'GMM07_metricas_comparativas',True)
    return {'arquivo':nome+'.'+ext,'mcc_fixo':mcc_fixo,'mcc_otimo':mcc_otimo,
            'origem_mcc':avaliacao['origem'], 'n_planos':len(planos)}
