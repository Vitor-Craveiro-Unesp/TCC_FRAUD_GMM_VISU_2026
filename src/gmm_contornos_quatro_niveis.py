# -*- coding: utf-8 -*-
"""GMM07/GMM08: QUATRO niveis diagnosticos sobre UMA superficie por dominio.

Contrato cientifico:
* Na geometria OOF: campo do GMM vencedor reajustado em F1-F9,
  e MCC/limiar calculados nas predicoes OOF do vencedor.
* Em F10: campo do MESMO modelo F1-F9 congelado, mas malha delimitada
  pelas coordenadas do proprio F10; MCC F10, tau* SEM nova otimizacao.
* Linhas 0.50/tau* sao desenhadas apenas onde existe intersecao verdadeira.
* O maximo da malha e marcado como ponto (extremo nao e curva generica).
* O ponto medio (min+max)/2 e nivel exclusivamente diagnostico.
* Todos os scores e rotulos sao usados nas metricas; apenas scatter legitimo
  e subamostrado. As fraudes REAIS sao todas desenhadas.
Nao faz fit, nao modifica GMM/PCA/t-SNE, nao escolhe candidato nem limiar.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Ellipse
from matplotlib import cm
from sklearn.metrics import matthews_corrcoef

VERSION = 'GMM_QUATRO_NIVEIS_F1F9_F10_V1'
COLORS = {'fixo': '#20242b', 'oof': '#d73535', 'medio': '#00b8c7', 'maximo': '#ef931e'}


def _validate(data, escolha):
    X = data['coord'].to_numpy(dtype=float)
    y = np.asarray(data['y']).ravel()
    gm = data['gmm'].modelo_
    riscos = np.asarray(data['rh'].riscos_, dtype=float).ravel()
    tau = float(escolha['limiar_mcc_oof'])
    if X.ndim != 2 or X.shape[1] not in (1, 2, 3):
        raise ValueError('GMM07/GMM08 exige 1D, 2D ou 3D.')
    if X.shape[0] != len(y) or not X.size or not np.isfinite(X).all():
        raise ValueError('Coordenadas invalidas ou desalinhadas.')
    if not np.isin(y, (0, 1)).all():
        raise ValueError('Rotulos precisam ser 0/1.')
    if gm.n_components != len(riscos) or not np.isfinite(riscos).all():
        raise ValueError('Riscos gaussianos e GMM incompatíveis.')
    if not (np.all((riscos >= 0) & (riscos <= 1)) and 0 <= tau <= 1):
        raise ValueError('Riscos ou tau* fora de [0,1].')
    if gm.covariance_type != 'full':
        raise ValueError('Esta visualizacao exige covariancia FULL.')
    return X, y, gm, riscos, tau


def _pontuar(gm, riscos, pontos):
    pontos = np.asarray(pontos, dtype=float)
    if len(pontos) == 0:
        raise ValueError('Malha vazia.')
    # GMM full do proprio ajuste. Sem rotulos, sem fit ou transformacoes extras.
    valores = gm.predict_proba(pontos) @ riscos
    if not np.isfinite(valores).all() or (valores < -1e-8).any() or (valores > 1+1e-8).any():
        raise ValueError('Scores da malha invalidos.')
    return np.clip(valores, 0, 1)


def _faixa(q, margem):
    # F10_BOUNDARY_CERTIFICATE_V2: cobrir TODOS os pontos F10, nao percentis.
    # Inclui extremos observados e uma margem em cada direcao.
    a, b = float(np.min(q)), float(np.max(q))
    amplitude = float(b - a)
    if amplitude <= 1e-12:
        amplitude = max(abs(float(a)), 1.)
    return float(a - amplitude*margem), float(b + amplitude*margem)


def _eixo_com_ancora(valores, n, margem, referencia):
    eixo = np.linspace(*_faixa(valores, margem), n)
    if referencia is not None:
        extras = np.atleast_1d(np.asarray(referencia, dtype=float))
        extras = extras[(extras >= eixo[0]) & (extras <= eixo[-1])]
        # Acrescenta nos exatos para manter extremos e certificados, sem perder
        # resolucao da malha regular nem alterar a ordem monotona dos eixos.
        eixo = np.unique(np.concatenate((eixo, extras)))
    return eixo


def _planos(X, gm, riscos, d, n, margem, *, ancora=None, referencias=None):
    # Com F10+ e F10- o ponto de fronteira e obtido por bissecao no MESMO GMM.
    # Os tres planos em 3D passam por este ponto, nao por uma media arbitraria.
    medias = X.mean(axis=0) if ancora is None else np.asarray(ancora, dtype=float).ravel()
    pares = [(0,)] if d == 1 else ([(0, 1)] if d == 2 else [(0, 1), (0, 2), (1, 2)])
    planos = []
    for par in pares:
        def refs(j):
            pontos = [] if ancora is None else [medias[j]]
            if referencias is not None:
                pontos.extend(float(v[j]) for v in referencias)
            return pontos if pontos else None
        if len(par) == 1:
            xx = _eixo_com_ancora(X[:, par[0]], n, margem, refs(par[0]))
            pts = np.tile(medias, (len(xx), 1))
            pts[:, par[0]] = xx
            risco = _pontuar(gm, riscos, pts)
            grade = (xx,)
        else:
            e0 = _eixo_com_ancora(X[:, par[0]], n, margem, refs(par[0]))
            e1 = _eixo_com_ancora(X[:, par[1]], n, margem, refs(par[1]))
            xx, yy = np.meshgrid(e0, e1)
            pts = np.tile(medias, (xx.size, 1))
            pts[:, par[0]], pts[:, par[1]] = xx.ravel(), yy.ravel()
            risco = _pontuar(gm, riscos, pts).reshape(xx.shape)
            grade = (xx, yy)
        planos.append((par, grade, risco))
    return planos, medias


def _fraudes_todas_legitimas_amostradas(y, max_legit, semente):
    fraudes = np.flatnonzero(y == 1)
    legit = np.flatnonzero(y == 0)
    if max_legit < len(legit):
        legit = np.sort(np.random.default_rng(semente).choice(legit, size=max_legit, replace=False))
    return legit, fraudes


def _fonte(destino, avaliacao, y, data, tau):
    # A pasta F10 e a fonte confiavel quando legados repassam origem='oof' no renderizador.
    caminho = str(destino).replace('\\', '/').lower()
    f10 = '/teste_final/' in caminho or caminho.endswith('/teste_final')
    if f10:
        scores = np.asarray(data['scores'], dtype=float).ravel()
        if len(scores) != len(y) or not np.isfinite(scores).all():
            raise ValueError('Scores F10 fora de ordem.')
        return 'F10', (float(matthews_corrcoef(y, scores >= .5)),
                       float(matthews_corrcoef(y, scores >= tau))), 'MCC F10'
    origem = (avaliacao or {}).get('origem')
    if origem == 'simulado':
        o = 'SIMULACAO'
    else:
        o = 'OOF'
    if 'y' in (avaliacao or {}) and 'p' in (avaliacao or {}):
        yval = np.asarray(avaliacao['y']).ravel()
        pval = np.asarray(avaliacao['p'], dtype=float).ravel()
        if yval.shape != pval.shape or not np.isfinite(pval).all():
            raise ValueError('Avaliacao OOF desalinhada.')
        mc = (float(matthews_corrcoef(yval, pval >= .5)),
              float(matthews_corrcoef(yval, pval >= tau)))
        return o, mc, 'MCC OOF' if o == 'OOF' else 'MCC simulado'
    # Respeita a interface historica que pode estar temporariamente monkeypatched
    # no finalizador F10. Sem tentar inferir resultados OOF pelo modelo final.
    from src import visualizacao_geometrica as geo
    if avaliacao is not None:
        mc = geo._avaliar_cortes_oof(avaliacao, tau)
        return o, tuple(float(v) for v in mc), 'MCC OOF' if o == 'OOF' else 'MCC simulado'
    return o, (float('nan'), float('nan')), 'MCC indisponivel'


def _nome(grafico, d):
    return (f'GMM07_sensibilidade_{d}d_comparativo' if grafico == 'GMM07'
            else f'GMM08_mapa_risco_{d}d_comparativo')


def _status_minmax(scores, tau):
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if tau > hi:
        return 'ausente: limiar acima do maximo'
    if tau < lo:
        return 'ausente: limiar abaixo do minimo'
    if np.isclose(lo, hi, rtol=0, atol=1e-13):
        return 'campo constante (sem curva)'
    return 'possivel cruzamento'


def _rotulos_planos(par, nomes, medias):
    if len(par) == 1:
        return nomes[0]
    titulo = f'{nomes[par[0]]} × {nomes[par[1]]}'
    if len(nomes) == 3:
        z = next(i for i in range(3) if i not in par)
        titulo += f'\n{nomes[z]} = {medias[z]:.4g} (valor fixado)'
    return titulo


def _elin_2d(ax, gm, par, fixos):
    # Seção da elipsoide FULL 95% em 3D (não simples projeção marginal).
    from scipy.stats import chi2
    k_n = gm.n_components
    for k in range(k_n):
        mu = np.asarray(gm.means_[k], dtype=float)
        cov = np.asarray(gm.covariances_[k], dtype=float)
        if len(par) == 2 and cov.shape == (2, 2):
            S = cov
            center = mu
            lim = float(chi2.ppf(.95, 2))
        elif len(par) == 2 and cov.shape == (3, 3):
            rest = next(i for i in range(3) if i not in par)
            # secao exata (Schur complement) no plano fixado
            S = cov[np.ix_(par, par)] - (
                cov[np.ix_(par, [rest])] @ cov[np.ix_([rest], par)] / cov[rest, rest])
            delta = float(fixos[rest] - mu[rest])
            center = mu[list(par)] + cov[np.ix_(par, [rest])].ravel() * delta / cov[rest, rest]
            lim = float(chi2.ppf(.95, 3) - delta*delta / cov[rest, rest])
            if lim <= 0:
                continue
        else:
            continue
        eigen, vec = np.linalg.eigh(S)
        if np.any(eigen <= 0):
            continue
        width, height = 2*np.sqrt(lim*eigen[1]), 2*np.sqrt(lim*eigen[0])
        theta = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
        ax.add_patch(Ellipse(center, width, height, angle=theta, ls='--', lw=1.2,
                             fc='none', ec=plt.get_cmap('tab10')(k % 10), alpha=.78, zorder=6))


def _intervalos_gaussianos_95_1d(ax, gm, xx):
    """GMM08 1D: equivalentes unidimensionais às elipses FULL 95% do 2D.

    Faixas centrais de CADA gaussiana, não um intervalo de confiança
    estimado dos dados nem um intervalo 95% da mistura inteira.
    """
    # GMM_1D_CLEAN_GAUSS95_V3
    from scipy.stats import norm

    if gm.covariance_type != 'full' or gm.means_.shape[1] != 1:
        raise ValueError('Intervalos gaussianos 1D exigem GMM de covariância FULL em 1D.')

    z = float(norm.ppf(.975))
    handles = []
    for k in range(gm.n_components):
        media = float(gm.means_[k, 0])
        variancia = float(gm.covariances_[k, 0, 0])
        if not (np.isfinite(media) and np.isfinite(variancia) and variancia > 0):
            raise ValueError(f'Parâmetros inválidos na gaussiana G{k+1}.')
        delta = z * np.sqrt(variancia)
        esquerda, direita = media - delta, media + delta
        cor = plt.get_cmap('tab10')(k % 10)
        ax.axvspan(esquerda, direita, fc=cor, ec='none', alpha=.10, zorder=.3)
        ax.axvline(esquerda, color=cor, ls='--', lw=1.05, alpha=.8, zorder=1.3)
        ax.axvline(direita, color=cor, ls='--', lw=1.05, alpha=.8, zorder=1.3)
        handles.append(Line2D([0], [0], color=cor, ls='--', lw=2,
                              label=f'G{k+1} 95%: [{esquerda:.2f}, {direita:.2f}]'))

    # A adição das gaussianas não deve estender o domínio visual além
    # da malha F10 congelada; regiões externas são simplesmente recortadas.
    ax.set_xlim(float(xx[0]), float(xx[-1]))
    ax.legend(handles=handles, loc='upper right',
              title='Gaussianas — intervalo central 95%',
              fontsize=8.0, title_fontsize=8.2, framealpha=.94)


def _painel_png(config, data, escolha, destino, ext, dpi, grafico, origem, mcc, mccnome,
                planos, medias, lo, hi, *, substituir_staging=False, avaliacao_diagnostico=None, certificado=None):
    X, y, gm, riscos, tau = _validate(data, escolha)
    nomes = list(data['coord'].columns)
    d = X.shape[1]
    op = config.get('visualizacao', {}).get('graficos', {}).get(
        'sensibilidade_fronteira' if grafico == 'GMM07' else 'mapa_risco_fronteira', {})
    max_legit = int(op.get('n_pontos_scatter', 1500 if grafico == 'GMM07' else 5000))
    seed = int(config.get('experimento', {}).get('semente', 42))
    il, iff = _fraudes_todas_legitimas_amostradas(y, max_legit, seed)
    mid = (hi + lo) / 2
    niveis = [('fixo', .5, 'τ = 0,50'), ('oof', tau, 'τ* OOF'),
              ('medio', mid, 'ponto médio'), ('maximo', hi, 'máximo da malha')]
    ncols = len(planos)
    fig, axs = plt.subplots(1, ncols,
        figsize=((14, 8) if d==1 else ((12, 8.3) if d==2 else (19.8, 8.8))),
        squeeze=False)
    axs = axs[0]
    norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-8)
    estados = []
    for ax, (par, grade, risco) in zip(axs, planos):
        local_lo, local_hi = float(np.min(risco)), float(np.max(risco))
        if d == 1:
            xx = grade[0]
            ax.plot(xx, risco, lw=2.7, color='#2868a7', zorder=4)
            ax.scatter(X[il, 0], np.asarray(data['scores'])[il], color='#2678bd',
                       s=5, alpha=.28, rasterized=True, zorder=3)
            ax.scatter(X[iff, 0], np.asarray(data['scores'])[iff], color='#da3131',
                       marker='*', s=24, alpha=.8, rasterized=True, zorder=5)
            for key, level, title in niveis:
                if key == 'maximo':
                    j = int(np.argmax(risco))
                    ax.scatter([xx[j]], [risco[j]], marker='D', s=80,
                               color=COLORS[key], edgecolors='white', linewidths=.7, zorder=9)
                    estados.append((title, 'maximo marcado como ponto'))
                    continue
                if local_lo <= level <= local_hi:
                    ax.axhline(level, color=COLORS[key], ls='--' if key=='medio' else '-',
                               lw=1.15, alpha=.75, zorder=2)
                    dd = risco - level
                    inds = np.flatnonzero(dd[:-1]*dd[1:] < 0)
                    for j in inds:
                        t = -dd[j] / (dd[j+1] - dd[j])
                        xroot = xx[j] + t*(xx[j+1]-xx[j])
                        ax.axvline(xroot, color=COLORS[key], lw=2.2, zorder=7)
                    estados.append((title, 'cruzamento' if len(inds) else 'nivel interno sem cruzamento discretizado'))
                else:
                    estados.append((title, _status_minmax(risco, level)))
            ymin, ymax = local_lo, local_hi
            pad = max((ymax-ymin)*.22, .0001)
            ax.set_ylim(max(0, ymin-pad), min(1, ymax+pad))
            ax.set_xlabel(nomes[0]); ax.set_ylabel('Score de fraude s(x)')
            if grafico == 'GMM08':
                _intervalos_gaussianos_95_1d(ax, gm, xx)
        else:
            xx, yy = grade
            ax.pcolormesh(xx, yy, risco, cmap='viridis', norm=norm,
                          shading='auto', rasterized=True, zorder=0)
            ax.scatter(X[il, par[0]], X[il, par[1]], color='#2678bd',
                       s=7, alpha=.24, rasterized=True, zorder=3)
            ax.scatter(X[iff, par[0]], X[iff, par[1]], color='#da3131',
                       marker='*', s=30, alpha=.68, rasterized=True, zorder=4)
            if grafico == 'GMM08':
                _elin_2d(ax, gm, par, medias)
            for key, level, title in niveis:
                if key == 'maximo':
                    if np.isclose(local_hi, hi, atol=1e-14, rtol=0):
                        j, k = np.unravel_index(int(np.argmax(risco)), risco.shape)
                        ax.scatter([xx[j, k]], [yy[j, k]], s=94, color=COLORS[key],
                                   edgecolors='black', linewidths=.5, marker='D', zorder=10)
                    estados.append((title, 'maximo marcado como ponto'))
                    continue
                if local_lo < level < local_hi:
                    cs = ax.contour(xx, yy, risco, levels=[level], colors=[COLORS[key]],
                                    linewidths=2.5 if key=='oof' else 1.9,
                                    linestyles='--' if key=='medio' else '-', zorder=8)
                    cross = bool(cs.allsegs[0] and any(len(seg) >= 2 for seg in cs.allsegs[0]))
                    estados.append((title, 'curva encontrada' if cross else 'nao cruzou na resolucao'))
                else:
                    estados.append((title, _status_minmax(risco, level)))
            ax.set_xlabel(nomes[par[0]]); ax.set_ylabel(nomes[par[1]])
            ax.set_xlim(xx[0,0], xx[0,-1]); ax.set_ylim(yy[0,0], yy[-1,0])
        # F10_BOUNDARY_CERTIFICATE_V2: ponto REAL da funcao score=tau*.
        # No 3D os planos passam por esse ponto, pois suas dimensoes restantes
        # foram fixadas na propria coordenada da raiz de bissecao.
        if certificado is not None and certificado['cruzamento_confirmado']:
            raiz = certificado['ponto_fronteira']
            if d == 1:
                ax.scatter([raiz[0]], [tau], marker='X', s=120,
                           color=COLORS['oof'], edgecolors='white', linewidths=.7,
                           zorder=14)
            else:
                ax.scatter([raiz[par[0]]], [raiz[par[1]]], marker='X', s=105,
                           color=COLORS['oof'], edgecolors='white', linewidths=.7,
                           zorder=14)
        ax.set_title(_rotulos_planos(par, nomes, medias), fontsize=11, pad=10)
        ax.grid(alpha=.07)
        ax.text(.02, .02, f'Malha deste plano: [{local_lo:.7f}, {local_hi:.7f}]',
                transform=ax.transAxes, ha='left', va='bottom', fontsize=8,
                bbox={'facecolor':'white', 'edgecolor':'#d5deea','alpha':.9}, zorder=11)
    title = ('SENSIBILIDADE: QUATRO NÍVEIS DE RISCO' if grafico == 'GMM07'
             else 'MAPA DE RISCO: QUATRO NÍVEIS E GEOMETRIA (95%)')
    fig.suptitle(title, fontsize=16.3, fontweight='bold', y=.984)
    desc = ('OOF: modelo vencedor reajustado com F1–F9; métricas nas predições OOF'
            if origem == 'OOF' else
            'F10: modelo F1–F9 congelado; limiar τ* escolhido exclusivamente em OOF'
            if origem == 'F10' else 'Prévia sintética; NÃO é avaliação OOF')
    # GMM_2D_3D_LIMPO_F10_V4_2: subtítulo removido dos PNGs 1D/2D/3D.
    # A anotacao de estado nunca afirma que um contorno ausente seja positivo.
    global_note = (f'Malha s(x): mínimo={lo:.7f} | máximo={hi:.7f} | '
                   f'médio={(lo+hi)/2:.7f} | τ*= {tau:.7f} | '
                   f'{mccnome}(0,50)={mcc[0]:.7f} | {mccnome}(τ*)={mcc[1]:.7f}')
    fig.text(.5, (.884 if d == 1 else .932), global_note, ha='center', fontsize=9.3, color='#24364c')
    handles = [Line2D([0],[0], color=COLORS[k], lw=2.3,
                      ls='--' if k=='medio' else '-', label=(nm+' = '+format(v,'.7f') if k!='maximo' else 'Máximo da malha (ponto)'))
               for k,v,nm in niveis]
    handles += [Line2D([0],[0],marker='*',ls='',color='#da3131',markersize=8,
                       label=f'Fraudes reais: todas ({len(iff)})'),
                Line2D([0],[0],marker='o',ls='',color='#2678bd',markersize=6,
                       label=f'Legítimas: amostra ({len(il)}/{np.count_nonzero(y==0)})')]
    if grafico == 'GMM08' and d > 1:
        handles.append(Line2D([0],[0],ls='--',color='#8b8196',label='Gaussianas FULL 95%'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .85 if d == 1 else .890),
               ncol=3 if d>1 else 2, fontsize=8.4, frameon=False)
    # HOTFIX_GMM_QUATRO_NIVEIS_BARRA_FORA_DOS_EIXOS_20261104
    fig.subplots_adjust(left=.07 if d!=3 else .055,
                        right=.96 if d==1 else (.885 if d==2 else .895),
                        top=(.72 if d==1 else .77),
                        bottom=(.12 if d==1 else .10),
                        wspace=.28 if d==3 else .33)
    if d > 1:
        from matplotlib.cm import ScalarMappable
        # Eixo independente: nao ocupa espaco de nenhum painel de risco.
        cax = fig.add_axes([.92, .20, .013, .50])
        fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),
                     cax=cax, label='Risco estimado na malha')
    # Resumo: pontos positivos da malha p/ cada corte, sem refazer classificacoes oficiais
    tot = int(sum(p[2].size for p in planos))
    nfix = int(sum(np.count_nonzero(p[2]>=.5) for p in planos))
    nstar = int(sum(np.count_nonzero(p[2]>=tau) for p in planos))
    nmid = int(sum(np.count_nonzero(p[2]>=mid) for p in planos))
    msg = (f'Positivos na MALHA: τ=0,50: {nfix}/{tot} ({100*nfix/tot:.2f}%); '
           f'τ*={tau:.7f}: {nstar}/{tot} ({100*nstar/tot:.2f}%); '
           f'nível médio: {nmid}/{tot} ({100*nmid/tot:.2f}%).')
    if origem == 'OOF' and avaliacao_diagnostico is not None:
        # Não atribuir ao modelo final as classificações feitas por 9 modelos OOF.
        n_oof = int(np.count_nonzero(np.asarray(avaliacao_diagnostico['p']) >= tau))
        n_refit = int(np.count_nonzero(np.asarray(data['scores']) >= tau))
        msg += (f'\nPredições positivas OOF (9 modelos): {n_oof}; '
                f'modelo reajustado F1–F9 sobre desenvolvimento: {n_refit}.')
    elif origem == 'F10':
        n_f10 = int(np.count_nonzero(np.asarray(data['scores']) >= tau))
        msg += f'\nPredições positivas F10 (modelo congelado): {n_f10}/{len(y)}.'
    else:
        msg += '\nMáximo marcado como ponto; níveis ausentes não são inventados.'
    if origem == 'F10' and certificado is not None:
        if certificado['cruzamento_confirmado']:
            msg += ' | X vermelho: fronteira τ* confirmada por bissecao F10.'
        elif certificado['positivos_f10']:
            msg += ' | Positivos F10 sem par negativo: contorno nao garantido.'
    # GMM_2D_3D_LIMPO_F10_V4_2: rodapé visual removido; métricas e dados preservados.
    destino = Path(destino)
    destino.mkdir(parents=True,exist_ok=True)
    filename = _nome(grafico,d) + '.' + ext
    caminho = destino/filename
    if caminho.exists() and not substituir_staging:
        plt.close(fig)
        raise FileExistsError('PNG original protegido, nao sobrescrever: '+str(caminho))
    try:
        fig.savefig(caminho, dpi=dpi, facecolor='white')
    finally:
        plt.close(fig)
    return filename


def _html3d(config, data, escolha, destino, grafico, origem, mcc, planos, lo_plano, hi_plano, *, substituir_staging=False, certificado=None):
    """HTML offline com a malha VOLUMÉTRICA do mesmo GMM congelado.

    Usa curvas como ISOSUPERFÍCIES em 3D, não como planos artificiais.
    """
    import plotly.graph_objects as go
    from scipy.stats import chi2
    X, y, gm, riscos, tau = _validate(data, escolha)
    op = config.get('visualizacao',{}).get('graficos',{}).get(
        'sensibilidade_fronteira' if grafico=='GMM07' else 'mapa_risco_fronteira',{})
    n = int(op.get('resolucao_volume_html_3d', 32))
    n = min(45,max(20,n))
    margem = float(op.get('margem', .15))
    axes = [np.linspace(*_faixa(X[:,i], margem), n) for i in range(3)]
    xx,yy,zz = np.meshgrid(*axes, indexing='ij')
    pts = np.column_stack((xx.ravel(),yy.ravel(),zz.ravel()))
    scores = _pontuar(gm,riscos,pts)
    lo, hi = float(scores.min()), float(scores.max())
    mid = (lo+hi)/2
    fig = go.Figure()
    il, iff = _fraudes_todas_legitimas_amostradas(y, int(op.get('n_pontos_scatter_html_3d', 2500)),
                                                  int(config.get('experimento',{}).get('semente',42)))
    for indices,cor,s,rotulo in [(il, '#2678bd',2.3,'Legítimas (amostra)'),
                                (iff,'#da3131',3.6,'Fraudes (todas)')]:
        fig.add_trace(go.Scatter3d(x=X[indices,0],y=X[indices,1],z=X[indices,2],
                                  mode='markers', name=rotulo,
                                  marker={'size':s,'color':cor,'opacity':.6}))
    texto=[]
    for key,level,nome in [('fixo',.5,'τ = 0,50'),('oof',tau,'τ* OOF'),('medio',mid,'Ponto médio da malha')]:
        if lo < level < hi:
            eps=max((hi-lo)*.00005,1e-9)
            fig.add_trace(go.Isosurface(x=pts[:,0],y=pts[:,1],z=pts[:,2],value=scores,
                         isomin=level-eps,isomax=level+eps, surface_count=1,
                         colorscale=[[0,COLORS[key]],[1,COLORS[key]]],
                         caps={'x_show':False,'y_show':False,'z_show':False},
                         opacity=.25, showscale=False, name=f'{nome} = {level:.7f}'))
            texto.append(f'{nome}: superfície disponível')
        else:
            texto.append(f'{nome}: {_status_minmax(scores,level)}')
    j = int(np.argmax(scores)); pmax = pts[j]
    fig.add_trace(go.Scatter3d(x=[pmax[0]],y=[pmax[1]],z=[pmax[2]],mode='markers',
        marker={'size':6,'color':COLORS['maximo'],'symbol':'diamond'},
        name=f'Máximo da malha volumétrica = {hi:.7f}'))
    # F10_BOUNDARY_CERTIFICATE_V2: marcacao exata de cruzamento em 3D.
    if certificado is not None and certificado['cruzamento_confirmado']:
        raiz = certificado['ponto_fronteira']
        fig.add_trace(go.Scatter3d(
            x=[raiz[0]], y=[raiz[1]], z=[raiz[2]], mode='markers',
            marker={'size':8, 'color':COLORS['oof'], 'symbol':'diamond'},
            name='Cruzamento real τ* confirmado por bissecao F10'))
    if grafico=='GMM08':
        for k in range(gm.n_components):
            cov = np.asarray(gm.covariances_[k], dtype=float)
            eig,V=np.linalg.eigh(cov)
            if np.any(eig<=0):
                raise ValueError(f'Covariancia FULL G{k+1} nao positiva definida: nao e possivel desenhar 95%.')
            u=np.linspace(0,2*np.pi,26)
            v=np.linspace(0,np.pi,19)
            sphere=np.stack((np.outer(np.cos(u),np.sin(v)),
                             np.outer(np.sin(u),np.sin(v)),
                             np.outer(np.ones_like(u),np.cos(v))),axis=-1)
            mat=V@np.diag(np.sqrt(eig*chi2.ppf(.95,3)))
            P=np.einsum('ij,abj->abi',mat,sphere)+gm.means_[k]
            # GMM08_HTML_3D_ELIPSOIDES_FULL_95_V5
            # Cor idêntica às seções gaussianas 95% nos PNGs (tab10).
            from matplotlib.colors import to_hex
            color = to_hex(plt.get_cmap('tab10')(k % 10))
            # Elipsoide FULL: (x-mu)^T Sigma^-1 (x-mu) = chi2_3(0.95).
            # Antes o trace era 'legendonly': não aparecia ao abrir o HTML.
            fig.add_trace(go.Surface(x=P[:,:,0], y=P[:,:,1], z=P[:,:,2],
                colorscale=[[0,color],[1,color]], opacity=.23,
                showscale=False, showlegend=True, visible=True,
                name=f'G{k+1} — elipsoide FULL 95%',
                hovertemplate=f'Gaussiana G{k+1} | elipsoide central FULL 95%<extra></extra>'))
            # Centro visível facilita comparar componentes e suas elipsoides.
            fig.add_trace(go.Scatter3d(
                x=[float(gm.means_[k,0])], y=[float(gm.means_[k,1])],
                z=[float(gm.means_[k,2])], mode='markers',
                marker={'size':5,'color':color,'symbol':'diamond',
                        'line':{'color':'#101827','width':1}},
                name=f'Centro G{k+1}',showlegend=False,
                hovertemplate=f'Centro da gaussiana G{k+1}<extra></extra>'))
    titulo = ('SENSIBILIDADE GMM07 — QUATRO NÍVEIS' if grafico=='GMM07'
              else 'MAPA DE RISCO GMM08 — QUATRO NÍVEIS E GAUSSIANAS 95%')
    origem_texto = ('OOF: modelo final F1–F9, limiar OOF' if origem=='OOF' else
                    'F10: modelo congelado F1–F9, limiar OOF congelado' if origem=='F10'
                    else 'Prévia sintética')
    subtitulo = (' | '.join(texto) +
                f' | Malha VOLUMÉTRICA: mínimo={lo:.7f}, máximo={hi:.7f}, médio={mid:.7f}; '
                f'cortes PNG: mínimo={lo_plano:.7f}, máximo={hi_plano:.7f}.')
    fig.update_layout(title={'text':titulo+'<br><sup>'+origem_texto+'<br>'+subtitulo+'</sup>',
                             'x':.5,'font':{'size':15}},
        scene={'xaxis_title':data['coord'].columns[0],
               'yaxis_title':data['coord'].columns[1],
               'zaxis_title':data['coord'].columns[2],
               'aspectmode':'data'},
        legend={'orientation':'h','y':-.09},margin={'l':0,'r':0,'b':100,'t':150},
        height=820, paper_bgcolor='white')
    path=Path(destino)/(_nome(grafico,3)+'_INTERATIVA.html')
    if path.exists() and not substituir_staging:
        raise FileExistsError('HTML original protegido, nao sobrescrever: '+str(path))
    fig.write_html(path, include_plotlyjs=True, full_html=True,
                   config={'responsive':True,'scrollZoom':True,'displaylogo':False})
    return path


def gerar(config, data, escolha, destino, ext, dpi, csv, *, tipo, avaliacao=None, substituir_staging=False):
    """Interface compativel com GMM07/GMM08 existentes; nenhum ajuste de modelo."""
    if tipo not in ('GMM07','GMM08'):
        raise ValueError('tipo deve ser GMM07 ou GMM08.')
    X,y,gm,riscos,tau = _validate(data,escolha)
    d=X.shape[1]
    op = config.get('visualizacao',{}).get('graficos',{}).get(
        'sensibilidade_fronteira' if tipo=='GMM07' else 'mapa_risco_fronteira',{})
    n = int(op.get('resolucao_1d', 500) if d==1 else op.get('resolucao_2d',180))
    n = min(n, 450 if d>1 else 20000)
    if n < 20:
        raise ValueError('Resolução da malha inferior a 20.')
    margem = float(op.get('margem',.15))
    if not 0 <= margem <= 1:
        raise ValueError('Margem de malha invalida.')
    origem,mcc,nome_mcc=_fonte(destino,avaliacao,y,data,tau)
    certificado = None
    if origem == 'F10':
        from src.gmm_fronteira_f10 import diagnosticar
        certificado = diagnosticar(gm, riscos, X, data['scores'], tau)
    referencias = ([certificado['ponto_positivo'], certificado['ponto_negativo']]
                   if certificado is not None and certificado['cruzamento_confirmado'] else None)
    planos,medias=_planos(X,gm,riscos,d,n,margem,
                          ancora=(certificado['ponto_fronteira']
                                  if certificado is not None else None),
                          referencias=referencias)
    lo = min(float(np.min(item[2])) for item in planos)
    hi = max(float(np.max(item[2])) for item in planos)
    filename = _painel_png(config,data,escolha,destino,ext,dpi,tipo,origem,mcc,nome_mcc,
                          planos,medias,lo,hi,substituir_staging=substituir_staging,
                          avaliacao_diagnostico=avaliacao if origem=='OOF'
                          and isinstance(avaliacao,dict) and 'p' in avaliacao else None,
                          certificado=certificado)
    dest=Path(destino)
    if csv:
        for par, grade, scores in planos:
            base={'score_risco':scores.ravel()}
            fix=(scores.ravel()>=.5).astype(np.uint8)
            otim=(scores.ravel()>=tau).astype(np.uint8)
            if tipo=='GMM07':
                base.update(pred_fixo=fix,pred_otimo=otim,
                            mudou=(fix!=otim).astype(np.uint8))
            else:
                base.update(classe_corte_fixo=fix,classe_corte_mcc=otim,
                            mudou_classe=(fix!=otim).astype(np.uint8))
            base['classe_corte_medio']=(scores.ravel()>=(lo+hi)/2).astype(np.uint8)
            tab=pd.DataFrame(base)
            for j,dim in enumerate(par): tab[data['coord'].columns[dim]] = grade[j].ravel()
            suffix='_'.join(data['coord'].columns[i] for i in par)
            name=(f'GMM07_malha_{suffix}.csv' if tipo=='GMM07' else f'GMM08_malha_{suffix}.csv')
            arquivo_csv=dest/name
            if arquivo_csv.exists():
                raise FileExistsError('CSV original protegido, nao sobrescrever: '+str(arquivo_csv))
            tab.to_csv(arquivo_csv,index=False,float_format='%.17g')
        if tipo=='GMM07':
            pd.DataFrame([
                {'corte':'fixo_050','limiar':.5,'mcc':mcc[0],'origem_mcc':origem},
                {'corte':'otimo_mcc','limiar':tau,'mcc':mcc[1],'origem_mcc':origem},
            ]).to_csv(dest/'GMM07_metricas_comparativas.csv',index=False,float_format='%.17g')
    if d==3:
        _html3d(config,data,escolha,dest,tipo,origem,mcc,planos,lo,hi,
                substituir_staging=substituir_staging, certificado=certificado)
    if tipo=='GMM07':
        return {'arquivo':filename,'mcc_fixo':mcc[0],'mcc_otimo':mcc[1],
                'origem_mcc':origem,'n_planos':len(planos)}
    return None


def substituir_png_html_tsne3d(coord,y,modelo,riscos,destino,*,tau,origem,resolucao_cortes,dpi):
    """Hook t-SNE3D F10 (a rota separada), SEM recalcular predicoes oficiais."""
    X=np.asarray(coord,dtype=float)
    r=np.asarray(riscos,dtype=float)
    # Wrappers somente para render; usam o mesmo modelo sklearn ajustado.
    nomes=[f'TSNE{i+1}' for i in range(3)]
    gm=SimpleNamespace(modelo_=modelo)
    rh=SimpleNamespace(riscos_=pd.Series(r,index=[f'G{i+1}' for i in range(len(r))]))
    data={'coord':pd.DataFrame(X,columns=nomes), 'y':np.asarray(y),
          'scores':_pontuar(modelo,r,X),'gmm':gm,'rh':rh}
    cfg={'experimento':{'semente':42},'visualizacao':{'graficos':{
         'sensibilidade_fronteira':{'resolucao_2d':int(resolucao_cortes)},
         'mapa_risco_fronteira':{'resolucao_2d':int(resolucao_cortes)}},
         'geometria':{'pares_2d':[[1,2],[1,3],[2,3]]}}}
    es={'limiar_mcc_oof':float(tau)}
    # Para F10, o renderizador calcula MCC F10 exclusivamente dos scores F10
    # para os DOIS cortes oficiais. Para simulado, sem fabricar MCC real.
    if origem=='f10':
        av={'origem':'f10_amostral_congelado'}
    else:
        av={'origem':'simulado','y':np.asarray(y),'p':_pontuar(modelo,r,X)}
    for tipo in ('GMM07','GMM08'):
        gerar(cfg,data,es,Path(destino),'png',int(dpi),False,tipo=tipo,avaliacao=av,
              substituir_staging=True)
