"""GMM 04 — dominancia entre componentes do GMM por responsabilidades soft.

k*(x) = argmax_k gamma_k(x). NAO e fronteira de fraude/legitima.
Geometria ajustada no desenvolvimento F1–F9; nenhum uso de F10.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def calcular_dominancia(gmm, coordenadas):
    """Prediz componentes dominantes sem usar y, RH, score ou limiar."""
    from src import visualizacao_geometrica as geo
    arr = np.asarray(coordenadas, dtype=float)
    _ = getattr(gmm, 'colunas_', None)
    geo._exigir(arr.ndim == 2 and len(arr) > 0 and np.isfinite(arr).all(),
                'Malha de dominancia invalida.')
    medias = np.asarray(gmm.modelo_.means_, dtype=float)
    geo._exigir(medias.ndim == 2 and arr.shape[1] == medias.shape[1],
                'Dimensoes de malha/GMM divergentes.')
    gamma = np.asarray(gmm.modelo_.predict_proba(arr), dtype=float)
    geo._exigir(gamma.shape == (len(arr), len(medias)) and np.isfinite(gamma).all()
                and np.all(gamma >= -1e-12) and np.allclose(gamma.sum(axis=1), 1., atol=1e-6),
                'Responsabilidades soft invalidas.')
    # Empates exatos: a primeira gaussiana no indice oficial do GMM prevalece.
    return np.argmax(gamma, axis=1).astype(np.int32), gamma


def _cores(k):
    tab = plt.get_cmap('tab10')
    cores = [tab(i % 10) for i in range(k)]
    suaves = [tuple(.52 + .48*np.asarray(to_rgb(c))) for c in cores]
    return cores, ListedColormap(suaves)


def _legenda(nomes, cores, *, pontos=False):
    items = [Patch(facecolor=tuple(.52+.48*np.array(to_rgb(c))), edgecolor=c,
                   label=f'{nome} — dominante') for nome,c in zip(nomes,cores)]
    items.append(Line2D([], [], color='#172b46', lw=2., label='Fronteira entre componentes'))
    if pontos:
        items.append(Line2D([], [], marker='.', ls='', color='#344359', label='Observações (projeção)'))
    return items


def gerar(config, data, destino, ext, dpi, csv):
    """1D: intervalos e curva; 2D: areas; 3D: tres planos de corte."""
    from src import visualizacao_geometrica as geo
    from src.visualizacao_gmm01 import calcular_marginal
    op = config['visualizacao']['graficos']['regioes_dominancia']
    gm = data['gmm']
    coords = data['coord']
    X = coords.to_numpy(dtype=float)
    nomes_dim = list(coords.columns)
    d = X.shape[1]
    geo._exigir(d in (1,2,3) and np.isfinite(X).all(), 'GMM04 exige dados finitos em 1D, 2D ou 3D.')
    nomes = list(gm.saidas_)
    k = len(nomes)
    geo._exigir(k == gm.modelo_.means_.shape[0] and k >= 1,
                'Componentes gaussianos invalidos para GMM04.')
    n1 = int(op.get('resolucao_1d',600))
    n2 = int(op.get('resolucao_2d',180))
    margem = float(op.get('margem',.15))
    geo._exigir(100 <= n1 <= 10000 and 25 <= n2 <= 450 and 0 <= margem <= 1,
                'Resolucao/margem do GMM04 invalida.')
    cores, mapa = _cores(k)
    origem = 'F1–F9 (desenvolvimento); F10 não utilizado'
    p = gm.modelo_
    centro_desenvolvimento = X.mean(axis=0)

    if d == 1:
        eixo = np.linspace(*geo._faixa(X[:,0],margem),n1)
        dom, gamma = calcular_dominancia(gm, eixo[:,None])
        contribuicoes, mistura = calcular_marginal(gm,0,eixo)
        fig, ax = plt.subplots(figsize=(11.5,6.8))
        inicio = np.r_[0, np.flatnonzero(np.diff(dom) != 0)+1]
        fim = np.r_[inicio[1:], len(eixo)]
        for i,j in zip(inicio,fim):
            esquerda = eixo[i] if i == 0 else .5*(eixo[i-1]+eixo[i])
            direita = eixo[-1] if j == len(eixo) else .5*(eixo[j-1]+eixo[j])
            ax.axvspan(esquerda,direita,color=cores[int(dom[i])],alpha=.16,zorder=0)
            if j < len(eixo): ax.axvline(direita,color='#172b46',ls=':',lw=1.3,alpha=.75)
        for g in range(k):
            ax.plot(eixo,contribuicoes[g],ls='--',lw=1.35,color=cores[g],alpha=.9,
                    label=f'{nomes[g]} — contribuição ponderada')
            media = p.means_[g,0]
            altura = float(np.interp(media,eixo,mistura))
            if eixo[0] <= media <= eixo[-1]:
                ax.plot(media,altura,'x',ms=8,mew=2,color=cores[g])
        ax.plot(eixo,mistura,color='#172b46',lw=2.7,label='Mistura marginal total',zorder=4)
        ax.set(xlabel=nomes_dim[0], ylabel='Densidade marginal', xlim=(eixo[0],eixo[-1]))
        ax.set_ylim(bottom=0)
        ax.set_title(f'GMM 04 — Regiões de dominância | {d}D | {p.covariance_type.upper()}',
                     fontsize=15,weight='bold',pad=12)
        handles=[Patch(facecolor=cores[g],alpha=.4,label=f'{nomes[g]} — região dominante') for g in range(k)]
        handles += [Line2D([],[],color='#172b46',lw=2.5,label='Mistura total')]
        ax.legend(handles=handles,loc='upper right',fontsize=8.5)
        ax.grid(alpha=.1)
        fig.text(.5,.045,'Faixas: maior responsabilidade γₖ(x). Curvas tracejadas: densidades dos componentes. '
                 f'Dominância ≠ fraude. {origem}.',ha='center',color='#3c506a',fontsize=8.7)
        fig.subplots_adjust(left=.09,right=.975,top=.87,bottom=.16)
        stem='GMM04_dominancia_1d'
        geo._guardar(fig,destino,stem,ext,dpi)
        if csv:
            tabela=pd.DataFrame({nomes_dim[0]:eixo,'gaussiana_dominante':[nomes[i] for i in dom],
                                 'densidade_mistura':mistura})
            for i,nome in enumerate(nomes):
                tabela[f'responsabilidade_{nome}']=gamma[:,i]
                tabela[f'densidade_ponderada_{nome}']=contribuicoes[i]
            geo._salvar_dados(tabela,destino,stem,True)
        return

    pares=[(0,1)] if d==2 else [(0,1),(0,2),(1,2)]
    nscatter=int(op.get('pontos_scatter',1000))
    geo._exigir(0<=nscatter<=50000,'Amostra visual GMM04 invalida.')
    idx=geo._amostra_plot(len(X),np.asarray(data['y']),config,
                          min(len(X),max(1,nscatter))) if nscatter else np.array([],dtype=int)
    fig,axset=plt.subplots(1,len(pares),figsize=(11,7.6) if d==2 else (19.3,7.4),squeeze=False)
    axes=axset.ravel()
    for ax,par in zip(axes,pares):
        i,j=par
        x=np.linspace(*geo._faixa(X[:,i],margem),n2)
        y=np.linspace(*geo._faixa(X[:,j],margem),n2)
        xx,yy=np.meshgrid(x,y)
        malha=np.tile(centro_desenvolvimento,(n2*n2,1))
        malha[:,i]=xx.ravel();malha[:,j]=yy.ravel()
        dom,gamma=calcular_dominancia(gm,malha)
        z=dom.reshape(n2,n2)
        ax.pcolormesh(xx,yy,z,cmap=mapa,vmin=-.5,vmax=k-.5,shading='auto',rasterized=True)
        if z.min()!=z.max():
            # Frontiers between discrete argmax regions, not risk thresholds.
            ax.contour(xx,yy,z,levels=np.arange(.5,k-.5,1.),colors=['#172b46'],linewidths=.85)
        if len(idx):
            ax.scatter(X[idx,i],X[idx,j],s=5,c='#344359',alpha=.18,rasterized=True)
        for g,nome in enumerate(nomes):
            ax.plot(p.means_[g,i],p.means_[g,j],marker='x',ms=8,mew=2,color=cores[g])
            ax.annotate(nome,(p.means_[g,i],p.means_[g,j]),xytext=(5,5),
                        textcoords='offset points',fontsize=8,fontweight='bold',color='#182b46')
        ax.set(xlabel=nomes_dim[i],ylabel=nomes_dim[j],xlim=(x[0],x[-1]),ylim=(y[0],y[-1]))
        if d==3:
            fixado=next(h for h in range(3) if h not in par)
            ax.set_title(f'{nomes_dim[i]} × {nomes_dim[j]}\n'
                         f'{nomes_dim[fixado]} fixada na média = {centro_desenvolvimento[fixado]:.3g}',
                         fontsize=10.2,pad=10)
        else:
            ax.set_title(f'{nomes_dim[i]} × {nomes_dim[j]}',fontsize=12,pad=12)
        if csv:
            stem_csv='GMM04_malha_'+nomes_dim[i]+'_'+nomes_dim[j]
            tabela=pd.DataFrame({nomes_dim[i]:xx.ravel(),nomes_dim[j]:yy.ravel(),
                                 'gaussiana_dominante':[nomes[t] for t in dom],
                                 'indice_dominante':dom+1})
            if d==3: tabela[f'{nomes_dim[fixado]}_fixada']=centro_desenvolvimento[fixado]
            for g,nome in enumerate(nomes): tabela[f'responsabilidade_{nome}']=gamma[:,g]
            geo._salvar_dados(tabela,destino,stem_csv,True)

    fig.suptitle(f'GMM 04 — Regiões de dominância | {d}D | {p.covariance_type.upper()}',
                 fontsize=15.5,fontweight='bold',y=.975)
    fig.legend(handles=_legenda(nomes,cores,pontos=bool(len(idx))),
               loc='upper center',bbox_to_anchor=(.5,.917),ncol=min(k+2,6),
               fontsize=9,frameon=False)
    texto=('Os centros e as observações são projeções; a dominância é calculada em cortes '
           'com a dimensão omitida fixada na média.' if d==3 else
           'Fronteiras = mudança da gaussiana de maior responsabilidade.')
    fig.text(.5,.065,texto,ha='center',fontsize=9.1,color='#293c54')
    fig.text(.5,.033,f'Dominância entre componentes ≠ fronteira fraude/não fraude. {origem}.',
             ha='center',fontsize=8.8,color='#43556d')
    fig.subplots_adjust(left=.065 if d==3 else .1,right=.985 if d==3 else .95,
                        top=.79,bottom=.16,wspace=.24)
    geo._guardar(fig,destino,f'GMM04_dominancia_{d}d',ext,dpi)
