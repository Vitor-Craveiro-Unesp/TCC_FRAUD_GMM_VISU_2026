# -*- coding: utf-8 -*-
"""GMM FULL 3D sobre coordenadas t-SNE congeladas — Fase 7.3.

Modulo independente da finalizacao oficial. Recebe modelo ja ajustado,
coordenadas 3D e resultados OOF; nunca ajusta GMM nem altera o limiar.
Os cortes 3D representam secoes de um UNICO GMM FULL (nao marginais 2D).
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2, multivariate_normal
from sklearn.metrics import matthews_corrcoef

from src.visualizacao_geometrica import _elipse_95_corte

EIXOS = ('TSNE1', 'TSNE2', 'TSNE3')
PARES = ((0, 1, 2), (0, 2, 1), (1, 2, 0))
CORES = ('#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
         '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf')
NOMES = {
    'GMM01': ('GMM01_densidade_conjunta_TSNE3D.png', 'GMM01_densidade_conjunta_TSNE3D_INTERATIVA.html'),
    'GMM04': ('GMM04_dominancia_tsne3d.png', 'GMM04_dominancia_tsne3d_INTERATIVA.html'),
    'GMM07': ('GMM07_sensibilidade_3d_comparativo.png', 'GMM07_sensibilidade_3d_comparativo_INTERATIVA.html'),
    'GMM08': ('GMM08_mapa_risco_3d_comparativo.png', 'GMM08_mapa_risco_3d_comparativo_INTERATIVA.html'),
}
MANIFESTO = 'GMM_TSNE3D_manifesto_fase73.json'


def _validar(coord, y, modelo, riscos, tau, y_oof, scores_oof, origem, fracao_f10):
    if not isinstance(coord, pd.DataFrame) or list(coord.columns) != list(EIXOS):
        raise ValueError('As coordenadas devem ser DataFrame TSNE1, TSNE2, TSNE3 (nessa ordem).')
    X = coord.to_numpy(dtype=float)
    if X.ndim != 2 or len(X) < 8 or not np.isfinite(X).all():
        raise ValueError('Coordenadas TSNE 3D invalidas.')
    y = np.asarray(y)
    if y.shape != (len(X),) or not np.isin(y, [0,1]).all():
        raise ValueError('Classes reais invalidas.')
    if getattr(modelo, 'covariance_type', None) != 'full':
        raise ValueError('GMM TSNE3D exige covariancia FULL.')
    mu = np.asarray(modelo.means_, dtype=float)
    cov = np.asarray(modelo.covariances_, dtype=float)
    w = np.asarray(modelo.weights_, dtype=float)
    if (mu.ndim != 2 or mu.shape[1] != 3 or cov.shape != (len(mu),3,3)
        or w.shape != (len(mu),) or len(mu) > len(CORES)
        or not (np.isfinite(mu).all() and np.isfinite(cov).all() and
                np.isfinite(w).all() and np.all(w>0) and np.isclose(w.sum(),1))
        or np.any(np.linalg.eigvalsh(cov) <= 0)):
        raise ValueError('Parametros do GMM FULL 3D invalidos.')
    risco = np.asarray(riscos, dtype=float)
    if risco.shape != (len(mu),) or not np.isfinite(risco).all() or np.any((risco < 0) | (risco > 1)):
        raise ValueError('Riscos por gaussiana invalidos.')
    tau = float(tau)
    if not np.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError('Limiar tau OOF invalido.')
    yo, po = np.asarray(y_oof), np.asarray(scores_oof, dtype=float)
    if (yo.ndim != 1 or po.shape != yo.shape or not len(yo)
        or not np.isin(yo, [0,1]).all() or
        not np.isfinite(po).all() or np.any((po<0)|(po>1))):
        raise ValueError('Predicoes OOF incompatíveis ou invalidas.')
    if origem not in {'simulado', 'oof', 'f10'}:
        raise ValueError('Origem invalida: simulado, oof ou f10.')
    if origem == 'f10' and (fracao_f10 is None or not 0 < float(fracao_f10) <= 1):
        raise ValueError('F10 requer fracao_f10 declarada pelo YAML.')
    return X, y.astype(int), mu, cov, w, risco, tau, yo, po


def _escores(modelo, riscos, X):
    resp = np.asarray(modelo.predict_proba(X), dtype=float)
    if resp.ndim != 2 or resp.shape != (len(X),len(riscos)) or not np.isfinite(resp).all():
        raise ValueError('Responsabilidades invalidas.')
    score = resp @ riscos
    if not np.isfinite(score).all() or np.any((score < -1e-10) | (score > 1+1e-10)):
        raise ValueError('Scores invalidos.')
    return np.clip(score, 0, 1), resp


def _secoes(X, modelo, riscos, n):
    centro = X.mean(axis=0)
    lim = []
    for j in range(3):
        q1, q2 = np.quantile(X[:,j],[.005,.995])
        a = max(float(q2-q1), 1e-5)
        lim.append((float(q1-.18*a),float(q2+.18*a)))
    secoes = []
    for a,b,om in PARES:
        xx, yy = np.meshgrid(np.linspace(*lim[a], n), np.linspace(*lim[b],n))
        pts = np.tile(centro, (n*n,1))
        pts[:,a],pts[:,b] = xx.ravel(),yy.ravel()
        s, resp = _escores(modelo,riscos,pts)
        dens = np.zeros(len(pts))
        for k in range(len(modelo.weights_)):
            dens += modelo.weights_[k]*multivariate_normal.pdf(pts, mean=modelo.means_[k],cov=modelo.covariances_[k])
        secoes.append((a,b,om,xx,yy,pts,s.reshape(xx.shape),
                       dens.reshape(xx.shape),resp.argmax(axis=1).reshape(xx.shape)))
    return centro,lim,secoes


def _malha(lim,modelo,riscos,n):
    xx,yy,zz=np.meshgrid(*(np.linspace(*lim[j],n) for j in range(3)),indexing='ij')
    pts=np.column_stack([xx.ravel(),yy.ravel(),zz.ravel()])
    scores,resp=_escores(modelo,riscos,pts)
    dens=np.zeros(len(pts))
    for k in range(len(modelo.weights_)):
        dens += modelo.weights_[k]*multivariate_normal.pdf(pts,mean=modelo.means_[k],cov=modelo.covariances_[k])
    return pts,scores,dens,resp


def _elipsoide(mu,cov,nu=27,nv=19):
    vals,vec=np.linalg.eigh(cov)
    u=np.linspace(0,2*np.pi,nu)
    v=np.linspace(0,np.pi,nv)
    base=np.stack((np.outer(np.cos(u),np.sin(v)),np.outer(np.sin(u),np.sin(v)),
                   np.outer(np.ones_like(u),np.cos(v))),axis=-1)
    mat=vec @ np.diag(np.sqrt(vals*chi2.ppf(.95,df=3)))
    out=np.einsum('ij,uvj->uvi',mat,base)+mu
    return out[...,0],out[...,1],out[...,2]


def _rotulo_origem(origem, fracao_f10=None):
    if origem == 'simulado':
        return 'PREVIA SINTETICA — sem dados reais'
    if origem == 'f10':
        return f'F10 EXPLORATORIO — {float(fracao_f10)*100:.2f}% do proprio F10'
    return 'OOF F1-F9 — sem F10'


def _html(fig,titulo,descricao,arquivo):
    fig.update_layout(
        title={'text':titulo+'<br><sup>'+descricao+'</sup>','x':.5},height=800,
        margin=dict(l=10,r=12,t=105,b=14),paper_bgcolor='white',
        scene=dict(xaxis_title='TSNE1',yaxis_title='TSNE2',zaxis_title='TSNE3',
                   aspectmode='data',dragmode='orbit',camera=dict(eye=dict(x=1.55,y=1.65,z=1.1))))
    pagina=fig.to_html(full_html=True,include_plotlyjs=True,div_id='gmm_tsne3d',
                      config={'responsive':True,'displaylogo':False,'scrollZoom':True})
    arquivo.write_text(pagina,encoding='utf-8')


def _densidade(X,modelo,centro,secoes,malha,destino,dpi,rotulo_origem):
    f,ax=plt.subplots(1,3,figsize=(18,5.8),layout='constrained')
    for a, (i,j,om,xx,yy,pts,s,den,dom) in enumerate(secoes):
        img=ax[a].pcolormesh(xx,yy,den,cmap='viridis',shading='auto',rasterized=True)
        ax[a].set(xlabel=EIXOS[i],ylabel=EIXOS[j],title=f'{EIXOS[om]} fixada na media dos dados')
        f.colorbar(img,ax=ax[a],shrink=.75)
    f.suptitle('GMM01 — Cortes da densidade conjunta FULL em t-SNE 3D\n'+rotulo_origem,fontweight='bold')
    png=destino/NOMES['GMM01'][0]
    f.savefig(png,dpi=dpi);plt.close(f)
    pts,scores,dens,resp=malha
    fig=go.Figure()
    if float(dens.max())>float(dens.min()):
        fig.add_trace(go.Isosurface(x=pts[:,0],y=pts[:,1],z=pts[:,2],value=dens,
            isomin=float(dens.max())*.12,isomax=float(dens.max())*.72,surface_count=3,
            opacity=.20,colorscale='Viridis',caps=dict(x_show=False,y_show=False,z_show=False),
            name='Densidade conjunta'))
    for k in range(len(modelo.weights_)):
        x,y,z=_elipsoide(modelo.means_[k],modelo.covariances_[k])
        fig.add_trace(go.Surface(x=x,y=y,z=z,showscale=False,opacity=.15,
            surfacecolor=np.full_like(x,k),colorscale=[[0,CORES[k]],[1,CORES[k]]],
            name=f'G{k+1} FULL 95%'))
    _html(fig,'GMM01 — Densidade conjunta t-SNE 3D',
          rotulo_origem+' | Elipsoides 95% indicam concentracao gaussiana, nao risco de fraude',
          destino/NOMES['GMM01'][1])


def _dominancia(X,y,modelo,secoes,destino,dpi,rotulo_origem):
    resp=modelo.predict_proba(X)
    domin=resp.argmax(axis=1)
    idx=np.linspace(0,len(X)-1,min(1600,len(X)),dtype=int)
    fig=plt.figure(figsize=(10,8))
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(X[idx,0],X[idx,1],X[idx,2],c=[CORES[z] for z in domin[idx]],s=7,alpha=.4,depthshade=False)
    for k,mu in enumerate(modelo.means_):
        ax.scatter(*mu,marker='X',s=100,color=CORES[k],label=f'G{k+1}')
    ax.set(xlabel='TSNE1',ylabel='TSNE2',zlabel='TSNE3',title='GMM04 — Dominancia por responsabilidade\n'+rotulo_origem)
    ax.legend(loc='upper left')
    png=destino/NOMES['GMM04'][0]
    fig.savefig(png,dpi=dpi);plt.close(fig)
    fig=go.Figure()
    fig.add_trace(go.Scatter3d(x=X[idx,0],y=X[idx,1],z=X[idx,2],mode='markers',
        marker=dict(size=2.5,opacity=.3,color=[CORES[z] for z in domin[idx]]),
        name='Observacoes por gaussiana dominante'))
    for k,mu in enumerate(modelo.means_):
        x,yy,z=_elipsoide(mu,modelo.covariances_[k])
        fig.add_trace(go.Surface(x=x,y=yy,z=z,opacity=.15,showscale=False,
            colorscale=[[0,CORES[k]],[1,CORES[k]]],surfacecolor=np.zeros_like(x),
            name=f'G{k+1} — FULL 95%'))
        fig.add_trace(go.Scatter3d(x=[mu[0]],y=[mu[1]],z=[mu[2]],mode='markers+text',
            text=[f'G{k+1}'],marker=dict(color=CORES[k],symbol='diamond',size=7),
            name=f'Centro G{k+1}'))
    _html(fig,'GMM04 — Dominancia t-SNE 3D',
          rotulo_origem+' | Cor por responsabilidade posterior maxima, nao por risco de fraude',
          destino/NOMES['GMM04'][1])


def _painel_fronteiras(X,y,modelo,sec,tau,mccfix,mccstar,rotulo_origem,destino,dpi,com_geometria):
    """Tres planos ortogonais; duas colunas para 0.50 e tau* OOF."""
    titulo='GMM08 — Mapa de risco e geometria FULL 95%' if com_geometria else 'GMM07 — Sensibilidade ao limiar'
    key='GMM08' if com_geometria else 'GMM07'
    fig, axs=plt.subplots(3,2,figsize=(13.5,17),layout='constrained')
    teto=max(float(np.max(item[6])) for item in sec)
    norm=Normalize(vmin=0,vmax=max(teto,1e-10))
    centro=X.mean(axis=0)
    for row,(a,b,om,xx,yy,pts,s,dens,dom) in enumerate(sec):
        for col,limiar in enumerate((.5,tau)):
            ax=axs[row,col]
            ax.pcolormesh(xx,yy,s,cmap='viridis',norm=norm,shading='auto',rasterized=True)
            mudou=(s>=.5)!=(s>=tau)
            if np.any(mudou):
                ax.contourf(xx,yy,mudou.astype(float),levels=[.5,1.5],colors='none',hatches=['////'])
            if float(s.min()) < limiar < float(s.max()):
                ax.contour(xx,yy,s,levels=[limiar],colors='black',linewidths=1.8)
            else:
                ax.text(.03,.03,'Sem cruzamento neste corte',transform=ax.transAxes,fontsize=8,
                        bbox=dict(facecolor='white',alpha=.8))
            if com_geometria:
                # Intersecao verdadeira do elipsoide 3D FULL com o plano.
                envoltorio=SimpleNamespace(modelo_=modelo)
                for k in range(len(modelo.weights_)):
                    ellipse=_elipse_95_corte(envoltorio,k,(a,b),centro)
                    if ellipse is not None:
                        (cx,cy),width,height,angle=ellipse
                        ax.add_patch(Ellipse((cx,cy),width,height,angle=angle,
                            fill=False,ls='--',color=CORES[k],lw=1.2))
            ax.set(xlabel=EIXOS[a],ylabel=EIXOS[b],title=f'{EIXOS[a]} × {EIXOS[b]} | {EIXOS[om]} = {centro[om]:.5f}')
            ax.tick_params(labelsize=8)
    axs[0,0].text(0,1.21,f'CORTE FIXO: τ=0.5000000 | MCC={mccfix:.7f}',
                  transform=axs[0,0].transAxes,fontsize=11.3,fontweight='bold')
    axs[0,1].text(0,1.21,f'CORTE OTIMO: τ*={tau:.7f} | MCC={mccstar:.7f}',
                  transform=axs[0,1].transAxes,fontsize=11.3,fontweight='bold')
    fig.suptitle(titulo+' — '+rotulo_origem,fontweight='bold')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=axs,shrink=.65,label='Risco s(x), escala comum')
    png=destino/NOMES[key][0]
    fig.savefig(png,dpi=dpi);plt.close(fig)
    return png


def _html_fronteiras(X,y,modelo,malha,tau,mccfix,mccstar,rotulo_origem,destino,com_geometria):
    """HTML com duas cenas 3D independentes e mesma malha de risco."""
    key='GMM08' if com_geometria else 'GMM07'
    pts,scores,dens,resp=malha
    fig=make_subplots(rows=1,cols=2,specs=[[{'type':'scene'},{'type':'scene'}]],
        subplot_titles=[f'τ=0.5000000 | MCC={mccfix:.7f}',
                        f'τ*={tau:.7f} | MCC={mccstar:.7f}'],
        horizontal_spacing=.055)
    lo,hi=float(scores.min()),float(scores.max())
    idx=np.linspace(0,len(X)-1,min(1300,len(X)),dtype=int)
    for col, (nome,th,cor) in enumerate((('Corte fixo',.5,'#253446'),
                                         ('Limiar MCC OOF',tau,'#d19c24')),start=1):
        if lo < th < hi:
            largura=max((hi-lo)*1e-5,1e-8)
            fig.add_trace(go.Isosurface(x=pts[:,0],y=pts[:,1],z=pts[:,2],
                value=scores,isomin=max(0,th-largura),isomax=min(1,th+largura),
                surface_count=1,showscale=False,opacity=.36,
                colorscale=[[0,cor],[1,cor]],caps=dict(x_show=False,y_show=False,z_show=False),
                name=nome),row=1,col=col)
        change=(scores>=.5)!=(scores>=tau)
        ids=np.flatnonzero(change)
        if len(ids)>1200:ids=np.random.default_rng(42).choice(ids,1200,replace=False)
        fig.add_trace(go.Scatter3d(x=pts[ids,0],y=pts[ids,1],z=pts[ids,2],mode='markers',
            marker=dict(color='#c6952d',opacity=.2,size=2),name='Regiao de mudanca',
            showlegend=(col==1)),row=1,col=col)
        if com_geometria:
            fig.add_trace(go.Volume(x=pts[:,0],y=pts[:,1],z=pts[:,2],value=scores,
                isomin=lo,isomax=hi,surface_count=6,opacity=.035,colorscale='Viridis',
                showscale=False,name='Volume de risco',visible='legendonly',
                showlegend=(col==1)),row=1,col=col)
            for k,mu in enumerate(modelo.means_):
                x,yy,z=_elipsoide(mu,modelo.covariances_[k])
                fig.add_trace(go.Surface(x=x,y=yy,z=z,opacity=.16,showscale=False,
                    surfacecolor=np.zeros_like(x),colorscale=[[0,CORES[k]],[1,CORES[k]]],
                    name=f'G{k+1} FULL 95%',visible='legendonly',
                    showlegend=(col==1)),row=1,col=col)
        for cls,c,txt in ((0,'#2378b9','Legitimas'),(1,'#e03535','Fraudes')):
            ids=idx[y[idx]==cls]
            fig.add_trace(go.Scatter3d(x=X[ids,0],y=X[ids,1],z=X[ids,2],mode='markers',
                marker=dict(color=c,opacity=.35,size=2.5),name=txt,
                showlegend=(col==1)),row=1,col=col)
    titulo='GMM08 — Mapa de risco e geometria t-SNE 3D' if com_geometria else 'GMM07 — Sensibilidade t-SNE 3D'
    fig.update_layout(title={'text':titulo+'<br><sup>'+rotulo_origem+
        ' | Mesma superficie de risco; limiares congelados do OOF</sup>','x':.5},
        height=750,margin=dict(l=5,r=5,t=135,b=5),paper_bgcolor='white',
        legend=dict(x=.01,y=.95),
        scene=dict(xaxis_title='TSNE1',yaxis_title='TSNE2',zaxis_title='TSNE3',
                   aspectmode='data',dragmode='orbit'),
        scene2=dict(xaxis_title='TSNE1',yaxis_title='TSNE2',zaxis_title='TSNE3',
                    aspectmode='data',dragmode='orbit'))
    pagina=fig.to_html(full_html=True,include_plotlyjs=True,div_id='gmm_comparativo_tsne3d',
                       config={'responsive':True,'displaylogo':False,'scrollZoom':True})
    (destino/NOMES[key][1]).write_text(pagina,encoding='utf-8')


def gerar_geometria_tsne3d(coord, y, modelo_gmm, riscos_gaussianos, destino, *,
    limiar_oof, y_oof, scores_oof, origem='simulado', fracao_f10=None,
    dpi=90, resolucao_cortes=60, resolucao_volume=21, salvar_csv=True):
    """Gera GMM01/04/07/08 em PNG e HTML, sem fit ou selecao de limiar.

    Origem simulado usa OOF sintetico explicitamente marcado. Para f10,
    y_oof/scores_oof DEVEM vir do desenvolvimento; nao do F10.
    """
    X,y,mu,cov,w,riscos,tau,yo,po=_validar(coord,y,modelo_gmm,riscos_gaussianos,
                                             limiar_oof,y_oof,scores_oof,origem,fracao_f10)
    if not 45 <= int(resolucao_cortes) <= 200:
        raise ValueError('Resolucao dos cortes fora de 45..200.')
    if not 15 <= int(resolucao_volume) <= 37:
        raise ValueError('Resolucao volumetrica fora de 15..37.')
    if not 50 <= int(dpi) <= 360:
        raise ValueError('DPI invalido.')
    destino=Path(destino)
    arquivos=[n for par in NOMES.values() for n in par]+[MANIFESTO]
    if salvar_csv: arquivos.append('GMM_TSNE3D_malha_risco.csv')
    if any((destino/n).exists() for n in arquivos):
        raise FileExistsError('Saida existente: nao sobrescrever graficos do experimento.')
    destino.mkdir(parents=True,exist_ok=True)
    mccfix=float(matthews_corrcoef(yo,po>=.5))
    mccstar=float(matthews_corrcoef(yo,po>=tau))
    centro,lim,secoes=_secoes(X,modelo_gmm,riscos,int(resolucao_cortes))
    malha=_malha(lim,modelo_gmm,riscos,int(resolucao_volume))
    rotulo_origem=_rotulo_origem(origem,fracao_f10)
    _densidade(X,modelo_gmm,centro,secoes,malha,destino,int(dpi),rotulo_origem)
    _dominancia(X,y,modelo_gmm,secoes,destino,int(dpi),rotulo_origem)
    for geom in (False,True):
        _painel_fronteiras(X,y,modelo_gmm,secoes,tau,mccfix,mccstar,rotulo_origem,destino,int(dpi),geom)
        _html_fronteiras(X,y,modelo_gmm,malha,tau,mccfix,mccstar,rotulo_origem,destino,geom)
    pts,scores,dens,resp=malha
    if salvar_csv:
        tab=pd.DataFrame(pts,columns=EIXOS)
        tab['score_risco']=scores
        tab['densidade_conjunta']=dens
        tab['gaussiana_dominante']=resp.argmax(axis=1)+1
        tab['classe_tau_050']=(scores>=.5).astype(int)
        tab['classe_tau_oof']=(scores>=tau).astype(int)
        tab['regiao_mudanca']=(tab['classe_tau_050']!=tab['classe_tau_oof']).astype(int)
        tab.to_csv(destino/'GMM_TSNE3D_malha_risco.csv',index=False,float_format='%.17g')
    sha={n:sha256((destino/n).read_bytes()).hexdigest() for n in arquivos if n!=MANIFESTO}
    manifesto={
        'fase':'7.3','metodo':'TSNE','dimensoes':3,'origem':origem,
        'avisos':['F10 e exploratorio; nao usar para selecionar modelo/limiar'] if origem=='f10' else [],
        'fracao_f10_dentro_do_fold':float(fracao_f10) if origem=='f10' else None,
        'n_observacoes':int(len(X)),'n_gaussianas':int(len(w)),
        'limiar_oof_congelado':tau,'mcc_tau_fixo_origem_oof':mccfix,
        'mcc_tau_estrela_origem_oof':mccstar,
        'origem_mcc':'simulado' if origem=='simulado' else 'oof_f1_a_f9',
        'nenhum_modelo_ajustado_durante_geracao':True,'nenhum_limiar_otimizado':True,
        'sha256_coordenadas':sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
        'sha256_parametros_gmm':sha256(np.ascontiguousarray(np.concatenate((mu.ravel(),cov.ravel(),w))).tobytes()).hexdigest(),
        'sha256_riscos_gaussianos':sha256(np.ascontiguousarray(riscos).tobytes()).hexdigest(),
        'sha256_predicoes_oof_fornecidas':sha256(np.ascontiguousarray(np.column_stack((yo,po))).tobytes()).hexdigest(),
        'hashes_sha256':sha,
    }
    (destino/MANIFESTO).write_text(json.dumps(manifesto,indent=2,ensure_ascii=False),encoding='utf-8')
    return {n:destino/n for n in arquivos}


def _demo(destino):
    from sklearn.mixture import GaussianMixture
    rng=np.random.default_rng(42)
    X=np.vstack([rng.normal(loc=(0,0,0),scale=.8,size=(115,3)),
                 rng.normal(loc=(2,2,-1),scale=.55,size=(75,3)),
                 rng.normal(loc=(-1,1,2),scale=.4,size=(70,3))])
    y=(rng.uniform(size=len(X)) < .08).astype(int)
    modelo=GaussianMixture(n_components=4,covariance_type='full',n_init=2,random_state=42).fit(X)
    risco=np.array([.012,.17,.41,.69])
    prob=modelo.predict_proba(X) @ risco
    return gerar_geometria_tsne3d(pd.DataFrame(X,columns=EIXOS),y,modelo,risco,destino,
          limiar_oof=.12,y_oof=y,scores_oof=prob,origem='simulado',
          resolucao_cortes=50,resolucao_volume=17,dpi=75)


def main():
    p=argparse.ArgumentParser(description='Fase 7.3: previa sintetica GMM FULL t-SNE 3D')
    p.add_argument('--demo',action='store_true')
    p.add_argument('--saida',type=Path,required=True)
    a=p.parse_args()
    if not a.demo:
        p.error('Somente --demo liberado. Integracao F10 do t-SNE 3D ainda nao implementada.')
    for nome, caminho in _demo(a.saida).items():
        print('[GERADO]',caminho)

if __name__=='__main__':
    main()
