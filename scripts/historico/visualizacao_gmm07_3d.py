# -*- coding: utf-8 -*-
"""GMM07 PCA 3D: PNG organizado + HTML Plotly offline, a partir do vencedor.

Nao ajusta modelos, nao seleciona cortes, nao utiliza F10.
O mesmo risco s(x) e os mesmos MCCs sao empregados nos dois limiares.
"""
from __future__ import annotations

from pathlib import Path
import html as html_lib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import plotly.graph_objects as go
from scipy.stats import chi2

from src import visualizacao_geometrica as geo
from src import visualizacao_gmm07 as gmm07

PREFIXO = 'GMM07_sensibilidade_3d_comparativo'
CORES_GAUSS = ('#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
              '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf')


def _elipsoide_95(media, cov, nu=32, nv=21):
    autovalores, autovetores = np.linalg.eigh(np.asarray(cov, dtype=float))
    if not np.all(autovalores > 0):
        raise ValueError('GMM07 3D exige matrizes FULL positivas definidas.')
    a = np.linspace(0, 2*np.pi, nu)
    b = np.linspace(0, np.pi, nv)
    esfera = np.stack((np.outer(np.cos(a),np.sin(b)),
                       np.outer(np.sin(a),np.sin(b)),
                       np.outer(np.ones_like(a),np.cos(b))), axis=-1)
    matriz = autovetores @ np.diag(np.sqrt(autovalores*chi2.ppf(.95,df=3)))
    p = np.einsum('ij,uvj->uvi', matriz, esfera) + media
    return p[...,0],p[...,1],p[...,2]


def _png(config, data, escolha, planos, medias, destino, ext, dpi, avaliacao,
         mcc_fixo, mcc_otimo):
    X = data['coord'].to_numpy(dtype=float)
    y = np.asarray(data['y'], dtype=np.int8)
    tau = float(escolha['limiar_mcc_oof'])
    nome_mcc, nota = gmm07._rotulo_mcc(avaliacao)
    if len(planos) != 3 or X.shape[1] != 3:
        raise ValueError('GMM07 3D requer os tres cortes PC1-PC2, PC1-PC3, PC2-PC3.')
    alto = max(max(float(np.max(pl[2])) for pl in planos),1e-8)
    norm = Normalize(vmin=0,vmax=alto)
    op = config['visualizacao']['graficos']['sensibilidade_fronteira']
    idx = geo._amostra_plot(len(y), y, config, int(op.get('n_pontos_scatter',1500)))
    ilegit = idx[y[idx]==0]
    ifraude = idx[y[idx]==1]
    nomes=list(data['coord'].columns)

    fig=plt.figure(figsize=(19.6,11.5),facecolor='white')
    cols=(.063,.349,.635)
    linhas=(.507,.155)
    for linha,limiar in enumerate((.5,tau)):
        for coluna,(par,grade,risco,pred_f,pred_m,mudou) in enumerate(planos):
            ax=fig.add_axes([cols[coluna],linhas[linha],.239,.254])
            xx,yy=grade
            ax.pcolormesh(xx,yy,risco,cmap='viridis',norm=norm,
                          shading='auto',rasterized=True,zorder=0)
            if np.any(mudou):
                h=ax.contourf(xx,yy,mudou.astype(float),levels=[.5,1.5],
                              colors='none',hatches=['////'],zorder=2)
                h.set_edgecolor('#a77a11')
                h.set_linewidth(.11)
            if len(ilegit):
                ax.scatter(X[ilegit,par[0]],X[ilegit,par[1]],c='#2477b5',
                           marker='o',s=7.5,alpha=.25,linewidths=0,
                           rasterized=True,zorder=3)
            if len(ifraude):
                ax.scatter(X[ifraude,par[0]],X[ifraude,par[1]],c='#dc3636',
                           marker='*',s=29,alpha=.75,linewidths=.1,
                           rasterized=True,zorder=4)
            if float(np.min(risco)) < limiar < float(np.max(risco)):
                ax.contour(xx,yy,risco,levels=[limiar],colors=['#161616'],
                           linewidths=1.9,zorder=5)
            else:
                ax.text(.025,.04,f'Sem cruzamento com τ = {limiar:.7f}',
                        transform=ax.transAxes,fontsize=8.4,
                        bbox=dict(facecolor='white',alpha=.92,
                                  edgecolor='#d9dee5',boxstyle='round,pad=.25'))
            ax.set_xlim(xx[0,0],xx[0,-1])
            ax.set_ylim(yy[0,0],yy[-1,0])
            ax.grid(alpha=.085)
            ax.set_xlabel(nomes[par[0]],fontsize=9.5,labelpad=3)
            ax.set_ylabel(nomes[par[1]],fontsize=9.5,labelpad=3)
            ax.tick_params(labelsize=8.2)
            omitida=next(i for i in range(3) if i not in par)
            ax.set_title(f'{nomes[par[0]]} × {nomes[par[1]]}  |  '
                         f'{nomes[omitida]} = {medias[omitida]:.7f}',
                         fontsize=10.2,pad=8,fontweight='medium')
    fig.text(.5,.979,'SENSIBILIDADE DA FRONTEIRA AO LIMIAR',ha='center',va='top',
             fontsize=19,fontweight='bold',color='#14243b')
    fig.text(.5,.939,f'FULL | PCA 3D | K = {data["gmm"].modelo_.n_components}',
             ha='center',fontsize=10.5,color='#687b92')
    handles=[
        Line2D([0],[0],color='#181818',lw=2,label='Fronteira (s(x) = τ)'),
        Patch(facecolor='none',edgecolor='#a77a11',hatch='////',label='Região de mudança'),
        Line2D([0],[0],marker='o',color='#2477b5',linestyle='',markersize=5,
               label='Legítimas (projeção)'),
        Line2D([0],[0],marker='*',color='#dc3636',linestyle='',markersize=7,
               label='Fraudes (projeção)')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.909),
               ncol=4,frameon=False,fontsize=10,columnspacing=2.5)
    fig.text(.5,.811,f'A — CORTE FIXO  |  τ = {0.5:.7f}  |  '
             f'{nome_mcc} = {mcc_fixo:.7f}',ha='center',va='center',
             fontsize=11.6,fontweight='bold',color='#263b50')
    fig.text(.5,.459,f'B — LIMIAR ÓTIMO DO MCC  |  τ* = {tau:.7f}  |  '
             f'{nome_mcc} = {mcc_otimo:.7f}',ha='center',va='center',
             fontsize=11.6,fontweight='bold',color='#263b50')
    cbax=fig.add_axes([.906,.156,.013,.605])
    colorbar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),cax=cbax)
    colorbar.ax.tick_params(labelsize=9)
    colorbar.set_label('Score de risco s(x) — mesma escala',fontsize=10,labelpad=10)
    direcao=('legítima → fraude' if tau < .5 else
             'fraude → legítima' if tau > .5 else 'nenhuma mudança')
    fig.text(.5,.086,f'Região hachurada: mudança de classe ({direcao}); '
             'mesma superfície de risco nos dois limiares.',
             ha='center',fontsize=10,fontweight='medium',color='#233c57')
    fig.text(.5,.056,f'{nota} Planos fixados na média do desenvolvimento.',
             ha='center',fontsize=9.3,color='#596d84')
    saida = Path(destino)/(PREFIXO+'.'+ext)
    try:
        fig.savefig(saida,dpi=dpi,facecolor='white')
    finally:
        plt.close(fig)
    return saida


def _html(config, data, escolha, destino, avaliacao, mcc_fixo, mcc_otimo):
    X=data['coord'].to_numpy(dtype=float)
    y=np.asarray(data['y'],dtype=np.int8)
    gm,rh=data['gmm'],data['rh']
    modelo=gm.modelo_
    ncomp=int(modelo.n_components)
    if modelo.covariance_type != 'full' or X.shape[1]!=3:
        raise ValueError('HTML GMM07 requer GMM FULL e PCA 3D.')
    if len(X)!=len(y) or not set(np.unique(y)).issubset({0,1}):
        raise ValueError('Rotulos de desenvolvimento invalidos para GMM07 HTML.')
    tau=float(escolha['limiar_mcc_oof'])
    if not 0 <= tau <= 1:
        raise ValueError('Limiar tau* invalido.')
    nome_mcc, nota=gmm07._rotulo_mcc(avaliacao)
    op=config['visualizacao']['graficos']['sensibilidade_fronteira']
    n=int(op.get('resolucao_volume_html_3d', 35))
    n=max(20,min(45,n))
    margem=float(op.get('margem',.15))
    ranges=[geo._faixa(X[:,i],margem) for i in range(3)]
    axes=[np.linspace(a,b,n) for a,b in ranges]
    xx,yy,zz=np.meshgrid(*axes,indexing='ij')
    pontos=np.c_[xx.ravel(),yy.ravel(),zz.ravel()]
    risco=np.asarray(geo._pontuar_gmm_array(gm,rh,pontos),dtype=float)
    if not np.isfinite(risco).all():
        raise ValueError('Risco nao finito na malha volumetrica GMM07.')
    classe_f,classe_m,mudou=geo._mudanca_classes(risco,tau)
    if not np.array_equal(mudou,classe_f!=classe_m):
        raise ValueError('Inconsistencia entre classes e regiao de mudanca.')
    fig=go.Figure()
    layer_idx={}
    def add(key,trace,visible):
        fig.add_trace(trace)
        layer_idx[key]=len(fig.data)-1
    vmin,vmax=float(risco.min()),float(risco.max())
    largura=max((vmax-vmin)*0.00002,1e-8)
    for key,limiar,cor,nome in (
        ('fixo',.5,'#253446','Fronteira fixa τ = 0,5000000'),
        ('otimo',tau,'#e49b29',f'Fronteira ótima τ* = {tau:.7f}')):
        if vmin < limiar < vmax:
            add(key,go.Isosurface(
                x=pontos[:,0],y=pontos[:,1],z=pontos[:,2],value=risco,
                isomin=limiar-largura,isomax=limiar+largura,
                surface_count=1,colorscale=[[0,cor],[1,cor]],
                showscale=False,opacity=.33,
                caps=dict(x_show=False,y_show=False,z_show=False),
                name=nome,showlegend=False,
                hovertemplate='PC1=%{x:.3f}<br>PC2=%{y:.3f}'
                              '<br>PC3=%{z:.3f}<br>Risco=%{value:.7f}<extra></extra>'),True)
        else:
            layer_idx[key]=None
    indices=np.flatnonzero(mudou)
    if len(indices)>3000:
        indices=np.random.default_rng(20261027).choice(indices,3000,replace=False)
    add('mudanca',go.Scatter3d(
        x=pontos[indices,0],y=pontos[indices,1],z=pontos[indices,2],
        mode='markers',marker=dict(size=2,color='#c89a1c',opacity=.22),
        name='Amostra da malha que muda de classe',visible=False,
        showlegend=False,customdata=risco[indices],
        hovertemplate='PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}'
                      '<br>Risco=%{customdata:.7f}<extra></extra>'),False)
    for k in range(ncomp):
        cor=CORES_GAUSS[k%len(CORES_GAUSS)]
        cov=geo._covariancia(gm,k)
        E=_elipsoide_95(np.asarray(modelo.means_[k]),cov)
        add(f'g{k+1}',go.Surface(x=E[0],y=E[1],z=E[2],
            surfacecolor=np.zeros_like(E[0]),
            colorscale=[[0,cor],[1,cor]],opacity=.25,showscale=False,
            visible=False,showlegend=False,
            name=f'G{k+1} — elipsoide FULL 95%',
            hovertemplate=f'Elipsoide FULL 95% — G{k+1}<extra></extra>'),False)
    # Mostrar uma subamostra controlada para manter a navegacao responsiva.
    idx=geo._amostra_plot(len(y),y,config,
            int(op.get('n_pontos_scatter_html_3d',min(len(y),2500))))
    for classe,cor,tamanho,opacidade,key,desc in (
        (0,'#2378b9',2.1,.38,'legitimas','Observações legítimas'),
        (1,'#e03535',3.6,.75,'fraudes','Observações fraudulentas')):
        inds=idx[y[idx]==classe]
        add(key,go.Scatter3d(x=X[inds,0],y=X[inds,1],z=X[inds,2],
            mode='markers',marker=dict(color=cor,size=tamanho,opacity=opacidade),
            name=desc,showlegend=False,
            hovertemplate=desc+'<br>PC1=%{x:.5f}<br>PC2=%{y:.5f}'
                               '<br>PC3=%{z:.5f}<extra></extra>'),True)
    for k in range(ncomp):
        cor=CORES_GAUSS[k%len(CORES_GAUSS)]
        c=np.asarray(modelo.means_[k])
        add(f'centro{k+1}',go.Scatter3d(
            x=[float(c[0])],y=[float(c[1])],z=[float(c[2])],
            mode='markers+text',marker=dict(color=cor,size=6,symbol='diamond'),
            text=[f'G{k+1}'],textposition='top center',
            name=f'Centro G{k+1}',showlegend=False,
            hovertemplate=f'Centro G{k+1}<br>PC1=%{{x:.7f}}<br>PC2=%{{y:.7f}}'
                          '<br>PC3=%{{z:.7f}}<extra></extra>'),True)
    fig.update_layout(autosize=True,height=720,margin=dict(l=0,r=0,t=12,b=0),
        scene=dict(xaxis_title='PC1',yaxis_title='PC2',zaxis_title='PC3',
                   aspectmode='data',camera=dict(eye=dict(x=1.55,y=1.6,z=1.15)),
                   bgcolor='rgb(248,250,252)'),paper_bgcolor='white',
        showlegend=False,uirevision='gmm07_oficial')
    graph=fig.to_html(full_html=False,include_plotlyjs=True,div_id='gmm07_3d',
                      config={'responsive':True,'scrollZoom':True,'displaylogo':False})
    gauss_check=''.join(
        f'<label class="chk"><input type="checkbox" data-layer="g{k+1}"> '
        f'<span class="dot" style="background:{CORES_GAUSS[k%len(CORES_GAUSS)]}"></span>'
        f'Gaussiana G{k+1}</label>' for k in range(ncomp))
    escape=html_lib.escape
    html=r'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GMM07 — Sensibilidade da fronteira PCA 3D</title>
<style>
*{box-sizing:border-box}body{font-family:Inter,system-ui,-apple-system,'Segoe UI',sans-serif;margin:0;background:#f3f6fb;color:#17283f}
.container{max-width:1600px;margin:auto;padding:22px}.head{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:20px 24px;margin-bottom:16px}
h1{font-size:clamp(20px,2vw,30px);margin:0 0 8px}.sub{color:#65758d;font-size:14px;margin:0}
.metric{display:inline-block;padding:7px 11px;border:1px solid #dfe6f0;border-radius:8px;margin:10px 8px 0 0;background:#fafcff;font-variant-numeric:tabular-nums;font-size:13px}
.layout{display:grid;grid-template-columns:260px minmax(0,1fr);gap:15px;align-items:start}
.panel,.viz{border-radius:14px;background:white;border:1px solid #e2e8f0;box-shadow:0 2px 6px #15233b0b}
.panel{padding:17px;display:grid;gap:15px}h2{font-size:13px;text-transform:uppercase;letter-spacing:.055em;color:#607187;margin:0 0 7px}
.chk{display:flex;align-items:center;gap:9px;padding:7px 4px;font-size:13px;line-height:1.25;cursor:pointer}
.chk input{accent-color:#275ca9;width:16px;height:16px;flex:none}.dot{width:12px;height:12px;border-radius:50%;display:inline-block;flex:none}
.controls{display:flex;flex-wrap:wrap;gap:7px}.controls button{background:#f5f8fc;border:1px solid #cad6e6;padding:9px 10px;border-radius:7px;cursor:pointer;color:#233c58;font-weight:550}
.controls button:hover{background:#e7eef8}.viz{min-width:0;padding:4px 7px 10px}
#gmm07_3d{width:100%!important;height:75vh!important;min-height:580px;max-height:900px}
.note{font-size:12px;color:#64758b;margin:8px 15px 3px;line-height:1.5}
@media(max-width:900px){.layout{grid-template-columns:1fr}.panel{grid-template-columns:repeat(auto-fit,minmax(205px,1fr))}.container{padding:10px}#gmm07_3d{height:67vh!important;min-height:480px}}
</style></head><body><div class="container"><header class="head">
<h1>SENSIBILIDADE DA FRONTEIRA AO LIMIAR</h1>
<p class="sub">PCA 3D · GMM FULL · K=__K__ · __NOTA__</p>
<div class="metric">τ fixo: <b>0,5000000</b> · __MCCNOME__: <b>__MCCFIX__</b></div>
<div class="metric">τ* ótimo: <b>__TAU__</b> · __MCCNOME__: <b>__MCCSTAR__</b></div>
</header><div class="layout"><aside class="panel">
<section><h2>Fronteiras de decisão</h2>
<label class="chk"><input type="checkbox" data-layer="fixo" checked> τ = 0,50</label>
<label class="chk"><input type="checkbox" data-layer="otimo" checked> τ* (MCC ótimo)</label>
<label class="chk"><input type="checkbox" data-layer="mudanca"><span class="dot" style="background:#e4b63c"></span> Região de mudança</label>
</section><section><h2>Elipsoides FULL de 95%</h2>__GAUSSIANAS__</section>
<section><h2>Observações do desenvolvimento</h2>
<label class="chk"><input type="checkbox" data-layer="legitimas" checked><span class="dot" style="background:#2378b9"></span> Legítimas</label>
<label class="chk"><input type="checkbox" data-layer="fraudes" checked><span class="dot" style="background:#e03535"></span> Fraudes</label>
<label class="chk"><input type="checkbox" data-layer="centros" checked> Centros gaussianos</label>
</section><section><h2>Visualização</h2><div class="controls">
<button id="auto">▶ Girar</button><button id="reset">Restaurar câmera</button>
<button id="todos">Mostrar tudo</button><button id="nenhum">Limpar camadas</button>
</div></section></aside><main class="viz">__CHART__
<p class="note">Arraste a cena para girar e use a roda do mouse para zoom. Os elipsoides têm 95% de concentração gaussiana e <b>não</b> indicam regiões de fraude. A região de mudança é uma amostra da malha volumétrica; a superfície de risco é a mesma em ambos os limiares.</p>
</main></div></div><script>
(function(){const gd=document.getElementById('gmm07_3d');const traceIndex=__TRACE_INDEX__;
const boxes=Array.from(document.querySelectorAll('[data-layer]'));
function indices(key){if(key==='centros')return Object.entries(traceIndex).filter(([k,v])=>k.startsWith('centro')&&v!==null).map(([k,v])=>v);
const i=traceIndex[key];return i===null||i===undefined?[]:[i];}
function sync(key,val){for(const i of indices(key))Plotly.restyle(gd,{visible:val},[i]);}
boxes.forEach(el=>{const key=el.dataset.layer;if(indices(key).length===0){el.disabled=true;el.parentElement.title='Sem geometria disponivel nesse corte';}el.addEventListener('change',()=>sync(key,el.checked));});
function bulk(val){for(const el of boxes){if(el.disabled)continue;el.checked=val;sync(el.dataset.layer,val);}}
document.getElementById('todos').onclick=()=>bulk(true);
document.getElementById('nenhum').onclick=()=>bulk(false);
const camera={eye:{x:1.55,y:1.6,z:1.15}},b=document.getElementById('auto');let timer=null,angle=0;
function stop(){if(timer){clearInterval(timer);timer=null;}b.textContent='▶ Girar';}
b.onclick=()=>{if(timer){stop();return;}b.textContent='■ Parar rotação';timer=setInterval(()=>{angle+=.028;Plotly.relayout(gd,{'scene.camera.eye':{x:2.35*Math.cos(angle),y:2.35*Math.sin(angle),z:1.18}});},100);};
document.getElementById('reset').onclick=()=>{stop();Plotly.relayout(gd,{'scene.camera':camera});};
window.addEventListener('beforeunload',stop);
})();</script></body></html>'''
    for key,value in {
        '__K__':str(ncomp),'__NOTA__':escape(nota),
        '__MCCNOME__':escape(nome_mcc),
        '__MCCFIX__':f'{mcc_fixo:.7f}',
        '__MCCSTAR__':f'{mcc_otimo:.7f}',
        '__TAU__':f'{tau:.7f}',
        '__GAUSSIANAS__':gauss_check,
        '__CHART__':graph,
        '__TRACE_INDEX__':json.dumps(layer_idx),
    }.items():
        html=html.replace(key,value)
    if any(marker in html for marker in ('__CHART__','__TRACE_INDEX__','__MCCFIX__')):
        raise ValueError('Template GMM07 HTML nao resolvido.')
    arquivo=Path(destino)/(PREFIXO+'_INTERATIVA.html')
    arquivo.write_text(html,encoding='utf-8')
    return arquivo


def gerar_3d(config,data,escolha,planos,medias,destino,ext,dpi,avaliacao,
             mcc_fixo,mcc_otimo):
    destino=Path(destino)
    destino.mkdir(parents=True,exist_ok=True)
    _png(config,data,escolha,planos,medias,destino,ext,dpi,avaliacao,
         mcc_fixo,mcc_otimo)
    _html(config,data,escolha,destino,avaliacao,mcc_fixo,mcc_otimo)
    return PREFIXO
