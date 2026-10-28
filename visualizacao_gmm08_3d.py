# -*- coding: utf-8 -*-
"""GMM08 em PCA 3D: PNG comparativo e HTML rotativo do modelo vencedor.

Usa exclusivamente o GMM e os riscos ajustados no desenvolvimento F1-F9.
Não treina, não seleciona limiar, não utiliza F10.
"""
from __future__ import annotations
from pathlib import Path
import html as html_lib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Ellipse
from matplotlib.colors import Normalize
import plotly.graph_objects as go
from scipy.stats import chi2
from src import visualizacao_geometrica as geo

PREFIXO='GMM08_mapa_risco_3d_comparativo'
CORES=('#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
       '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf')

def _origem_mcc(avaliacao):
    origem=avaliacao.get('origem')
    if origem=='oof':
        return 'MCC OOF','Resultados OOF F1-F9; sem F10'
    if origem=='simulado':
        return 'MCC da simulação','Prévia sintética; não é OOF real'
    return 'MCC (origem desconhecida)','Origem não informada; não afirmar que é OOF'

def png_organizado(config, data, escolha, avaliacao, destino, ext, dpi):
    """Reorganiza SOMENTE o PNG da pasta de teste, usando os mesmos scores FULL."""
    X = data['coord'].to_numpy(dtype=float)
    y = np.asarray(data['y'], dtype=np.int8)
    tau = float(escolha['limiar_mcc_oof'])
    mcc_fixo, mcc_otimo = geo._avaliar_cortes_oof(avaliacao, tau)
    op = config['visualizacao']['graficos']['mapa_risco_fronteira']
    n = int(op.get('resolucao_2d', 180))
    margem = float(op.get('margem', .15))
    media = X.mean(axis=0)
    planos = []
    for par in [(0,1),(0,2),(1,2)]:
        i,j=par
        xx,yy=np.meshgrid(np.linspace(*geo._faixa(X[:,i],margem),n),
                          np.linspace(*geo._faixa(X[:,j],margem),n))
        pts=np.tile(media,(n*n,1))
        pts[:,i]=xx.ravel();pts[:,j]=yy.ravel()
        risk=np.asarray(geo._pontuar_gmm_array(data['gmm'],data['rh'],pts)).reshape(n,n)
        fixo,otimo,mudou=geo._mudanca_classes(risk,tau)
        assert np.array_equal(mudou,fixo!=otimo)
        planos.append((par,(xx,yy),risk,mudou))
    maior=max(float(np.max(plano[2])) for plano in planos)
    assert maior>0
    norm=Normalize(vmin=0,vmax=maior)
    rng=np.random.default_rng(20261027)
    ix0=np.flatnonzero(y==0);ix1=np.flatnonzero(y==1)
    if len(ix0)>1000:ix0=rng.choice(ix0,1000,replace=False)
    if len(ix1)>450:ix1=rng.choice(ix1,450,replace=False)
    fig=plt.figure(figsize=(19.7,12.2),facecolor='white')
    cols=[.058,.345,.632];linhas=[.52,.17]
    largura,altura=.235,.248
    cores = CORES
    for row,threshold in enumerate((.5,tau)):
        for col,(par,grade,risk,mudou) in enumerate(planos):
            xx,yy=grade
            ax=fig.add_axes([cols[col],linhas[row],largura,altura])
            ax.pcolormesh(xx,yy,risk,shading='auto',cmap='viridis',norm=norm,
                          rasterized=True,zorder=0)
            if np.any(mudou):
                hatch=ax.contourf(xx,yy,mudou.astype(float),levels=[.5,1.5],
                                 colors='none',hatches=['////'],zorder=1)
                hatch.set_edgecolor('#bc880a');hatch.set_linewidth(.09)
            ax.scatter(X[ix0,par[0]],X[ix0,par[1]],s=7,c='#2879b6',
                       alpha=.3,linewidths=0,rasterized=True,zorder=3)
            ax.scatter(X[ix1,par[0]],X[ix1,par[1]],s=27,c='#df3939',marker='*',
                       alpha=.85,linewidths=.12,rasterized=True,zorder=4)
            if op.get('mostrar_gaussianas',True):
                for k in range(data['gmm'].modelo_.n_components):
                    recorte=geo._elipse_95_corte(data['gmm'],k,par,media)
                    if recorte is None:
                        continue
                    c,w,h,ang=recorte
                    ax.add_patch(Ellipse(c,w,h,angle=ang,ec=cores[k % len(cores)],fc='none',
                                         linestyle='--',lw=1.75,zorder=5))
                    # Projection of the 3D mean may not lie on the actual 95% slice.
                    ax.plot(data['gmm'].modelo_.means_[k,par[0]],
                            data['gmm'].modelo_.means_[k,par[1]],
                            marker='x',ms=6.5,color=cores[k % len(cores)],mew=1.7,zorder=6)
            if float(risk.min())<threshold<float(risk.max()):
                ax.contour(xx,yy,risk,levels=[threshold],colors=['#182130'],
                           linewidths=2.05,zorder=7)
            else:
                ax.text(.026,.04,f'Sem contorno para τ = {threshold:.7f}',
                        transform=ax.transAxes,fontsize=8.2,zorder=8,
                        bbox=dict(boxstyle='round,pad=.25',facecolor='white',
                                  edgecolor='#d2dae5',alpha=.94))
            ax.set_xlim(xx[0,0],xx[0,-1]);ax.set_ylim(yy[0,0],yy[-1,0])
            ax.set_xlabel(f'PC{par[0]+1}',fontsize=9.5,labelpad=3)
            ax.set_ylabel(f'PC{par[1]+1}',fontsize=9.5,labelpad=3)
            ax.tick_params(labelsize=8.4)
            faltando=next(k for k in range(3) if k not in par)
            ax.set_title(f'PC{par[0]+1} × PC{par[1]+1}  |  '
                         f'PC{faltando+1} = {media[faltando]:.7f}',
                         fontsize=10,pad=9)
            ax.grid(alpha=.08)
    fig.text(.5,.983,'MAPA DE RISCO E GEOMETRIA GAUSSIANA (95%)',
             ha='center',va='top',fontsize=19,fontweight='bold',color='#14243b')
    origem_mcc, nota_mcc = _origem_mcc(avaliacao)
    fig.text(.5,.943,f'PCA 3D  |  GMM FULL  |  K = {data["gmm"].modelo_.n_components}  |  {nota_mcc}',
             ha='center',fontsize=10.6,color='#657990')
    handles=[
        Line2D([0],[0],color='#182130',lw=2,label='Fronteira de decisão'),
        Patch(facecolor='none',edgecolor='#bc880a',hatch='////',label='Região de mudança'),
        Line2D([0],[0],marker='o',ls='',color='#2879b6',markersize=5,label='Legítimas'),
        Line2D([0],[0],marker='*',ls='',color='#df3939',markersize=7,label='Fraudes')]
    handles.extend(Line2D([0],[0],ls='--',color=cores[k % len(cores)],lw=1.75,
                          label=f'G{k+1} — seção FULL 95%')
                   for k in range(data['gmm'].modelo_.n_components))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.49,.907),
               ncol=7,frameon=False,fontsize=9.2,columnspacing=1.6,
               handlelength=2.2)
    fig.text(.5,.817,
             f'A — CORTE FIXO  |  τ = {0.5:.7f}  |  {origem_mcc} = {mcc_fixo:.7f}',
             ha='center',va='center',fontsize=11.5,weight='bold',color='#263c53')
    fig.text(.5,.465,
             f'B — LIMIAR ÓTIMO DO MCC  |  τ* = {tau:.7f}  |  '
             f'{origem_mcc} = {mcc_otimo:.7f}',
             ha='center',va='center',fontsize=11.5,weight='bold',color='#263c53')
    cax=fig.add_axes([.908,.172,.012,.595])
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),cax=cax)
    bar.ax.tick_params(labelsize=9)
    bar.set_label('Probabilidade estimada de fraude — mesma escala',fontsize=10,
                  labelpad=9)
    direcao=('legítima → fraude' if tau < .5 else
             'fraude → legítima' if tau > .5 else 'nenhuma mudança')
    fig.text(.5,.091,
             f'Hachura: mudança de classe ({direcao}); mesma superfície de risco '
             'nos dois limiares.',
             ha='center',fontsize=10,color='#233c57',weight='bold')
    fig.text(.5,.061,
             f'{nota_mcc}. Planos com coordenada omitida fixada na média do desenvolvimento.',
             ha='center',fontsize=9.2,color='#596d84')
    fig.text(.5,.037,
             'Elipses tracejadas: seções dos elipsoides FULL 95%; '
             'não representam regiões de fraude. F10 não utilizado.',
             ha='center',fontsize=9,color='#596d84')
    png=destino/(PREFIXO+"."+ext)
    fig.savefig(png,dpi=dpi,facecolor='white')
    plt.close(fig)
    return png


def elipsoide_95(media, cov, nu=39, nv=25):
    """Superficie de Mahalanobis q = chi2.ppf(.95, df=3), sem aproximar FULL."""
    autovalores, autovetores = np.linalg.eigh(cov)
    assert np.all(autovalores > 0), 'Matriz FULL nao positiva definida.'
    a = np.linspace(0,2*np.pi,nu)
    b = np.linspace(0,np.pi,nv)
    esfera = np.stack([
        np.outer(np.cos(a),np.sin(b)),
        np.outer(np.sin(a),np.sin(b)),
        np.outer(np.ones_like(a),np.cos(b))], axis=-1)
    escala = chi2.ppf(.95,df=3)
    transformacao = autovetores @ np.diag(np.sqrt(autovalores*escala))
    p = np.einsum('ij,uvj->uvi',transformacao,esfera)+media
    q = np.einsum('...i,ij,...j->...',p-media,np.linalg.inv(cov),p-media)
    assert np.max(np.abs(q-escala)) < 1e-7
    return p[...,0],p[...,1],p[...,2]


def html_interativo_v2(config,data,escolha,avaliacao,destino,csv):
    X=data['coord'].to_numpy(dtype=float)
    y=np.asarray(data['y'],dtype=np.int8)
    gm,rh=data['gmm'],data['rh']
    tau=float(escolha['limiar_mcc_oof'])
    mcc_fixo,mcc_tau=geo._avaliar_cortes_oof(avaliacao,tau)
    limites=[geo._faixa(X[:,j],.13) for j in range(3)]
    n=int(config['visualizacao']['graficos']['mapa_risco_fronteira'].get('resolucao_volume_html_3d', 35))
    if not 18 <= n <= 80:
        raise ValueError('Resolução volumétrica deve estar entre 18 e 80.')
    axes=[np.linspace(*limites[j],n) for j in range(3)]
    xx,yy,zz=np.meshgrid(*axes,indexing='ij')
    pontos=np.c_[xx.ravel(),yy.ravel(),zz.ravel()]
    risco=np.asarray(geo._pontuar_gmm_array(gm,rh,pontos))
    fixo,otimo,mudou=geo._mudanca_classes(risco,tau)
    assert np.array_equal(mudou,fixo!=otimo)
    assert np.all(np.isfinite(risco))
    if csv:
        pd.DataFrame({
        'PC1':pontos[:,0],'PC2':pontos[:,1],'PC3':pontos[:,2],
        'score_risco':risco,
        'classe_corte_050':fixo.astype(np.uint8),
        'classe_corte_mcc':otimo.astype(np.uint8),
        'mudou_classe':mudou.astype(np.uint8),
        }).to_csv(destino/'GMM08_malha_volumetrica_3d.csv',index=False,float_format='%.17g')

    fig=go.Figure()
    componentes=[]
    def add(kind, trace, enabled):
        fig.add_trace(trace)
        componentes.append(dict(key=kind,index=len(fig.data)-1,checked=bool(enabled)))

    # Volume probabilistico com cores de risco: amostra volumetrica FULL.
    add('volume',go.Volume(
        x=pontos[:,0],y=pontos[:,1],z=pontos[:,2],value=risco,
        isomin=float(np.min(risco)),isomax=float(np.max(risco)),
        surface_count=9,opacity=.065,colorscale='Viridis',
        showscale=True,colorbar=dict(title='Risco s(x)',len=.7,thickness=13),
        caps=dict(x_show=False,y_show=False,z_show=False),
        visible=False,showlegend=False,name='Volume de risco estimado',
        hovertemplate='PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}'
                      '<br>Risco=%{value:.7f}<extra></extra>'
    ),False)

    amplitude=float(risco.max()-risco.min())
    largura=min(.0005,max(amplitude*.00002,1e-7))
    for key,limiar,cor,nome in (
        ('fixo',.5,'#253446','Fronteira fixa τ = 0,5000000'),
        ('otimo',tau,'#e49b29',f'Fronteira ótima τ* = {tau:.7f}'),
    ):
        if float(risco.min()) < limiar < float(risco.max()):
            add(key,go.Isosurface(
                x=pontos[:,0],y=pontos[:,1],z=pontos[:,2],value=risco,
                isomin=limiar-largura,isomax=limiar+largura,
                surface_count=1,colorscale=[[0,cor],[1,cor]],
                showscale=False,opacity=.33,
                caps=dict(x_show=False,y_show=False,z_show=False),
                name=nome,showlegend=False,
                hovertemplate='PC1=%{x:.3f}<br>PC2=%{y:.3f}'
                              '<br>PC3=%{z:.3f}<br>Risco=%{value:.7f}<extra></extra>'
            ),True)
        else:
            componentes.append(dict(key=key,index=None,checked=False))

    idx=np.flatnonzero(mudou)
    if len(idx)>3000:
        idx=np.random.default_rng(20261027).choice(idx,3000,replace=False)
    add('mudanca',go.Scatter3d(
        x=pontos[idx,0],y=pontos[idx,1],z=pontos[idx,2],
        mode='markers',marker=dict(size=2,color='#c89a1c',opacity=.22),
        name='Pontos da malha que mudam de classe',visible=False,
        showlegend=False,
        customdata=risco[idx],
        hovertemplate='PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}'
                      '<br>Risco=%{customdata:.7f}<extra></extra>'
    ),False)

    cores = CORES
    modelo=gm.modelo_
    for k in range(modelo.n_components):
        E=elipsoide_95(modelo.means_[k],modelo.covariances_[k])
        add('g'+str(k+1),go.Surface(
            x=E[0],y=E[1],z=E[2],
            surfacecolor=np.zeros_like(E[0]),
            colorscale=[[0,cores[k % len(cores)]],[1,cores[k % len(cores)]]],
            opacity=.25,showscale=False,visible=False,showlegend=False,
            name=f'G{k+1} — elipsoide FULL 95%',
            hovertemplate=f'Elipsoide FULL 95% — G{k+1}<extra></extra>'
        ),False)

    # Observacoes reais do desenvolvimento, com amostragem somente visual.
    for classe,cor,tamanho,opacidade,key,desc in (
        (0,'#2378b9',2.1,.38,'legitimas','Observações legítimas'),
        (1,'#e03535',3.6,.75,'fraudes','Observações fraudulentas'),
    ):
        inds=np.flatnonzero(y==classe)
        max_pontos=int(config['visualizacao']['graficos']['mapa_risco_fronteira'].get('n_pontos_scatter_html_3d', 1800))
        if len(inds)>max_pontos:
            inds=np.random.default_rng(20261027+classe).choice(inds,max_pontos,replace=False)
        add(key,go.Scatter3d(
            x=X[inds,0],y=X[inds,1],z=X[inds,2],
            mode='markers',
            marker=dict(color=cor,size=tamanho,opacity=opacidade),
            name=desc,showlegend=False,
            hovertemplate=desc+'<br>PC1=%{x:.5f}<br>PC2=%{y:.5f}'
                               '<br>PC3=%{z:.5f}<extra></extra>'
        ),True)

    for k in range(modelo.n_components):
        cor=cores[k % len(cores)]
        centro=modelo.means_[k]
        add('centro'+str(k+1),go.Scatter3d(
            x=[float(centro[0])],y=[float(centro[1])],z=[float(centro[2])],
            mode='markers+text',marker=dict(color=cor,size=6,symbol='diamond'),
            text=[f'G{k+1}'],textposition='top center',
            name=f'Centro de G{k+1}',showlegend=False,
            hovertemplate=f'Centro G{k+1}<br>PC1=%{{x:.7f}}'
                          '<br>PC2=%{y:.7f}<br>PC3=%{z:.7f}<extra></extra>'
        ),True)

    fig.update_layout(
        autosize=True,height=720,
        margin=dict(l=0,r=0,t=12,b=0),
        scene=dict(xaxis_title='PC1',yaxis_title='PC2',zaxis_title='PC3',
                   aspectmode='data',
                   camera=dict(eye=dict(x=1.55,y=1.6,z=1.15)),
                   bgcolor='rgb(248,250,252)'),
        paper_bgcolor='white',showlegend=False,uirevision='gmm08_auditoria'
    )
    # HTML offline: a legenda visual e os botoes ficam FORA da cena 3D.
    chart=fig.to_html(full_html=False,include_plotlyjs=True,div_id='gmm08_3d',
                      config={'responsive':True,'scrollZoom':True,'displaylogo':False})
    import json
    maps={r['key']:r['index'] for r in componentes}
    html=r'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GMM08 — Mapa de risco PCA 3D</title>
<style>
*{box-sizing:border-box}body{font-family:Inter,system-ui,-apple-system,'Segoe UI',sans-serif;margin:0;background:#f3f6fb;color:#17283f}
.container{max-width:1600px;margin:auto;padding:22px}.head{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:20px 24px;margin-bottom:16px}
h1{font-size:clamp(20px,2vw,30px);margin:0 0 8px} .sub{color:#65758d;font-size:14px;margin:0}
.metric{display:inline-block;padding:7px 11px;border:1px solid #dfe6f0;border-radius:8px;margin:10px 8px 0 0;background:#fafcff;font-variant-numeric:tabular-nums;font-size:13px}
.layout{display:grid;grid-template-columns:260px minmax(0,1fr);gap:15px;align-items:start}
.panel,.viz{border-radius:14px;background:white;border:1px solid #e2e8f0;box-shadow:0 2px 6px #15233b0b}
.panel{padding:17px;display:grid;gap:15px}h2{font-size:13px;text-transform:uppercase;letter-spacing:.055em;color:#607187;margin:0 0 7px}
.chk{display:flex;align-items:center;gap:9px;padding:7px 4px;font-size:13px;line-height:1.25;cursor:pointer}
.chk input{accent-color:#275ca9;width:16px;height:16px;flex:none}.dot{width:12px;height:12px;border-radius:50%;display:inline-block;flex:none}
.controls{display:flex;flex-wrap:wrap;gap:7px}.controls button{background:#f5f8fc;border:1px solid #cad6e6;padding:9px 10px;border-radius:7px;cursor:pointer;color:#233c58;font-weight:550}
.controls button:hover{background:#e7eef8}.viz{min-width:0;padding:4px 7px 10px}
#gmm08_3d{width:100%!important;height:75vh!important;min-height:580px;max-height:900px}
.note{font-size:12px;color:#64758b;margin:8px 15px 3px;line-height:1.5}
@media(max-width:900px){.layout{grid-template-columns:1fr}.panel{grid-template-columns:repeat(auto-fit,minmax(205px,1fr))}.container{padding:10px}#gmm08_3d{height:67vh!important;min-height:480px}}
</style></head><body><div class="container">
<header class="head"><h1>MAPA DE RISCO E GEOMETRIA GAUSSIANA (95%)</h1>
<p class="sub">PCA 3D · GMM FULL · K=__K__ · __ORIGEM__</p>
<div class="metric">τ fixo: <b>0,5000000</b> · __METRIC__: <b>__MCCFIX__</b></div>
<div class="metric">τ* ótimo: <b>__TAU__</b> · __METRIC__: <b>__MCCSTAR__</b></div>
</header><div class="layout"><aside class="panel">
<section><h2>Mapa volumétrico de risco</h2>
<label class="chk"><input type="checkbox" data-layer="volume"> Volume colorido pelo score s(x)</label>
</section>
<section><h2>Fronteiras de decisão</h2>
<label class="chk"><input type="checkbox" data-layer="fixo" checked> τ = 0,50</label>
<label class="chk"><input type="checkbox" data-layer="otimo" checked> τ* (MCC ótimo)</label>
<label class="chk"><input type="checkbox" data-layer="mudanca"> <span class="dot" style="background:#e4b63c"></span>Região de mudança</label>
</section>
<section><h2>Elipsoides FULL de 95%</h2>
__GAUSSIAN_CONTROLS__
</section>
<section><h2>Observações da simulação</h2>
<label class="chk"><input type="checkbox" data-layer="legitimas" checked><span class="dot" style="background:#2378b9"></span> Legítimas</label>
<label class="chk"><input type="checkbox" data-layer="fraudes" checked><span class="dot" style="background:#e03535"></span> Fraudes</label>
<label class="chk"><input type="checkbox" data-layer="centros" checked> Centros das gaussianas</label>
</section>
<section><h2>Visualização</h2><div class="controls">
<button id="auto">▶ Girar</button><button id="reset">Restaurar câmera</button>
<button id="todos">Mostrar tudo</button><button id="nenhum">Limpar camadas</button>
</div></section>
</aside><main class="viz">__CHART__
<p class="note">Arraste a cena para girar e use a roda do mouse para zoom. Os elipsoides têm 95% de concentração gaussiana e <b>não</b> indicam regiões de fraude. A região de mudança corresponde a células da malha que mudam de classe; a superfície de risco é a mesma em ambos os limiares. O volume colorido representa s(x).</p>
</main></div></div>
<script>
(function(){
const gd=document.getElementById('gmm08_3d');const traceIndex=__TRACE_INDEX__;
const boxes=Array.from(document.querySelectorAll('[data-layer]'));
function indices(key){if(key==='centros')return __CENTER_KEYS__.map(k=>traceIndex[k]).filter(i=>i!==null&&i!==undefined);
const i=traceIndex[key];return i===null||i===undefined?[]:[i];}
function sync(key,val){for(const i of indices(key))Plotly.restyle(gd,{visible:val},[i]);}
boxes.forEach(el=>{const key=el.dataset.layer;if(indices(key).length===0){el.disabled=true;el.parentElement.title='Limiar fora da faixa da malha';}
el.addEventListener('change',()=>sync(key,el.checked));});
function bulk(val){for(const el of boxes){if(el.disabled)continue;el.checked=val;sync(el.dataset.layer,val);}}
document.getElementById('todos').onclick=()=>bulk(true);
document.getElementById('nenhum').onclick=()=>bulk(false);
const camera={eye:{x:1.55,y:1.6,z:1.15}},b=document.getElementById('auto');let timer=null,angle=0;
function stop(){if(timer){clearInterval(timer);timer=null;}b.textContent='▶ Girar';}
b.onclick=()=>{if(timer){stop();return;} b.textContent='■ Parar rotação';timer=setInterval(()=>{angle+=.028;
Plotly.relayout(gd,{'scene.camera.eye':{x:2.35*Math.cos(angle),y:2.35*Math.sin(angle),z:1.18}});},100);};
document.getElementById('reset').onclick=()=>{stop();Plotly.relayout(gd,{'scene.camera':camera});};
window.addEventListener('beforeunload',stop);
})();
</script></body></html>'''
    html=html.replace('__MCCFIX__',f'{mcc_fixo:.7f}').replace('__TAU__',f'{tau:.7f}')
    html=html.replace('__MCCSTAR__',f'{mcc_tau:.7f}')
    origem_mcc,nota_mcc=_origem_mcc(avaliacao)
    html=html.replace('__ORIGEM__',html_lib.escape(nota_mcc)).replace('__K__',str(modelo.n_components))
    html=html.replace('__METRIC__',html_lib.escape(origem_mcc))
    controles='\n'.join('<label class="chk"><input type="checkbox" data-layer="g{0}"> <span class="dot" style="background:{1}"></span>Gaussiana G{0}</label>'.format(k+1,cores[k % len(cores)]) for k in range(modelo.n_components))
    html=html.replace('__GAUSSIAN_CONTROLS__',controles)
    html=html.replace('__CENTER_KEYS__',json.dumps(['centro'+str(k+1) for k in range(modelo.n_components)]))
    html=html.replace('__CHART__',chart).replace('__TRACE_INDEX__',json.dumps(maps))
    path=destino/(PREFIXO+'_INTERATIVA.html')
    path.write_text(html,encoding='utf-8')
    assert path.stat().st_size>100000
    return path



def gerar_3d(config, data, escolha, destino, ext, dpi, csv, *, avaliacao):
    """Substitui apenas o GMM08 do vencedor PCA3D; gera PNG/PDF/SVG e HTML."""
    destino=Path(destino)
    destino.mkdir(parents=True,exist_ok=True)
    X=data['coord'].to_numpy(dtype=float)
    if X.ndim!=2 or X.shape[1]!=3 or list(data['coord'].columns)!=['PC1','PC2','PC3']:
        raise ValueError('GMM08 interativo exige PCA 3D com PC1, PC2 e PC3.')
    gm=data['gmm'].modelo_
    if gm.covariance_type!='full' or gm.means_.shape!=(gm.n_components,3):
        raise ValueError('GMM08 PCA3D exige covariancia FULL 3x3.')
    if not np.isclose(float(config['visualizacao']['graficos']['mapa_risco_fronteira'].get('cobertura_gaussiana',.95)),.95):
        raise ValueError('GMM08 3D exige elipsoides de 95%.')
    tau=float(escolha['limiar_mcc_oof'])
    if not 0<=tau<=1:raise ValueError('Limiar MCC invalido.')
    op=config['visualizacao']['graficos']['mapa_risco_fronteira']
    n=int(op.get('resolucao_2d',180))
    if not 20<=n<=450:raise ValueError('Resolucao 2D invalida.')
    margem=float(op.get('margem',.15))
    if not 0<=margem<=1:raise ValueError('Margem invalida.')
    media=X.mean(axis=0)
    for i,j in ((0,1),(0,2),(1,2)):
        xx,yy=np.meshgrid(np.linspace(*geo._faixa(X[:,i],margem),n),
                          np.linspace(*geo._faixa(X[:,j],margem),n))
        pontos=np.tile(media,(n*n,1))
        pontos[:,i]=xx.ravel();pontos[:,j]=yy.ravel()
        risco=np.asarray(geo._pontuar_gmm_array(data['gmm'],data['rh'],pontos)).reshape(n,n)
        fixo,otimo,mudou=geo._mudanca_classes(risco,tau)
        if csv:
            # Convenção original GMM08: CSV por plano, precisão completa.
            geo._salvar_dados(pd.DataFrame({
                'score_risco':risco.ravel(),
                'classe_corte_fixo':fixo.ravel().astype(int),
                'classe_corte_mcc':otimo.ravel().astype(int),
                'mudou_classe':mudou.ravel().astype(int),
                f'PC{i+1}':xx.ravel(),f'PC{j+1}':yy.ravel(),
            }),destino,f'GMM08_malha_PC{i+1}_PC{j+1}',True)
    caminho=png_organizado(config,data,escolha,avaliacao,destino,ext,dpi)
    pagina=html_interativo_v2(config,data,escolha,avaliacao,destino,csv)
    assert caminho.is_file() and pagina.is_file()
    print('[GERADO]',caminho.name)
    print('[GERADO]',pagina.name)
