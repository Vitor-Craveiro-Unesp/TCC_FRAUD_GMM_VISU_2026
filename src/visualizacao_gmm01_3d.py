# GMM01_GEOMETRIA_HTML_3D_V1
# GMM01_LAYOUT_LIMPO_V1
# -*- coding: utf-8 -*-
"""GMM01 - complementos da densidade conjunta em PCA 3D.

Usa exclusivamente GMM ajustado no desenvolvimento F1-F9 recebido de
src.visualizacao_gmm01.gerar. Nao ajusta modelos nem acessa F10.

Gera, para PCA 3D / GMM FULL / K=3:
- 3 imagens de cortes: cada imagem tem 3 planos, fixando a coordenada
  omitida na media de cada gaussiana (cada painel mostra a mistura inteira);
- 1 PNG com 3 elipsoides FULL 95%;
- 1 HTML offline interativo da densidade conjunta em 3D;
- CSVs numericos dos 9 planos e da malha volumetrica.
As tres marginais continuam sendo geradas pelo modulo GMM01 original.
"""
from __future__ import annotations

from pathlib import Path
from scipy.stats import chi2, multivariate_normal
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = ('#1685bb', '#f0842d', '#25a277')
PARES = ((0,1,2),(0,2,1),(1,2,0))
NOMES = ('PC1','PC2','PC3')


def _validar(data):
    coord = data['coord']
    gm = data['gmm']
    model = gm.modelo_
    if getattr(model,'covariance_type',None) != 'full':
        raise ValueError('GMM01 3D: necessario GMM FULL.')
    mu=np.asarray(model.means_,dtype=float)
    cov=np.asarray(model.covariances_,dtype=float)
    weights=np.asarray(model.weights_,dtype=float)
    # K3 e o experimento de auditoria aprovado; outros K exigem novo layout.
    if mu.shape!=(3,3) or cov.shape!=(3,3,3) or weights.shape!=(3,):
        raise ValueError('GMM01 3D: este layout validado requer K=3 e PCA 3D.')
    xyz=np.asarray(coord,dtype=float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
        raise ValueError('GMM01 3D: coordenadas PCA invalidas.')
    if not (np.isfinite(mu).all() and np.isfinite(cov).all() and
            np.isfinite(weights).all() and (weights>0).all() and
            np.isclose(weights.sum(),1.,atol=1e-6)):
        raise ValueError('GMM01 3D: parametros gaussianos invalidos.')
    if any(np.linalg.eigvalsh(c).min()<=0 for c in cov):
        raise ValueError('GMM01 3D: matrizes FULL nao positivas definidas.')
    return xyz,mu,cov,weights


def _faixas(xyz,mu,cov):
    ranges=[]
    for j in range(3):
        qlo,qhi=np.quantile(xyz[:,j],[.005,.995])
        sd=np.sqrt(cov[:,j,j])
        lower=min(qlo,float(np.min(mu[:,j]-3.4*sd)))
        upper=max(qhi,float(np.max(mu[:,j]+3.4*sd)))
        pad=.08*(upper-lower)
        ranges.append((lower-pad,upper+pad))
    return ranges


def _densidade_componentes(pts,mu,cov,weights):
    arr=np.column_stack([
        weights[k]*multivariate_normal.pdf(
            pts,mean=mu[k],cov=cov[k],allow_singular=False)
        for k in range(3)
    ])
    if not np.isfinite(arr).all() or (arr<0).any():
        raise ValueError('GMM01 3D: densidade invalida.')
    return arr


def _planos(xyz,mu,cov,weights,n=125):
    """Nove seções da MESMA densidade conjunta: 3 pares x 3 alturas."""
    ranges=_faixas(xyz,mu,cov)
    nivel95=float(chi2.ppf(.95,df=3))
    results=[]
    inv_cov=np.linalg.inv(cov)
    for a,b,om in PARES:
        gx,gy=np.meshgrid(np.linspace(*ranges[a],n),np.linspace(*ranges[b],n))
        for h in range(3):
            fixed=float(mu[h,om])
            pts=np.empty((n*n,3),dtype=float)
            pts[:,a]=gx.ravel()
            pts[:,b]=gy.ravel()
            pts[:,om]=fixed
            comp=_densidade_componentes(pts,mu,cov,weights)
            f=comp.sum(axis=1).reshape(gx.shape)
            distances=[]
            for k in range(3):
                d=pts-mu[k]
                distances.append(np.einsum('ni,ij,nj->n',d,inv_cov[k],d).reshape(gx.shape))
            results.append((a,b,om,h,fixed,gx,gy,pts,comp,f,distances,nivel95))
    return results


def _cortes(xyz,mu,cov,weights,dest,dpi=165,csv=True):
    mapas=_planos(xyz,mu,cov,weights)
    # Mesma escala de densidade para TODOS os 9 planos.
    vmax=max(float(entry[9].max()) for entry in mapas)
    if vmax<=0:raise ValueError('GMM01 3D: maximo conjunto invalido.')
    output=[]
    for par in PARES:
        a,b,om=par
        sel=[m for m in mapas if m[:3]==par]
        fig,axs=plt.subplots(1,3,figsize=(19.5,6.4),sharex=True,sharey=True)
        fig.suptitle('CORTES DA DENSIDADE CONJUNTA GAUSSIANA',
                     fontweight='bold',fontsize=17,y=.965)
        im=None
        for h,m in enumerate(sel):
            _,_,_,_,fixed,gx,gy,pts,comp,f,distances,level=m
            ax=axs[h]
            im=ax.pcolormesh(gx,gy,f,cmap='viridis',vmin=0.,vmax=vmax,shading='auto',rasterized=True)
            ax.set_title(f'Plano de G{h+1}  |  {NOMES[om]} = {fixed:.7f}',fontsize=11,fontweight='bold')
            for k,q in enumerate(distances):
                if float(q.min())<level<float(q.max()):
                    ax.contour(gx,gy,q,levels=[level],colors=[COLORS[k]],
                               linestyles='--',linewidths=1.75)
            ax.set_xlabel(NOMES[a])
            if h==0:ax.set_ylabel(NOMES[b])
            ax.grid(alpha=.09,ls=':')
            if csv:
                tab=pd.DataFrame(pts,columns=NOMES)
                tab['densidade_conjunta']=f.ravel()
                for k in range(3):
                    tab[f'contribuicao_G{k+1}']=comp[:,k]
                    tab[f'dentro_elipsoide_95_G{k+1}']=(distances[k].ravel()<=level)
                tab.to_csv(dest/f'GMM01_conjunta_corte_{NOMES[a]}_{NOMES[b]}_plano_G{h+1}.csv',
                           index=False,float_format='%.17g')
        cax=fig.add_axes([.917,.215,.012,.575])
        cb=fig.colorbar(im,cax=cax,format='%.7f')
        cb.set_label('Densidade conjunta f(PC1,PC2,PC3)')
        handles=[Line2D([0],[0],color=COLORS[k],lw=1.7,ls='--',
                        label=f'G{k+1} — seção FULL 95%') for k in range(3)]
        fig.legend(handles=handles,ncol=3,loc='lower center',
                   bbox_to_anchor=(.48,.070),frameon=False,fontsize=9)
        fig.subplots_adjust(left=.065,right=.895,bottom=.200,top=.865,wspace=.13)
        file=dest/f'GMM01_conjunta_corte_{NOMES[a]}_{NOMES[b]}.png'
        fig.savefig(file,dpi=dpi,facecolor='white')
        plt.close(fig)
        output.append(file)
    return output


def _elipsoide(mu,cov,nu=39,nv=25):
    vals,vec=np.linalg.eigh(cov)
    u=np.linspace(0,2*np.pi,nu)
    v=np.linspace(0,np.pi,nv)
    surface=np.stack((np.outer(np.cos(u),np.sin(v)),
                      np.outer(np.sin(u),np.sin(v)),
                      np.outer(np.ones_like(u),np.cos(v))),axis=-1)
    matrix=vec @ np.diag(np.sqrt(vals*chi2.ppf(.95,df=3)))
    p=np.einsum('ij,uvj->uvi',matrix,surface)+mu
    return p[:,:,0],p[:,:,1],p[:,:,2]


def _geometria_png(xyz,mu,cov,dest,dpi=165):
    fig=plt.figure(figsize=(12.4,8.8))
    ax=fig.add_subplot(111,projection='3d')
    # Amostra visual deterministica independente dos rótulos.
    idx=np.linspace(0,len(xyz)-1,min(len(xyz),2800),dtype=int)
    ax.scatter(xyz[idx,0],xyz[idx,1],xyz[idx,2],s=5,color='#456987',
               alpha=.095,depthshade=False,label='Observações projetadas')
    for k in range(3):
        x,y,z=_elipsoide(mu[k],cov[k])
        ax.plot_wireframe(x,y,z,color=COLORS[k],linewidth=.60,alpha=.47,rstride=2,cstride=2)
        ax.scatter([mu[k,0]],[mu[k,1]],[mu[k,2]],color=COLORS[k],marker='X',
                   s=85,depthshade=False,label=f'G{k+1} — 95%')
    ax.set(xlabel='PC1',ylabel='PC2',zlabel='PC3')
    ax.view_init(elev=23,azim=-59)
    ax.legend(loc='upper left',frameon=True,fontsize=9)
    fig.suptitle('GEOMETRIA DA MISTURA GAUSSIANA — PCA 3D',
                 fontsize=15,fontweight='bold',y=.965)
    fig.text(.5,.915,'FULL | K=3 | elipsoides de concentração gaussiana de 95%',
             ha='center',fontsize=11,color='#4c617c')
    fig.subplots_adjust(left=.06,right=.87,top=.86,bottom=.15)
    file=dest/'GMM01_geometria_gaussianas_FULL_3d_95.png'
    fig.savefig(file,dpi=dpi,facecolor='white')
    plt.close(fig)
    return file


def _malha_3d(xyz,mu,cov,weights,n=29):
    ranges=_faixas(xyz,mu,cov)
    axes=[np.linspace(*ranges[j],n) for j in range(3)]
    aa,bb,cc=np.meshgrid(*axes,indexing='ij')
    pts=np.column_stack([aa.ravel(),bb.ravel(),cc.ravel()])
    components=_densidade_componentes(pts,mu,cov,weights)
    tab=pd.DataFrame(pts,columns=NOMES)
    tab['densidade_conjunta']=components.sum(axis=1)
    for k in range(3):tab[f'contribuicao_G{k+1}']=components[:,k]
    return tab


def _interativo_html(mu,cov,weights,malha,dest):
    try:import plotly.graph_objects as go
    except ImportError as e:raise ImportError('GMM01 3D requer plotly para gerar o HTML interativo.') from e
    val=malha['densidade_conjunta'].to_numpy(dtype=float)
    mx=float(val.max())
    fig=go.Figure()
    fig.add_trace(go.Isosurface(
        x=malha['PC1'].to_numpy(),y=malha['PC2'].to_numpy(),z=malha['PC3'].to_numpy(),
        value=val,isomin=.13*mx,isomax=.72*mx,surface_count=4,opacity=.23,
        colorscale='Viridis',caps={'x_show':False,'y_show':False,'z_show':False},
        showscale=True,colorbar={'title':{'text':'Densidade','side':'top'},'x':1.02,'y':.5,'len':.69,'thickness':16,'tickformat':'.3f'},name='Isossuperfícies da mistura',
    ))
    for k in range(3):
        x,y,z=_elipsoide(mu[k],cov[k])
        fig.add_trace(go.Surface(x=x,y=y,z=z,surfacecolor=np.full_like(x,k,dtype=float),
            colorscale=[[0,COLORS[k]],[1,COLORS[k]]],showscale=False,opacity=.23,
            name=f'G{k+1} — 95%',showlegend=True))
        fig.add_trace(go.Scatter3d(x=[mu[k,0]],y=[mu[k,1]],z=[mu[k,2]],
            mode='markers+text',text=[f'G{k+1}'],textposition='top center',
            marker={'size':6,'color':COLORS[k],'symbol':'diamond'},
            name=f'Centro G{k+1}',hovertemplate=(f'G{k+1} | peso = {weights[k]:.7f}<extra></extra>')))
    v_iso=[True]+[False,True]*3
    v_ell=[False]+[True,True]*3
    v_all=[True]+[True,True]*3
    fig.update_layout(
        height=730,margin={'l':10,'r':110,'t':95,'b':15},showlegend=False,
        scene={'xaxis':{'title':'PC1'},'yaxis':{'title':'PC2'},'zaxis':{'title':'PC3'},
               'aspectmode':'data','dragmode':'orbit',
               'camera':{'eye':{'x':1.8,'y':1.6,'z':.95}}},
        updatemenus=[{'type':'buttons','direction':'right','x':.50,'y':1.06,
                     'xanchor':'center','yanchor':'bottom','buttons':[
            {'label':'Densidade conjunta','method':'update','args':[{'visible':v_iso}]},
            {'label':'Elipsoides FULL 95%','method':'update','args':[{'visible':v_ell}]},
            {'label':'Ambos','method':'update','args':[{'visible':v_all}]},
        ]}])
    for tr,state in zip(fig.data,v_iso):tr.visible=state
    div_id='gmm01_3d'
    chart=fig.to_html(include_plotlyjs=True,full_html=False,div_id=div_id,
                      config={'displaylogo':False,'scrollZoom':True,'responsive':True})
    html='''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GMM01 PCA 3D</title>
<style>body{font-family:system-ui,sans-serif;background:#f4f7fb;color:#182538;margin:0}main{max-width:1380px;margin:20px auto;padding:0 14px}h1{font-size:21px}button{padding:9px 14px;border:1px solid #c8d2dd;background:white;border-radius:8px;cursor:pointer}.header{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}.panel{background:white;padding:10px;border-radius:12px;box-shadow:0 2px 10px #19283e18}</style>
</head><body><main><div class="header"><div><h1>Densidade Conjunta PCA 3D</h1><p>Covariância FULL | K=3 | PCA 3D</p></div><div><button id="spin">▶ Girar automaticamente</button> <button id="reset">↺ Restaurar</button></div></div><div class="panel">__CHART__</div>
<p>Isossuperfícies: densidade conjunta da mistura. Elipsoides: concentração FULL 95% de cada gaussiana. Não representam risco de fraude.</p></main><script>(()=>{const gd=document.getElementById('gmm01_3d');const spin=document.getElementById('spin');let timer=null,a=.7;function stop(){if(timer!==null){clearInterval(timer);timer=null;}spin.textContent='▶ Girar automaticamente';}spin.onclick=()=>{if(timer!==null){stop();return;}spin.textContent='⏸ Pausar';timer=setInterval(()=>{a+=.025;Plotly.relayout(gd,{'scene.camera.eye':{x:2.1*Math.cos(a),y:2.1*Math.sin(a),z:.9}})},75)};document.getElementById('reset').onclick=()=>{stop();a=.7;Plotly.relayout(gd,{'scene.camera.eye':{x:1.8,y:1.6,z:.95}})};gd.addEventListener('pointerdown',stop);window.addEventListener('pagehide',stop)})();</script></body></html>'''
    file=dest/'GMM01_densidade_conjunta_PCA3D_INTERATIVA.html'
    file.write_text(html.replace('__CHART__',chart),encoding='utf-8')
    return file




def _geometria_interativa_html(xyz, mu, cov, weights, dest):
    """Geometria FULL95 do GMM PCA3D, separada da densidade volumetrica.

    Todas as coordenadas, medias e covariancias vem do mesmo ajuste recebido.
    Nao estima modelos nem usa classes, scores, labels ou F10.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError('GMM01 geometria 3D requer plotly instalado.') from exc

    import numpy as np
    from pathlib import Path

    destino = Path(dest)
    destino.mkdir(parents=True, exist_ok=True)
    figura = go.Figure()

    indices = np.linspace(0, len(xyz)-1, min(len(xyz), 2800), dtype=int)
    amostra = xyz[indices, :]
    figura.add_trace(go.Scatter3d(
        x=amostra[:, 0], y=amostra[:, 1], z=amostra[:, 2],
        mode='markers',
        marker={'color': '#637d97', 'size': 2.4, 'opacity': 0.24},
        name='Observações projetadas', legendgroup='obs',
        hovertemplate='PC1=%{x:.7f}<br>PC2=%{y:.7f}<br>PC3=%{z:.7f}<extra></extra>',
    ))

    for k in range(3):
        x, y, z = _elipsoide(mu[k], cov[k], nu=49, nv=31)
        px, py, pz = [], [], []

        # Linhas de longitude e latitude. Superficie elipsoidal FULL95.
        for ii in range(0, x.shape[0], 4):
            px.extend(x[ii, :].tolist() + [None])
            py.extend(y[ii, :].tolist() + [None])
            pz.extend(z[ii, :].tolist() + [None])
        for jj in range(0, x.shape[1], 4):
            px.extend(x[:, jj].tolist() + [None])
            py.extend(y[:, jj].tolist() + [None])
            pz.extend(z[:, jj].tolist() + [None])

        figura.add_trace(go.Scatter3d(
            x=px, y=py, z=pz, mode='lines',
            line={'color': COLORS[k], 'width': 2.3},
            name=f'G{k+1} — elipsoide FULL 95%',
            legendgroup=f'g{k+1}',
            hoverinfo='skip',
        ))
        figura.add_trace(go.Scatter3d(
            x=[float(mu[k, 0])], y=[float(mu[k, 1])],
            z=[float(mu[k, 2])], mode='markers+text',
            text=[f'G{k+1}'], textposition='top center',
            marker={'color': COLORS[k], 'size': 7, 'symbol': 'diamond'},
            name=f'Centro G{k+1}', legendgroup=f'g{k+1}',
            showlegend=False,
            hovertemplate=(
                f'G{k+1} | peso={weights[k]:.7f}<br>'
                'PC1=%{x:.7f}<br>PC2=%{y:.7f}<br>PC3=%{z:.7f}<extra></extra>'
            ),
        ))

    figura.update_layout(
        height=760,
        margin={'l': 5, 'r': 185, 't': 10, 'b': 15},
        paper_bgcolor='white',
        showlegend=True,
        legend={
            'x': 1.02, 'xanchor': 'left', 'y': 0.98, 'yanchor': 'top',
            'groupclick': 'togglegroup',
            'font': {'size': 12},
        },
        scene={
            'xaxis': {'title': 'PC1'},
            'yaxis': {'title': 'PC2'},
            'zaxis': {'title': 'PC3'},
            'aspectmode': 'data',
            'dragmode': 'orbit',
            'camera': {'eye': {'x': 1.65, 'y': 1.65, 'z': 0.90}},
        },
    )
    grafico = figura.to_html(
        include_plotlyjs=True, full_html=False,
        div_id='gmm01_geometria_3d',
        config={'displaylogo': False, 'scrollZoom': True, 'responsive': True},
    )
    pagina = r"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Geometria da Mistura Gaussiana — PCA 3D</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#1d2c43;font-family:system-ui,Arial,sans-serif}
main{max-width:1550px;margin:18px auto;padding:0 16px}header{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:16px;margin-bottom:14px}
h1{font-size:23px;line-height:1.35;margin:0 0 5px}p{margin:0;color:#52647e;font-size:14px}
.toolbar{display:flex;flex-wrap:wrap;gap:8px}button{background:white;border:1px solid #bfccdb;border-radius:7px;padding:9px 12px;cursor:pointer;color:#233c5b}
button:hover{background:#ebf3fc}.plot{background:white;border-radius:12px;box-shadow:0 2px 12px #1b324016;padding:8px}
@media(max-width:780px){h1{font-size:19px}.toolbar{width:100%}}
</style></head><body><main><header><div><h1>Geometria da Mistura Gaussiana — PCA 3D</h1>
<p>Covariância FULL | K=3 | Elipsoides gaussianos de 95%</p></div>
<div class="toolbar"><button id="girar">▶ Girar automaticamente</button>
<button id="restaurar">↺ Restaurar câmera</button>
<button id="todos">Mostrar tudo</button>
<button id="gaussianas">Só gaussianas</button>
<button id="observacoes">Só observações</button></div></header>
<div class="plot">__GRAFICO__</div></main>
<script>(()=>{
const plot=document.getElementById('gmm01_geometria_3d');
const girar=document.getElementById('girar');let relogio=null,angulo=.79;
function parar(){if(relogio!==null){clearInterval(relogio);relogio=null;}girar.textContent='▶ Girar automaticamente';}
function visiveis(tipo){parar();const v=tipo==='todos'?[true,true,true,true,true,true,true]:
(tipo==='gaussianas'?[false,true,true,true,true,true,true]:[true,false,false,false,false,false,false]);
Plotly.restyle(plot,{visible:v});}
document.getElementById('todos').onclick=()=>visiveis('todos');
document.getElementById('gaussianas').onclick=()=>visiveis('gaussianas');
document.getElementById('observacoes').onclick=()=>visiveis('observacoes');
girar.onclick=()=>{if(relogio!==null){parar();return;}girar.textContent='⏸ Pausar';
relogio=setInterval(()=>{angulo+=.022;Plotly.relayout(plot,{'scene.camera.eye':
{x:2.1*Math.cos(angulo),y:2.1*Math.sin(angulo),z:1.05}})},80)};
document.getElementById('restaurar').onclick=()=>{parar();angulo=.79;Plotly.relayout(plot,{'scene.camera.eye':{x:1.65,y:1.65,z:.90}})};
plot.addEventListener('pointerdown',parar);window.addEventListener('pagehide',parar);
})();</script></body></html>"""
    arquivo = destino / 'GMM01_geometria_gaussianas_FULL_3d_95_INTERATIVA.html'
    arquivo.write_text(pagina.replace('__GRAFICO__', grafico), encoding='utf-8')
    return arquivo

def gerar_complementares(config,data,destino,ext,dpi,csv):
    """Complementos do GMM01 em PCA 3D; chamado apos as tres marginais oficiais."""
    if ext.lower().lstrip('.')!='png':
        raise ValueError('GMM01 PCA 3D: use formato PNG para a galeria com HTML interativo.')
    dest=Path(destino)
    dest.mkdir(parents=True,exist_ok=True)
    xyz,mu,cov,weights=_validar(data)
    outputs=_cortes(xyz,mu,cov,weights,dest,int(dpi),bool(csv))
    outputs.append(_geometria_png(xyz,mu,cov,dest,int(dpi)))
    outputs.append(_geometria_interativa_html(xyz,mu,cov,weights,dest))
    malha=_malha_3d(xyz,mu,cov,weights)
    if csv:malha.to_csv(dest/'GMM01_malha_densidade_conjunta_PCA3D.csv',
                       index=False,float_format='%.17g')
    outputs.append(_interativo_html(mu,cov,weights,malha,dest))
    if not all(p.is_file() for p in outputs):
        raise RuntimeError('GMM01 PCA 3D: saídas incompletas.')
    return outputs
