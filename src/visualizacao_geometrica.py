"""Passo 07.02 — diagnosticos geometricos do VENCEDOR OOF.

    python -m src.visualizacao_geometrica --experimento outputs/ensaio_pca_gmm_... --planejar
    python -m src.visualizacao_geometrica --experimento outputs/ensaio_pca_gmm_... --executar

Refaz UM ajuste final do vencedor no MESMO desenvolvimento (F1..F9) do
experimento/ensaio. Nao lê F10 para treino, graficos ou selecao. A pasta
"teste_final/graficos_gerados" segue o YAML e contem diagnósticos do
modelo ajustado no desenvolvimento; NAO sao metricas ou graficos de F10.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from src import orquestracao_oof as orq
from src import visualizacao_oof as vis
from src.amostragem import selecionar_indices_gmm
from src.preprocessing import selecionar_features, EncodersPorSufixo, ImputadorPadronizador
from src.reducao import RedutorPCA
from src.reducao_tsne import RedutorTSNE
from src.gmm import GMMNaoSupervisionado
from src.risco import RiscoHistoricoGaussianas
from src.probabilidade import PreditorProbabilisticoGMM

TIPOS = (
    'risco_por_gaussiana', 'pesos_responsabilidades', 'medias_gaussianas',
    'covariancias_gaussianas', 'pca_cotovelo', 'pca_projecoes',
    'tsne_mapa', 'tsne_diagnostico', 'mapa_risco_fronteira',
)
NUM = {nome: i for i, nome in enumerate(TIPOS, start=7)}


def _exigir(cond, msg):
    if not cond:
        raise ValueError(msg)


def _ativo(config, nome):
    # Graficos geometricos estao na fase 'teste_final' do YAML, MAS
    # sua origem e exclusivamente o desenvolvimento F1..F9.
    return vis._ativado(config, nome, fase='teste_final')


def _estilo(config):
    p = config['visualizacao']['padrao']
    ext = str(p.get('formato', 'png')).lower()
    _exigir(ext in ('png', 'pdf', 'svg'), 'Formato do grafico invalido.')
    dpi = int(p.get('dpi', 180))
    _exigir(50 <= dpi <= 1200, 'DPI fora da faixa permitida.')
    return ext, dpi, p.get('salvar_dados_grafico', True) is True


def _fig(config):
    fig, ax = vis._figura(config)
    return fig, ax


def _guardar(fig, destino, base, ext, dpi):
    path = destino / f'{base}.{ext}'
    return vis._salvar(fig, path, dpi=dpi)


def _salvar_dados(df, destino, nome, habilitado=True):
    if habilitado:
        vis._csv_protegido(df, destino/f'{nome}.csv')


def _selecao(experimento, raiz):
    """Audita vencedor, dados, hashes, folds e IDs ANTES de qualquer fit."""
    cfg, grade, escolha, y_oof, _ = vis._ler_resultados(experimento)
    manifesto = json.loads((experimento/'manifesto.json').read_text(encoding='utf-8-sig'))
    _exigir(manifesto.get('status') == 'concluido_06_03', 'Experimento OOF ainda nao concluido.')
    _exigir(manifesto.get('tipo') in ('ensaio','oficial'), 'Tipo de experimento nao reconhecido.')
    _exigir(not escolha.get('teste_final_usado'), 'F10 ja utilizado na selecao.')
    cid = escolha['id_candidato']
    caminho = experimento/'oof'/'predicoes'/f'{cid}.npz'
    with np.load(caminho,allow_pickle=False) as a:
        ids = np.asarray(a['linha_original'],dtype=np.int64)
        f_oof = np.asarray(a['fold'],dtype=np.uint8)
    _exigir(len(ids) == len(y_oof), 'Indices nao coincidem com o OOF.')
    _exigir(np.all(np.diff(ids)>0), 'Indices de origem precisam estar estritamente ordenados.')
    _exigir(set(np.unique(f_oof)) == set(range(1,10)), 'OOF nao contem os nove folds.')
    arquivo_base = raiz / cfg['dados']['arquivo']
    folds, arq_folds, hash_base = orq._identificar_folds(raiz,cfg,arquivo_base)
    _exigir(manifesto.get('sha256_parquet') == hash_base, 'Parquet diverge da execucao OOF.')
    _exigir(manifesto.get('sha256_folds') == orq.hash_arquivo(arq_folds), 'Folds divergem do OOF.')
    _exigir(np.array_equal(folds[ids],f_oof) and (folds[ids]!=10).all(), 'F10 ou folds trocados nos indices de origem.')
    _exigir(int(manifesto['registros_desenvolvimento']) == len(ids), 'Quantidade de desenvolvimento diverge do manifesto.')
    _exigir(escolha['configuracao']['metodo'].upper() == cfg['reducao']['metodo'].upper(),
            'Metodo do vencedor diverge do YAML do experimento.')
    return cfg, escolha, ids, y_oof, arquivo_base, folds


def _treinar_vencedor(df, config, escolha):
    """Ajusta preprocessamento/reducao em TODO desenvolvimento selecionado;
    GMM pode usar RUS, RH usa necessariamente todos os labels originais.
    """
    cand = escolha['configuracao']
    t0 = perf_counter()
    x = selecionar_features(df,config)
    encoder = EncodersPorSufixo(config).fit(x)
    x_enc = encoder.transform(x)
    imp = ImputadorPadronizador(config,encoder).fit(x_enc)
    x_pre = imp.transform(x_enc)
    del x, x_enc
    t_red = perf_counter()
    if cand['metodo'] == 'PCA':
        red = RedutorPCA(config,cand['reducao']).fit(x_pre)
        coord = red.transform(x_pre)
    elif cand['metodo'] == 'TSNE':
        red = RedutorTSNE(config,cand['reducao']).fit(x_pre)
        coord = red.transform_treino()
    else:
        raise ValueError('Metodo de reducao desconhecido.')
    y = df[config['dados']['coluna_alvo']].copy()
    _exigir(coord.index.equals(y.index), 'Desalinhamento entre coordenadas e labels.')
    amostra = selecionar_indices_gmm(y.to_numpy(),config,modo=cand['modo'],
                                   razao=cand['razao'],fold_validacao=0)
    modelo = GMMNaoSupervisionado(config,cand['gmm']).fit(coord.iloc[amostra])
    gamma = modelo.responsabilidades(coord)
    rh = RiscoHistoricoGaussianas(config,cand['suavizacao']).fit(gamma,y)
    probs = PreditorProbabilisticoGMM(modelo,rh).prever_probabilidade(coord)
    _exigir(np.isfinite(probs).all(), 'Probabilidades de desenvolvimento invalidas.')
    return {'coord':coord,'gamma':gamma,'rh':rh,'gmm':modelo,'redutor':red,
            'pre':x_pre,'y':y.to_numpy(dtype=np.int8), 'scores':probs.to_numpy(),
            'tempo_reducao':t_red-t0,'tempo_total':perf_counter()-t0,
            'n_treino_gmm':len(amostra),'limiar_oof':float(escolha['limiar_mcc_oof'])}


def _covariancia(gmm,k):
    gm=gmm.modelo_
    if gm.covariance_type=='full': return np.asarray(gm.covariances_[k],dtype=float)
    if gm.covariance_type=='tied': return np.asarray(gm.covariances_,dtype=float)
    if gm.covariance_type=='diag': return np.diag(np.asarray(gm.covariances_[k],dtype=float))
    if gm.covariance_type=='spherical': return np.eye(gm.means_.shape[1])*float(gm.covariances_[k])
    raise ValueError('Tipo de covariancia invalido.')


def _amostra_plot(n,y,config,limite=None):
    geo=config['visualizacao']['geometria']
    limite=int(limite or geo.get('pontos_maximos_amostra',20000))
    _exigir(limite>=1, 'Limite de pontos deve ser positivo.')
    if n<=limite: return np.arange(n)
    rng=np.random.default_rng(int(config['experimento']['semente']))
    if geo.get('amostragem_estratificada', True):
        i0=np.flatnonzero(y==0); i1=np.flatnonzero(y==1)
        n1=max(1,round(limite*len(i1)/n))
        n1=min(n1,len(i1),limite)
        n0=min(limite-n1,len(i0))
        if n0+n1<limite:
            faltam=limite-n0-n1
            n1=min(n1+faltam,len(i1))
        return np.sort(np.r_[rng.choice(i0,n0,replace=False),rng.choice(i1,n1,replace=False)])
    return np.sort(rng.choice(n,size=limite,replace=False))


def _pares(n_dim,config):
    geom=config['visualizacao']['geometria']
    if n_dim==1: return [(0,)]
    pares=[]
    for a,b in geom.get('pares_2d',[[1,2]]):
        _exigir(isinstance(a,int) and isinstance(b,int) and a!=b,
                'Pares de geometria incorretos.')
        if 1<=a<=n_dim and 1<=b<=n_dim:
            pares.append((a-1,b-1))
    _exigir(pares, 'Nenhum par de dimensoes de geometria valido.')
    return list(dict.fromkeys(pares))


def _scatter(ax,coords,y,indices,config,par):
    cc=config['visualizacao']['classes']
    for classe,nome in ((0,'legitima'),(1,'fraude')):
        i=indices[y[indices]==classe]
        estilo=cc[nome]
        if not len(i): continue
        xy=coords[i]
        if len(par)==1:
            ax.scatter(xy[:,par[0]],np.zeros(len(i))+classe*.008,
                       s=estilo.get('tamanho',12),alpha=estilo.get('alpha',.5),
                       color=estilo.get('cor'),marker=estilo.get('marcador','o'),
                       label=estilo.get('nome',nome),rasterized=True)
        else:
            ax.scatter(xy[:,par[0]],xy[:,par[1]],
                       s=estilo.get('tamanho',12),alpha=estilo.get('alpha',.5),
                       color=estilo.get('cor'),marker=estilo.get('marcador','o'),
                       label=estilo.get('nome',nome),rasterized=True)


def _risco_por_gaussiana(config,data,dest,ext,dpi,csv):
    op=config['visualizacao']['graficos']['risco_por_gaussiana']
    rh=data['rh']; df=rh.tabela().reset_index()
    ordenacao=op.get('ordenar_por','risco_decrescente')
    if ordenacao=='risco_decrescente': df=df.sort_values('risco_historico',ascending=False)
    elif ordenacao=='risco_crescente':df=df.sort_values('risco_historico')
    elif ordenacao!='indice':raise ValueError('Ordenacao desconhecida de risco_por_gaussiana.')
    fig,ax=_fig(config)
    if op.get('tipo','barra_horizontal')=='barra_horizontal':
        ax.barh(df['gaussiana'],df['risco_historico'])
        ax.invert_yaxis();ax.set_xlabel('Risco historico de fraude')
    else:
        ax.bar(df['gaussiana'],df['risco_historico']);ax.set_ylabel('Risco historico de fraude')
    if op.get('mostrar_risco',True):
        if op.get('tipo','barra_horizontal')=='barra_horizontal':
            for i,v in enumerate(df['risco_historico']):
                ax.text(float(v)+0.0002,i,f'{float(v):.5g}',va='center',fontsize=8)
        else:
            for i,v in enumerate(df['risco_historico']):
                ax.text(i,float(v)+0.0002,f'{float(v):.5g}',ha='center',fontsize=8)
    if op.get('mostrar_prevalencia_desenvolvimento',True):
        func=ax.axvline if op.get('tipo','barra_horizontal')=='barra_horizontal' else ax.axhline
        func(rh.prevalencia_treino_,ls='--',color='black',label='Prevalencia desenvolvimento')
        ax.legend()
    ax.set_title('07 — Risco por gaussiana | modelo reajustado no desenvolvimento')
    _guardar(fig,dest,'07_risco_por_gaussiana',ext,dpi)
    _salvar_dados(df,dest,'07_risco_por_gaussiana',csv)


def _pesos(config,data,dest,ext,dpi,csv):
    gm=data['gmm'].modelo_;rh=data['rh'];nomes=list(rh.riscos_.index)
    op=config['visualizacao']['graficos']['pesos_responsabilidades']
    massas=rh.massas_.to_numpy();fraude=rh.fraudes_ponderadas_.to_numpy()
    legitimas=massas-fraude
    dados=pd.DataFrame({'gaussiana':nomes,'peso_mistura':gm.weights_,
                        'massa_total':massas,'massa_fraude':fraude,'massa_legitima':legitimas})
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    ax=axs[0];x=np.arange(len(nomes));larg=.36
    if op.get('mostrar_peso_mistura',True):
        ax.bar(x-larg/2,gm.weights_,width=larg,label='Peso de mistura GMM')
    if op.get('mostrar_massa_total',True):
        ax.bar(x+larg/2,massas/rh.n_treino_,width=larg,label='Massa soft / N')
    ax.set(xticks=x,xticklabels=nomes,ylabel='Proporcao')
    if ax.patches:ax.legend(fontsize=8)
    ax=axs[1]
    if op.get('mostrar_massa_legitima',True):
        ax.bar(x,legitimas,label='Massa soft legitima',color='#1f77b4')
    if op.get('mostrar_massa_fraude',True):
        bottom=legitimas if op.get('mostrar_massa_legitima',True) else None
        ax.bar(x,fraude,bottom=bottom,label='Massa soft fraude',color='#d62728')
    ax.set(xticks=x,xticklabels=nomes,ylabel='Massa de responsabilidades')
    if ax.patches:ax.legend(fontsize=8)
    fig.suptitle('08 — Pesos e massas soft | desenvolvimento (sem F10)')
    _guardar(fig,dest,'08_pesos_responsabilidades',ext,dpi)
    _salvar_dados(dados,dest,'08_pesos_responsabilidades',csv)


def _medias(config,data,dest,ext,dpi,csv):
    gm=data['gmm'].modelo_; cols=list(data['coord'].columns)
    mat=np.asarray(gm.means_,dtype=float)
    tab=pd.DataFrame(mat,columns=cols);tab.insert(0,'gaussiana',data['gmm'].saidas_)
    maxdim=int(config['visualizacao']['graficos']['medias_gaussianas'].get('maximo_dimensoes_exibidas',12))
    _exigir(maxdim>0,'maximo_dimensoes_exibidas deve ser positivo.')
    fig,ax=_fig(config);rec=mat[:,:maxdim]
    im=ax.imshow(rec,aspect='auto',cmap='coolwarm')
    ax.set(xticks=np.arange(rec.shape[1]), xticklabels=cols[:maxdim],
           yticks=np.arange(len(gm.weights_)),yticklabels=data['gmm'].saidas_)
    ax.set_title('09 — Medias do GMM | desenvolvimento');fig.colorbar(im,ax=ax)
    if config['visualizacao']['graficos']['medias_gaussianas'].get('anotacoes_numericas',True):
        if rec.size<=144:
            for i in range(rec.shape[0]):
                for j in range(rec.shape[1]):ax.text(j,i,f'{rec[i,j]:.2g}',ha='center',va='center',fontsize=8)
    _guardar(fig,dest,'09_medias_gaussianas',ext,dpi)
    _salvar_dados(tab,dest,'09_medias_gaussianas_completo',csv)


def _covariancias(config,data,dest,ext,dpi,csv):
    gm=data['gmm'].modelo_;n=len(gm.weights_)
    mat=np.asarray([_covariancia(data['gmm'],k) for k in range(n)])
    tipo=gm.covariance_type
    limite=int(config['visualizacao']['graficos']['covariancias_gaussianas'].get('maximo_gaussianas',16))
    _exigir(limite>=1,'maximo_gaussianas deve ser positivo.')
    if tipo in ('diag','spherical'):
        vals=np.diagonal(mat,axis1=1,axis2=2)
        fig,ax=_fig(config)
        im=ax.imshow(vals[:limite],aspect='auto',cmap='viridis')
        ax.set(yticks=np.arange(min(n,limite)),yticklabels=data['gmm'].saidas_[:limite],
               xticks=np.arange(vals.shape[1]),xticklabels=data['coord'].columns)
        ax.set_title(f'10 — Variancias GMM ({tipo}; sem covariancias cruzadas)')
        fig.colorbar(im,ax=ax);_guardar(fig,dest,'10_variancias_gaussianas',ext,dpi)
    else:
        op=config['visualizacao']['graficos']['covariancias_gaussianas']
        for k in range(min(n,limite)):
            fig,ax=_fig(config)
            vismat=np.asarray(mat[k],dtype=float).copy()
            if not op.get('mostrar_fora_diagonal',True):
                vismat[~np.eye(len(vismat),dtype=bool)]=np.nan
            if not op.get('mostrar_diagonal',True):np.fill_diagonal(vismat,np.nan)
            _exigir(np.isfinite(vismat).any(),'Covariancias: diags e fora-diagonal ocultados simultaneamente.')
            im=ax.imshow(vismat,cmap='coolwarm')
            ax.set_title(f'10 — Covariancia {data["gmm"].saidas_[k]} ({tipo})')
            fig.colorbar(im,ax=ax)
            _guardar(fig,dest,f'10_covariancia_{data["gmm"].saidas_[k]}',ext,dpi)
    if csv:
        # CSV longo, sem truncar dimensoes, inclusive para diag/esferica.
        linhas=[(data['gmm'].saidas_[k],i+1,j+1,float(mat[k,i,j]))
                 for k in range(n) for i in range(mat.shape[1]) for j in range(mat.shape[2])]
        _salvar_dados(pd.DataFrame(linhas,columns=['gaussiana','dim_i','dim_j','covariancia']),
                      dest,'10_covariancias_completas',True)


def _scree(config,data,dest,ext,dpi,csv):
    # Calculo diagnostico so no desenvolvimento. Nao modifica K vencedor.
    x=data['pre'].to_numpy(dtype=np.float32,copy=False)
    k=min(len(x),x.shape[1])
    pca=PCA(n_components=k,svd_solver='full')
    pca.fit(x)
    ratio=np.asarray(pca.explained_variance_ratio_,dtype=float)
    nsel=data['coord'].shape[1]
    fig,ax=_fig(config)
    xaxis=np.arange(1,len(ratio)+1)
    opt=config['visualizacao']['graficos']['pca_cotovelo']
    if opt.get('mostrar_variancia_individual',True):ax.plot(xaxis,ratio,'o-',label='Variancia individual')
    if opt.get('mostrar_variancia_acumulada',True):ax.plot(xaxis,np.cumsum(ratio),'s-',label='Variancia acumulada')
    if opt.get('destacar_componentes_selecionados',True):ax.axvline(nsel,color='black',ls='--',label=f'PCA vencedor = {nsel}')
    ax.set(xlabel='Numero de componentes',ylabel='Proporcao variancia',ylim=(0,1.02))
    ax.set_title('11 — Scree PCA | ajuste diagnostico no desenvolvimento')
    ax.legend();_guardar(fig,dest,'11_pca_cotovelo',ext,dpi)
    _salvar_dados(pd.DataFrame({'componente':xaxis,'variancia_individual':ratio,
                               'variancia_acumulada':np.cumsum(ratio)}),
                  dest,'11_pca_cotovelo',csv)


def _projecoes(config,data,dest,ext,dpi,csv,metodo):
    coord=data['coord'].to_numpy(); y=data['y'];names=list(data['coord'].columns)
    op=config['visualizacao']['graficos']['pca_projecoes' if metodo=='PCA' else 'tsne_mapa']
    limite=int(config['visualizacao']['geometria'].get('pontos_maximos_amostra',20000))
    idx=_amostra_plot(len(y),y,config,limite)
    nome_g='12_pca_projecao' if metodo=='PCA' else '13_tsne_mapa'
    colorir=op.get('colorir_por','classe_real')
    _exigir(colorir in ('classe_real','classe_predita','risco','gaussiana_dominante'),
            'Projecao: colorir_por desconhecido.')
    for par in _pares(coord.shape[1],config):
        fig,ax=_fig(config)
        if colorir=='classe_real':
            _scatter(ax,coord,y,idx,config,par)
        elif colorir=='classe_predita':
            classe=(data['scores']>=data['limiar_oof']).astype(np.uint8)
            _scatter(ax,coord,classe,idx,config,par)
        elif colorir=='risco':
            yplot=np.zeros(len(idx)) if len(par)==1 else coord[idx,par[1]]
            graf=ax.scatter(coord[idx,par[0]],yplot,c=data['scores'][idx],
                            cmap='viridis',s=12,alpha=.65,rasterized=True)
            fig.colorbar(graf,ax=ax,label='Risco estimado no desenvolvimento')
        else:
            grupo=np.argmax(data['gamma'].to_numpy(),axis=1)
            yplot=np.zeros(len(idx)) if len(par)==1 else coord[idx,par[1]]
            for k,nome in enumerate(data['gmm'].saidas_):
                sub=idx[grupo[idx]==k]
                if len(sub):ax.scatter(coord[sub,par[0]],
                    (np.zeros(len(sub))+k*.008) if len(par)==1 else coord[sub,par[1]],
                    s=10,alpha=.5,label=nome,rasterized=True)
        if op.get('mostrar_contornos_gaussianas',False):
            gm=data['gmm']; nstd=2.
            for k in range(len(gm.saidas_)):
                if len(par)==1:
                    mean=gm.modelo_.means_[k,par[0]]
                    sd=np.sqrt(_covariancia(gm,k)[par[0],par[0]])
                    ax.axvspan(mean-nstd*sd,mean+nstd*sd,alpha=.05)
                else:
                    cov=_covariancia(gm,k)[np.ix_(par,par)]
                    eig,V=np.linalg.eigh(cov)
                    if (eig>0).all():
                        theta=np.degrees(np.arctan2(V[1,1],V[0,1]))
                        ax.add_patch(Ellipse(gm.modelo_.means_[k,list(par)],
                             width=2*nstd*np.sqrt(eig[1]),height=2*nstd*np.sqrt(eig[0]),
                             angle=theta,facecolor='none',edgecolor='black',alpha=.4))
        if len(par)==1:
            ax.set(xlabel=names[par[0]],ylabel='Separacao vertical apenas visual')
        else:
            ax.set(xlabel=names[par[0]],ylabel=names[par[1]])
        ax.set_title(f'{nome_g[:2]} — {metodo} | desenvolvimento (sem F10)')
        handles,labels=ax.get_legend_handles_labels()
        if handles:ax.legend(loc='best',fontsize=8)
        nome=nome_g+'_'+('_'.join(names[i] for i in par))
        _guardar(fig,dest,nome,ext,dpi)
        _salvar_dados(pd.DataFrame({'classe_real':y[idx],
            'risco_estimado_treino':data['scores'][idx],
            'gaussiana_dominante':np.argmax(data['gamma'].to_numpy()[idx],axis=1)+1,
            **{names[i]:coord[idx,i] for i in par}}),dest,nome,csv)


def _tsne_diagnostico(config,data,dest,ext,dpi,csv):
    red=data['redutor']
    kl=getattr(red,'kl_divergence_',None)
    fig,ax=_fig(config)
    ax.axis('off')
    detalhes=["14 — Diagnostico t-SNE (desenvolvimento)",
              f"Divergencia KL final: {kl if kl is not None else 'indisponivel'}",
              f"Tempo ajuste/reducao: {data['tempo_reducao']:.1f} s",
              'Historico por iteracao: nao fornecido pelo openTSNE utilizado']
    ax.text(.08,.8,'\n'.join(detalhes),transform=ax.transAxes,va='top',fontsize=12)
    _guardar(fig,dest,'14_tsne_diagnostico',ext,dpi)
    _salvar_dados(pd.DataFrame([{'kl_final':kl,'tempo_reducao_s':data['tempo_reducao'],
                                'historico_disponivel':False}]),dest,'14_tsne_diagnostico',csv)


def _pontuar_gmm_array(gmm,rh,arr):
    # Entrada pode ser lotes de malha; DataFrame preserva nomes/ordem das dimensoes.
    x=pd.DataFrame(arr,columns=gmm.colunas_)
    return gmm.modelo_.predict_proba(x.to_numpy(dtype=float)) @ rh.riscos_.to_numpy(dtype=float)


def _limiares(config,escolha):
    requested=config['visualizacao']['graficos']['mapa_risco_fronteira'].get('limiares',['mcc_oof'])
    if not isinstance(requested,list):requested=[requested]
    ret=[]
    for v in requested:
        a=float(escolha['limiar_mcc_oof']) if v=='mcc_oof' else float(v)
        _exigir(0<=a<=1 and np.isfinite(a),'Limiar de fronteira invalido.')
        if not any(np.isclose(a,b[1]) for b in ret):ret.append((str(v),a))
    return ret


def _faixa(v,margem):
    mi,ma=np.quantile(v,[.005,.995])
    amp=ma-mi
    if amp<=0:amp=max(abs(mi),1.)
    return mi-amp*margem, ma+amp*margem


def _fronteira(config,data,escolha,dest,ext,dpi,csv):
    op=config['visualizacao']['graficos']['mapa_risco_fronteira']
    X=data['coord'].to_numpy(dtype=np.float64);y=data['y']
    gm=data['gmm'];rh=data['rh'];nomes=data['coord'].columns
    nvis=int(op.get('n_pontos_scatter',5000))
    idx=_amostra_plot(len(y),y,config,nvis)
    margem=float(op.get('margem',.15));_exigir(0<=margem<=1,'Margem invalida.')
    pares=_pares(X.shape[1],config)
    for par in pares:
        lim=_limiares(config,escolha)
        if len(par)==1:
            p=par[0]; n=int(op.get('resolucao_1d',500));_exigir(20<=n<=20000,'Resolucao 1D fora dos limites.')
            eixo=np.linspace(*_faixa(X[:,p],margem),n)
            malha=np.repeat(X.mean(axis=0)[None,:],n,axis=0)
            malha[:,p]=eixo
            risco=_pontuar_gmm_array(gm,rh,malha)
            fig,ax=_fig(config)
            ax.plot(eixo,risco,label='Risco estimado no espaço reduzido',lw=2)
            risco_scatter=data['scores'][idx]
            ax.scatter(X[idx,p],risco_scatter,s=4,alpha=.08,color='gray',
                       label='Amostra de scores no desenvolvimento')
            # Manter visual legivel se 0.5 estiver muito acima de todos os scores.
            limiar_oof=float(escolha['limiar_mcc_oof'])
            teto=min(1.0,max(.025,1.2*float(risco.max()),1.5*limiar_oof))
            fora=[]
            for nome,thr in lim:
                if thr<=teto:
                    ax.axhline(thr,linestyle='--',label=f'Limiar {nome}={thr:.5g}')
                else:
                    fora.append(f'{nome}={thr:.5g}')
            if op.get('mostrar_gaussianas',True):
                for k in range(len(gm.saidas_)):
                    mean=gm.modelo_.means_[k,p]
                    sd=np.sqrt(_covariancia(gm,k)[p,p])
                    ax.axvspan(mean-float(op.get('n_desvios',2))*sd,
                               mean+float(op.get('n_desvios',2))*sd,alpha=.06)
            ax.set(xlabel=nomes[p],ylabel='Probabilidade de fraude',ylim=(0,teto))
            if fora:
                ax.text(.02,.98,'Fora da escala: '+', '.join(fora),va='top',ha='left',
                        fontsize=8,transform=ax.transAxes)
            ax.set_title('15 — Risco 1D | modelo desenvolvimento; limiar OOF')
            ax.legend(fontsize=8,loc='best')
            _guardar(fig,dest,'15_mapa_risco_1d_'+nomes[p],ext,dpi)
            _salvar_dados(pd.DataFrame({nomes[p]:eixo,'risco_estimado':risco}),
                          dest,'15_mapa_risco_1d_'+nomes[p],csv)
        else:
            i,j=par;n=int(op.get('resolucao_2d',180));_exigir(20<=n<=450,'Resolucao 2D fora de limites.')
            xx=np.linspace(*_faixa(X[:,i],margem),n)
            yy=np.linspace(*_faixa(X[:,j],margem),n)
            xmesh,ymesh=np.meshgrid(xx,yy)
            malha=np.tile(X.mean(axis=0),(n*n,1))
            malha[:,i]=xmesh.ravel();malha[:,j]=ymesh.ravel()
            risco=_pontuar_gmm_array(gm,rh,malha).reshape(n,n)
            fig,ax=_fig(config)
            m=ax.pcolormesh(xmesh,ymesh,risco,shading='auto',cmap='viridis')
            fig.colorbar(m,ax=ax,label='Risco estimado')
            for nome,thr in lim:
                if risco.min()<=thr<=risco.max():
                    ax.contour(xmesh,ymesh,risco,levels=[thr],linewidths=1.4,
                               colors=['white' if nome=='mcc_oof' else 'red'])
            _scatter(ax,X,y,idx,config,par)
            if op.get('mostrar_gaussianas',True):
                for k in range(len(gm.saidas_)):
                    cov=_covariancia(gm,k)[np.ix_(par,par)]
                    eig,V=np.linalg.eigh(cov)
                    if (eig>0).all():
                        theta=np.degrees(np.arctan2(V[1,1],V[0,1]))
                        e=Ellipse(gm.modelo_.means_[k,list(par)],
                                  width=2*float(op.get('n_desvios',2))*np.sqrt(eig[1]),
                                  height=2*float(op.get('n_desvios',2))*np.sqrt(eig[0]),
                                  angle=theta,edgecolor='white',facecolor='none',lw=1)
                        ax.add_patch(e)
            desc='corte com demais dimensoes fixadas na media do desenvolvimento' if X.shape[1]>2 else 'espaco completo 2D'
            ax.set(xlabel=nomes[i],ylabel=nomes[j]);ax.set_title('15 — Risco e fronteira | '+desc)
            ax.legend(loc='best')
            nome='15_mapa_risco_'+nomes[i]+'_'+nomes[j]
            _guardar(fig,dest,nome,ext,dpi)
            _salvar_dados(pd.DataFrame({nomes[i]:malha[:,i],nomes[j]:malha[:,j],
                                       'risco_estimado':risco.ravel()}),dest,nome,csv)


def _relatorio(config,data,dest):
    op=config['visualizacao'].get('relatorio_gaussianas',{})
    if not op.get('habilitar') or 'teste_final' not in op.get('gerar_em',[]):return
    rh=data['rh'];gm=data['gmm'];n=len(gm.saidas_)
    fator=float(op.get('rotulo_descritivo',{}).get('multiplicador_prevalencia',2.))
    limiar_abs=float(op.get('rotulo_descritivo',{}).get('limiar_absoluto',.5))
    estrategia=op.get('rotulo_descritivo',{}).get('estrategia','relativo_prevalencia')
    rows=[]
    for k,nome in enumerate(gm.saidas_):
        r=float(rh.riscos_.iloc[k]);prev=float(rh.prevalencia_treino_)
        elevado=r>= (prev*fator if estrategia=='relativo_prevalencia' else limiar_abs)
        rows.append({'gaussiana':nome,'peso_mistura':float(gm.modelo_.weights_[k]),
            'massa_responsabilidade_total':float(rh.massas_.iloc[k]),
            'massa_fraude':float(rh.fraudes_ponderadas_.iloc[k]),
            'massa_legitima':float(rh.massas_.iloc[k]-rh.fraudes_ponderadas_.iloc[k]),
            'risco_historico':r,'razao_risco_prevalencia':r/prev if prev else None,
            'rotulo_descritivo':op.get('rotulo_descritivo',{}).get('rotulo_alto','alto_risco_fraude') if elevado
                                  else op.get('rotulo_descritivo',{}).get('rotulo_baixo','baixo_risco_fraude'),
            'media_vetor_completo':gm.modelo_.means_[k].tolist(),
            'covariancia_completa':_covariancia(gm,k).tolist(),
            'covariance_type':gm.modelo_.covariance_type})
    arq=dest/'relatorio_gaussianas_desenvolvimento.json'
    if op.get('salvar_json',True) and not arq.exists():
        arq.write_text(json.dumps({'origem':'desenvolvimento_F1_F9','gaussianas':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if op.get('salvar_csv',True):
        txt=pd.DataFrame(rows)
        for col in ('media_vetor_completo','covariancia_completa'):
            txt[col]=txt[col].apply(json.dumps)
        vis._csv_protegido(txt,dest/'relatorio_gaussianas_desenvolvimento.csv')
    if op.get('salvar_markdown',True):
        linhas=['# Relatorio de gaussianas — desenvolvimento (F1..F9)','',
                'Nao ha resultados do F10 neste relatorio.','',
                '| Gaussiana | Peso GMM | Massa soft | Risco historico | Rotulo |',
                '|---|---:|---:|---:|---|']
        linhas += [f'| {r["gaussiana"]} | {r["peso_mistura"]:.5g} | {r["massa_responsabilidade_total"]:.5g} | '
                   f'{r["risco_historico"]:.6g} | {r["rotulo_descritivo"]} |' for r in rows]
        path=dest/'relatorio_gaussianas_desenvolvimento.md'
        if not path.exists():path.write_text('\n'.join(linhas)+'\n',encoding='utf-8')


def executar(experimento, *, raiz=None, somente_planejar=False):
    experimento=Path(experimento).resolve()
    raiz=Path(raiz).resolve() if raiz is not None else Path(__file__).resolve().parents[1]
    cfg,grade,escolha,_,_ = vis._ler_resultados(experimento)
    manifest=json.loads((experimento/'manifesto.json').read_text(encoding='utf-8-sig'))
    _exigir(manifest.get('status')=='concluido_06_03', 'Execucao OOF ainda nao concluida.')
    _exigir(escolha.get('teste_final_usado') is False,'Nao e permitido usar F10 na selecao.')
    ativos=[n for n in TIPOS if _ativo(cfg,n) and (n not in ('pca_cotovelo','pca_projecoes','tsne_mapa','tsne_diagnostico')
             or ('pca_' in n and escolha['configuracao']['metodo']=='PCA')
             or ('tsne_' in n and escolha['configuracao']['metodo']=='TSNE'))]
    print('PASSO 07.02 — GEOMETRIA DO VENCEDOR')
    print('Vencedor:',escolha['id_candidato'],escolha['configuracao'])
    print('Graficos aplicaveis:',', '.join(ativos))
    print('Ajustes GMM a executar:', 0 if somente_planejar else 1)
    print('F10: nao entra em nenhuma etapa do Passo 07.02')
    if somente_planejar:return {'id_vencedor':escolha['id_candidato'],'graficos':ativos,'planejamento':True}
    cfg,escolha,ids,y_oof,arquivo,folds=_selecao(experimento,raiz)
    df=orq._ler_desenvolvimento(arquivo,ids)
    _exigir(np.array_equal(df[cfg['dados']['coluna_alvo']].to_numpy(dtype=np.int8),y_oof),
            'Alvo da base mudou desde o OOF; treino bloqueado.')
    _exigir((folds[ids]!=10).all(),'F10 detectado no desenvolvimento.')
    print('Reajustando vencedor com',len(df),'transacoes F1..F9. Aguarde...')
    cand=escolha['configuracao'];jobs=int(cfg['execucao'].get('n_jobs',1))
    with threadpool_limits(limits=1 if jobs == -1 else max(1,jobs)):
        data=_treinar_vencedor(df,cfg,escolha)
    destino=experimento/cfg['saidas']['pastas']['teste_final']/cfg['saidas']['pastas']['graficos']
    destino.mkdir(parents=True,exist_ok=True)
    ext,dpi,csv=_estilo(cfg)
    funcoes={
        'risco_por_gaussiana':lambda:_risco_por_gaussiana(cfg,data,destino,ext,dpi,csv),
        'pesos_responsabilidades':lambda:_pesos(cfg,data,destino,ext,dpi,csv),
        'medias_gaussianas':lambda:_medias(cfg,data,destino,ext,dpi,csv),
        'covariancias_gaussianas':lambda:_covariancias(cfg,data,destino,ext,dpi,csv),
        'pca_cotovelo':lambda:_scree(cfg,data,destino,ext,dpi,csv),
        'pca_projecoes':lambda:_projecoes(cfg,data,destino,ext,dpi,csv,'PCA'),
        'tsne_mapa':lambda:_projecoes(cfg,data,destino,ext,dpi,csv,'TSNE'),
        'tsne_diagnostico':lambda:_tsne_diagnostico(cfg,data,destino,ext,dpi,csv),
        'mapa_risco_fronteira':lambda:_fronteira(cfg,data,escolha,destino,ext,dpi,csv),
    }
    for nome in ativos:
        print('[GRAFICO]',nome,flush=True)
        funcoes[nome]()
    _relatorio(cfg,data,destino)
    meta={'etapa':'07.02','origem':'desenvolvimento_F1_F9','f10_utilizado':False,
          'tipo_experimento':manifest['tipo'],'id_vencedor':escolha['id_candidato'],
          'configuracao':cand,'limiar_congelado_oof':float(escolha['limiar_mcc_oof']),
          'n_desenvolvimento':len(df),'n_treino_gmm':data['n_treino_gmm'],
          'n_fraudes_desenvolvimento':int(data['y'].sum()),
          'tempo_ajuste_s':data['tempo_total'],'tipos_aplicaveis':ativos,
          'tipos_omitidos':[n for n in TIPOS if n not in ativos],
          'aviso':'Geometria modelo desenvolvimento; NAO sao graficos ou metricas F10.'}
    path=destino/'manifesto_geometria_07_02.json'
    if not path.exists():path.write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Passo 07.02 concluido; arquivos em:',destino)
    return meta


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experimento',required=True,type=Path)
    modo=p.add_mutually_exclusive_group(required=True)
    modo.add_argument('--planejar',action='store_true')
    modo.add_argument('--executar',action='store_true')
    a=p.parse_args(argv)
    executar(a.experimento,somente_planejar=a.planejar)


if __name__=='__main__':main()
