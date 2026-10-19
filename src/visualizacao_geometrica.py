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

# Numeracao explicita: os graficos avulsos 07, 08 e 10 foram substituidos
# pelo painel GMM 09, conforme o catalogo. Nao renumerar PCA/t-SNE.
TIPOS = (
    'painel_parametros_gaussianos', 'pca_cotovelo', 'pca_projecoes',
    'tsne_mapa', 'tsne_diagnostico', 'mapa_risco_fronteira',
)
NUM = {'painel_parametros_gaussianos': 9, 'pca_cotovelo': 11,
       'pca_projecoes': 12, 'tsne_mapa': 13, 'tsne_diagnostico': 14,
       'mapa_risco_fronteira': 15}



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


def _regra_gaussianas(config, rh):
    """Unica regra de rotulagem DESCRITIVA. Nao altera previsoes nem o GMM.

    pi aqui e a prevalencia EMPIRICA dos rotulos de treino ORIGINAL usados
    nos soft counts (fold-train no OOF; F1..F9 no reajuste final).
    """
    rot = config['visualizacao']['relatorio_gaussianas']['rotulo_descritivo']
    _exigir(rot.get('estrategia') == 'relativo_prevalencia',
            'Somente rotulagem relativo_prevalencia e permitida.')
    multiplicador = float(rot['multiplicador_prevalencia'])
    _exigir(np.isfinite(multiplicador) and multiplicador > 0,
            'Multiplicador de prevalencia invalido.')
    prevalencia = float(rh.prevalencia_treino_)
    _exigir(np.isfinite(prevalencia) and 0 <= prevalencia <= 1,
            'Prevalencia do treino invalida.')
    return {'estrategia': 'relativo_prevalencia',
            'multiplicador_prevalencia': multiplicador,
            'prevalencia_treino': prevalencia,
            'limiar_alto_risco_gaussiana': multiplicador * prevalencia,
            'rotulo_alto': rot['rotulo_alto'],
            'rotulo_baixo': rot['rotulo_baixo'],
            'uso': 'exclusivamente descritivo; nao classifica transacoes'}


def _rotulo_gaussiana(risco, regra):
    """Rotulo explicativo; nenhum componente se torna exclusivamente fraude."""
    alto = regra['prevalencia_treino'] > 0 and risco >= regra['limiar_alto_risco_gaussiana']
    return (regra['rotulo_alto'] if alto else regra['rotulo_baixo'],
            'Alto risco de fraude' if alto else 'Baixo risco de fraude')


def _linhas_gaussianas(config, data):
    gm = data['gmm']; rh = data['rh']
    regra = _regra_gaussianas(config, rh)
    nomes = list(gm.saidas_)
    _exigir(nomes == list(rh.riscos_.index), 'Ordem de riscos e gaussianas diverge.')
    _exigir(len(nomes) == len(gm.modelo_.weights_), 'Pesos GMM inconsistentes.')
    rows = []
    for k, nome in enumerate(nomes):
        r = float(rh.riscos_.iloc[k]); massa = float(rh.massas_.iloc[k])
        fraudes = float(rh.fraudes_ponderadas_.iloc[k])
        rotulo, descricao = _rotulo_gaussiana(r, regra)
        rows.append({'gaussiana': nome,
            'peso_mistura': float(gm.modelo_.weights_[k]),
            'massa_responsabilidade_total': massa,
            'massa_fraude': fraudes,
            'massa_legitima': massa - fraudes,
            'risco_historico': r,
            'prevalencia_treino': regra['prevalencia_treino'],
            'multiplicador_prevalencia': regra['multiplicador_prevalencia'],
            'limiar_alto_risco_gaussiana': regra['limiar_alto_risco_gaussiana'],
            'razao_risco_prevalencia': r / regra['prevalencia_treino']
                 if regra['prevalencia_treino'] > 0 else None,
            'rotulo_descritivo': rotulo, 'descricao_rotulo': descricao,
            'media_vetor_completo': gm.modelo_.means_[k].tolist(),
            'covariancia_completa': _covariancia(gm, k).tolist(),
            'covariance_type': gm.modelo_.covariance_type})
    return rows, regra


def _painel_parametros_gaussianos(config, data, dest, ext, dpi, csv):
    """GMM 09: pesos, risco, status, medias e matriz Sigma, sem heatmaps.

    Pagina ate quatro gaussianas, com cores estaveis para reconciliar mapas.
    """
    op = config['visualizacao']['graficos']['painel_parametros_gaussianos']
    por_pagina = int(op.get('gaussianas_por_pagina', 4))
    _exigir(1 <= por_pagina <= 8, 'gaussianas_por_pagina deve ficar entre 1 e 8.')
    rows, regra = _linhas_gaussianas(config, data)
    d = len(rows[0]['media_vetor_completo'])
    _exigir(1 <= d <= 3, 'O painel do PDF foi definido apenas para 1D, 2D e 3D.')
    n_paginas = (len(rows) + por_pagina - 1) // por_pagina
    for pag in range(n_paginas):
        bloco = rows[pag * por_pagina:(pag + 1) * por_pagina]
        ncol = 2 if len(bloco) > 1 else 1
        nrow = (len(bloco) + ncol - 1) // ncol
        fig, axs = plt.subplots(nrow, ncol, figsize=(13.8 if ncol == 2 else 7.4,
                                                  1.65 + nrow * 4.6), squeeze=False)
        fig.patch.set_facecolor('#0d192d')
        titulo = f'GMM 09 — Painel de parâmetros gaussianos | {d}D | {data["gmm"].modelo_.covariance_type.upper()}'
        fig.suptitle(titulo, color='#f2f7ff', fontweight='bold', fontsize=17, y=.985)
        for j, ax in enumerate(axs.flat):
            ax.set_facecolor('#172b46')
            ax.set_xticks([]); ax.set_yticks([])
            for borda in ax.spines.values():
                borda.set_edgecolor('#38516d'); borda.set_linewidth(1.5)
            if j >= len(bloco):
                ax.set_visible(False)
                continue
            info = bloco[j]
            k = pag * por_pagina + j
            cor = plt.get_cmap('tab10')(k % 10)
            ax.plot([.05,.05], [.06,.93],color=cor,lw=5,transform=ax.transAxes,solid_capstyle='round')
            ax.text(.09,.89,info['gaussiana'],transform=ax.transAxes,
                    fontsize=16,fontweight='bold',color='#f5f9ff',va='center')
            alto = info['rotulo_descritivo'] == regra['rotulo_alto']
            ax.text(.94,.89,info['descricao_rotulo'],transform=ax.transAxes,
                    ha='right',va='center', fontsize=10,fontweight='bold',
                    color='#ffc99f' if alto else '#b9dcff')
            ax.text(.09,.74,f"Peso da mistura  πₖ = {info['peso_mistura']:.6g}",
                    transform=ax.transAxes,color='#e7f1ff',fontsize=11)
            ax.text(.09,.65,f"Risco histórico  rₖ = {info['risco_historico']:.6g} ({100*info['risco_historico']:.3f}%)",
                    transform=ax.transAxes,color='#e7f1ff',fontsize=11)
            medias = ', '.join(f'{v:.5g}' for v in info['media_vetor_completo'])
            ax.text(.09,.53,'Vetor de médias μₖ',transform=ax.transAxes,
                    color='#a9d4ff',fontsize=11,fontweight='bold')
            ax.text(.11,.445,'[' + medias + ']',transform=ax.transAxes,
                    color='#f5f9ff',fontsize=10.5,family='monospace')
            ax.text(.09,.34,'Matriz de covariâncias Σₖ',transform=ax.transAxes,
                    color='#a9d4ff',fontsize=11,fontweight='bold')
            linhas = [('[ ' if i == 0 else '  ') + '   '.join(f'{v: .4g}' for v in linha) +
                      (' ]' if i == d - 1 else '')
                      for i,linha in enumerate(info['covariancia_completa'])]
            ax.text(.105,.265,'\n'.join(linhas),transform=ax.transAxes,
                    color='#f5f9ff',fontsize=10.2,family='monospace',va='top',linespacing=1.35)
            ax.set_xlim(0,1);ax.set_ylim(0,1)
        fig.text(
                .5, .045,
                f"Regra DESCRITIVA: alto risco se rₖ ≥ {regra['multiplicador_prevalencia']:.3g} × prevalência do treino "
                f"({regra['prevalencia_treino']:.6%}) = {regra['limiar_alto_risco_gaussiana']:.6%}  |  "
                "Não classifica transações; F10 não utilizado.",
                ha='center',
                color='#c8d8ec',
                fontsize=9.5
                )
        fig.subplots_adjust(left=.055,right=.945,top=.88,bottom=.12,hspace=.18,wspace=.09)
        stem = '09_painel_parametros_gaussianos'
        if n_paginas > 1: stem += f'_parte_{pag+1:02d}'
        _guardar(fig,dest,stem,ext,dpi)
    # CSV longo contem parametros completos, nao apenas os arredondados na figura.
    if csv:
        tabela = pd.DataFrame(rows)
        for nome in ('media_vetor_completo','covariancia_completa'):
            tabela[nome] = tabela[nome].apply(json.dumps)
        _salvar_dados(tabela,dest,'09_painel_parametros_gaussianos',True)


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
            regra = _regra_gaussianas(config, data['rh'])
            for k,nome in enumerate(data['gmm'].saidas_):
                sub=idx[grupo[idx]==k]
                r = float(data['rh'].riscos_.iloc[k])
                _,descricao = _rotulo_gaussiana(r,regra)
                if len(sub):ax.scatter(coord[sub,par[0]],
                    (np.zeros(len(sub))+k*.008) if len(par)==1 else coord[sub,par[1]],
                    s=10,alpha=.5,color=plt.get_cmap('tab10')(k % 10),
                    label=f'{nome} — {descricao}',rasterized=True)
        if op.get('mostrar_contornos_gaussianas',False):
            gm=data['gmm']; nstd=2.
            for k in range(len(gm.saidas_)):
                if len(par)==1:
                    mean=gm.modelo_.means_[k,par[0]]
                    sd=np.sqrt(_covariancia(gm,k)[par[0],par[0]])
                    ax.axvspan(mean-nstd*sd,mean+nstd*sd,alpha=.06,color=plt.get_cmap('tab10')(k%10))
                else:
                    cov=_covariancia(gm,k)[np.ix_(par,par)]
                    eig,V=np.linalg.eigh(cov)
                    if (eig>0).all():
                        theta=np.degrees(np.arctan2(V[1,1],V[0,1]))
                        ax.add_patch(Ellipse(gm.modelo_.means_[k,list(par)],
                             width=2*nstd*np.sqrt(eig[1]),height=2*nstd*np.sqrt(eig[0]),
                             angle=theta,facecolor='none',edgecolor=plt.get_cmap('tab10')(k%10),alpha=.55))
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


def _mudanca_classes(score, tau_star):
    """Compara os mesmos scores nos dois cortes; sem labels da malha."""
    v = np.asarray(score, dtype=float)
    _exigir(v.size > 0 and np.isfinite(v).all() and np.all((0 <= v) & (v <= 1)),
            'Scores invalidos no mapa GMM08.')
    tau = float(tau_star)
    _exigir(np.isfinite(tau) and 0 <= tau <= 1, 'Limiar MCC OOF invalido.')
    fixo = v >= 0.50
    otimo = v >= tau
    mudou = np.logical_xor(fixo, otimo)
    if tau < 0.50:
        _exigir(np.all(~fixo | otimo), 'Regiao fixa nao contida na regiao MCC.')
    elif tau > 0.50:
        _exigir(np.all(~otimo | fixo), 'Regiao MCC nao contida na regiao fixa.')
    else:
        _exigir(not mudou.any(), 'Limiar igual alterou alguma classificacao.')
    return fixo, otimo, mudou


def _avaliar_cortes_oof(avaliacao, tau_star):
    """MCC sempre nos MESMOS dados OOF, nunca no desenvolvimento ou na malha."""
    from sklearn.metrics import matthews_corrcoef
    _exigir(isinstance(avaliacao, dict) and 'y' in avaliacao and 'p' in avaliacao,
            'Predicoes OOF necessarias para avaliar os cortes GMM08.')
    y = np.asarray(avaliacao['y'])
    p = np.asarray(avaliacao['p'], dtype=float)
    _exigir(y.ndim == p.ndim == 1 and len(y) == len(p) and len(y) > 0,
            'Rotulos e probabilidades OOF incompativeis.')
    _exigir(np.isin(y, [0, 1]).all(), 'Rotulos OOF nao binarios.')
    fixo, otimo, _ = _mudanca_classes(p, tau_star)
    return (float(matthews_corrcoef(y, fixo)),
            float(matthews_corrcoef(y, otimo)))


def _elipse_95_corte(gmm, k, par, fixos):
    """Intersecao verdadeira do elipsoide 3D FULL com plano de corte.

    Em 2D, regioes centrais definidas por chi2(df=2).
    Em 3D, recorte do elipsoide definido por chi2(df=3), SEM confundir
    a elipse seccional com uma marginal 2D.
    """
    from scipy.stats import chi2
    mu = np.asarray(gmm.modelo_.means_[k], dtype=float)
    cov = _covariancia(gmm, k)
    d = mu.size
    _exigir(d in (2, 3), 'Elipses do catalogo GMM08 somente para 2D/3D.')
    _exigir(len(par) == 2, 'A elipse exige um par de dimensoes.')
    par = tuple(par)
    outros = tuple(i for i in range(d) if i not in par)
    inv = np.linalg.inv(cov)
    a = inv[np.ix_(par, par)]
    centro = mu[list(par)].copy()
    raio2 = float(chi2.ppf(.95, df=d))
    if outros:
        delta = np.asarray(fixos, dtype=float)[list(outros)] - mu[list(outros)]
        b = inv[np.ix_(par, outros)] @ delta
        c = float(delta @ inv[np.ix_(outros, outros)] @ delta)
        desloc = np.linalg.solve(a, b)
        centro -= desloc
        raio2 -= c - float(b @ desloc)
    if raio2 <= 0:
        return None  # O plano nao intercepta o elipsoide 95%.
    autovalores, autovetores = np.linalg.eigh(np.linalg.inv(a))
    _exigir(np.all(autovalores > 0), 'Covariancia FULL sem autovalores positivos.')
    largura, altura = 2 * np.sqrt(raio2 * autovalores[::-1])
    angulo = np.degrees(np.arctan2(autovetores[1, 1], autovetores[0, 1]))
    return centro, float(largura), float(altura), float(angulo)


def _fronteira_1d_comparativa(config, data, gm, op, plano, idx, tau_star,
                            mcc_fixo, mcc_otimo, dest, ext, dpi, teto):
    """Um único arquivo, eixos iguais; cortes da MESMA função de risco.

    Sombras 95% referem-se aos componentes, não à decisão binária.
    A hachura no painel MCC marca apenas a discordância com o corte 0,50.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.stats import norm

    (par, grade, campo, classe_fixa, classe_mcc, mudou) = plano
    x = grade[0]
    coluna = par[0]
    X = data['coord'].to_numpy(dtype=float)
    y = np.asarray(data['y'], dtype=np.uint8)
    n_comp = len(gm.saidas_)
    _exigir(len(x) == len(campo), 'Malha unidimensional inconsistente.')

    fig, axes = plt.subplots(1, 2, figsize=(16.4, 7.4), sharex=True, sharey=True)
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle('GMM 08 — Mapa de risco e geometria 95% | '
                 f'{gm.modelo_.covariance_type.upper()} | 1D',
                 fontsize=16, fontweight='bold', y=.983)
    nome_pc = data['coord'].columns[coluna]
    for j, (ax, tau, mcc, mascara) in enumerate((
        (axes[0], .5, mcc_fixo, classe_fixa),
        (axes[1], tau_star, mcc_otimo, classe_mcc),
    )):
        # Mesmos limites e mesmos dados em ambos os painéis.
        ax.plot(x, campo, lw=2.5, color='#2361a1', zorder=6)
        for classe, cor in ((0,'#4389c2'), (1,'#e58d31')):
            ii=idx[y[idx]==classe]
            if len(ii):
                ax.scatter(X[ii,coluna],np.asarray(data['scores'])[ii],
                           s=8, alpha=.15, color=cor, rasterized=True, zorder=5)

        if op.get('mostrar_gaussianas', True):
            for k in range(n_comp):
                media=float(gm.modelo_.means_[k,coluna])
                sigma=float(np.sqrt(_covariancia(gm,k)[coluna,coluna]))
                r=norm.ppf(.975)*sigma
                cor=plt.get_cmap('tab10')(k%10)
                ax.axvspan(media-r,media+r,facecolor=cor,alpha=.055,zorder=1)
                ax.axvline(media-r,color=cor,lw=1,ls='--',alpha=.65,zorder=2)
                ax.axvline(media+r,color=cor,lw=1,ls='--',alpha=.65,zorder=2)
                ax.axvline(media,color=cor,lw=1,ls=':',alpha=.75,zorder=2)

        # Classe positiva definida pelo limiar específico de cada painel.
        # Não confundir essa classificação com as faixas gaussianas.
        if mascara.any():
            ax.fill_between(x, 0, teto, where=mascara,
                            color='#efb35e', alpha=.085, step=None,zorder=0)

        if j==1 and mudou.any():
            limites=np.flatnonzero(np.diff(np.r_[False,mudou,False]))
            for start,end in limites.reshape(-1,2):
                ax.axvspan(x[start],x[min(end,len(x)-1)],
                           facecolor='#f3c747',alpha=.24,hatch='////',
                           edgecolor='#b38904',lw=.4,zorder=3)

        # Em 1D, as fronteiras são valores de PC1, NÃO linhas horizontais.
        if campo.min() <= tau <= campo.max():
            inds=np.flatnonzero((campo[:-1]-tau)*(campo[1:]-tau)<=0)
            for pos in inds:
                y1,y2=campo[pos],campo[pos+1]
                x0=x[pos] if y1==y2 else x[pos]+(tau-y1)*(x[pos+1]-x[pos])/(y2-y1)
                ax.axvline(x0,color='#171717',lw=2,zorder=7)
            if tau <= teto:
                ax.axhline(tau,color='#222222',ls=':',lw=1,alpha=.55,zorder=4)
        else:
            ax.text(.03,.96,
                    f'Sem fronteira: risco não cruza τ={tau:.6g} nesta faixa',
                    transform=ax.transAxes,va='top',fontsize=9.3,color='#353b45',
                    bbox={'facecolor':'white','edgecolor':'#ccd3dc','alpha':.92},
                    zorder=8)

        titulo=f'A — Corte fixo τ = 0,50' if j==0 else f'B — Corte ótimo MCC OOF τ* = {tau_star:.6g}'
        ax.set_title(f'{titulo}\nMCC OOF = {mcc:.5f}',fontsize=11.3,pad=10,fontweight='medium')
        ax.set(xlabel=nome_pc,xlim=(x[0],x[-1]),ylim=(0,teto))
        ax.grid(alpha=.12)
    axes[0].set_ylabel('Probabilidade estimada de fraude')

    legenda=[Line2D([0],[0],color='#2361a1',lw=2.6,label='Risco contínuo'),
             Line2D([0],[0],color='#171717',lw=2,label='Fronteira de decisão'),
             Patch(facecolor='#f3c747',edgecolor='#b38904',hatch='////',
                   label='Mudança entre cortes'),
             Line2D([0],[0],ls='',marker='o',markersize=5,color='#4389c2',label='Legítimas'),
             Line2D([0],[0],ls='',marker='o',markersize=5,color='#e58d31',label='Fraudes')]
    if op.get('mostrar_gaussianas', True):
        legenda += [Line2D([0],[0],color=plt.get_cmap('tab10')(k%10),
                            ls='--',lw=2,label=f'{gm.saidas_[k]} — intervalo 95%')
                    for k in range(n_comp)]
    fig.legend(handles=legenda,loc='upper center',bbox_to_anchor=(.5,.926),
               ncol=min(5,len(legenda)),fontsize=8.7,frameon=False,
               columnspacing=1.6,handlelength=2.7)
    direcao=('legítima → fraude' if tau_star < .5 else
             'fraude → legítima' if tau_star > .5 else 'nenhuma')
    fig.text(.5,.065,
             f'MCC OOF(0,50) = {mcc_fixo:.5f}    |    MCC OOF(τ*={tau_star:.6g}) = {mcc_otimo:.5f}'
             f'    |    Hachura: mudança de classe ({direcao})',
             fontsize=10,ha='center',color='#253652',fontweight='bold')
    fig.text(.5,.035,
             'Linhas coloridas = intervalos centrais gaussianos de 95%, não regiões de fraude. '
             'MCC calculado nas predições OOF; geometria ajustada em F1–F9; F10 não utilizado.',
             fontsize=8.6,ha='center',color='#506079')
    fig.subplots_adjust(left=.07,right=.985,top=.76,bottom=.18,wspace=.12)
    _guardar(fig,dest,'GMM08_mapa_risco_1d_comparativo',ext,dpi)


def _fronteira(config, data, escolha, dest, ext, dpi, csv, *, avaliacao):
    """GMM08 PDF: mapas fixo e MCC com geometria FULL e mudancas hachuradas.

    A malha e o campo de score sao identicos nos dois arquivos gerados.
    Somente a fronteira e a regiao hachurada dependem do corte.
    """
    from scipy.stats import norm
    op = config['visualizacao']['graficos']['mapa_risco_fronteira']
    X = data['coord'].to_numpy(dtype=float)
    y = np.asarray(data['y'], dtype=np.uint8)
    gm = data['gmm']; rh = data['rh']; nomes = list(data['coord'].columns)
    _exigir(X.shape[1] in (1, 2, 3), 'GMM08 PDF so descreve 1D, 2D, 3D.')
    _exigir(np.isclose(float(op.get('cobertura_gaussiana',.95)),.95),
            'GMM08 exige cobertura gaussiana 95%.')
    tau_star = float(escolha['limiar_mcc_oof'])
    mcc_fixo, mcc_otimo = _avaliar_cortes_oof(avaliacao, tau_star)
    idx = _amostra_plot(len(y), y, config, int(op.get('n_pontos_scatter', 5000)))
    margem = float(op.get('margem', .15))
    _exigir(0 <= margem <= 1, 'Margem invalida.')
    par_list = _pares(X.shape[1], config)
    # Comparacao conjunta exige os tres cortes quando o modelo e 3D.
    if X.shape[1] == 3:
        _exigir(set(par_list) == {(0,1),(0,2),(1,2)},
                'GMM08 3D exige os cortes PC1-PC2, PC1-PC3 e PC2-PC3.')
    n = int(op.get('resolucao_1d', 500) if X.shape[1] == 1 else op.get('resolucao_2d', 180))
    _exigir(20 <= n <= (20000 if X.shape[1] == 1 else 450), 'Resolucao de mapa invalida.')
    fixos = X.mean(axis=0)
    planos = []
    for par in par_list:
        if len(par) == 1:
            x = np.linspace(*_faixa(X[:,par[0]], margem), n)
            malha = np.tile(fixos, (n, 1))
            malha[:,par[0]] = x
            campo = _pontuar_gmm_array(gm, rh, malha)
            grade = (x,)
        else:
            i,j = par
            x = np.linspace(*_faixa(X[:,i], margem), n)
            z = np.linspace(*_faixa(X[:,j], margem), n)
            xx, yy = np.meshgrid(x, z)
            malha = np.tile(fixos, (n*n, 1))
            malha[:,i] = xx.ravel(); malha[:,j] = yy.ravel()
            campo = _pontuar_gmm_array(gm, rh, malha).reshape(n, n)
            grade = (xx, yy)
        fixo, otimo, mudou = _mudanca_classes(campo, tau_star)
        planos.append((par, grade, campo, fixo, otimo, mudou))
        # Uma tabela por corte geométrico preserva TODOS os pontos da malha.
        if csv:
            mapa = {'score_risco':campo.ravel(), 'classe_corte_fixo':fixo.ravel().astype(int),
                    'classe_corte_mcc':otimo.ravel().astype(int),
                    'mudou_classe':mudou.ravel().astype(int)}
            if len(par) == 1: mapa[nomes[par[0]]] = grade[0]
            else:
                mapa[nomes[par[0]]] = grade[0].ravel()
                mapa[nomes[par[1]]] = grade[1].ravel()
            _salvar_dados(pd.DataFrame(mapa), dest,
                          'GMM08_malha_'+'_'.join(nomes[i] for i in par), True)
    # Escala de risco SEMPRE identica nos dois cortes e nos tres planos de 3D.
    maior = max(float(np.max(campo)) for _,_,campo,_,_,_ in planos)
    _exigir(maior > 0, 'Campo de risco sem valores positivos.')
    teto_1d = min(1, max(maior*1.16, tau_star*1.12, .002))
    if X.shape[1] == 1:
        _fronteira_1d_comparativa(config,data,gm,op,planos[0],idx,tau_star,
                                mcc_fixo,mcc_otimo,dest,ext,dpi,teto_1d)
        return
    # A origem do MCC deve ser explícita. Testes sintéticos não são OOF.
    origem_mcc = avaliacao.get('origem')
    if origem_mcc is not None:
        _exigir(origem_mcc in ('oof', 'simulado'),
                'Origem da avaliação GMM08 desconhecida.')
    if origem_mcc == 'oof':
        nome_mcc = 'MCC OOF'
        nota_mcc = 'MCC das predições OOF (F1–F9)'
    elif origem_mcc == 'simulado':
        nome_mcc = 'MCC da simulação'
        nota_mcc = 'PRÉVIA SINTÉTICA: MCC calculado apenas nos dados simulados; não é OOF real'
    else:
        nome_mcc = 'MCC (origem não identificada)'
        nota_mcc = 'Origem da avaliação não informada; não afirmar que é OOF'

    for tipo, tau, mcc in [('fixo', .5, mcc_fixo), ('mcc', tau_star, mcc_otimo)]:
        if len(par_list) == 1:
            fig, axs = plt.subplots(1, 1, figsize=(10.8, 7.3))
        elif X.shape[1] == 3:
            fig, axs = plt.subplots(1, len(par_list), figsize=(18.8, 7.2))
        else:
            fig, axs = plt.subplots(1, len(par_list), figsize=(6*len(par_list), 6.2))
        axes = np.ravel(np.asarray(axs, dtype=object))
        for ax, (par, grade, campo, fixo, otimo, mudou) in zip(axes, planos):
            if len(par) == 1:
                x = grade[0]
                ax.plot(x, campo, color='#2361a1', lw=2.4, label='Risco conjunto do GMM')
                for classe,nome,cor in ((0,'Legítimas','#4389c2'),(1,'Fraudes','#e58d31')):
                    ii = idx[y[idx] == classe]
                    if len(ii): ax.scatter(X[ii,par[0]], data['scores'][ii],
                       color=cor, s=7, alpha=.18, label=nome, rasterized=True)
                if op.get('mostrar_gaussianas', True):
                    for k in range(len(gm.saidas_)):
                        media = float(gm.modelo_.means_[k, par[0]])
                        sigma = np.sqrt(_covariancia(gm,k)[par[0],par[0]])
                        r = norm.ppf(.975) * sigma
                        cor = plt.get_cmap('tab10')(k % 10)
                        ax.axvspan(media-r, media+r, facecolor=cor, alpha=.06)
                        ax.axvline(media-r, color=cor, lw=1, ls='--', alpha=.7)
                        ax.axvline(media+r, color=cor, lw=1, ls='--', alpha=.7)
                        ax.axvline(media, color=cor, lw=1, ls=':', alpha=.8)
                # Destacar intervalos contiguos de mudanca na curva 1D.
                if tipo == 'mcc' and mudou.any():
                    limites = np.flatnonzero(np.diff(np.r_[False, mudou, False]))
                    for start, end in limites.reshape(-1,2):
                        ax.axvspan(x[start], x[min(end,len(x)-1)],
                                   facecolor='#f3c747', alpha=.25, hatch='////',
                                   edgecolor='#b38904', lw=.3)
                if campo.min() < tau < campo.max():
                    # Em 1D, a FRONTEIRA sao as coordenadas x em que s(x)=tau.
                    inds = np.flatnonzero((campo[:-1]-tau)*(campo[1:]-tau) <= 0)
                    for pos in inds:
                        y1, y2 = campo[pos], campo[pos+1]
                        x0 = x[pos] if y1 == y2 else x[pos]+(tau-y1)*(x[pos+1]-x[pos])/(y2-y1)
                        ax.axvline(x0, color='black', lw=1.8, ls='-',
                                   label=f'Fronteira s(x)={tau:.6g}' if pos==inds[0] else None)
                    ax.axhline(tau, color='black', ls=':', lw=1, alpha=.55)
                else:
                    ax.text(.02,.96, f'Sem fronteira: score não cruza τ={tau:.6g} neste intervalo',
                            transform=ax.transAxes, va='top', fontsize=9)
                ax.set(xlabel=nomes[par[0]], ylabel='Probabilidade estimada de fraude',
                       ylim=(0, teto_1d), xlim=(x[0],x[-1]))
                ax.legend(fontsize=8,loc='upper right')
            else:
                xx, yy = grade
                quad = ax.pcolormesh(xx,yy,campo,cmap='viridis',shading='auto',vmin=0,vmax=maior)
                _scatter(ax, X, y, idx, config, par)
                if op.get('mostrar_gaussianas', True):
                    for k in range(len(gm.saidas_)):
                        geo = _elipse_95_corte(gm,k,par,fixos)
                        if geo is None: continue
                        centro,largura,altura,angulo = geo
                        cor = plt.get_cmap('tab10')(k % 10)
                        ax.add_patch(Ellipse(centro,largura,altura,angle=angulo,
                                           ec=cor,fc='none',ls='--',lw=1.6))
                        ax.plot(centro[0],centro[1],marker='x',color=cor,ms=7,mew=2)
                if tipo == 'mcc' and mudou.any():
                    h = ax.contourf(xx,yy,mudou.astype(float), levels=[.5,1.5],
                                    colors='none',hatches=['////'])
                    h.set_edgecolor('#ffd247')
                    h.set_linewidth(.1)
                if campo.min() < tau < campo.max():
                    ax.contour(xx,yy,campo,levels=[tau],colors='black',linewidths=2)
                else:
                    ax.text(.02,.03, f'Sem contorno s(x)={tau:.5g} neste plano',
                            transform=ax.transAxes,color='white',fontsize=8,
                            bbox={'facecolor':'#152233','alpha':.8,'edgecolor':'none'})
                ax.set(xlabel=nomes[par[0]],ylabel=nomes[par[1]],
                       xlim=(xx[0,0],xx[0,-1]),ylim=(yy[0,0],yy[-1,0]))
                if X.shape[1] == 3:
                    omitida = next(k for k in range(3) if k not in par)
                    ax.set_title(
                        f'Corte {nomes[par[0]]} × {nomes[par[1]]}\n'
                        f'{nomes[omitida]} fixada na média do desenvolvimento = {fixos[omitida]:.3g}',
                        fontsize=9.5, pad=10)
                else:
                    ax.set_title(f'Corte {nomes[par[0]]} × {nomes[par[1]]}',fontsize=10)
        if X.shape[1] > 1:
            # No 3D, reservar margem DIREITA incluindo ticklabels e rótulo.
            cax = (fig.add_axes([.888,.23,.012,.54]) if X.shape[1] == 3
                   else fig.add_axes([.925,.21,.012,.57]))
            fig.colorbar(quad, cax=cax,
                         label='Probabilidade estimada de fraude (mesma escala)')
        nome_corte = 'Fixo 0,50' if tipo == 'fixo' else 'Ótimo MCC OOF'
        if X.shape[1] == 3 and origem_mcc != 'oof':
            nome_corte = 'Fixo 0,50' if tipo == 'fixo' else 'Ótimo MCC — ' + (
                'simulação' if origem_mcc == 'simulado' else 'avaliação sem origem')
        fig.suptitle(f'GMM 08 — Mapa de risco e geometria 95% | {nome_corte} | '
                     f'{gm.modelo_.covariance_type.upper()} | {X.shape[1]}D',
                     fontsize=15,fontweight='bold',y=.984)
        direcao=('Legítima → fraude' if tau_star < .5 else
                 'Fraude → legítima' if tau_star > .5 else 'Nenhuma mudança')
        if X.shape[1] == 3:
            # Mesmas cores tab10 do Painel GMM 09, para TODOS os componentes.
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            legenda = [
                Line2D([0],[0],color='black',lw=2,label='Fronteira da decisão'),
                Patch(facecolor='none',edgecolor='#d6a915',hatch='////',
                      label='Mudança entre os cortes'),
                Line2D([0],[0],ls='',marker='o',color='#4389c2',
                       label='Legítimas — projeção dos pontos'),
                Line2D([0],[0],ls='',marker='*',color='#d62728',
                       label='Fraudes — projeção dos pontos'),
            ]
            if op.get('mostrar_gaussianas',True):
                legenda.extend(Line2D([0],[0],color=plt.get_cmap('tab10')(k%10),
                          ls='--',lw=1.8,label=f'{nome} — seção 95%')
                          for k,nome in enumerate(gm.saidas_))
            fig.legend(handles=legenda,loc='upper center',
                       bbox_to_anchor=(.48,.928),ncol=min(6,len(legenda)),
                       frameon=False,fontsize=8.5,columnspacing=1.1,
                       handlelength=2.4)
            fig.text(.49,.095,
                     f'{nome_mcc}(τ={tau:.6g}) = {mcc:.5f}   |   '
                     f'Hachura amarela: mudança de classe ({direcao})',
                     fontsize=9.7,ha='center',color='#23334a',weight='bold')
            fig.text(.49,.063,
                     f'{nota_mcc}. Pontos = projeções das observações; os campos são cortes '
                     'com a terceira dimensão fixada na média de F1–F9.',
                     fontsize=8.5,ha='center',color='#4a5970')
            fig.text(.49,.037,
                     'Curvas tracejadas = seções dos elipsoides FULL de 95%, '
                     'não regiões classificadas como fraude. F10 não utilizado.',
                     fontsize=8.3,ha='center',color='#4a5970')
            fig.subplots_adjust(left=.06,right=.855,top=.765,bottom=.18,wspace=.30)
        else:
            fig.text(.5,.035,
                     f'MCC OOF(τ={tau:.6g})={mcc:.5f}  |  '
                     f'Hachura amarela: mudança entre cortes ({direcao}); '
                     'gaussianas 95% ≠ regiões de fraude. Modelo geométrico: desenvolvimento F1–F9.',
                     fontsize=9,ha='center',color='#23334a')
            fig.subplots_adjust(left=.075,right=.895,top=.87,bottom=.16,wspace=.38)
        stem=f'GMM08_mapa_risco_{X.shape[1]}d_{tipo}'
        _guardar(fig,dest,stem,ext,dpi)



def _relatorio(config,data,dest):
    op=config['visualizacao'].get('relatorio_gaussianas',{})
    if not op.get('habilitar') or 'teste_final' not in op.get('gerar_em',[]):return
    rows, regra = _linhas_gaussianas(config,data)
    arq=dest/'relatorio_gaussianas_desenvolvimento.json'
    if op.get('salvar_json',True) and not arq.exists():
        arq.write_text(json.dumps({'origem':'desenvolvimento_F1_F9',
              'regra_rotulagem_descritiva':regra, 'gaussianas':rows},
              ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if op.get('salvar_csv',True):
        txt=pd.DataFrame(rows)
        for col in ('media_vetor_completo','covariancia_completa'):
            txt[col]=txt[col].apply(json.dumps)
        vis._csv_protegido(txt,dest/'relatorio_gaussianas_desenvolvimento.csv')
    if op.get('salvar_markdown',True):
        linhas=['# Relatorio de gaussianas — desenvolvimento (F1..F9)','',
                'Nao ha resultados do F10 neste relatorio.','',
                '## Regra de rotulagem EXCLUSIVAMENTE descritiva','',
                f"Prevalencia no treino original: {regra['prevalencia_treino']:.8%}.",
                f"Multiplicador: {regra['multiplicador_prevalencia']:.6g}.",
                f"Limiar de alto risco: {regra['limiar_alto_risco_gaussiana']:.8%}.",
                'Alto risco se r_k >= multiplicador * prevalencia do treino original; ',
                'caso contrario, baixo risco. Esta etiqueta nao altera soft assignment,',
                'probabilidade de fraude nem limiar MCC/0,50 das transacoes.','',
                '| Gaussiana | Peso GMM | Massa soft | Risco historico | Rotulo |',
                '|---|---:|---:|---:|---|']
        linhas += [f'| {r["gaussiana"]} | {r["peso_mistura"]:.5g} | {r["massa_responsabilidade_total"]:.5g} | '
                   f'{r["risco_historico"]:.6g} | {r["rotulo_descritivo"]} |' for r in rows]
        path=dest/'relatorio_gaussianas_desenvolvimento.md'
        if not path.exists():path.write_text('\n'.join(linhas)+'\n',encoding='utf-8')


def executar(experimento, *, raiz=None, somente_planejar=False):
    experimento=Path(experimento).resolve()
    raiz=Path(raiz).resolve() if raiz is not None else Path(__file__).resolve().parents[1]
    cfg,grade,escolha,y_av_oof,p_av_oof = vis._ler_resultados(experimento)
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
        'painel_parametros_gaussianos':lambda:_painel_parametros_gaussianos(cfg,data,destino,ext,dpi,csv),
        'pca_cotovelo':lambda:_scree(cfg,data,destino,ext,dpi,csv),
        'pca_projecoes':lambda:_projecoes(cfg,data,destino,ext,dpi,csv,'PCA'),
        'tsne_mapa':lambda:_projecoes(cfg,data,destino,ext,dpi,csv,'TSNE'),
        'tsne_diagnostico':lambda:_tsne_diagnostico(cfg,data,destino,ext,dpi,csv),
        'mapa_risco_fronteira':lambda:_fronteira(cfg,data,escolha,destino,ext,dpi,csv,
                                               avaliacao={'y':y_av_oof, 'p':p_av_oof, 'origem':'oof'}),
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
          'tempo_ajuste_s':data['tempo_total'],
           'regra_rotulagem_descritiva':_regra_gaussianas(cfg,data['rh']),
           'tipos_aplicaveis':ativos,
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
