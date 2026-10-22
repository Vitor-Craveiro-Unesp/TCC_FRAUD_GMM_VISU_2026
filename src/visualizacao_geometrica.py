"""Passo 07.02 — diagnosticos geometricos do VENCEDOR OOF.

    python -m src.visualizacao_geometrica --experimento outputs/ensaio_pca_gmm_... --planejar
    python -m src.visualizacao_geometrica --experimento outputs/ensaio_pca_gmm_... --executar

Refaz UM ajuste final do vencedor no MESMO desenvolvimento (F1..F9) do
experimento/ensaio. Nao lê F10 para treino, graficos ou selecao. A pasta
"vencedor/visualizacao/geometria_desenvolvimento" contem diagnósticos do
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
from src import estrutura_saida as layout
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
    'painel_parametros_gaussianos', 'pca_cotovelo', 'pca_projecoes', 'pca_biplot',
    'tsne_trustworthiness', 'tsne_mapa', 'tsne_diagnostico', 'mapa_risco_fronteira',
    'sensibilidade_fronteira', 'densidades_marginais', 'regioes_dominancia',
)
NUM = {'densidades_marginais': 1, 'regioes_dominancia': 4, 'painel_parametros_gaussianos': 9, 'pca_cotovelo': 11,
       'pca_projecoes': 12, 'tsne_trustworthiness': 2, 'tsne_mapa': 13, 'tsne_diagnostico': 14,
       'mapa_risco_fronteira': 8, 'sensibilidade_fronteira': 7}



def _exigir(cond, msg):
    if not cond:
        raise ValueError(msg)


def _ativo(config, nome):
    # Graficos geometricos estao na fase 'teste_final' do YAML, MAS
    # sua origem e exclusivamente o desenvolvimento F1..F9.
    # Experimentos antigos podem nao ter este novo grafico no YAML congelado.
    if nome in ('sensibilidade_fronteira','densidades_marginais','regioes_dominancia') and nome not in config.get('visualizacao', {}).get('graficos', {}):
        return False
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
    caminho = layout.predicoes_vencedor(experimento)
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
    inicio_reducao = perf_counter()
    if cand['metodo'] == 'PCA':
        red = RedutorPCA(config,cand['reducao']).fit(x_pre)
        coord = red.transform(x_pre)
    elif cand['metodo'] == 'TSNE':
        red = RedutorTSNE(config,cand['reducao']).fit(x_pre)
        coord = red.transform_treino()
    else:
        raise ValueError('Metodo de reducao desconhecido.')
    tempo_reducao = perf_counter() - inicio_reducao
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
            'tempo_reducao':tempo_reducao,'tempo_total':perf_counter()-t0,
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

    def numero7(valor):
        valor = float(valor)
        return f'{0.0 if abs(valor) < 0.00000005 else valor:.7f}'
    d = len(rows[0]['media_vetor_completo'])

    _exigir(1 <= d <= 3, 'O painel do PDF foi definido apenas para 1D, 2D e 3D.')
    n_paginas = (len(rows) + por_pagina - 1) // por_pagina
    for pag in range(n_paginas):
        bloco = rows[pag * por_pagina:(pag + 1) * por_pagina]
        ncol = 2 if len(bloco) > 1 else 1
        nrow = (len(bloco) + ncol - 1) // ncol



        altura_fig = 1.65 + nrow * 4.6

        fig, axs = plt.subplots(
            nrow,
            ncol,
            figsize=(13.8 if ncol == 2 else 7.4, altura_fig),
            squeeze=False
        )

        fig.patch.set_facecolor('#0d192d')

        fig.suptitle(
            'PAINEL DE PARÂMETROS GAUSSIANOS',
            color='#f2f7ff',
            fontweight='bold',
            fontsize=18,
            y=.985
        )

        y_subtitulo = .985 - (0.50 / altura_fig)

        fig.text(
            .5,
            y_subtitulo,
            f'PARTE {pag + 1:02d} DE {n_paginas:02d}  |  '
            f'{d}D  |  GMM {data["gmm"].modelo_.covariance_type.upper()}',
            ha='center',
            va='top',
            color='#b9dcff',
            fontsize=10.5
        )


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

            status = (
                'Acima do critério de alto risco'
                if alto else
                'Abaixo do critério de alto risco'
            )

            ax.text(
                .94, .89, status,
                transform=ax.transAxes,
                ha='right',
                va='center',
                fontsize=9.2,
                fontweight='bold',
                color='#ffc99f' if alto else '#b9dcff'
            )

            if not alto and info['risco_historico'] > regra['prevalencia_treino']:
                ax.text(
                    .94, .815,
                    'Acima da prevalência do treino',
                    transform=ax.transAxes,
                    ha='right',
                    va='center',
                    fontsize=8.3,
                    color='#b9dcff'
                )

            
            ax.text(
                .09, .74,
                f"Peso da mistura  πₖ = {numero7(info['peso_mistura'])}",
                transform=ax.transAxes,
                color='#e7f1ff',
                fontsize=11
            )

            ax.text(
                .09, .65,
                f"Risco histórico  rₖ = {numero7(info['risco_historico'])}",
                transform=ax.transAxes,
                color='#e7f1ff',
                fontsize=11
            )


            
            medias = ', '.join(
                numero7(v) for v in info['media_vetor_completo']
            )


            ax.text(.09,.53,'Vetor de médias μₖ',transform=ax.transAxes,
                    color='#a9d4ff',fontsize=11,fontweight='bold')
            ax.text(.11,.445,'[' + medias + ']',transform=ax.transAxes,
                    color='#f5f9ff',fontsize=10.5,family='monospace')
            ax.text(.09,.34,'Matriz de covariâncias Σₖ',transform=ax.transAxes,
                    color='#a9d4ff',fontsize=11,fontweight='bold')

            
            linhas = [
                ('[ ' if i == 0 else '  ')
                + '  '.join(f'{numero7(v):>11}' for v in linha)
                + (' ]' if i == d - 1 else '')
                for i, linha in enumerate(info['covariancia_completa'])
            ]

            
            ax.text(.105,.265,'\n'.join(linhas),transform=ax.transAxes,
                    color='#f5f9ff',fontsize=10.2,family='monospace',va='top',linespacing=1.35)
            ax.set_xlim(0,1);ax.set_ylim(0,1)





        origem_gmm = (
            'RUS'
            if data['n_treino_gmm'] < len(data['y'])
            else 'amostra completa'
        )

        fig.text(
            .5, .069,
            f"Critério descritivo: rₖ ≥ "
            f"{numero7(regra['multiplicador_prevalencia'])} × "
            f"{numero7(regra['prevalencia_treino'])} = "
            f"{numero7(regra['limiar_alto_risco_gaussiana'])}",
            ha='center',
            color='#f2f7ff',
            fontsize=9.5
        )

        fig.text(
            .5, .037,
            f"πₖ, μₖ e Σₖ: GMM ajustado com {origem_gmm}; "
            "rₖ: responsabilidades e rótulos do treino original.",
            ha='center',
            color='#c8d8ec',
            fontsize=9.0
        )


        
        topo_cartoes = y_subtitulo - (0.95 / altura_fig)

        fig.subplots_adjust(
            left=.055,
            right=.945,
            top=topo_cartoes,
            bottom=.14,
            hspace=.18,
            wspace=.09
        )


        stem = '09_painel_parametros_gaussianos'
        if n_paginas > 1: stem += f'_parte_{pag+1:02d}'
        _guardar(fig,dest,stem,ext,dpi)
    # CSV longo contem parametros completos, nao apenas os arredondados na figura.
    if csv:
        tabela = pd.DataFrame(rows)
        for nome in ('media_vetor_completo','covariancia_completa'):
            tabela[nome] = tabela[nome].apply(json.dumps)
        _salvar_dados(tabela,dest,'09_painel_parametros_gaussianos',True)


def _scree(config, data, dest, ext, dpi, csv, *, origem='desenvolvimento'):
    """PCA 11 (PDF): barras individuais + curva acumulada em eixos distintos.

    PCA diagnostico sobre desenvolvimento pre-processado. O vencedor NAO muda;
    o grafico exibe o espectro completo, mesmo quando o vencedor tem PC1 so.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    x = data['pre'].to_numpy(dtype=np.float32, copy=False)
    n_disponiveis = min(len(x), x.shape[1])
    _exigir(n_disponiveis >= 1, 'PCA diagnostico sem componentes.')
    pca = PCA(n_components=n_disponiveis, svd_solver='full').fit(x)
    ratio = np.asarray(pca.explained_variance_ratio_, dtype=float)
    absoluto = np.asarray(pca.explained_variance_, dtype=float)
    acumulada = np.cumsum(ratio)
    _exigir(np.isfinite(ratio).all() and (ratio >= 0).all(),
            'Variancia explicada PCA invalida.')
    nsel = int(data['coord'].shape[1])
    _exigir(1 <= nsel <= len(ratio), 'Numero de PCs vencedor invalido.')
    op = config['visualizacao']['graficos']['pca_cotovelo']
    nums = np.arange(1, len(ratio)+1)
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    ax_acum = ax.twinx()
    barras = op.get('mostrar_variancia_individual', True)
    linha = op.get('mostrar_variancia_acumulada', True)
    if barras:
        ax.bar(nums, ratio, width=.74, color='#83bb79', alpha=.82,
               edgecolor='#5c9a55', linewidth=.45, zorder=2)
    if linha:
        ax_acum.plot(nums, acumulada, color='#d92e2e', linewidth=2.2,
                     marker='o', markersize=3.0, zorder=5)
        for alvo in op.get('linhas_referencia_acumulada', [0.80, 0.90, 0.95]):
            valor = float(alvo)
            _exigir(np.isfinite(valor) and 0 < valor <= 1,
                    'Referencia de variancia acumulada invalida.')
            ax_acum.axhline(valor, color='#64748b', linestyle=':',
                            linewidth=.85, alpha=.55, zorder=1)

            
            ax_acum.text(
                .995, valor,
                f'{valor:.7f}',
                transform=ax_acum.get_yaxis_transform(),
                ha='right',
                va='bottom',
                fontsize=7.8,
                color='#69788b'
            )

            
        for j in op.get('marcar_acumulados', [1, 2, 3]):
            k = int(j)
            if 1 <= k <= len(acumulada):
                ax_acum.scatter([k], [acumulada[k-1]], s=30, zorder=7,
                                facecolor='#d92e2e', edgecolor='white', linewidth=.5)
    if op.get('destacar_componentes_selecionados', True):
        ax.axvline(nsel, color='#101820', linewidth=1.65,
                   linestyle='--', alpha=.85, zorder=6)

        
        
    ax.set_xlabel('Componente principal', fontsize=11)

    ax.set_ylabel(
        'Variância individual explicada',
        fontsize=11,
        labelpad=12
    )

    ax_acum.set_ylabel(
        'Variância explicada acumulada',
        fontsize=11,
        color='#c62828',
        labelpad=14
    )

    from matplotlib.ticker import FormatStrFormatter

    ax.yaxis.set_major_formatter(
        FormatStrFormatter('%.7f')
    )

    ax_acum.yaxis.set_major_formatter(
        FormatStrFormatter('%.7f')
    )


    ax.set_ylim(0, max(.03, min(1.0, float(max(ratio))*1.18)))
    ax_acum.set_ylim(0, 1.045)
    ax.set_xlim(.35, len(ratio)+.7)
    if len(ratio) <= 15:
        xt = nums
    else:
        passo = 5 if len(ratio) <= 65 else 10
        xt = sorted(set([1, 2, 3, len(ratio)] + list(range(5, len(ratio)+1, passo))))
    ax.set_xticks(xt)
    ax.grid(axis='y', linestyle=':', color='#9ca3af', alpha=.22)


    fig.suptitle(
        'Variância explicada / Scree plot',
        fontsize=16,
        weight='bold',
        color='#26364a',
        y=.978
    )

    
    legenda = []

    if barras:
        legenda.append(
            Patch(
                facecolor='#83bb79',
                label='Variância individual explicada'
            )
        )

    if linha:
        legenda.append(
            Line2D(
                [0], [0],
                color='#d92e2e',
                lw=2.2,
                marker='o',
                label='Variância explicada acumulada'
            )
        )

        
    if op.get('destacar_componentes_selecionados', True):
        legenda.append(Line2D([0], [0], color='#101820', lw=1.65,
                              ls='--', label=f'PCA selecionado: {nsel}'))
    fig.legend(handles=legenda, ncol=3, loc='upper center',
               bbox_to_anchor=(.5, .905), fontsize=9.4, frameon=False)
    _exigir(origem in ('desenvolvimento', 'simulado'),
            'Origem do Scree PCA desconhecida.')
    nota = ('PRÉVIA SINTÉTICA: dados simulados; sem métricas OOF reais.'
            if origem == 'simulado' else
            'Espectro diagnóstico calculado apenas no desenvolvimento F1–F9.')
    fig.text(.5, .035,
             nota + ' Variância explicada não mede separação de fraudes; F10 não utilizado.',
             ha='center', fontsize=8.7, color='#506079')
    
    fig.subplots_adjust(
        left=.105,
        right=.88,
        top=.83,
        bottom=.15
    )

    _guardar(fig, dest, '11_pca_cotovelo', ext, dpi)
    _salvar_dados(pd.DataFrame({'componente': nums,
        'variancia_individual': ratio,
        'variancia_acumulada': acumulada,
        'variancia_absoluta': absoluto}), dest, '11_pca_cotovelo', csv)



def _pca_histograma_1d(config, data, dest, ext, dpi, csv, op, *, origem='desenvolvimento'):
    """PCA 12 (PDF): PC1 como histograma, nunca como eixo Y artificial.

    Usa as coordenadas do PCA VENCEDOR ajustado somente em F1–F9.
    Distribuicoes condicionais a cada grupo sao normalizadas para permitir
    a comparacao visual mesmo em dados fortemente desbalanceados.
    """
    from scipy.ndimage import gaussian_filter1d
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    _exigir(origem in ('desenvolvimento', 'simulado'), 'Origem da projecao PCA invalida.')

    scores_pca = data['coord'].to_numpy(dtype=float)
    _exigir(scores_pca.ndim == 2 and scores_pca.shape[1] == 1,
            'Histograma PCA 1D requer somente PC1.')
    x = scores_pca[:, 0]
    y = np.asarray(data['y']).ravel()
    _exigir(len(x) == len(y) and np.isfinite(x).all(),
            'Dados invalidos na projecao PC1.')
    pca = data['redutor'].modelo_
    ratio = np.asarray(pca.explained_variance_ratio_, dtype=float)
    _exigir(len(ratio) == 1 and np.isfinite(ratio[0]),
            'Variancia explicada do PCA vencedor indisponivel.')
    n_bins = int(op.get('bins_histograma_1d', 36))
    _exigir(5 <= n_bins <= 200, 'Quantidade de bins do PCA 1D invalida.')
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        margem = max(abs(lo) * .05, .5)
        lo -= margem
        hi += margem
    edges = np.linspace(lo, hi, n_bins + 1)
    mids = .5 * (edges[1:] + edges[:-1])
    width = float(edges[1] - edges[0])
    fig, ax = _fig(config)
    colorir = op.get('colorir_por', 'classe_real')
    classes = config['visualizacao']['classes']
    rotulos = None
    if colorir in ('classe_real', 'classe_predita'):
        valores = (y if colorir == 'classe_real' else
                   (np.asarray(data['scores']) >= float(data['limiar_oof'])).astype(int))
        rotulos = [(0, classes['legitima'].get('nome', 'Legitima'),
                    classes['legitima'].get('cor', '#438fc9')),
                   (1, classes['fraude'].get('nome', 'Fraude'),
                    classes['fraude'].get('cor', '#d43d3d'))]
        for grupo, nome, cor in rotulos:
            subset = x[valores == grupo]
            if not len(subset):
                continue
            valores_hist, _ = np.histogram(subset, bins=edges, density=True)



            ax.bar(
                mids, valores_hist,
                width=width * .93,
                color=cor,
                alpha=.38,
                edgecolor=cor,
                linewidth=.45,
                label=f'{nome} (n={len(subset)})',
                zorder=2
            )

            
            if op.get('curvas_suavizadas_1d', True) and len(subset) >= 3:
                curva = gaussian_filter1d(valores_hist, sigma=1.15, mode='nearest')
                ax.plot(mids, curva, color=cor, linewidth=1.9, zorder=4)





        ax.set_ylabel(
            'Densidade por classe (área = 1)',
            fontsize=11,
            labelpad=10
        )




        
    elif colorir == 'gaussiana_dominante':
        grupos = np.argmax(data['gamma'].to_numpy(), axis=1)
        _exigir(len(grupos) == len(x), 'Responsabilidades GMM nao alinhadas ao PCA.')
        for k in np.unique(grupos):
            subset = x[grupos == k]
            valores_hist, _ = np.histogram(subset, bins=edges, density=True)
            cor = plt.get_cmap('tab20')(int(k) % 20)
            ax.bar(mids, valores_hist, width=width * .93,
                   
                   color=cor, alpha=.23, label=f'G{k+1} (n={len(subset)})')
            
            if op.get('curvas_suavizadas_1d', True) and len(subset) >= 3:
                ax.plot(mids, gaussian_filter1d(valores_hist, sigma=1.15, mode='nearest'),
                        color=cor, linewidth=1.5)
        ax.set_ylabel('Densidade por gaussiana (área = 1)')
        
    elif colorir == 'risco':
        riscos = np.asarray(data['scores'], dtype=float)
        _exigir(len(riscos) == len(x) and np.isfinite(riscos).all() and
                ((riscos >= 0) & (riscos <= 1)).all(),
                'Riscos invalidos na projeção PCA.')
        contagem, _ = np.histogram(x, bins=edges)
        soma, _ = np.histogram(x, bins=edges, weights=riscos)
        medios = np.divide(soma, contagem, out=np.zeros(n_bins), where=contagem > 0)
        cor = plt.get_cmap('viridis')
        norma = Normalize(vmin=0, vmax=1)
        ax.bar(mids, contagem, width=width * .93,
               color=cor(norma(medios)), edgecolor='white', linewidth=.2)
        fig.colorbar(ScalarMappable(norm=norma, cmap=cor), ax=ax,
                     label='Risco médio estimado por faixa PC1')
        ax.set_ylabel('Frequência (observações)')

    else:
        raise ValueError('Projecao PCA 1D: colorir_por desconhecido.')



    # Media global da primeira componente principal
    if op.get('marcar_media_pc1_1d', True):
        media = float(x.mean())
        media_exibida = 0.0 if abs(media) < 0.00000005 else media

        ax.axvline(
            media,
            color='#253247',
            linestyle='--',
            linewidth=1.25,
            label=f'Média global da PC1 = {media_exibida:.7f}'
        )

    # Titulos e eixos
    ax.set_xlabel(
        f'PC1 (variância explicada = {float(ratio[0]):.7f})',
        fontsize=11,
        labelpad=10
    )

    ax.set_title(
        'PCA 1D | Histograma da projeção em PC1',
        fontsize=15,
        pad=12
    )

    # Padronizacao numerica
    from matplotlib.ticker import FuncFormatter

    def sete_casas(valor, _pos):
        valor = 0.0 if abs(valor) < 0.00000005 else valor
        return f'{valor:.7f}'

    ax.xaxis.set_major_formatter(FuncFormatter(sete_casas))

    if colorir == 'risco':
        # Neste modo o eixo vertical representa contagens inteiras.
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda valor, _pos: f'{valor:.0f}')
        )
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(sete_casas))

    ax.tick_params(axis='both', labelsize=8.5)

    ax.grid(
        axis='y',
        alpha=.16,
        linestyle=':'
    )

    ax.legend(
        loc='best',
        fontsize=9,
        frameon=True
    )

    fig.subplots_adjust(
        left=.12,
        right=.97,
        bottom=.16,
        top=.89
    )

    # Identificar dados simulados sem adicionar rodape no experimento real.
    if origem == 'simulado':
        fig.text(
            .5, .035,
            'PRÉVIA SINTÉTICA: dados simulados.',
            ha='center',
            fontsize=8,
            color='#4a5568'
        )



    nome = '12_pca_projecao_PC1'
    _guardar(fig, dest, nome, ext, dpi)
    if csv:
        _salvar_dados(pd.DataFrame({
            'PC1': x,
            'classe_real': y,
            'risco_estimado_treino': np.asarray(data['scores'], dtype=float),
            'gaussiana_dominante': np.argmax(data['gamma'].to_numpy(), axis=1) + 1,
            'variancia_explicada_PC1': np.full(len(x), float(ratio[0])),
        }), dest, nome, True)


def _pca_projecao_2d3d(config, data, dest, ext, dpi, csv, op, *, origem='desenvolvimento'):
    """PCA 12 do catálogo: 2 componentes = dispersão 2D, 3 = dispersão 3D.

    Componentes e percentuais são do MESMO PCA ajustado nos dados de
    desenvolvimento (ou numa simulação sintética de teste); nunca se
    acrescentam eixos artificiais a um PCA de menor dimensão.
    """
    _exigir(origem in ('desenvolvimento', 'simulado'), 'Origem do PCA desconhecida.')
    coords = data['coord']
    z = coords.to_numpy(dtype=float)
    d = z.shape[1]
    _exigir(d in (2, 3), 'Projeção PCA multivariada exige 2 ou 3 PCs.')
    _exigir(list(coords.columns) == [f'PC{i+1}' for i in range(d)],
            'Colunas PCA devem ser PC1, PC2 e PC3, na ordem.')
    pca = data['redutor'].modelo_
    ratios = np.asarray(pca.explained_variance_ratio_, dtype=float)
    _exigir(getattr(pca, 'n_components_', None) == d and ratios.shape == (d,),
            'Dimensão dos scores diverge do PCA realmente ajustado.')
    _exigir(np.isfinite(z).all() and np.isfinite(ratios).all(),
            'Scores ou variância PCA não finitos.')
    y = np.asarray(data['y']).ravel()
    _exigir(len(z) == len(y) and np.isin(y, [0, 1]).all(),
            'Observações e classes PCA desalinhadas.')
    limite = int(config['visualizacao']['geometria'].get('pontos_maximos_amostra', 20000))
    idx = _amostra_plot(len(y), y, config, limite)
    modo = op.get('colorir_por', 'classe_real')
    _exigir(modo in ('classe_real', 'classe_predita', 'risco', 'gaussiana_dominante'),
            'Modo de coloração PCA desconhecido.')
    legendas = config['visualizacao']['classes']
    if d == 2:
        fig, ax = _fig(config)
    else:
        fig = plt.figure(figsize=(11.5, 7.2))
        ax = fig.add_subplot(111, projection='3d')
    def desenhar(indices, *, cor, marcador, titulo, alpha=.55, tamanho=12):
        if not len(indices):
            return
        kw = dict(s=tamanho, c=[cor], marker=marcador, alpha=alpha,
                  label=titulo, rasterized=True)
        if d == 2:
            ax.scatter(z[indices, 0], z[indices, 1], **kw)
        else:
            ax.scatter(z[indices, 0], z[indices, 1], z[indices, 2], **kw)
    if modo in ('classe_real', 'classe_predita'):
        classes = (y if modo == 'classe_real' else
                   (np.asarray(data['scores']) >= float(data['limiar_oof'])).astype(int))
        for grupo, key in ((0, 'legitima'), (1, 'fraude')):
            estilo = legendas[key]
            indices = idx[classes[idx] == grupo]
            desenhar(indices, cor=estilo.get('cor', '#438fc9'),
                     marcador=estilo.get('marcador', 'o'),
                     titulo=f"{estilo.get('nome', key)} (n={len(indices):,} exibidas)",
                     alpha=float(estilo.get('alpha', .55)),
                     tamanho=float(estilo.get('tamanho', 12)))
        descricao = ('Classes reais (sobreposição descritiva; PCA não supervisionado)'
                     if modo == 'classe_real' else
                     'Classes previstas pelo limiar OOF; PCA não supervisionado')
    elif modo == 'risco':
        r = np.asarray(data['scores'], dtype=float).ravel()
        _exigir(len(r) == len(y) and np.isfinite(r).all() and
                ((r >= 0) & (r <= 1)).all(), 'Riscos PCA inválidos.')
        kw = dict(c=r[idx], cmap='viridis', vmin=0, vmax=1,
                  s=13, alpha=.7, rasterized=True)
        if d == 2:
            scatter = ax.scatter(z[idx, 0], z[idx, 1], **kw)
        else:
            scatter = ax.scatter(z[idx, 0], z[idx, 1], z[idx, 2], **kw)
        fig.colorbar(scatter, ax=ax, shrink=.65 if d == 3 else 1,
                     pad=.12 if d == 3 else .02, label='Risco estimado')
        descricao = 'Cor = risco estimado; não é fronteira fraude/não fraude'
    else:
        grupos = np.argmax(data['gamma'].to_numpy(), axis=1)
        _exigir(len(grupos) == len(y), 'Gaussianas e projeções não alinhadas.')
        for k in np.unique(grupos[idx]):
            indices = idx[grupos[idx] == k]
            desenhar(indices, cor=plt.get_cmap('tab20')(int(k) % 20),
                     marcador='o', titulo=f'G{int(k)+1} (n={len(indices):,} exibidas)',
                     alpha=.52, tamanho=11)
        descricao = 'Dominância por responsabilidade GMM; não é classe de fraude'
    # Preserva os contornos opcionais do PCA 2D sem atribuir cobertura 95%.
    if d == 2 and op.get('mostrar_contornos_gaussianas', False):
        gm = data['gmm']
        for k in range(len(gm.saidas_)):
            cov = _covariancia(gm, k)[:2, :2]
            eig, V = np.linalg.eigh(cov)
            if (eig > 0).all():
                theta = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
                ax.add_patch(Ellipse(gm.modelo_.means_[k, :2],
                             width=4*np.sqrt(eig[1]), height=4*np.sqrt(eig[0]),
                             angle=theta, facecolor='none',
                             edgecolor=plt.get_cmap('tab10')(k % 10), alpha=.48))
    ax.set_xlabel(f'PC1 (variância: {ratios[0]:.1%})', labelpad=9)
    ax.set_ylabel(f'PC2 (variância: {ratios[1]:.1%})', labelpad=9)
    if d == 3:
        ax.set_zlabel(f'PC3 (variância: {ratios[2]:.1%})', labelpad=9)
        ax.view_init(elev=22, azim=-64)
        ax.set_box_aspect((1, 1, .85))
    else:
        ax.grid(alpha=.15, linestyle=':')
    total = sum(float(v) for v in ratios)
    titulo = f'12 — PCA {d}D | Projeção PC1 × PC2' + (' × PC3' if d == 3 else '')
    ax.set_title(titulo + f' | variância acumulada: {total:.1%}', pad=15)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best' if d == 2 else 'upper left', fontsize=8)
    nota = f'{descricao}. Desenvolvimento F1–F9; F10 não utilizado.'
    if d == 3 and op.get('mostrar_contornos_gaussianas', False):
        nota += ' Elipsoides GMM não exibidos neste painel PCA.'
    if len(idx) < len(y):
        nota += f' Exibição amostrada: {len(idx):,} de {len(y):,} pontos; CSV completo.'
    if origem == 'simulado':
        nota = 'PRÉVIA SINTÉTICA: PCA ajustado em dados simulados. ' + nota
    fig.text(.5, .035, nota, ha='center', fontsize=7.3, color='#46566a')
    if d == 2:
        fig.subplots_adjust(top=.88, bottom=.18)
    else:
        fig.subplots_adjust(left=.04, right=.94, bottom=.12, top=.9)
    nome = '12_pca_projecao_PC1_PC2' + ('_PC3' if d == 3 else '')
    _guardar(fig, dest, nome, ext, dpi)
    if csv:
        # Salvar TODAS as observações, mesmo quando a imagem usa amostragem.
        campos = {f'PC{i+1}': z[:, i] for i in range(d)}
        campos.update({
            'classe_real': y,
            'risco_estimado_treino': np.asarray(data['scores'], dtype=float),
            'gaussiana_dominante': np.argmax(data['gamma'].to_numpy(), axis=1) + 1,
            'exibido_na_imagem': np.isin(np.arange(len(y)), idx),
        })
        campos.update({f'variancia_explicada_PC{i+1}': np.full(len(y), float(ratios[i]))
                       for i in range(d)})
        _salvar_dados(pd.DataFrame(campos), dest, nome, True)

def _tsne_histograma_1d(config, data, dest, ext, dpi, csv, op, *, origem='desenvolvimento'):
    """Grafico 13 PDF: embedding t-SNE 1D em histograma, sem eixo Y ficticio.

    Coordenadas vindas do t-SNE ja ajustado no desenvolvimento. Os rotulos
    entram SOMENTE na coloracao/descricao, nunca no ajuste do embedding.
    O histograma usa todos os pontos; o CSV preserva todas as linhas.
    """
    from scipy.ndimage import gaussian_filter1d
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    _exigir(origem in ('desenvolvimento', 'simulado'),
            'Origem do embedding t-SNE desconhecida.')
    coord = data['coord']
    z = coord.to_numpy(dtype=float)
    redutor = data['redutor']
    parametros = getattr(redutor, 'parametros', None)
    _exigir(isinstance(parametros, dict) and int(parametros.get('n_components', 0)) == 1,
            'Histograma t-SNE 1D exige redutor ajustado para 1 dimensao.')
    _exigir(z.ndim == 2 and z.shape[1] == 1 and list(coord.columns) == ['TSNE1'],
            'Scores t-SNE devem conter apenas TSNE1.')
    y = np.asarray(data['y']).ravel()
    _exigir(len(z) == len(y) and len(y) >= 2 and np.isfinite(z).all() and
            np.isin(y, [0, 1]).all(), 'Classes e coordenadas t-SNE invalidas.')
    x = z[:, 0]
    risco = np.asarray(data['scores'], dtype=float).ravel()
    _exigir(len(risco) == len(y) and np.isfinite(risco).all() and
            np.all((risco >= 0) & (risco <= 1)), 'Riscos desalinhados ao t-SNE.')
    grupos = np.argmax(data['gamma'].to_numpy(), axis=1)
    _exigir(len(grupos) == len(x), 'Gaussianas desalinhadas ao t-SNE.')
    n_bins = int(op.get('bins_histograma_1d', 36))
    _exigir(5 <= n_bins <= 200, 'Bins t-SNE 1D fora de 5..200.')
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        folga = max(0.5, abs(lo) * .05)
        lo, hi = lo - folga, hi + folga
    edges = np.linspace(lo, hi, n_bins + 1)
    centros = .5 * (edges[:-1] + edges[1:])
    largura = float(edges[1] - edges[0])
    fig, ax = _fig(config)
    colorir = op.get('colorir_por', 'classe_real')
    estilos = config['visualizacao']['classes']
    if colorir in ('classe_real', 'classe_predita'):
        if colorir == 'classe_predita':
            limiar = float(data['limiar_oof'])
            _exigir(np.isfinite(limiar) and 0 <= limiar <= 1,
                    'Limiar OOF invalido no embedding t-SNE.')
            categorias = (risco >= limiar).astype(int)
        else:
            categorias = y
        for classe, chave in ((0, 'legitima'), (1, 'fraude')):
            subset = x[categorias == classe]
            if not len(subset):
                continue
            e = estilos[chave]
            cor = e.get('cor', '#438fc9' if classe == 0 else '#d43d3d')
            dens, _ = np.histogram(subset, bins=edges, density=True)
            ax.bar(centros, dens, width=largura * .94, color=cor,
                   alpha=.38, edgecolor=cor, linewidth=.4,
                   label=f"{e.get('nome', chave)} (n={len(subset):,})", zorder=2)
            if len(subset) >= 3 and op.get('curvas_suavizadas_1d', True):
                ax.plot(centros, gaussian_filter1d(dens, 1.15, mode='nearest'),
                        color=cor, linewidth=1.8, zorder=4)
        ax.set_ylabel('Densidade por grupo (área = 1)')
        significado = ('Classes observadas, apenas sobrepostas ao t-SNE não supervisionado'
                       if colorir == 'classe_real' else
                       'Classes previstas por limiar MCC OOF; não são rótulos do ajuste t-SNE')
        nota = 'Densidades normalizadas por grupo; alturas não são prevalência. ' + significado
    elif colorir == 'gaussiana_dominante':
        for k in np.unique(grupos):
            subset = x[grupos == k]
            dens, _ = np.histogram(subset, bins=edges, density=True)
            cor = plt.get_cmap('tab20')(int(k) % 20)
            ax.bar(centros, dens, width=largura * .94,
                   color=cor, alpha=.26, label=f'G{k+1} (n={len(subset)})')
            if len(subset) >= 3 and op.get('curvas_suavizadas_1d', True):
                ax.plot(centros, gaussian_filter1d(dens, 1.15, mode='nearest'),
                        color=cor, linewidth=1.6)
        ax.set_ylabel('Densidade por componente (área = 1)')
        nota = ('Grupos por maior responsabilidade GMM, não classes de fraude. '
                'Densidades normalizadas por componente.')
    elif colorir == 'risco':
        contagem, _ = np.histogram(x, bins=edges)
        somas, _ = np.histogram(x, bins=edges, weights=risco)
        medias = np.divide(somas, contagem, out=np.zeros(n_bins), where=contagem > 0)
        cmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=0, vmax=1)
        ax.bar(centros, contagem, width=largura * .94,
               color=cmap(norm(medias)), edgecolor='white', linewidth=.3)
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     label='Risco médio estimado por faixa t-SNE 1')
        ax.set_ylabel('Frequência (observações)')
        nota = ('Cor indica risco médio por intervalo do embedding; '
                'não é fronteira de classificação.')
    else:
        raise ValueError('t-SNE 13: colorir_por invalido.')

    ax.set_xlabel('t-SNE 1 (coordenada do embedding; sem variância explicada)')
    ax.set_title('13 — t-SNE 1D | Histograma do embedding')
    ax.grid(axis='y', alpha=.18, linestyle=':')
    if colorir != 'risco':
        ax.legend(loc='best', fontsize=8)
    p = parametros
    subtitulo = (f"Configuração: perplexity={p.get('perplexity', 'n/d')} | "
                 f"metric={p.get('metric', 'n/d')} | 1 dimensão")
    fig.text(.5, .925, subtitulo, ha='center', fontsize=9, color='#374151')
    rodape = ('Leitura local: distâncias globais não devem ser superinterpretadas. '
              + nota + ' Desenvolvimento F1–F9; F10 não utilizado.')
    if origem == 'simulado':
        rodape = 'PRÉVIA SINTÉTICA: t-SNE ajustado em dados simulados. ' + rodape
    # Texto dividido em duas linhas para caber em resoluções menores.
    fig.text(.5, .05, rodape, ha='center', va='center', fontsize=7.3,
             color='#526077', wrap=True)
    fig.subplots_adjust(top=.87, bottom=.19)
    nome = '13_tsne_mapa_TSNE1'
    _guardar(fig, dest, nome, ext, dpi)
    if csv:
        _salvar_dados(pd.DataFrame({
            'TSNE1': x, 'classe_real': y,
            'classe_predita_limiar_oof': (risco >= float(data['limiar_oof'])).astype(int),
            'risco_estimado_desenvolvimento': risco,
            'gaussiana_dominante': grupos + 1,
        }), dest, nome, True)

def _projecoes(config,data,dest,ext,dpi,csv,metodo):
    coord=data['coord'].to_numpy(); y=data['y'];names=list(data['coord'].columns)
    op=config['visualizacao']['graficos']['pca_projecoes' if metodo=='PCA' else 'tsne_mapa']
    limite=int(config['visualizacao']['geometria'].get('pontos_maximos_amostra',20000))
    idx=_amostra_plot(len(y),y,config,limite)
    nome_g='12_pca_projecao' if metodo=='PCA' else '13_tsne_mapa'
    colorir=op.get('colorir_por','classe_real')
    _exigir(colorir in ('classe_real','classe_predita','risco','gaussiana_dominante'),
            'Projecao: colorir_por desconhecido.')
    if metodo == 'TSNE' and coord.shape[1] == 1:
        _tsne_histograma_1d(config, data, dest, ext, dpi, csv, op)
        return
    if metodo == 'TSNE' and coord.shape[1] == 3:
        from src import visualizacao_tsne_pdf as tsne_pdf
        tsne_pdf.projecao_3d(config, data, dest, ext, dpi, csv, op)
        return
    if metodo == 'PCA' and coord.shape[1] == 1:
        _pca_histograma_1d(config, data, dest, ext, dpi, csv, op)
        return
    if metodo == 'PCA' and coord.shape[1] in (2, 3):
        _pca_projecao_2d3d(config, data, dest, ext, dpi, csv, op)
        return
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


def _tsne_diagnostico(config, data, dest, ext, dpi, csv):
    from src import visualizacao_tsne_pdf as tsne_pdf
    return tsne_pdf.diagnostico_kl(config, data, dest, ext, dpi, csv)


def _tsne_trustworthiness(config, data, dest, ext, dpi, csv):
    from src import visualizacao_tsne_pdf as tsne_pdf
    return tsne_pdf.trustworthiness_perplexity(config, data, dest, ext, dpi, csv)


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
    """GMM08: 1D comparativo, 2D (1x2) e 3D (3x2), sem alterar o GMM.

    Ambos os limiares partilham malha, campo de risco, escala e seções FULL
    das gaussianas de 95%. Só fronteira/hachura dependem do corte.
    """
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

    # Uma figura comparativa por dimensionalidade, com linhas = cortes do
    # espaco 3D e colunas = limiares fixo / otimo. Mesmo campo em ambas.
    dim = X.shape[1]
    fig_size = (17.2, 17.8) if dim == 3 else (16.0, 7.8)
    fig, axes = plt.subplots(len(planos), 2, figsize=fig_size,
                             sharex='row', sharey='row', squeeze=False)
    fig.patch.set_facecolor('white')
    cortes = (('fixo', .5, mcc_fixo), ('mcc', tau_star, mcc_otimo))
    for linha, (par, grade, campo, fixo, otimo, mudou) in enumerate(planos):
        xx, yy = grade
        for coluna, (tipo, tau, mcc) in enumerate(cortes):
            ax = axes[linha, coluna]
            quad = ax.pcolormesh(xx, yy, campo, cmap='viridis',
                                 shading='auto', vmin=0, vmax=maior)
            _scatter(ax, X, y, idx, config, par)
            if op.get('mostrar_gaussianas', True):
                for k in range(len(gm.saidas_)):
                    geo = _elipse_95_corte(gm, k, par, fixos)
                    if geo is None:
                        continue
                    centro, largura, altura, angulo = geo
                    cor = plt.get_cmap('tab10')(k % 10)
                    ax.add_patch(Ellipse(centro, largura, altura, angle=angulo,
                                         ec=cor, fc='none', ls='--', lw=1.6))
                    ax.plot(centro[0], centro[1], marker='x',
                            color=cor, ms=7, mew=2)
            # A mudanca e XOR(score>=.50, score>=tau_star): mostrar apenas
            # na coluna MCC; nao alterar scores nem probabilidades.
            if tipo == 'mcc' and mudou.any():
                h = ax.contourf(xx, yy, mudou.astype(float), levels=[.5, 1.5],
                                colors='none', hatches=['////'])
                h.set_edgecolor('#ffd247')
                h.set_linewidth(.1)
            if campo.min() < tau < campo.max():
                ax.contour(xx, yy, campo, levels=[tau], colors='black',
                           linewidths=2)
            else:
                ax.text(.02, .03, f'Sem contorno s(x)={tau:.5g} neste plano',
                        transform=ax.transAxes, color='white', fontsize=8,
                        bbox={'facecolor':'#152233','alpha':.8,'edgecolor':'none'})
            ax.set(xlabel=nomes[par[0]], ylabel=nomes[par[1]],
                   xlim=(xx[0, 0], xx[0, -1]),
                   ylim=(yy[0, 0], yy[-1, 0]))
            if dim == 3:
                omitida = next(k for k in range(3) if k not in par)
                ax.set_title(
                    f'Corte {nomes[par[0]]} × {nomes[par[1]]}\n'
                    f'{nomes[omitida]} fixada na média do desenvolvimento = '
                    f'{fixos[omitida]:.3g}', fontsize=9.5, pad=10)
            else:
                ax.set_title(f'Corte {nomes[par[0]]} × {nomes[par[1]]}',
                             fontsize=10)
    # Barra única e mesma escala de risco em todos os 2 ou 6 painéis.
    cax = fig.add_axes([.91, .26 if dim == 3 else .25, .012,
                        .47 if dim == 3 else .48])
    fig.colorbar(quad, cax=cax,
                 label='Probabilidade estimada de fraude (mesma escala)')

    fig.suptitle(f'GMM 08 — Mapa de risco e geometria 95% | '
                 f'{gm.modelo_.covariance_type.upper()} | {dim}D',
                 fontsize=16, fontweight='bold', y=.984)
    # Titulos das colunas identificam a ORIGEM correta dos MCCs.
    fig.text(.265, .937, f'Limiar fixo τ=0,50  |  {nome_mcc}={mcc_fixo:.5f}',
             ha='center', fontsize=12, weight='bold', color='#23334a')
    fig.text(.685, .937, f'Limiar ótimo τ*={tau_star:.6g}  |  '
             f'{nome_mcc}={mcc_otimo:.5f}',
             ha='center', fontsize=12, weight='bold', color='#23334a')

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legenda = [
        Line2D([0], [0], color='black', lw=2, label='Fronteira de decisão'),
        Patch(facecolor='none', edgecolor='#d6a915', hatch='////',
              label='Mudança entre os cortes (coluna MCC)'),
        Line2D([0], [0], ls='', marker='o', color='#4389c2',
               label='Legítimas — pontos projetados'),
        Line2D([0], [0], ls='', marker='*', color='#d62728',
               label='Fraudes — pontos projetados'),
    ]
    if op.get('mostrar_gaussianas', True):
        legenda.extend(Line2D([0], [0],
                       color=plt.get_cmap('tab10')(k % 10), ls='--', lw=1.8,
                       label=f'{nome} — seção 95%')
                       for k, nome in enumerate(gm.saidas_))
    fig.legend(handles=legenda, loc='upper center',
               bbox_to_anchor=(.49, .897), ncol=min(6, len(legenda)),
               frameon=False, fontsize=8.4, columnspacing=1.1)

    direcao = ('Legítima → fraude' if tau_star < .5 else
               'Fraude → legítima' if tau_star > .5 else 'Nenhuma mudança')
    fig.text(.49, .075, f'Hachura amarela: mudança de classe ({direcao}); '
             'mesma superfície de risco nos dois limiares.',
             fontsize=9.4, ha='center', color='#23334a', weight='bold')
    fig.text(.49, .050, f'{nota_mcc}. '
             + ('Os planos 3D fixam a coordenada omitida na média de F1–F9.'
                if dim == 3 else 'Visualização do desenvolvimento F1–F9.'),
             fontsize=8.5, ha='center', color='#4a5970')
    fig.text(.49, .027, 'Elipses tracejadas: seções de gaussianas FULL 95%, '
             'não regiões classificadas como fraude. F10 não utilizado.',
             fontsize=8.2, ha='center', color='#4a5970')
    fig.subplots_adjust(left=.07, right=.875, top=.82, bottom=.145,
                        wspace=.23, hspace=.45 if dim == 3 else .27)
    _guardar(fig, dest, f'GMM08_mapa_risco_{dim}d_comparativo', ext, dpi)



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
    ativos=[n for n in TIPOS if _ativo(cfg,n) and (n not in ('pca_cotovelo','pca_projecoes','pca_biplot','tsne_trustworthiness','tsne_mapa','tsne_diagnostico')
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
    destino=layout.imagens_geometria(experimento)
    destino.mkdir(parents=True,exist_ok=True)
    ext,dpi,csv=_estilo(cfg)
    from src import visualizacao_gmm07 as sensibilidade
    from src import visualizacao_gmm01 as marginais
    from src import visualizacao_gmm04 as dominancia
    from src import visualizacao_pca_biplot as biplot
    funcoes={
        'densidades_marginais':lambda:marginais.gerar(cfg,data,destino,ext,dpi,csv),
        'regioes_dominancia':lambda:dominancia.gerar(cfg,data,destino,ext,dpi,csv),
        'painel_parametros_gaussianos':lambda:_painel_parametros_gaussianos(cfg,data,destino,ext,dpi,csv),
        'sensibilidade_fronteira':lambda:sensibilidade.gerar(cfg,data,escolha,destino,ext,dpi,csv,
                          avaliacao={'y':y_av_oof,'p':p_av_oof,'origem':'oof'}),
        'pca_cotovelo':lambda:_scree(cfg,data,destino,ext,dpi,csv),
        'pca_projecoes':lambda:_projecoes(cfg,data,destino,ext,dpi,csv,'PCA'),
        'pca_biplot':lambda:biplot.gerar(cfg,data,destino,ext,dpi,csv),
        'tsne_trustworthiness':lambda:_tsne_trustworthiness(cfg,data,destino,ext,dpi,csv),
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
