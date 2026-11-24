# -*- coding: utf-8 -*-
"""Projecao 2D do t-SNE (TSNE13) e PCA (PCA12) sem fit e sem F10 na selecao.

Reutilizavel em F1-F9 e F10, com posicionamento fixo de titulo/legenda/rodape.
Nao altera o modelo nem recalcula probabilidades. O grafico e descritivo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.amostragem_f10 import fracao_amostra_f10


def gerar_projecao_2d(config, data, destino, ext, dpi, csv, *, origem, metodo):
    """Gera PNG 2D e CSV das coordenadas observadas; requer modelo ja ajustado."""
    if origem not in ('desenvolvimento','f10') or metodo not in ('PCA','TSNE'):
        raise ValueError('Projecao 2D: origem/metodo invalidos.')
    nomes = (['PC1','PC2'] if metodo == 'PCA' else ['TSNE1','TSNE2'])
    coord = data['coord']
    if not isinstance(coord, pd.DataFrame) or list(coord.columns) != nomes:
        raise ValueError('Projecao 2D: coordenadas ausentes ou incorretas.')
    x = coord.to_numpy(dtype=float)
    y = np.asarray(data['y'],dtype=np.int8).reshape(-1)
    if (x.ndim != 2 or x.shape[1] != 2 or len(y) != len(x)
            or not np.isfinite(x).all() or not np.isin(y,[0,1]).all()):
        raise ValueError('Projecao 2D: dados, dimensao ou classes invalidos.')
    risco = np.asarray(data['scores'],dtype=float).reshape(-1)
    if len(risco) != len(y) or not np.isfinite(risco).all() or np.any((risco<0)|(risco>1)):
        raise ValueError('Projecao 2D: risco invalido.')
    from src import visualizacao_geometrica as geo
    limite = int(config['visualizacao']['geometria'].get('pontos_maximos_amostra',20000))
    idx = geo._amostra_plot(len(y), y, config, limite)
    fig, ax = plt.subplots(figsize=(11.5, 8.2),facecolor='white')
    for classe, rotulo, cor, marcador, tamanho in (
        (0,'Legítimas','#4389c2','o',10), (1,'Fraudes','#d62728','*',39)):
        ids = idx[y[idx] == classe]
        if len(ids):
            ax.scatter(x[ids,0],x[ids,1],s=tamanho,c=cor,marker=marcador,
                       alpha=.48 if classe==0 else .86,edgecolors='none',
                       rasterized=True,label=f'{rotulo} (n={int(np.sum(y==classe)):,})',
                       zorder=2 if classe==0 else 4)
    ax.set_xlabel(nomes[0] + (' (sem variância explicada)' if metodo=='TSNE' else ''))
    ax.set_ylabel(nomes[1] + (' (sem variância explicada)' if metodo=='TSNE' else ''))
    ax.grid(alpha=.16,linestyle=':')
    ax.legend(loc='upper right',fontsize=9,frameon=True)
    if metodo=='TSNE':
        parametros = getattr(data.get('redutor'),'parametros',{})
        subt = ('Embedding não supervisionado | '
                f'perplexity={parametros.get("perplexity", "n/d")} | '
                f'métrica={parametros.get("metric", "n/d")} | 2D')
        titulo='13 — t-SNE 2D | Projeção TSNE1 × TSNE2'
        aviso='Leitura local: distâncias globais do t-SNE não devem ser superinterpretadas.'
    else:
        titulo='12 — PCA 2D | Projeção PC1 × PC2'
        subt='Projeção PCA em duas componentes; observações coloridas pela classe real.'
        aviso='Classes observadas, sobrepostas descritivamente à projeção não supervisionada.'
    if origem == 'desenvolvimento':
        context = 'Desenvolvimento F1–F9; F10 não utilizado.'
    else:
        fracao_texto = (f'{fracao_amostra_f10(config):.2%} do fold' if
                        'fracao_amostra_f10' in config.get('divisao', {})
                        else 'fração histórica não registrada no YAML')
        context = ('F10 exploratório (' + fracao_texto +
                   '); redutor congelado em F1–F9; sem novo ajuste.')
    fig.suptitle(titulo,fontsize=16,weight='bold',y=.975)
    fig.text(.5,.927,subt,ha='center',fontsize=9.8,color='#41536b')
    fig.text(.5,.087,aviso,ha='center',fontsize=9.0,color='#475569')
    fig.text(.5,.060,context,ha='center',fontsize=9.0,color='#475569')
    fig.subplots_adjust(left=.10,right=.97,top=.865,bottom=.17)
    stem=('13_tsne_mapa_TSNE1_TSNE2' if metodo=='TSNE' else '12_pca_projecao_PC1_PC2')
    destino.mkdir(parents=True,exist_ok=True)
    geo._guardar(fig,destino,stem,ext,dpi)
    if csv:
        cols={nomes[0]:x[:,0],nomes[1]:x[:,1],
              'classe_real':y,'risco_estimado':risco}
        geo._salvar_dados(pd.DataFrame(cols),destino,stem,True)
    return stem
