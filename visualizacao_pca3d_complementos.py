# -*- coding: utf-8 -*-
"""Complementos oficiais do Biplot PCA 3D: coeficientes e HTML rotativo.

A entrada sao os objetos do ajuste de desenvolvimento ja existente.
Nao reajusta PCA/GMM, nao le F10 e nao altera o YAML.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

from src import visualizacao_oof as vis

MARCA = 'PCA3D_BIPLOT_COMPLEMENTOS_OFICIAIS_V1'
COEF = 'PCA_COEFICIENTES_PC1_PC2_PC3'
HTML = 'PCA_BIPLOT_3D_INTERATIVO.html'


def _fig_coeficientes(tabela, variancias):
    nomes = tabela['variavel_preprocessada'].astype(str).tolist()
    n = len(nomes)
    if n == 0:
        raise ValueError('Tabela de coeficientes PCA3D vazia.')
    valores = tabela[[f'coeficiente_PC{i}' for i in (1, 2, 3)]].to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        raise ValueError('Coeficientes PCA3D nao finitos.')
    limite = max(0.08, float(np.max(np.abs(valores))) * 1.52)
    altura = max(12.5, 2.6 + .405 * n)
    fig, axes = plt.subplots(1, 3, figsize=(25.5, altura), sharey=True)
    fig.patch.set_facecolor('white')
    yy = np.arange(n)
    cores = tabela['cor_hex'].tolist()

    for j, ax in enumerate(axes):
        ax.axvline(0, color='#526174', lw=1.25)
        ax.set_axisbelow(True)
        ax.grid(axis='x', linestyle=':', alpha=.27)
        for i, (valor, cor) in enumerate(zip(valores[:, j], cores)):
            ax.annotate('', xy=(valor, i), xytext=(0, i),
                        arrowprops={'arrowstyle': '-|>', 'lw': 1.75,
                                    'color': cor, 'mutation_scale': 12}, zorder=3)
            visivel = 0.0 if abs(valor) < 5e-8 else float(valor)
            deslocamento = limite * .022
            ax.text(valor + (deslocamento if valor >= 0 else -deslocamento), i,
                    f'{visivel:+.7f}', ha='left' if valor >= 0 else 'right',
                    va='center', fontsize=7.0, color=cor, fontweight='semibold')
        ax.set_xlim(-limite, limite)
        ax.set_ylim(n - .7, -.8)
        ax.set_title(f'PC{j+1} | Variância explicada: {variancias[j]:.7%}',
                     fontweight='bold', fontsize=12.5, pad=15)
        ax.set_xlabel('Coeficiente da combinação linear', fontsize=10.5, labelpad=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _pos: f'{(0.0 if abs(v) < 5e-8 else v):.7f}'))
        ax.tick_params(axis='x', labelsize=8.3, rotation=12)
        ax.tick_params(axis='y', length=0, pad=8)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color('#b3bece')

    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(nomes, fontsize=9.2)
    axes[0].tick_params(axis='y', labelleft=True)
    for ax in axes[1:]:
        ax.tick_params(axis='y', labelleft=False)
    fig.suptitle('COEFICIENTES DAS COMBINAÇÕES LINEARES DO PCA 3D',
                 fontsize=20, fontweight='bold', y=.995)
    fig.text(.5, .951,
             f'{n} variáveis preprocessadas | Mesma ordem, cor e escala nos três painéis',
             ha='center', fontsize=11, color='#506278')
    fig.subplots_adjust(left=.115, right=.986, top=.908, bottom=.072, wspace=.115)
    return fig


def _html_interativo(coord, y, idx, correlacoes, variancias, destino):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError('Plotly e necessario no PCA 3D: python -m pip install plotly') from exc

    z = np.asarray(coord, dtype=float)
    y = np.asarray(y, dtype=int)
    idx = np.asarray(idx, dtype=int)
    if len(idx) > 5000:
        # Amostra visual equilibrada em indice, nunca altera classificacao ou dados.
        idx = idx[np.linspace(0, len(idx)-1, 5000, dtype=int)]
    if z.ndim != 2 or z.shape[1] != 3 or y.shape[0] != z.shape[0]:
        raise ValueError('Scores PCA3D ou classes desalinhados para HTML.')

    fig = go.Figure()
    for classe, nome, cor, simbolo, tamanho, opac in (
        (0, 'Legítimas', '#2374b8', 'circle', 3.6, .50),
        (1, 'Fraudes', '#e34445', 'diamond', 5.2, .88),
    ):
        ids = idx[y[idx] == classe]
        fig.add_trace(go.Scatter3d(
            x=z[ids, 0], y=z[ids, 1], z=z[ids, 2],
            mode='markers', name=f'{nome} (n={len(ids)})',
            marker={'size': tamanho, 'color': cor, 'symbol': simbolo, 'opacity': opac},
            hovertemplate=(f'<b>{nome}</b><br>PC1: %{{x:.7f}}<br>'
                           'PC2: %{y:.7f}<br>PC3: %{z:.7f}<extra></extra>'),
        ))
    corr = correlacoes.copy()
    exigir = {'variavel_preprocessada', 'corr_PC1', 'corr_PC2', 'corr_PC3'}
    if not exigir.issubset(corr.columns):
        raise ValueError('CSV/tabela de correlacoes PCA3D incompleta.')
    corr['norma'] = np.linalg.norm(corr[[f'corr_PC{i}' for i in (1, 2, 3)]].to_numpy(), axis=1)
    corr = corr.sort_values(['norma', 'variavel_preprocessada'],
                            ascending=[False, True], kind='stable').reset_index(drop=True)
    escala = .55 * float(np.min(np.max(np.abs(z[idx]), axis=0)))
    num_top = min(6, len(corr))
    for j, r in corr.iterrows():
        vx, vy, vz = (escala * float(r[f'corr_PC{k}']) for k in (1, 2, 3))
        top = j < num_top
        fig.add_trace(go.Scatter3d(
            x=[0, vx], y=[0, vy], z=[0, vz],
            mode='lines+markers+text' if top else 'lines+markers',
            marker={'size': [1, 4] if top else [1, 3], 'color': '#7049bf'},
            line={'color': '#7049bf' if top else '#94a3b8', 'width': 6 if top else 3},
            text=[None, str(r['variavel_preprocessada'])] if top else None,
            textposition='top center', showlegend=False, visible=top,
            hovertemplate=(
                f'<b>{escape(str(r["variavel_preprocessada"]))}</b><br>'
                f'corr(PC1)={float(r["corr_PC1"]):.7f}<br>'
                f'corr(PC2)={float(r["corr_PC2"]):.7f}<br>'
                f'corr(PC3)={float(r["corr_PC3"]):.7f}<extra></extra>'),
        ))
    total = len(fig.data)
    fig.update_layout(
        title={'text': 'BIPLOT PCA 3D — OBSERVAÇÕES E VETORES', 'x': .5,
               'font': {'size': 18}},
        height=780, font={'family': 'Arial, sans-serif', 'size': 12},
        margin={'l': 8, 'r': 8, 'b': 8, 't': 90},
        legend={'orientation': 'h', 'y': 1.05, 'x': .04},
        scene={
            'xaxis': {'title': f'PC1 ({variancias[0]:.7%})'},
            'yaxis': {'title': f'PC2 ({variancias[1]:.7%})'},
            'zaxis': {'title': f'PC3 ({variancias[2]:.7%})'},
            'aspectmode': 'data', 'dragmode': 'orbit',
            'camera': {'eye': {'x': 1.7, 'y': 1.6, 'z': 1.0}},
        },
        updatemenus=[{
            'type': 'buttons', 'direction': 'right', 'x': .98, 'y': 1.13,
            'xanchor': 'right', 'yanchor': 'top', 'showactive': True,
            'buttons': [
                {'label': f'{num_top} vetores', 'method': 'update',
                 'args': [{'visible': [True, True] + [j < num_top for j in range(len(corr))]}]},
                {'label': f'{len(corr)} vetores (todos)', 'method': 'update',
                 'args': [{'visible': [True] * total}]},
            ],
        }],
    )
    snippet = fig.to_html(full_html=False, include_plotlyjs=True,
                          div_id='biplot3d',
                          config={'responsive': True, 'displaylogo': False, 'scrollZoom': True})
    html = '''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Biplot PCA 3D interativo</title><style>
body{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:#f4f7fb;color:#1c2c44}
main{max-width:1450px;margin:16px auto;padding:0 18px}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
h1{font-size:22px;margin:0 0 6px}p{color:#5a6a80;font-size:13px;margin:0}
button{border:1px solid #c7d2e0;border-radius:8px;background:white;padding:10px;cursor:pointer}
.painel{background:white;border-radius:12px;box-shadow:0 2px 12px #14234018;overflow:hidden}
</style></head><body><main><header><div><h1>Biplot PCA 3D interativo</h1>
<p>Seis vetores iniciais; todas as features disponíveis. Dados do desenvolvimento F1–F9.</p>
</div><div><button id="rodar">▶ Rotação automática</button>
<button id="resetar">↺ Reiniciar câmera</button></div></header>
<section class="painel">''' + snippet + '''</section>
<p>Arraste para girar · Scroll para zoom · As setas de correlação são escaladas apenas visualmente.</p>
</main><script>(function(){const gd=document.getElementById('biplot3d');
const bt=document.getElementById('rodar'),rs=document.getElementById('resetar');
let t=null,a=.6;function stop(){if(t!==null){clearInterval(t);t=null;}bt.textContent='▶ Rotação automática';}
bt.onclick=function(){if(t!==null){stop();return;}bt.textContent='⏸ Pausar rotação';
t=setInterval(function(){a+=.025;Plotly.relayout(gd,{'scene.camera.eye':
{x:2.05*Math.cos(a),y:2.05*Math.sin(a),z:.88}})},65)};
rs.onclick=function(){stop();a=.6;Plotly.relayout(gd,{'scene.camera.eye':{x:1.7,y:1.6,z:1}})};
gd.addEventListener('pointerdown',stop);window.addEventListener('pagehide',stop);
})();</script></body></html>'''
    if destino.is_file():
        print(f'[PRESERVADO] {destino.name}')
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(html, encoding='utf-8')
    print(f'[GERADO] {destino.name}')


def gerar_complementares(config, data, destino, ext, dpi, csv,
                         *, idx, tabela_correlacoes, mapa_cores):
    """Usa o mesmo PCA ajustado e indices de amostra do biplot oficial."""
    destino = Path(destino)
    x, coord = data['pre'], data['coord']
    modelo, y = data['redutor'].modelo_, np.asarray(data['y'], dtype=int)
    if not isinstance(x, pd.DataFrame) or not isinstance(coord, pd.DataFrame):
        raise ValueError('PCA3D requer pre e coord como DataFrames.')
    if int(coord.shape[1]) != 3 or np.shape(modelo.components_) != (3, x.shape[1]):
        raise ValueError('Complementos PCA3D exigem PCA ja ajustado em 3D.')
    if not np.array_equal(np.asarray(data['redutor'].colunas_), np.asarray(x.columns)):
        raise ValueError('Colunas PCA3D desalinhadas.')
    tabela = pd.DataFrame(np.asarray(modelo.components_, dtype=float).T,
                          columns=[f'coeficiente_PC{i}' for i in (1, 2, 3)])
    tabela.insert(0, 'variavel_preprocessada', list(x.columns))
    tabela['magnitude_conjunta'] = np.linalg.norm(
        tabela[[f'coeficiente_PC{i}' for i in (1, 2, 3)]].to_numpy(), axis=1)
    tabela = tabela.sort_values(['magnitude_conjunta', 'variavel_preprocessada'],
                                ascending=[False, True], kind='stable').reset_index(drop=True)
    tabela['cor_hex'] = tabela['variavel_preprocessada'].map(mapa_cores)
    if tabela['cor_hex'].isna().any():
        raise ValueError('Paleta PCA3D incompleta.')
    if csv:
        # Preserva precisao de double para os coeficientes originais.
        vis._csv_protegido(tabela, destino / f'{COEF}.csv')
    fig = _fig_coeficientes(tabela, np.asarray(modelo.explained_variance_ratio_, dtype=float))
    vis._salvar(fig, destino / f'{COEF}.{ext.lstrip(".")}', dpi=int(dpi))
    _html_interativo(coord.to_numpy(dtype=float), y, idx, tabela_correlacoes,
                     modelo.explained_variance_ratio_, destino / HTML)
    return {'coeficientes': f'{COEF}.{ext.lstrip(".")}', 'html': HTML,
            'features': len(x.columns), 'origem': 'desenvolvimento_F1_F9',
            'f10_utilizado': False}
