# -*- coding: utf-8 -*-
r"""Auditoria isolada PCA 3D - GMM01: marginais + densidade conjunta interativa.

Execute na raiz do TCC_EXPERIMENTAL_V3:
    .\.venv\Scripts\python.exe .\validar_pca3d_05_gmm01.py

Gera:
- 3 PNGs e 3 CSVs marginais pela implementacao oficial do projeto.
- 1 PNG 3D com elipsoides gaussianos FULL de 95%.
- 1 HTML offline Plotly com isossuperficies de densidade, 95% e rotacao automatica.
- 1 CSV da malha tridimensional da densidade conjunta.
Usa APENAS dados sinteticos. Nao altera YAML, nao usa F10 nem roda OOF.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from html import escape
from pathlib import Path
from types import SimpleNamespace
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from scipy.stats import chi2, multivariate_normal
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src import visualizacao_gmm01

COLORS = ['#1685bb', '#f0842d', '#25a277']


def dados_sinteticos():
    rng = np.random.default_rng(20261027)
    centros = np.array([[-2.5, -1.4, -.5], [2.5, -1., 1.3], [-.3, 2.6, -1.6]])
    covs = np.array([
        [[.6, .2, .05], [.2, .8, -.1], [.05, -.1, .65]],
        [[.8, -.2, .13], [-.2, .6, .02], [.13, .02, .70]],
        [[.75, .15, -.10], [.15, .9, .2], [-.10, .2, .5]],
    ])
    latente = np.vstack([rng.multivariate_normal(centros[k], covs[k], 420)
                         for k in range(3)])
    pesos = rng.normal(size=(3, 39))
    X = latente @ pesos + rng.normal(scale=.53, size=(len(latente), 39))
    Z = PCA(n_components=3, svd_solver='full').fit_transform(
        StandardScaler().fit_transform(X))
    coord = pd.DataFrame(Z, columns=['PC1', 'PC2', 'PC3'])
    modelo = GaussianMixture(n_components=3, covariance_type='full',
                             n_init=3, max_iter=300, random_state=20261027).fit(coord)
    assert modelo.covariance_type == 'full'
    assert modelo.means_.shape == (3, 3)
    assert modelo.covariances_.shape == (3, 3, 3)
    assert modelo.converged_, 'GMM nao convergiu na simulacao.'
    gm = SimpleNamespace(modelo_=modelo, saidas_=['G1', 'G2', 'G3'],
                         colunas_=['PC1', 'PC2', 'PC3'])
    return coord, gm


def malha_joint(coord, gm, n=31):
    Z = coord.to_numpy(dtype=float)
    a, b = np.quantile(Z, [0.002, 0.998], axis=0)
    extra = 0.12 * (b - a)
    eixos = [np.linspace(a[j] - extra[j], b[j] + extra[j], n)
             for j in range(3)]
    xx, yy, zz = np.meshgrid(*eixos, indexing='ij')
    pts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    model = gm.modelo_
    componentes = np.column_stack([
        float(model.weights_[k]) * multivariate_normal.pdf(
            pts, mean=model.means_[k], cov=model.covariances_[k],
            allow_singular=False)
        for k in range(model.n_components)
    ])
    densidade = componentes.sum(axis=1)
    assert densidade.size == n ** 3
    assert np.isfinite(densidade).all() and np.all(densidade >= 0)
    assert np.allclose(densidade, componentes.sum(axis=1), atol=1e-12)
    tab = pd.DataFrame(pts, columns=['PC1', 'PC2', 'PC3'])
    tab['densidade_conjunta'] = densidade
    for k in range(model.n_components):
        tab[f'contribuicao_G{k+1}'] = componentes[:, k]
    return tab


def elipsoide(mu, cov, cobertura=0.95, n_u=35, n_v=21):
    """Superficie Mahalanobis chi2(p=3), cobertura gaussiana de 95%."""
    vals, vecs = np.linalg.eigh(cov)
    assert np.all(vals > 0)
    u = np.linspace(0, 2*np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    ux = np.outer(np.cos(u), np.sin(v))
    uy = np.outer(np.sin(u), np.sin(v))
    uz = np.outer(np.ones_like(u), np.cos(v))
    normal = np.stack((ux, uy, uz), axis=-1)
    base = vecs @ np.diag(np.sqrt(vals * chi2.ppf(cobertura, df=3)))
    p = np.einsum('ij,klj->kli', base, normal) + mu
    return p[:, :, 0], p[:, :, 1], p[:, :, 2]


def png_estatico(coord, gm, saida):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(12.4, 8.8))
    ax = fig.add_subplot(111, projection='3d')
    pontos = coord.to_numpy(dtype=float)
    ax.scatter(pontos[:, 0], pontos[:, 1], pontos[:, 2],
               s=5, color='#456987', alpha=.095, depthshade=False,
               label='Observacoes sinteticas')
    for k in range(3):
        x, y, z = elipsoide(gm.modelo_.means_[k], gm.modelo_.covariances_[k])
        ax.plot_wireframe(x, y, z, color=COLORS[k], linewidth=.60, alpha=.47,
                          rstride=2, cstride=2)
        media = gm.modelo_.means_[k]
        ax.scatter([media[0]], [media[1]], [media[2]], color=COLORS[k],
                   marker='X', s=85, depthshade=False, label=f'G{k+1} — 95%')
    ax.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')
    ax.view_init(elev=23, azim=-59)
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    fig.suptitle('GEOMETRIA DA MISTURA GAUSSIANA — PCA 3D',
                 fontsize=15, fontweight='bold', y=.965)
    fig.text(.5, .915, 'FULL | K = 3 | elipsoides de concentracao gaussiana de 95%',
             ha='center', fontsize=11, color='#4c617c')
    # Não chamar de superfície de densidade: são elipsoides de probabilidade individual.
    fig.text(.5, .035, 'Elipsoides = cobertura de cada gaussiana, nao fronteiras de fraude.',
             ha='center', fontsize=9, color='#4c617c')
    fig.subplots_adjust(left=.06, right=.87, top=.86, bottom=.15)
    nome = saida / 'GMM01_geometria_gaussianas_FULL_3d_95.png'
    fig.savefig(nome, dpi=165, facecolor='white')
    plt.close(fig)
    return nome


def html_interativo(coord, gm, malha, saida):
    """Isossuperficies da mistura + elipsoides FULL; controles de exibicao."""
    m = gm.modelo_
    fig = go.Figure()
    dens = malha['densidade_conjunta'].to_numpy()
    mx = float(dens.max())
    # Isovalores de densidade comuns da MISTURA: nao equivalem a cobertura 95%.
    fig.add_trace(go.Isosurface(
        x=malha['PC1'].to_numpy(), y=malha['PC2'].to_numpy(),
        z=malha['PC3'].to_numpy(), value=dens,
        isomin=.13*mx, isomax=.72*mx, surface_count=4,
        opacity=.23, colorscale='Viridis',
        caps={'x_show':False, 'y_show':False, 'z_show':False},
        showscale=True, colorbar={'title':'Densidade'},
        name='Isossuperficies da mistura', hovertemplate=(
            'PC1=%{x:.7f}<br>PC2=%{y:.7f}<br>PC3=%{z:.7f}'
            '<br>Densidade conjunta=%{value:.7f}<extra></extra>'),
    ))
    for k in range(3):
        x, y, z = elipsoide(m.means_[k], m.covariances_[k])
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            surfacecolor=np.full_like(x, k, dtype=float),
            colorscale=[[0, COLORS[k]], [1, COLORS[k]]],
            showscale=False, opacity=.23,
            name=f'G{k+1} — elipsoide 95%', showlegend=True,
            hovertemplate=(f'G{k+1} — elipsoide 95%<br>'
                           'PC1=%{x:.7f}<br>PC2=%{y:.7f}<br>PC3=%{z:.7f}<extra></extra>'),
        ))
        center = m.means_[k]
        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode='markers+text', text=[f'G{k+1}'],
            textposition='top center',
            marker={'size':6,'color':COLORS[k],'symbol':'diamond'},
            name=f'Centro G{k+1}',
            hovertemplate=(f'G{k+1}<br>Peso={m.weights_[k]:.7f}'
                           '<br>PC1=%{x:.7f}<br>PC2=%{y:.7f}<br>PC3=%{z:.7f}<extra></extra>'),
        ))
    # Ordem: isosuperficie, 3 superf + 3 centros
    V_iso = [True] + [False, True]*3
    V_ell = [False] + [True, True]*3
    V_both = [True] + [True, True]*3
    fig.update_layout(
        title={'text':'GMM01 — DENSIDADE CONJUNTA E GEOMETRIA 3D', 'x':.5},
        height=730, margin={'l':0, 'r':0, 't':95, 'b':0},
        font={'family':'Arial, sans-serif'},
        legend={'orientation':'h', 'x':0, 'y':1.03},
        scene={
            'xaxis':{'title':'PC1'},'yaxis':{'title':'PC2'},'zaxis':{'title':'PC3'},
            'aspectmode':'data', 'dragmode':'orbit',
            'camera':{'eye':{'x':1.8,'y':1.6,'z':.95}}
        },
        updatemenus=[{
            'type':'buttons','direction':'right',
            'x':.98,'y':1.15,'xanchor':'right',
            'buttons':[
                {'label':'Densidade conjunta','method':'update',
                 'args':[{'visible':V_iso}]},
                {'label':'Elipsoides 95%','method':'update',
                 'args':[{'visible':V_ell}]},
                {'label':'Ambos','method':'update',
                 'args':[{'visible':V_both}]},
            ],
        }]
    )
    # O estado inicial e apenas a isossuperficie mais centros (sem transparencias sobrepostas).
    for tr, vis in zip(fig.data, V_iso):
        tr.visible = vis
    div_id = 'fig_gmm01_3d'
    grafico = fig.to_html(include_plotlyjs=True, full_html=False, div_id=div_id,
                          config={'displaylogo':False, 'scrollZoom':True, 'responsive':True})
    html = f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GMM01 3D — Interativo</title>
<style>body{{font-family:system-ui,Arial,sans-serif;background:#f4f7fb;color:#182538;margin:0}}
main{{max-width:1380px;margin:20px auto;padding:0 14px}}h1{{font-size:21px;margin-bottom:7px}}
.subtitle{{color:#52647c;font-size:13px}}button{{border:1px solid #c8d2dd;padding:9px 14px;border-radius:8px;background:white;cursor:pointer}}
.header{{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}}.panel{{border-radius:12px;background:white;padding:10px;box-shadow:0 2px 10px #19283e18}}</style>
</head><body><main><div class="header"><div><h1>GMM01 — Densidade Conjunta e Geometria PCA 3D</h1>
<div class="subtitle">K=3 • Covariância FULL • Rotacione com o mouse ou ative o movimento automático</div></div>
<div><button id="spin">▶ Rotação automática</button> <button id="reset">↺ Reiniciar câmera</button></div></div>
<div class="panel">{grafico}</div>
<p class="subtitle">As isossuperfícies mostram níveis de <b>densidade conjunta da mistura</b>.
Os elipsoides 95% delimitam concentração probabilística de <b>cada componente gaussiana</b>;
não representam risco de fraude, nem equivalem a isossuperfícies de densidade total.
Prévia exclusivamente sintética — sem F10.</p></main>
<script>(()=>{{ const gd=document.getElementById('{div_id}');const btn=document.getElementById('spin');
let ativo=false,id=null,angulo=.70;function parar(){{ativo=false;btn.textContent='▶ Rotação automática';if(id!==null){{clearInterval(id);id=null;}}}}
btn.addEventListener('click',()=>{{if(ativo){{parar();return;}}ativo=true;btn.textContent='⏸ Pausar';id=setInterval(()=>{{angulo+=.025;
Plotly.relayout(gd,{{'scene.camera.eye':{{x:2.1*Math.cos(angulo),y:2.1*Math.sin(angulo),z:.9}}}});}},75);}});
document.getElementById('reset').addEventListener('click',()=>{{parar();angulo=.70;Plotly.relayout(gd,{{'scene.camera.eye':{{x:1.8,y:1.6,z:.95}}}});}});
gd.addEventListener('pointerdown',parar);window.addEventListener('pagehide',parar);}})();</script>
</body></html>'''
    saida_html = saida / 'GMM01_densidade_conjunta_PCA3D_INTERATIVA.html'
    saida_html.write_text(html, encoding='utf-8')
    return saida_html


def main():
    raiz = Path.cwd()
    yaml_path = raiz / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    fonte = raiz / 'src' / 'visualizacao_gmm01.py'
    if not yaml_path.is_file() or not fonte.is_file():
        raise SystemExit('Execute na raiz do TCC_EXPERIMENTAL_V3 (YAML e src obrigatorios).')
    yaml_original = yaml_path.read_bytes()
    cfg = deepcopy(yaml.safe_load(yaml_original.decode('utf-8-sig')))
    cfg.setdefault('visualizacao', {}).setdefault('graficos', {}).setdefault(
        'densidades_marginais', {'pontos_curva':600, 'margem_sigmas':4.0})
    # Ajustar parametros apenas nesta simulacao, em memoria.
    cfg['visualizacao']['graficos']['densidades_marginais']['pontos_curva'] = 550
    coord, gm = dados_sinteticos()
    pasta = raiz / 'outputs' / (
        'validacao_pca3d_05_gmm01_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    pasta.mkdir(parents=True, exist_ok=False)

    # Figuras oficiais GMM01 existentes: exatamente 3 marginais para PCA 3D.
    visualizacao_gmm01.gerar(cfg, {'coord':coord, 'gmm':gm}, pasta, 'png', 165, True)
    for pc in ('PC1','PC2','PC3'):
        nome = f'GMM01_densidade_marginal_3d_{pc}'
        assert (pasta / f'{nome}.png').is_file(), nome + ' PNG nao gerado'
        assert (pasta / f'{nome}.csv').is_file(), nome + ' CSV nao gerado'
        tabela = pd.read_csv(pasta / f'{nome}.csv')
        assert len(tabela) == 550 and len(tabela.columns) == 5
        assert np.allclose(tabela['densidade_mistura'].to_numpy(),
                           tabela[['contribuicao_G1','contribuicao_G2','contribuicao_G3']].sum(axis=1),
                           atol=1e-12)
    joint = malha_joint(coord, gm, n=31)
    joint.to_csv(pasta / 'GMM01_malha_densidade_conjunta_PCA3D.csv', index=False,
                 float_format='%.17g')
    png_3d = png_estatico(coord, gm, pasta)
    html_3d = html_interativo(coord, gm, joint, pasta)
    assert png_3d.is_file() and html_3d.is_file()
    assert yaml_path.read_bytes() == yaml_original, 'YAML alterado!'

    print('GMM01 PCA 3D: VALIDACAO ISOLADA CONCLUIDA')
    print('Dados: 39 features sinteticas | 1260 observacoes')
    print('GMM: 3 gaussianas | FULL | PCA 3D')
    print('PNGs marginais: PC1, PC2, PC3')
    print('3D: elipsoides FULL 95%, isossuperficies de densidade conjunta')
    print('YAML preservado | F10 nao utilizado | Sem pipeline OOF')
    print('HTML=' + str(html_3d.resolve()))
    print('PNG_3D=' + str(png_3d.resolve()))
    print('PASTA=' + str(pasta.resolve()))


if __name__ == '__main__':
    main()
