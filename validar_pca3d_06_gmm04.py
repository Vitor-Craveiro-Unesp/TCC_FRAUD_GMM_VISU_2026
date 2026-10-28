# -*- coding: utf-8 -*-
"""Teste isolado do GMM04 em PCA 3D (K=3, covariância FULL).

Execute na raiz do TCC_EXPERIMENTAL_V3:
    .\\.venv\\Scripts\\python.exe .\\validar_pca3d_06_gmm04.py

Produz o PNG oficial dos três cortes, seus CSVs, um HTML 3D interativo
com planos móveis e elipsoides FULL 95%, e CSV da malha volumétrica.
Não modifica src/, YAML ou modelos/ensaios anteriores; não utiliza F10.
Apenas dados sintéticos. Dominância gaussiana NÃO é risco de fraude.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src import visualizacao_gmm04 as gmm04

SEED = 20261027
N_SLICE = 59
N_VOLUME = 23
OUTPUT_STEM = 'GMM04_dominancia_3d'


def sintetizar():
    rng = np.random.default_rng(SEED)
    centros = np.array([[-2.5, -1.4, -.5], [2.5, -1., 1.3], [-.3, 2.6, -1.6]])
    covs = np.array([
        [[.6, .2, .05], [.2, .8, -.1], [.05, -.1, .65]],
        [[.8, -.2, .13], [-.2, .6, .02], [.13, .02, .70]],
        [[.75, .15, -.10], [.15, .9, .2], [-.10, .2, .5]],
    ])
    latente = np.vstack([
        rng.multivariate_normal(centros[k], covs[k], 420) for k in range(3)
    ])
    pesos = rng.normal(size=(3, 39))
    X = latente @ pesos + rng.normal(scale=.53, size=(len(latente), 39))
    X = StandardScaler().fit_transform(X)
    Z = PCA(n_components=3, svd_solver='full').fit_transform(X)
    modelo = GaussianMixture(n_components=3, covariance_type='full',
                             n_init=3, max_iter=300, random_state=SEED).fit(Z)
    assert modelo.converged_
    assert modelo.means_.shape == (3, 3)
    assert modelo.covariances_.shape == (3, 3, 3)
    coord = pd.DataFrame(Z, columns=['PC1', 'PC2', 'PC3'])
    gmm = SimpleNamespace(modelo_=modelo, saidas_=['G1', 'G2', 'G3'],
                          colunas_=['PC1', 'PC2', 'PC3'])
    # Rótulos APENAS para o scatter do PNG, não usados no cálculo de dominância.
    prob = modelo.predict_proba(Z) @ np.array([.04, .18, .55])
    y = rng.binomial(1, prob)
    return coord, gmm, y


def avaliar(gmm, pontos):
    dom, resp = gmm04.calcular_dominancia(gmm, pontos)
    assert resp.shape == (len(pontos), 3)
    assert np.allclose(resp.sum(axis=1), 1.0, atol=1e-9)
    assert np.array_equal(dom, resp.argmax(axis=1))
    return dom, resp


def criar_planos(gmm, eixos, fixo, n=N_SLICE):
    """Três planos ortogonais PC1×PC2, PC1×PC3, PC2×PC3.

    Todos avaliam o MESMO GMM tridimensional, com a coordenada omitida
    fixada no vetor `fixo`. Não são marginais bivariadas.
    """
    planos = []
    for a, b, omitida in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
        A, B = np.meshgrid(eixos[a], eixos[b], indexing='xy')
        pts = np.tile(fixo, (A.size, 1))
        pts[:, a] = A.ravel()
        pts[:, b] = B.ravel()
        dom, _ = avaliar(gmm, pts)
        X = pts[:, 0].reshape(A.shape)
        Y = pts[:, 1].reshape(A.shape)
        Z = pts[:, 2].reshape(A.shape)
        planos.append({'x': X, 'y': Y, 'z': Z,
                       'surfacecolor': dom.reshape(A.shape).astype(float),
                       'a': a, 'b': b, 'omitida': omitida})
    return planos


def geometria_elipsoide(media, cov, n_u=38, n_v=24):
    """Mahalanobis 3D, q=chi2.ppf(0.95, df=3); usa Σ FULL completa."""
    vals, vecs = np.linalg.eigh(cov)
    assert np.all(vals > 0), 'Covariancia nao positiva definida.'
    u = np.linspace(0, 2*np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    esfera = np.stack((np.outer(np.cos(u), np.sin(v)),
                       np.outer(np.sin(u), np.sin(v)),
                       np.outer(np.ones_like(u), np.cos(v))), axis=-1)
    T = vecs @ np.diag(np.sqrt(vals * chi2.ppf(.95, df=3)))
    p = np.einsum('ij,klj->kli', T, esfera) + media
    q = np.einsum('...i,ij,...j->...', p-media, np.linalg.inv(cov), p-media)
    assert np.max(np.abs(q - chi2.ppf(.95, df=3))) < 1e-8
    return p[..., 0], p[..., 1], p[..., 2]


def html_interativo(coord, gm, destino):
    modelo = gm.modelo_
    X = coord.to_numpy(dtype=float)
    centro_global = X.mean(axis=0)
    # Mesma convenção usada na imagem oficial: limite de dados mais margem.
    menor = np.quantile(X, .005, axis=0)
    maior = np.quantile(X, .995, axis=0)
    extra = .20 * (maior - menor)
    limites = np.column_stack((menor-extra, maior+extra))
    eixos = [np.linspace(*limites[i], N_SLICE) for i in range(3)]
    cores = [to_hex(c) for c in gmm04._cores(3)[0]]
    cor_escala = [
        [0.0, cores[0]], [1/3, cores[0]],
        [1/3 + 1e-6, cores[1]], [2/3, cores[1]],
        [2/3 + 1e-6, cores[2]], [1.0, cores[2]],
    ]
    # Para three discrete classes with cmin=-.5,cmax=2.5, boundaries at 1/3, 2/3.
    fig = go.Figure()
    planos = criar_planos(gm, eixos, centro_global)
    for i, plano in enumerate(planos):
        fig.add_trace(go.Surface(
            x=plano['x'], y=plano['y'], z=plano['z'],
            surfacecolor=plano['surfacecolor'],
            colorscale=cor_escala, cmin=-.5, cmax=2.5, showscale=False,
            opacity=.72, name=f'Corte {i+1}: PC{plano["a"]+1} x PC{plano["b"]+1}',
            showlegend=True, hovertemplate=(
                'PC1=%{x:.7f}<br>PC2=%{y:.7f}<br>PC3=%{z:.7f}'
                '<br>Dominância (índice zero): %{surfacecolor:.0f}<extra></extra>'
            )))
    for k in range(3):
        U, V, W = geometria_elipsoide(modelo.means_[k], modelo.covariances_[k])
        fig.add_trace(go.Surface(
            x=U, y=V, z=W,
            surfacecolor=np.full_like(U, float(k)),
            colorscale=cor_escala, cmin=-.5, cmax=2.5,
            opacity=.16, showscale=False, name=f'G{k+1} — elipsoide FULL 95%',
            showlegend=True, hoverinfo='skip'))
    for k in range(3):
        mu = modelo.means_[k]
        fig.add_trace(go.Scatter3d(
            x=[mu[0]], y=[mu[1]], z=[mu[2]], mode='markers+text',
            marker={'size': 7, 'symbol': 'diamond', 'color': cores[k],
                    'line': {'color': '#142135', 'width': 1.3}},
            text=[f'G{k+1}'], textposition='top center',
            name=f'Centro G{k+1}', hovertemplate=(
                f'Centro G{k+1}<br>PC1=%{{x:.7f}}<br>PC2=%{{y:.7f}}'
                '<br>PC3=%{z:.7f}<extra></extra>'))) 

    # Amostragem deterministica de uma malha do volume: não pretende mostrar
    # fronteira contínua exata. Pontos ficam desativados inicialmente.
    seq = [np.linspace(*limites[j], N_VOLUME) for j in range(3)]
    vol = np.stack(np.meshgrid(*seq, indexing='ij'), axis=-1).reshape(-1, 3)
    classes, resp = avaliar(gm, vol)
    csv = pd.DataFrame(vol, columns=['PC1', 'PC2', 'PC3'])
    csv['gaussiana_dominante'] = [f'G{i+1}' for i in classes]
    for k in range(3):
        csv[f'responsabilidade_G{k+1}'] = resp[:, k]
    csv.to_csv(destino / 'GMM04_dominancia_volume_3d.csv', index=False,
               float_format='%.17g')
    rng = np.random.default_rng(SEED)
    indices = np.sort(rng.choice(len(vol), size=min(4500, len(vol)), replace=False))
    fig.add_trace(go.Scatter3d(
        x=vol[indices, 0], y=vol[indices, 1], z=vol[indices, 2],
        mode='markers', marker={'size': 2.1, 'opacity': .10,
                                'color': [cores[i] for i in classes[indices]]},
        name='Amostra da dominância volumétrica', visible='legendonly',
        hovertemplate='PC1=%{x:.7f}<br>PC2=%{y:.7f}<br>PC3=%{z:.7f}<extra></extra>'))

    controles = []
    for rotulo, alvo in [('Centro dos dados', centro_global)] + [
        (f'Centro G{k+1}', modelo.means_[k]) for k in range(3)]:
        ps = criar_planos(gm, eixos, alvo)
        controles.append(dict(label=rotulo, method='restyle', args=[{
            'x': [p['x'].tolist() for p in ps],
            'y': [p['y'].tolist() for p in ps],
            'z': [p['z'].tolist() for p in ps],
            'surfacecolor': [p['surfacecolor'].tolist() for p in ps],
        }, [0, 1, 2]]))

    fig.update_layout(
        title={'text': 'REGIÕES DE DOMINÂNCIA DAS GAUSSIANAS<br>'
               '<sup>PC1 × PC2 × PC3 | FULL | 3D | K=3 | '
               'argmax das responsabilidades γₖ(z), não é risco de fraude</sup>',
               'x': .5, 'xanchor': 'center'},
        width=1250, height=830, margin=dict(l=2, r=2, t=100, b=20),
        paper_bgcolor='#ffffff',
        scene=dict(
            xaxis=dict(title='PC1'), yaxis=dict(title='PC2'),
            zaxis=dict(title='PC3'), aspectmode='data',
            camera=dict(eye={'x': 1.6, 'y': 1.65, 'z': 1.1}),
        ),
        legend=dict(x=.005, y=.85, bgcolor='rgba(255,255,255,.8)',
                    bordercolor='#cad4df', borderwidth=1),
        updatemenus=[dict(type='dropdown', buttons=controles,
                          direction='down', showactive=True, x=.012,
                          xanchor='left', y=1.12, yanchor='top',
                          bgcolor='#f1f5f9', bordercolor='#cbd5e1')],
        annotations=[dict(x=.012, y=1.175, xref='paper', yref='paper',
                          text='Posição dos três cortes:', showarrow=False,
                          xanchor='left', font={'size': 12, 'color': '#344054'})],
    )
    html = fig.to_html(full_html=True, include_plotlyjs=True,
                       div_id='gmm04_3d', config={'responsive': True,
                                                 'displaylogo': False,
                                                 'scrollZoom': True})
    widgets = '''
<div style="position:fixed;top:12px;right:24px;z-index:1000;background:white;
 box-shadow:0 2px 12px #0002;padding:9px;border:1px solid #ddd;border-radius:8px">
 <button id="gmm04_rotar" style="padding:8px 12px;cursor:pointer">▶ Girar automaticamente</button>
 <span style="font:12px sans-serif;color:#45536c;margin-left:6px">
 Arraste para girar • scroll para zoom • clique na legenda para ocultar</span>
</div>
<script>
(function () {
 const g = document.getElementById('gmm04_3d');
 const b = document.getElementById('gmm04_rotar');
 let t = null, a = 0;
 b.onclick = function () {
   if (t !== null) {clearInterval(t);t=null;b.textContent='▶ Girar automaticamente';return;}
   b.textContent='■ Parar rotação';
   t = setInterval(function () {
     a += 0.025;
     Plotly.relayout(g, {'scene.camera.eye':
       {x:2.35*Math.cos(a),y:2.35*Math.sin(a),z:1.24}});
   }, 90);
 };
})();
</script>
'''
    assert '</body>' in html
    html = html.replace('</body>', widgets + '</body>')
    arquivo = destino / 'GMM04_dominancia_3d_INTERATIVA.html'
    arquivo.write_text(html, encoding='utf-8')
    assert arquivo.stat().st_size > 100_000
    return arquivo


def main():
    raiz = Path.cwd()
    yaml_path = raiz / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    assert yaml_path.is_file(), f'YAML nao encontrado: {yaml_path}'
    assert (raiz / 'src' / 'visualizacao_gmm04.py').is_file(), (
        'Execute na raiz do TCC_EXPERIMENTAL_V3.')
    yaml_bytes = yaml_path.read_bytes()
    checksum = hashlib.sha256(yaml_bytes).hexdigest()
    cfg = yaml.safe_load(yaml_bytes.decode('utf-8-sig'))
    coord, gm, y = sintetizar()
    op = cfg['visualizacao']['graficos']['regioes_dominancia']
    op['resolucao_2d'] = 145
    op['pontos_scatter'] = 500
    op['margem'] = .16
    saida = raiz / 'outputs' / ('validacao_pca3d_06_gmm04_' +
            datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    saida.mkdir(parents=True, exist_ok=False)
    gmm04.gerar(cfg, {'coord': coord, 'gmm': gm, 'y': y},
                saida, 'png', 170, True)
    estatico = saida / f'{OUTPUT_STEM}.png'
    assert estatico.is_file() and estatico.stat().st_size > 25_000, (
        'GMM04 oficial nao gerou PNG 3D.')
    for a, b in [('PC1','PC2'), ('PC1','PC3'), ('PC2','PC3')]:
        p = saida / f'GMM04_malha_{a}_{b}.csv'
        assert p.is_file(), f'CSV do corte ausente: {p.name}'
        tab = pd.read_csv(p)
        assert len(tab) == 145**2
        assert all(f'responsabilidade_G{k}' in tab for k in (1,2,3))
        assert np.allclose(tab[[f'responsabilidade_G{k}' for k in (1,2,3)]].sum(axis=1), 1)
    interativo = html_interativo(coord, gm, saida)
    assert (saida / 'GMM04_dominancia_volume_3d.csv').is_file()
    assert hashlib.sha256(yaml_path.read_bytes()).hexdigest() == checksum, (
        'YAML foi alterado durante a validacao!')
    print('GMM04 PCA 3D: VALIDACAO ISOLADA CONCLUIDA')
    print('3 gaussianas FULL / 39 features sinteticas')
    print('PNG oficial, 3 CSVs de cortes, CSV volumetrico e HTML rotativo gerados.')
    print('Nao houve treinamento completo, selecao OOF ou uso de F10.')
    print('YAML original preservado.')
    print('PNG=' + str(estatico.resolve()))
    print('HTML=' + str(interativo.resolve()))

if __name__ == '__main__':
    main()
