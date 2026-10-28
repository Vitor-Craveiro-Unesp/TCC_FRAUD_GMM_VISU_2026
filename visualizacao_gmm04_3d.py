# -*- coding: utf-8 -*-
"""Representacao interativa 3D da dominancia GMM (somente visualizacao).

Dominancia = argmax das responsabilidades posteriores; NAO e risco de fraude.
Cortes sao secoes do GMM FULL 3D, nao marginais bivariadas.
Nao acessa base original, OOF, resultados F10 ou altera o GMM ajustado.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from scipy.stats import chi2


def _planos(gm, eixos, fixo):
    """Avaliar tres secoes ortogonais de uma MESMA mistura 3D."""
    from src.visualizacao_gmm04 import calcular_dominancia
    saidas = []
    for i, j, om in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        A, B = np.meshgrid(eixos[i], eixos[j], indexing='xy')
        xyz = np.tile(fixo, (A.size, 1))
        xyz[:, i] = A.ravel()
        xyz[:, j] = B.ravel()
        dom, _ = calcular_dominancia(gm, xyz)
        saidas.append({
            'x': xyz[:, 0].reshape(A.shape),
            'y': xyz[:, 1].reshape(A.shape),
            'z': xyz[:, 2].reshape(A.shape),
            'classe': dom.reshape(A.shape).astype(float),
            'eixos': (i, j), 'omitida': om,
        })
    return saidas


def _elipsoide_95(media, cov, *, n_u=38, n_v=24):
    vals, vect = np.linalg.eigh(np.asarray(cov, dtype=float))
    if not np.all(vals > 0):
        raise ValueError('A covariancia FULL nao e positiva definida.')
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    esfera = np.stack((
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
    ), axis=-1)
    T = vect @ np.diag(np.sqrt(vals * chi2.ppf(.95, df=3)))
    xyz = np.einsum('ij,klj->kli', T, esfera) + media
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]


def _escala_discreta(cores):
    """Escala de k faixas, com cmin=-0.5 e cmax=k-0.5."""
    k = len(cores)
    escala = []
    for ind, cor in enumerate(cores):
        esquerda = ind / k
        direita = (ind + 1) / k
        escala.append([esquerda, cor])
        escala.append([direita, cor])
    return escala


def gerar_interativo(config, data, destino, *, salvar_csv=True):
    """Gera HTML offline a partir do modelo GMM ja ajustado, sem retreino.

    Executar apos GMM04.gerar() somente quando a coordenada tiver 3 colunas.
    Retorna o caminho do HTML; nao modifica PNGs, modelos ou arquivos OOF.
    """
    import plotly.graph_objects as go
    from src import visualizacao_gmm04 as base

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    coord = data['coord']
    X = np.asarray(coord, dtype=float)
    gm = data['gmm']
    p = gm.modelo_
    nomes = list(gm.saidas_)
    k = len(nomes)
    if (X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all()
            or k < 1 or np.asarray(p.means_).shape != (k, 3)
            or str(p.covariance_type).lower() != 'full'
            or np.asarray(p.covariances_).shape != (k, 3, 3)):
        raise ValueError('O HTML GMM04 3D requer GMM FULL de tres dimensoes.')
    for c in p.covariances_:
        if np.linalg.eigvalsh(c).min() <= 0:
            raise ValueError('Covariancia FULL invalida.')

    op = config['visualizacao']['graficos']['regioes_dominancia']
    n = int(op.get('resolucao_html_3d', 59))
    if n < 25 or n > 100:
        raise ValueError('Resolucao de cortes HTML 3D deve estar entre 25 e 100.')
    nv = int(op.get('resolucao_volume_html_3d', 23))
    if nv < 9 or nv > 40:
        raise ValueError('Resolucao de volume HTML 3D deve estar entre 9 e 40.')
    limite = float(op.get('margem', .15))
    if not 0 <= limite <= 1:
        raise ValueError('Margem invalida.')

    centro = X.mean(axis=0)
    menor, maior = np.quantile(X, (.005, .995), axis=0)
    amplitude = np.maximum(maior - menor, 1e-6)
    extremos = np.column_stack((menor - (.15 + limite) * amplitude,
                                maior + (.15 + limite) * amplitude))
    eixos = [np.linspace(*extremos[i], n) for i in range(3)]
    cores = [to_hex(c) for c in base._cores(k)[0]]
    escala = _escala_discreta(cores)
    fig = go.Figure()
    secoes = _planos(gm, eixos, centro)

    for secao in secoes:
        i, j = secao['eixos']
        om = secao['omitida']
        fig.add_trace(go.Surface(
            x=secao['x'], y=secao['y'], z=secao['z'],
            surfacecolor=secao['classe'], colorscale=escala,
            cmin=-.5, cmax=k-.5, showscale=False, opacity=.72,
            name=f'Corte PC{i+1} × PC{j+1} (PC{om+1} fixada)',
            showlegend=True,
            hovertemplate=(
                'PC1=%{x:.7f}<br>PC2=%{y:.7f}<br>PC3=%{z:.7f}'
                '<br>Índice da dominante (G = índice + 1): '
                '%{surfacecolor:.0f}<extra></extra>'
            ),
        ))

    for ind in range(k):
        x, y, z = _elipsoide_95(p.means_[ind], p.covariances_[ind])
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=np.full_like(x, float(ind)),
            colorscale=escala, cmin=-.5, cmax=k-.5, showscale=False,
            opacity=.16, name=f'{nomes[ind]} — elipsoide FULL 95%',
            showlegend=True, hoverinfo='skip',
        ))
    for ind, mu in enumerate(p.means_):
        fig.add_trace(go.Scatter3d(
            x=[mu[0]], y=[mu[1]], z=[mu[2]], mode='markers+text',
            marker={'size': 7, 'symbol': 'diamond', 'color': cores[ind],
                    'line': {'color': '#142135', 'width': 1.2}},
            text=[nomes[ind]], textposition='top center',
            name=f'Centro {nomes[ind]}',
            hovertemplate=(
                f'Centro {nomes[ind]}<br>PC1=%{{x:.7f}}<br>'
                'PC2=%{y:.7f}<br>PC3=%{z:.7f}<extra></extra>'
            ),
        ))

    # Volume amostrado: auxilia exploracao, NAO e uma fronteira continua.
    eixos_vol = [np.linspace(*extremos[i], nv) for i in range(3)]
    vol = np.stack(np.meshgrid(*eixos_vol, indexing='ij'), axis=-1).reshape(-1, 3)
    dominante, prob = base.calcular_dominancia(gm, vol)
    if salvar_csv:
        tab = pd.DataFrame(vol, columns=['PC1', 'PC2', 'PC3'])
        tab['gaussiana_dominante'] = [nomes[i] for i in dominante]
        for ind, nome in enumerate(nomes):
            tab[f'responsabilidade_{nome}'] = prob[:, ind]
        tab.to_csv(destino / 'GMM04_dominancia_volume_3d.csv', index=False,
                   float_format='%.17g')
    rng = np.random.default_rng(20261027)
    ids = np.sort(rng.choice(len(vol), size=min(4500, len(vol)), replace=False))
    fig.add_trace(go.Scatter3d(
        x=vol[ids, 0], y=vol[ids, 1], z=vol[ids, 2],
        mode='markers', marker={'size': 2, 'opacity': .11,
                                'color': [cores[i] for i in dominante[ids]]},
        name='Amostra volumétrica da dominância', visible='legendonly',
        hovertemplate='PC1=%{x:.7f}<br>PC2=%{y:.7f}<br>PC3=%{z:.7f}<extra></extra>',
    ))

    posicoes = [('Centro dos dados', centro)] + [
        (f'Centro {nomes[ind]}', p.means_[ind]) for ind in range(k)
    ]
    botoes = []
    for nome, fixo in posicoes:
        sec = _planos(gm, eixos, fixo)
        botoes.append(dict(
            label=nome, method='restyle',
            args=[{
                'x': [a['x'].tolist() for a in sec],
                'y': [a['y'].tolist() for a in sec],
                'z': [a['z'].tolist() for a in sec],
                'surfacecolor': [a['classe'].tolist() for a in sec],
            }, [0, 1, 2]],
        ))

    fig.update_layout(
        title=dict(text='REGIÕES DE DOMINÂNCIA DAS GAUSSIANAS<br>'
                        '<sup>PCA 3D | GMM FULL | '
                        f'K={k} | dominância não representa risco de fraude</sup>',
                   x=.5, xanchor='center', y=.975),
        width=1370, height=850, margin=dict(l=0, r=0, t=105, b=14),
        paper_bgcolor='white',
        scene=dict(xaxis=dict(title='PC1'), yaxis=dict(title='PC2'),
                   zaxis=dict(title='PC3'), aspectmode='data',
                   camera=dict(eye=dict(x=1.6, y=1.65, z=1.1))),
        legend=dict(x=.005, y=.86, bgcolor='rgba(255,255,255,.8)',
                    bordercolor='#cad4df', borderwidth=1),
        updatemenus=[dict(type='dropdown', buttons=botoes, direction='down',
                          showactive=True, x=.012, xanchor='left',
                          y=1.12, yanchor='top', bgcolor='#f1f5f9',
                          bordercolor='#cbd5e1')],
        annotations=[dict(x=.012, y=1.175, xref='paper', yref='paper',
                          text='Posição dos três cortes:', showarrow=False,
                          xanchor='left', font={'size': 12, 'color': '#344054'})],
    )
    pagina = fig.to_html(full_html=True, include_plotlyjs=True,
                         div_id='gmm04_3d',
                         config={'responsive': True, 'displaylogo': False,
                                 'scrollZoom': True})
    controles = '''
<div style="position:fixed;top:12px;right:24px;z-index:1000;background:#fff;
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
   if (t !== null) {clearInterval(t); t=null; b.textContent='▶ Girar automaticamente'; return;}
   b.textContent='■ Parar rotação';
   t=setInterval(function () {
     a += 0.025;
     Plotly.relayout(g, {'scene.camera.eye':
        {x:2.35*Math.cos(a),y:2.35*Math.sin(a),z:1.24}});
   },90);
 };
})();
</script>
'''
    if '</body>' not in pagina:
        raise RuntimeError('HTML gerado sem tag body.')
    pagina = pagina.replace('</body>', controles + '</body>')
    arq = destino / 'GMM04_dominancia_3d_INTERATIVA.html'
    if arq.exists():
        raise FileExistsError(f'HTML GMM04 preexistente: {arq}')
    arq.write_text(pagina, encoding='utf-8')
    if arq.stat().st_size < 100_000:
        raise RuntimeError('HTML offline Plotly incompleto.')
    print('[GERADO] GMM04_dominancia_3d_INTERATIVA.html')
    return arq
