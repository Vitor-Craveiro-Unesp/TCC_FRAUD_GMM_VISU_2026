# -*- coding: utf-8 -*-
"""Auditoria isolada GMM07 / PCA 3D / K=3 FULL.

Colocar na raiz do TCC_EXPERIMENTAL_V3 e executar:
    .\\.venv\\Scripts\\python.exe .\\validar_pca3d_07_gmm07.py

Produz PNG oficial comparando cortes 0,50 e tau* em 3 planos,
3 CSVs de malhas, CSV com MCC de simulacao, CSV do volume e HTML offline
rotativo das duas fronteiras e regiao que muda de classe.
Nao altera src/, YAML, nem ensaios existentes e nao utiliza F10.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src import visualizacao_gmm07 as gmm07
from src import visualizacao_geometrica as geo

SEED = 20261027
PREFIXO = 'GMM07_sensibilidade_3d_comparativo'


def preparar_dados():
    rng = np.random.default_rng(SEED)
    centros = np.array([[-2.5, -1.4, -.5], [2.5, -1., 1.3], [-.3, 2.6, -1.6]])
    covs = np.array([
        [[.6, .2, .05], [.2, .8, -.1], [.05, -.1, .65]],
        [[.8, -.2, .13], [-.2, .6, .02], [.13, .02, .70]],
        [[.75, .15, -.10], [.15, .9, .2], [-.10, .2, .5]],
    ])
    W = rng.normal(size=(3, 39))

    def amostrar(n):
        latent = np.vstack([
            rng.multivariate_normal(centros[k], covs[k], n) for k in range(3)
        ])
        return latent @ W + rng.normal(scale=.53, size=(len(latent), 39))

    X_train = amostrar(450)
    X_valid = amostrar(300)
    escalador = StandardScaler().fit(X_train)
    pca = PCA(n_components=3, svd_solver='full').fit(escalador.transform(X_train))
    Z_train = pca.transform(escalador.transform(X_train))
    Z_val = pca.transform(escalador.transform(X_valid))
    modelo = GaussianMixture(n_components=3, covariance_type='full', n_init=3,
                             max_iter=300, random_state=SEED).fit(Z_train)
    assert modelo.converged_
    assert modelo.covariances_.shape == (3, 3, 3)
    nomes = ['G1','G2','G3']
    gmm = SimpleNamespace(modelo_=modelo, saidas_=nomes,
                          colunas_=['PC1','PC2','PC3'])
    # Riscos sintéticos atribuídos por componente, não estimativas de OOF.
    riscos = np.array([.03, .32, .62])
    rh = SimpleNamespace(riscos_=pd.Series(riscos, index=nomes))
    p_train = modelo.predict_proba(Z_train) @ riscos
    p_val = modelo.predict_proba(Z_val) @ riscos
    y_train = rng.binomial(1, p_train).astype(np.int8)
    y_val = rng.binomial(1, p_val).astype(np.int8)

    # Otimo da simulacao computado apenas em amostra de avaliacao separada.
    # Este resultado NÃO deve ser descrito como MCC OOF verdadeiro.
    limiares = np.linspace(.05, .95, 901)
    mcc = np.array([matthews_corrcoef(y_val, p_val >= t) for t in limiares])
    tau = float(limiares[np.argmax(mcc)])
    assert 0 < tau < 1
    data = {
        'coord': pd.DataFrame(Z_train, columns=gmm.colunas_),
        'gmm': gmm, 'rh': rh, 'y': y_train, 'scores': p_train,
    }
    avaliacao = {'origem': 'simulado', 'y': y_val, 'p': p_val}
    escolha = {'limiar_mcc_oof': tau}  # Nome de contrato do modulo; simulacao.
    return data, escolha, avaliacao


def html_interativo(data, escolha, avaliacao, destino):
    """Mesma funcao de risco 3D, duas isosuperficies e regiao de mudanca."""
    X = data['coord'].to_numpy(dtype=float)
    gm, rh = data['gmm'], data['rh']
    tau = float(escolha['limiar_mcc_oof'])
    mcc_fixo, mcc_tau = geo._avaliar_cortes_oof(avaliacao, tau)
    # Malha volumétrica única, não combinação dos três cortes 2D.
    limites = [geo._faixa(X[:, j], .13) for j in range(3)]
    ns = 37
    eixos = [np.linspace(*limites[j], ns) for j in range(3)]
    xx, yy, zz = np.meshgrid(*eixos, indexing='ij')
    pontos = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    risco = geo._pontuar_gmm_array(gm, rh, pontos)
    cf, co, mudou = geo._mudanca_classes(risco, tau)
    assert np.array_equal(mudou, cf != co)
    assert np.isfinite(risco).all()

    tabela = pd.DataFrame(pontos, columns=['PC1','PC2','PC3'])
    tabela['score_risco'] = risco
    tabela['classe_corte_050'] = cf.astype(np.uint8)
    tabela['classe_corte_mcc_simulado'] = co.astype(np.uint8)
    tabela['mudou_classe'] = mudou.astype(np.uint8)
    tabela.to_csv(destino / 'GMM07_malha_volumetrica_3d.csv', index=False,
                  float_format='%.17g')

    fig = go.Figure()
    var = risco.max() - risco.min()
    def superficie(t, cor, nome, initial):
        if not (risco.min() < t < risco.max()):
            return None
        eps = min(0.0005, max(var * 0.00002, 1e-7))
        fig.add_trace(go.Isosurface(
            x=pontos[:, 0], y=pontos[:, 1], z=pontos[:, 2],
            value=risco, isomin=t-eps, isomax=t+eps, surface_count=1,
            colorscale=[[0,cor],[1,cor]], showscale=False,
            opacity=.36, caps=dict(x_show=False,y_show=False,z_show=False),
            visible=initial, name=nome, showlegend=True,
            hovertemplate='PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}'
                          '<br>Score=%{value:.7f}<extra></extra>'
        ))
        return len(fig.data)-1

    i_fixo = superficie(.5, '#15191f', 'Fronteira τ=0,5000000', True)
    i_otimo = superficie(tau, '#d17b00', f'Fronteira τ*={tau:.7f}', True)

    indices = np.flatnonzero(mudou)
    if len(indices) > 3600:
        indices = np.random.default_rng(SEED).choice(indices, 3600, replace=False)
    fig.add_trace(go.Scatter3d(
        x=pontos[indices,0],y=pontos[indices,1],z=pontos[indices,2],
        mode='markers', marker=dict(size=2.2,color='#e3b531',opacity=.22),
        visible='legendonly', name='Região que muda de classe (amostra da malha)',
        customdata=risco[indices],
        hovertemplate='PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}'
                      '<br>Score=%{customdata:.7f}<extra></extra>'
    ))

    palette=['#1f77b4','#ff7f0e','#2ca02c']
    for k in range(3):
        mu = gm.modelo_.means_[k]
        fig.add_trace(go.Scatter3d(
            x=[mu[0]],y=[mu[1]],z=[mu[2]],mode='markers+text',
            marker=dict(size=6,color=palette[k],symbol='diamond'),
            text=[f'G{k+1}'], textposition='top center',
            name=f'Centro G{k+1}', visible=True,
            hovertemplate=f'Centro G{k+1}<br>PC1=%{{x:.7f}}<br>PC2=%{{y:.7f}}'
                          '<br>PC3=%{z:.7f}<extra></extra>'
        ))

    # Botões para isosuperfícies: o grafico 2x3 permanece PNG oficial.
    def visivel(qual):
        v=[True]*len(fig.data)
        if i_fixo is not None: v[i_fixo] = qual in ('ambas','fixo')
        if i_otimo is not None: v[i_otimo] = qual in ('ambas','otimo')
        v[2 if i_fixo is not None and i_otimo is not None else (1 if i_fixo is not None or i_otimo is not None else 0)]='legendonly'
        return v

    buttons=[dict(label=nome,method='restyle',args=[{'visible':visivel(chave)}])
             for chave,nome in [('ambas','Mostrar ambas'),('fixo','Somente τ=0,50'),
                                ('otimo','Somente τ*')]]
    fig.update_layout(
        title=dict(text=('SENSIBILIDADE DA FRONTEIRA AO LIMIAR'
                         '<br><sup>PCA 3D | FULL | K=3 | PRÉVIA SINTÉTICA — NÃO É OOF REAL'
                         f' | MCC(0,50)={mcc_fixo:.7f} | MCC(τ*={tau:.7f})={mcc_tau:.7f}</sup>'),
                   x=.5, xanchor='center'),
        width=1250,height=850,margin=dict(l=5,r=5,t=135,b=10),
        scene=dict(xaxis_title='PC1',yaxis_title='PC2',zaxis_title='PC3',
                   aspectmode='data',camera=dict(eye=dict(x=1.65,y=1.55,z=1.1))),
        legend=dict(x=.005,y=.90,bgcolor='rgba(255,255,255,.80)'),
        updatemenus=[dict(type='buttons',direction='right',buttons=buttons,
                          x=.02,y=1.08,xanchor='left',yanchor='top',
                          bgcolor='#f4f6fa')]
    )
    html=fig.to_html(full_html=True,include_plotlyjs=True,
                     div_id='gmm07_3d',config={'responsive':True,'scrollZoom':True,
                                              'displaylogo':False})
    widgets='''
<div style="position:fixed;right:22px;top:14px;z-index:1000;background:white;
 padding:9px 12px;border:1px solid #ddd;border-radius:8px;font:12px sans-serif">
 <button id="gmm07_auto" style="padding:6px 9px;cursor:pointer">▶ Girar automaticamente</button>
 <span style="color:#4b5563;margin-left:7px">Arraste para girar • scroll para zoom • legenda para alternar</span>
</div>
<script>
(function(){const g=document.getElementById('gmm07_3d');
 const b=document.getElementById('gmm07_auto');let id=null,a=0;
 b.onclick=function(){if(id){clearInterval(id);id=null;b.textContent='▶ Girar automaticamente';return;}
 b.textContent='■ Parar rotação';id=setInterval(function(){a+=.023;
 Plotly.relayout(g,{'scene.camera.eye':{x:2.35*Math.cos(a),y:2.35*Math.sin(a),z:1.25}});
 },90);};})();
</script>
'''
    assert '</body>' in html
    html=html.replace('</body>',widgets+'</body>')
    path=destino/(PREFIXO+'_INTERATIVA.html')
    path.write_text(html,encoding='utf-8')
    assert path.stat().st_size>100_000
    return path


def main():
    raiz=Path.cwd()
    cfg_path=raiz/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    assert cfg_path.is_file(), 'Execute na raiz do TCC_EXPERIMENTAL_V3.'
    assert (raiz/'src'/'visualizacao_gmm07.py').is_file(), 'GMM07 oficial nao encontrado.'
    digest=hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    config=yaml.safe_load(cfg_path.read_text(encoding='utf-8-sig'))
    config['visualizacao']['geometria']['pares_2d']=[[1,2],[1,3],[2,3]]
    op=config['visualizacao']['graficos']['sensibilidade_fronteira']
    op['resolucao_2d']=120
    op['n_pontos_scatter']=900
    op['margem']=.15
    data,escolha,avaliacao=preparar_dados()
    saida=raiz/'outputs'/('validacao_pca3d_07_gmm07_'+
         datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    saida.mkdir(parents=True,exist_ok=False)
    resultado=gmm07.gerar(config,data,escolha,saida,'png',160,True,
                          avaliacao=avaliacao)
    assert resultado['n_planos']==3 and resultado['origem_mcc']=='simulado'
    png=saida/(PREFIXO+'.png')
    assert png.is_file() and png.stat().st_size>30000
    for a,b in [('PC1','PC2'),('PC1','PC3'),('PC2','PC3')]:
        caminho=saida/f'GMM07_malha_{a}_{b}.csv'
        assert caminho.is_file(),f'CSV ausente: {caminho.name}'
        tab=pd.read_csv(caminho)
        assert len(tab)==120**2
        assert np.array_equal(tab['mudou'].to_numpy(dtype=bool),
                              (tab['pred_fixo']!=tab['pred_otimo']).to_numpy())
    assert (saida/'GMM07_metricas_comparativas.csv').is_file()
    interativo=html_interativo(data,escolha,avaliacao,saida)
    assert hashlib.sha256(cfg_path.read_bytes()).hexdigest()==digest, 'YAML foi alterado.'
    print('GMM07 PCA 3D — VALIDACAO ISOLADA CONCLUIDA')
    print(f'K=3 FULL | tau*: {escolha["limiar_mcc_oof"]:.7f}')
    print(f'MCC da simulacao, 0,50: {resultado["mcc_fixo"]:.7f}')
    print(f'MCC da simulacao, tau*: {resultado["mcc_otimo"]:.7f}')
    print('PNG + 3 CSVs cortes + CSV MCC + CSV volume + HTML offline.')
    print('YAML e outros ensaios preservados, F10 nao utilizado.')
    print('PNG='+str(png.resolve()))
    print('HTML='+str(interativo.resolve()))

if __name__=='__main__':
    main()
