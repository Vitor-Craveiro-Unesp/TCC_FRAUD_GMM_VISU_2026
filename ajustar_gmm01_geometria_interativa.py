# -*- coding: utf-8 -*-
"""GMM01 PCA3D: retira rodape do PNG de geometria e acrescenta HTML Plotly rotativo.

Instalacao preservadora, com backup e rollback; nao modifica YAML, nao executa OOF/F10.
Usar na raiz do TCC_EXPERIMENTAL_V3:
  .venv\\Scripts\\python.exe ajustar_gmm01_geometria_interativa.py --instalar
  .venv\\Scripts\\python.exe ajustar_gmm01_geometria_interativa.py --testar
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
from pathlib import Path
import re
import sys

MARCA = 'GMM01_GEOMETRIA_HTML_3D_V1'
ARQ_HTML = 'GMM01_geometria_gaussianas_FULL_3d_95_INTERATIVA.html'

NOVA_FUNCAO = r'''

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

'''


def modificar_modulo(texto: str) -> str:
    if MARCA in texto:
        return texto
    tree = ast.parse(texto)
    lines = texto.splitlines(keepends=True)
    f_geo = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_geometria_png'), None)
    f_gerar = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gerar_complementares'), None)
    if f_geo is None or f_gerar is None:
        raise RuntimeError('Funcoes do GMM01 3D nao encontradas. Nenhum arquivo alterado.')
    # Rodape do PNG, caso ainda exista: remover apenas a chamada com texto alvo.
    chamadas = []
    for stmt in f_geo.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == 'text':
                if any(isinstance(arg, ast.Constant) and isinstance(arg.value, str) and
                       'Elipsoides: concentração de cada gaussiana' in arg.value for arg in call.args):
                    chamadas.append(stmt)
    if len(chamadas) > 1:
        raise RuntimeError('Rodape aparece mais de uma vez; nenhum arquivo alterado.')
    if chamadas:
        s = chamadas[0]
        lines[s.lineno - 1:s.end_lineno] = ['']

    source = ''.join(lines)
    if 'def _geometria_interativa_html(' in source:
        raise RuntimeError('Gerador interativo ja existe sem o marcador esperado. Nada alterado.')
    # Insert generator before official entry point using AST after changing footer.
    new_tree = ast.parse(source)
    entry = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gerar_complementares')
    ln = source.splitlines(keepends=True)
    ln.insert(entry.lineno - 1, NOVA_FUNCAO)
    source = ''.join(ln)
    call = '    outputs.append(_geometria_png(xyz,mu,cov,dest,int(dpi)))'
    if source.count(call) != 1:
        raise RuntimeError('Ponto de geracao PNG nao localizado. Nenhum arquivo alterado.')
    source = source.replace(call, call + '\n    outputs.append(_geometria_interativa_html(xyz,mu,cov,weights,dest))', 1)
    source = '# ' + MARCA + '\n' + source
    compile(source, 'src/visualizacao_gmm01_3d.py', 'exec')
    return source


def modificar_galeria(texto: str) -> str:
    if MARCA in texto:
        return texto
    if "FORMATS = ('.png', '.pdf', '.svg', '.html')" not in texto:
        raise RuntimeError('Galeria nao reconhece HTML. Necessaria versao integrada do PCA3D.')
    if '# PCA3D_GMM01_GALERIA_V1' not in texto:
        raise RuntimeError('Integracao original GMM01 3D nao encontrada. Nada alterado.')
    # Colocar a regra junto das regras de whitelist, sem mudar as anteriores.
    linha_anchor = '            if permitido:'
    if texto.count(linha_anchor) != 1:
        raise RuntimeError('Ponto de inclusao na galeria nao localizado de forma unica.')
    inserir = f'''                # {MARCA}: geometria 3D interativa do vencedor.
                if ('densidades_marginais' in ativos_geo
                        and escolha['configuracao']['metodo'].upper() == 'PCA'
                        and int(escolha['configuracao']['reducao']['n_components']) == 3
                        and arq.name == '{ARQ_HTML}'):
                    permitido = True
                # Mostrar o HTML em vez do PNG estatico na galeria.
                if ('densidades_marginais' in ativos_geo
                        and escolha['configuracao']['metodo'].upper() == 'PCA'
                        and int(escolha['configuracao']['reducao']['n_components']) == 3
                        and arq.name == 'GMM01_geometria_gaussianas_FULL_3d_95.png'):
                    permitido = False

'''
    novo = texto.replace(linha_anchor, inserir + linha_anchor, 1)
    compile(novo, 'src/visualizacao_integrada.py', 'exec')
    return novo


def instalar() -> None:
    raiz = Path.cwd()
    fonte = raiz / 'src' / 'visualizacao_gmm01_3d.py'
    galeria = raiz / 'src' / 'visualizacao_integrada.py'
    for arquivo in [fonte, galeria]:
        if not arquivo.is_file():
            raise RuntimeError(f'Arquivo ausente: {arquivo}')
    fbytes = fonte.read_bytes()
    gbytes = galeria.read_bytes()
    ft = fbytes.decode('utf-8-sig')
    gt = gbytes.decode('utf-8-sig')
    if MARCA in ft and MARCA in gt:
        print('JA INSTALADO. Nenhum arquivo modificado.')
        return
    if MARCA in ft or MARCA in gt:
        raise RuntimeError('Instalacao parcial detectada. Nenhum arquivo alterado.')
    nf = modificar_modulo(ft)
    ng = modificar_galeria(gt)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    bf = fonte.with_name(fonte.name + '.ANTES_GEOMETRIA_HTML_' + stamp + '.bak')
    bg = galeria.with_name(galeria.name + '.ANTES_GEOMETRIA_HTML_' + stamp + '.bak')
    bf.write_bytes(fbytes)
    bg.write_bytes(gbytes)
    try:
        fonte.write_bytes((b'\xef\xbb\xbf' if fbytes.startswith(b'\xef\xbb\xbf') else b'') + nf.encode('utf-8'))
        galeria.write_bytes((b'\xef\xbb\xbf' if gbytes.startswith(b'\xef\xbb\xbf') else b'') + ng.encode('utf-8'))
        compile(fonte.read_text(encoding='utf-8-sig'), str(fonte), 'exec')
        compile(galeria.read_text(encoding='utf-8-sig'), str(galeria), 'exec')
    except Exception:
        fonte.write_bytes(fbytes)
        galeria.write_bytes(gbytes)
        raise
    print('GMM01 GEOMETRIA 3D INTERATIVA INSTALADA')
    print('Modulo:', fonte)
    print('Galeria:', galeria)
    print('Backup do modulo:', bf)
    print('Backup da galeria:', bg)
    print('YAML, OOF e F10 preservados.')


def testar() -> None:
    from types import SimpleNamespace
    import numpy as np
    from sklearn.mixture import GaussianMixture
    from src import visualizacao_gmm01_3d as vis
    import tempfile

    assert MARCA in (Path('src') / 'visualizacao_gmm01_3d.py').read_text(encoding='utf-8-sig'), 'Execute --instalar primeiro.'
    rng = np.random.default_rng(20261028)
    centres = np.array([[-1.3, 0.3, 1.0], [1.2, -0.8, 0.0], [-0.6, 1.4, -1.1]])
    xyz = np.vstack([rng.multivariate_normal(centres[k], np.eye(3) * (0.35 + 0.12*k), size=75) for k in range(3)])
    gm = GaussianMixture(n_components=3, covariance_type='full', n_init=2, max_iter=150, random_state=20261028).fit(xyz)
    import pandas as pd
    data = {'coord':pd.DataFrame(xyz, columns=['PC1','PC2','PC3']), 'gmm':SimpleNamespace(modelo_=gm)}
    xyz, mu, cov, weights = vis._validar(data)
    from datetime import datetime
    saida = Path('outputs') / ('validacao_geometria_interativa_gmm01_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    saida.mkdir(parents=True, exist_ok=False)
    p_png = vis._geometria_png(xyz, mu, cov, saida, dpi=100)
    p_html = vis._geometria_interativa_html(xyz, mu, cov, weights, saida)
    assert p_png.is_file() and p_png.stat().st_size > 10000
    assert p_html.is_file() and p_html.stat().st_size > 500000
    content = p_html.read_text(encoding='utf-8')
    import json
    for required in ['Observações projetadas', 'G1 — elipsoide FULL 95%', 'G2 — elipsoide FULL 95%', 'G3 — elipsoide FULL 95%', 'Girar automaticamente', 'Só gaussianas']:
        if (required not in content and json.dumps(required, ensure_ascii=True)[1:-1] not in content):
            raise RuntimeError('HTML incompleto: ' + required)
    print('PNG_VALIDADO=', p_png.resolve())
    print('HTML_VALIDADO=', p_html.resolve())
    print('GMM01 GEOMETRIA INTERATIVA APROVADA (DADOS SINTETICOS).')
    print('Nenhum OOF ou F10 executado.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--instalar', action='store_true')
    grp.add_argument('--testar', action='store_true')
    args = parser.parse_args()
    if args.instalar:
        instalar()
    else:
        testar()


if __name__ == '__main__':
    main()
