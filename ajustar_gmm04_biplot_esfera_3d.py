# -*- coding: utf-8 -*-
"""Ajusta GMM04 3D e adiciona esfera de correlações PCA3D offline (V3).

Exige integracoes previas GMM04 e Biplot 3D. Criacao de backups,
verificacao AST e rollback. Nao treina, nao altera YAML ou F10.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
from pathlib import Path
import re
import sys

MARCA_GMM = '# GMM04_PCA3D_CORTES_GAUSSIANOS_95_V1'
MARCA_ESFERA = '# PCA3D_ESFERA_CORRELACOES_HTML_V1'
MARCA_GALERIA = '# PCA3D_ESFERA_CORRELACOES_GALERIA_V1'
ARQUIVO_HTML = 'PCA_ESFERA_CORRELACOES_3D_INTERATIVA.html'

NOVA_FUNCAO = r'''

# PCA3D_ESFERA_CORRELACOES_HTML_V1
def gerar_esfera_correlacoes_interativa(correlacoes, destino):
    """Esfera unitária com coordenadas CORR(X_j, PC1/2/3) reais; sem ajuste.

    Seis vetores mais relevantes visíveis inicialmente; todos disponíveis.
    """
    import plotly.graph_objects as go
    from html import escape as escape_html

    destino = Path(destino)
    cols = ['variavel_preprocessada', 'corr_PC1', 'corr_PC2', 'corr_PC3']
    if not all(col in correlacoes.columns for col in cols):
        raise ValueError('Correlacoes PCA3D incompletas para esfera interativa.')
    t = correlacoes.loc[:, cols].copy()
    for col in cols[1:]:
        t[col] = pd.to_numeric(t[col], errors='raise')
    valores = t[cols[1:]].to_numpy(dtype=float)
    if len(t) == 0 or not np.isfinite(valores).all() or (np.abs(valores) > 1.000001).any():
        raise ValueError('Correlacoes fora do intervalo [-1,1] ou nao finitas.')
    t['norma'] = np.linalg.norm(valores, axis=1)
    t = t.sort_values(['norma', 'variavel_preprocessada'],
                      ascending=[False, True], kind='stable').reset_index(drop=True)
    fig = go.Figure()
    az = np.linspace(0, 2*np.pi, 50)
    pol = np.linspace(0, np.pi, 25)
    xx = np.outer(np.cos(az), np.sin(pol))
    yy = np.outer(np.sin(az), np.sin(pol))
    zz = np.outer(np.ones_like(az), np.cos(pol))
    fig.add_trace(go.Surface(x=xx,y=yy,z=zz,
        surfacecolor=np.zeros_like(xx),
        colorscale=[[0,'#90abc8'],[1,'#90abc8']],
        opacity=.09, showscale=False, name='Esfera de raio 1',hoverinfo='skip'))
    for k, linha in t.iterrows():
        cx,cy,cz = [float(linha[col]) for col in cols[1:]]
        nome = str(linha['variavel_preprocessada'])
        top = k < min(6, len(t))
        fig.add_trace(go.Scatter3d(
            x=[0,cx],y=[0,cy],z=[0,cz], mode='lines+markers+text' if top else 'lines+markers',
            line={'color':'#2563a6' if top else '#849ab3','width':6 if top else 3},
            marker={'size':[1,5] if top else [1,3],'color':'#2563a6' if top else '#849ab3'},
            text=['', f'V{k+1}'] if top else None,
            textposition='top center', visible=top, showlegend=False,
            name=f'V{k+1}: {nome}',
            hovertemplate=(f'<b>{escape_html(nome)}</b><br>'
                           f'corr(X,PC1)={cx:.7f}<br>corr(X,PC2)={cy:.7f}'
                           f'<br>corr(X,PC3)={cz:.7f}<extra></extra>')))
    n = len(t)
    fig.update_layout(
        margin={'l':0,'r':0,'t':55,'b':0},height=760,
        paper_bgcolor='white',showlegend=False,
        scene={'xaxis':{'title':'corr(X,PC1)','range':[-1.05,1.05]},
               'yaxis':{'title':'corr(X,PC2)','range':[-1.05,1.05]},
               'zaxis':{'title':'corr(X,PC3)','range':[-1.05,1.05]},
               'aspectmode':'cube',
               'camera':{'eye':{'x':1.6,'y':1.6,'z':1.1}}},
        updatemenus=[{'type':'buttons','direction':'right','x':.5,'y':1.09,
             'xanchor':'center','yanchor':'bottom',
             'buttons':[
               {'label':'6 vetores principais','method':'update',
                 'args':[{'visible':[True]+[k<6 for k in range(n)]}]},
               {'label':f'Todos os {n} vetores','method':'update',
                 'args':[{'visible':[True]*(n+1)}]},
             ]}],
    )
    grafico = fig.to_html(full_html=False,include_plotlyjs=True,
        div_id='esfera_correlacoes_3d',
        config={'responsive':True,'scrollZoom':True,'displaylogo':False})
    linhas = []
    for k, linha in t.iterrows():
        nome = escape_html(str(linha['variavel_preprocessada']))
        c = [float(linha[col]) for col in cols[1:]]
        linhas.append('<tr><td>V%d</td><td>%s</td>%s</tr>'%(
            k+1,nome,''.join('<td>%.7f</td>'%v for v in c)))
    pagina = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Esfera de correlações PCA 3D</title><style>
body{margin:0;background:#f3f6fa;color:#192d45;font:14px system-ui,Segoe UI,sans-serif}
main{max-width:1350px;margin:18px auto;padding:0 18px}
h1{font-size:25px;margin:0 0 8px}p{color:#526579}
.cartao{background:white;border:1px solid #dfe6ee;border-radius:12px;padding:9px}
header{display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;margin-bottom:15px}
button{background:white;padding:10px;border:1px solid #b9c8dc;border-radius:8px;cursor:pointer;margin-left:5px}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{border-bottom:1px solid #e2e8f0;text-align:left;padding:6px}td:nth-child(n+3){font-variant-numeric:tabular-nums}
details{background:white;border-radius:9px;padding:13px;margin:13px 0}
</style></head><body><main><header><div><h1>ESFERA DE CORRELAÇÕES — PCA 3D</h1>
<p>Escala real [-1, 1]; seis vetores destacados. Dados de desenvolvimento F1–F9.</p></div>
<div><button id="girar">▶ Girar</button><button id="restaurar">Restaurar câmera</button></div></header>
<div class="cartao">""" + grafico + """</div>
<details><summary>Correspondência dos vetores (V1, V2, ...) e correlações</summary>
<table><thead><tr><th>Vetor</th><th>Variável</th><th>PC1</th><th>PC2</th><th>PC3</th></tr></thead>
<tbody>""" + '\n'.join(linhas) + """</tbody></table></details>
<p>Arraste para girar, use a roda para aproximar. Os vetores usam as correlações originais, sem ampliação visual.</p>
</main><script>(function(){const gd=document.getElementById('esfera_correlacoes_3d');
const b=document.getElementById('girar');let t=null,a=0;
function stop(){if(t!==null){clearInterval(t);t=null;}b.textContent='▶ Girar';}
b.onclick=function(){if(t!==null){stop();return;}b.textContent='■ Parar';
t=setInterval(function(){a+=.022;Plotly.relayout(gd,{'scene.camera.eye':
{x:2.3*Math.cos(a),y:2.3*Math.sin(a),z:1.15}});},100);};
document.getElementById('restaurar').onclick=function(){stop();Plotly.relayout(gd,{'scene.camera.eye':{x:1.6,y:1.6,z:1.1}})};
gd.addEventListener('pointerdown',stop);window.addEventListener('pagehide',stop);
})();</script></body></html>"""
    if destino.exists():
        print('[PRESERVADO]', destino.name)
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(pagina,encoding='utf-8')
    if destino.stat().st_size < 100000:
        raise RuntimeError('Esfera PCA 3D: HTML offline incompleto.')
    print('[GERADO]',destino.name)
    return destino
'''


def preparar_gmm04(orig: str) -> str:
    if MARCA_GMM in orig:
        return orig
    # O nome GMM04 e redundante no titulo do PNG.
    novo, count = re.subn(r'GMM\s*0?4\s*[—–-]\s*Regiões de dominância',
                          'REGIÕES DE DOMINÂNCIA DAS GAUSSIANAS', orig,
                          flags=re.IGNORECASE)
    if count != 1:
        raise RuntimeError('Titulo GMM04 3D nao localizado unicamente; nada alterado.')
    chave = '        # Fronteiras entre gaussianas dominantes.'
    if novo.count(chave) != 1:
        raise RuntimeError('Trecho dos planos GMM04 nao encontrado unicamente.')
    bloco = '''        # GMM04_PCA3D_CORTES_GAUSSIANOS_95_V1
        # Intersecao verdadeira do elipsoide FULL 95% com cada plano.
        if d == 3:
            from matplotlib.patches import Ellipse
            for g in range(k):
                secao = geo._elipse_95_corte(gm, g, (i, j), centro_desenvolvimento)
                if secao is None:
                    continue  # O plano nao intersecta o elipsoide de 95%.
                centro_el, largura_el, altura_el, angulo_el = secao
                ax.add_patch(Ellipse(
                    xy=centro_el, width=largura_el, height=altura_el,
                    angle=angulo_el, fill=False, linestyle='--',
                    linewidth=1.65, edgecolor=cores[g], zorder=4,
                ))

'''
    novo = novo.replace(chave,bloco+chave,1)
    # Completar a legenda 3D pela ARVORE SINTATICA: suporta espacos,
    # comentarios e formatacao diferente dos modulos ja integrados.
    tree = ast.parse(novo)
    cand = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'fig'
                and node.func.attr == 'legend'):
            continue
        for kw in node.keywords:
            if kw.arg != 'handles':
                continue
            expr = kw.value
            if (isinstance(expr, ast.Call)
                    and isinstance(expr.func, ast.Name)
                    and expr.func.id == '_legenda'):
                cand.append(expr)
    if len(cand) != 1:
        raise RuntimeError(
            'Legenda GMM04 3D nao identificada de forma univoca: '
            f'{len(cand)} encontrada(s). Nada alterado.'
        )
    alvo = cand[0]
    expr = ast.get_source_segment(novo, alvo)
    if not expr:
        raise RuntimeError('Nao foi possivel extrair a expressao da legenda GMM04.')
    novo_expr = (
        '(' + expr + " + [Line2D([], [], color='#495c74', ls='--', lw=1.65, "
        "label='Seções gaussianas FULL 95%')])"
    )
    linhas = novo.encode('utf-8').splitlines(keepends=True)
    comeco = sum(map(len, linhas[:alvo.lineno - 1])) + alvo.col_offset
    fim = sum(map(len, linhas[:alvo.end_lineno - 1])) + alvo.end_col_offset
    b = novo.encode('utf-8')
    if b[comeco:fim] != expr.encode('utf-8'):
        raise RuntimeError('Posicao da legenda nao confere; nada alterado.')
    novo = (b[:comeco] + novo_expr.encode('utf-8') + b[fim:]).decode('utf-8')
    return novo


def preparar_biplot(orig: str) -> str:
    if MARCA_ESFERA in orig:
        return orig
    marca = "    return {'coeficientes':"
    if orig.count(marca) != 1 or 'def gerar_complementares(' not in orig:
        raise RuntimeError('Ponto de integracao PCA3D Biplot nao identificado.')
    chamada = '''    # PCA3D_ESFERA_CORRELACOES_HTML_V1: usando tabela PCA existente.
    gerar_esfera_correlacoes_interativa(
        tabela_correlacoes, destino / 'PCA_ESFERA_CORRELACOES_3D_INTERATIVA.html'
    )
'''
    return orig.replace(marca, chamada+marca,1) + NOVA_FUNCAO



def preparar_biplot_png(orig: str) -> str:
    marca = '# PCA3D_BIPLOT_PNG_SEM_RODAPE_V1'
    if marca in orig:
        return orig
    arvore = ast.parse(orig)
    fns = [n for n in arvore.body
           if isinstance(n, ast.FunctionDef) and n.name == '_fig_3d']
    if len(fns) != 1:
        raise RuntimeError('Biplot PNG: _fig_3d nao encontrada.')
    f = fns[0]
    rodapes = [n for n in ast.walk(f)
               if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Attribute)
               and isinstance(n.value.func.value, ast.Name)
               and n.value.func.value.id == 'fig'
               and n.value.func.attr == 'text'
               and 'Correlacoes (PC1, PC2, PC3)' in
               (ast.get_source_segment(orig,n) or '')]
    if len(rodapes) != 1:
        raise RuntimeError('Biplot: rodape de correlacoes 3D nao localizado unicamente.')
    linhas=orig.splitlines(keepends=True)
    n=rodapes[0]
    linhas[n.lineno-1:n.end_lineno] = ['    # PCA3D_BIPLOT_PNG_SEM_RODAPE_V1: tabela no HTML interativo.\n']
    novo=''.join(linhas)
    # Evitar sobrepor nomes longos na esfera do PNG: V1..V6.
    alvo = "av.text(u * 1.1, v * 1.1, w * 1.1, str(r['variavel_preprocessada']), fontsize=7)"
    if novo.count(alvo) == 1:
        novo=novo.replace(alvo,"av.text(u * 1.1, v * 1.1, w * 1.1, f'V{i}', fontsize=8)",1)
    else:
        raise RuntimeError('Biplot 3D: rotulo das variaveis mudou; nada alterado.')
    novo=novo.replace('fig.tight_layout(rect=[0,.07,1,.95])',
                      'fig.tight_layout(rect=[0,.02,1,.95])',1)
    return novo

def preparar_galeria(orig: str) -> str:
    """Inclui SOMENTE esfera PCA3D no ponto que copia arquivos da galeria.

    Localiza o bloco por AST: sem exigir indentacao fixa ou numero total
    de ocorrencias de `if permitido:` no integrador existente.
    """
    if MARCA_GALERIA in orig:
        return orig
    if "'.html'" not in orig and '".html"' not in orig:
        raise RuntimeError('Integrador nao admite extensao .html; nada alterado.')
    if ARQUIVO_HTML in orig:
        raise RuntimeError(
            'HTML da esfera ja citado sem marcador oficial; revisar manualmente.'
        )

    arvore = ast.parse(orig)
    funcoes = [n for n in arvore.body if isinstance(n, ast.FunctionDef)
               and n.name == 'integrar']
    if len(funcoes) != 1:
        raise RuntimeError('Galeria: funcao integrar() nao localizada unicamente.')

    # A escolha correta eh o `if permitido` que efetivamente copia o
    # arquivo e registra a entrada no indice HTML; outros ifs sao ignorados.
    blocos = []
    for no in ast.walk(funcoes[0]):
        if not isinstance(no, ast.If):
            continue
        if not (isinstance(no.test, ast.Name) and no.test.id == 'permitido'):
            continue
        corpo = '\n'.join(ast.get_source_segment(orig, filho) or ''
                          for filho in no.body)
        if 'salvos.append' in corpo and (
                '_arquivo_seguro_copiar' in corpo
                or 'copy2(' in corpo or 'copyfile(' in corpo
                or 'relativo' in corpo):
            blocos.append(no)
    if len(blocos) != 1:
        raise RuntimeError(
            'Galeria: bloco final de copia nao identificado de forma '
            f'univoca ({len(blocos)} encontrado(s)). Nenhum arquivo alterado.'
        )

    alvo = blocos[0]
    linhas = orig.splitlines(keepends=True)
    linha = linhas[alvo.lineno - 1]
    recuo = linha[:len(linha) - len(linha.lstrip(' \t'))]
    if not linha.lstrip().startswith('if permitido:'):
        raise RuntimeError('Galeria: comando de copia inesperado; bloqueado.')

    trecho = [
        '# PCA3D_ESFERA_CORRELACOES_GALERIA_V1',
        "if ('pca_biplot' in ativos_geo",
        "        and escolha['configuracao']['metodo'].upper() == 'PCA'",
        "        and int(escolha['configuracao']['reducao']['n_components']) == 3",
        "        and secao == 'geometria_desenvolvimento'",
        "        and arq.name == 'PCA_ESFERA_CORRELACOES_3D_INTERATIVA.html'):",
        '    permitido = True',
        '',
    ]
    linhas[alvo.lineno - 1:alvo.lineno - 1] = [
        recuo + l + '\n' for l in trecho
    ]
    novo = ''.join(linhas)
    compile(novo, 'visualizacao_integrada.py', 'exec')
    return novo


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--verificar',action='store_true')
    ap.add_argument('--aplicar',action='store_true')
    args=ap.parse_args()
    if not args.verificar and not args.aplicar:
        ap.error('Use --verificar ou --aplicar.')
    root=Path.cwd()
    caminhos = {
        'GMM04':root/'src'/'visualizacao_gmm04.py',
        'Biplot':root/'src'/'visualizacao_pca3d_complementos.py',
        'Biplot PNG':root/'src'/'visualizacao_pca_biplot.py',
        'Galeria':root/'src'/'visualizacao_integrada.py',
    }
    for nome, p in caminhos.items():
        if not p.is_file():
            raise RuntimeError(f'Modulo ausente: {nome}: {p}. Nenhum arquivo alterado.')
    before = {p:p.read_bytes() for p in caminhos.values()}
    textos = {p: before[p].decode('utf-8-sig') for p in before}
    funcs = {'GMM04':preparar_gmm04,
             'Biplot':preparar_biplot,
             'Biplot PNG':preparar_biplot_png,'Galeria':preparar_galeria}
    novos = {}
    for nome, p in caminhos.items():
        novo=funcs[nome](textos[p])
        compile(novo,str(p),'exec')
        ast.parse(novo)
        novos[p]=novo
        print(f'[VALIDADO] {nome}:', 'ja instalado' if novo==textos[p] else 'alteracao pronta')
    if args.verificar:
        print('PREFLIGHT APROVADO. Nada modificado. Nao utilizou F10.')
        return
    modificados=[p for p in novos if novos[p]!=textos[p]]
    if not modificados:
        print('ALTERACOES JA INSTALADAS. Nada modificado.')
        return
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backups={p:p.with_name(p.name+'.ANTES_ESFERA_GMM04_'+stamp+'.bak') for p in modificados}
    for p,b in backups.items():
        if b.exists():
            raise RuntimeError(f'Backup ja existe: {b}')
    for p,b in backups.items():
        b.write_bytes(before[p])
    try:
        for p in modificados:
            bom=b'\xef\xbb\xbf' if before[p].startswith(b'\xef\xbb\xbf') else b''
            p.write_bytes(bom+novos[p].encode('utf-8'))
            compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
    except BaseException:
        for p in modificados:
            p.write_bytes(before[p])
        raise
    for p in modificados:
        print('[ATUALIZADO]',p)
        print('  Backup:',backups[p].name)
    print('CONCLUIDO: PCA 1D/2D, YAML, OOF e F10 preservados.')
    print('Novos ensaios PCA3D irao criar a esfera interativa e os contornos GMM04.')


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print('BLOQUEADO:',exc,file=sys.stderr)
        sys.exit(1)
