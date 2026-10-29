r"""Cria HTMLs 3D rotacionaveis a partir dos CSVs da auditoria PCA3D.

Uso na raiz do TCC_EXPERIMENTAL_V3:
    .\.venv\Scripts\python.exe .\validar_pca3d_interativo.py

Nao treina, nao escreve YAML, nao toca em F10 e nao modifica PNGs.
Requer: plotly, pandas, numpy.
"""
from __future__ import annotations

from pathlib import Path
from html import escape
import sys
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.colors import qualitative
except ImportError as exc:
    raise SystemExit(
        'Plotly nao instalado neste ambiente. Execute: '
        r'.\.venv\Scripts\python.exe -m pip install plotly'
    ) from exc


def ultimo_csv(raiz: Path, prefixo_pasta: str, arquivo: str) -> Path:
    pasta_outputs = raiz / 'outputs'
    if not pasta_outputs.is_dir():
        raise FileNotFoundError(f'Pasta outputs nao encontrada: {pasta_outputs}')
    candidatos = [
        p / arquivo for p in pasta_outputs.iterdir()
        if p.is_dir() and p.name.startswith(prefixo_pasta)
        and (p / arquivo).is_file()
    ]
    if not candidatos:
        raise FileNotFoundError(
            f'CSV nao encontrado: outputs/{prefixo_pasta}*/{arquivo}. '
            'Execute primeiro a validacao correspondente.'
        )
    return max(candidatos, key=lambda p: p.stat().st_mtime_ns)


def html_plot(fig: go.Figure, destino: Path, titulo: str, nota: str) -> None:
    # Biblioteca Plotly integral no arquivo HTML. Funciona offline.
    div_id = 'grafico3d'
    grafico = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=div_id,
        config={'responsive': True, 'displaylogo': False, 'scrollZoom': True},
    )
    conteudo = f'''<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>{escape(titulo)}</title>
<style>
:root {{color-scheme:light}} body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#f5f7fb;color:#17243c;margin:0}}
main{{max-width:1350px;margin:20px auto;padding:0 14px}}
header{{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:12px}}
h1{{font-size:22px;margin:0 0 6px}}p{{color:#53637a;font-size:13px;margin:0}}
.controls{{display:flex;gap:10px;align-items:center}}button{{border:1px solid #c7d2e0;border-radius:9px;background:white;padding:9px 14px;font-size:14px;cursor:pointer}}
button:hover{{background:#e7edf6}}.painel{{background:white;box-shadow:0 2px 12px #17243c14;border-radius:14px;padding:9px}}
</style></head><body><main>
<header><div><h1>{escape(titulo)}</h1><p>{escape(nota)}</p></div>
<div class="controls"><button id="rodar" type="button">▶ Rotação automática</button>
<button id="resetar" type="button">↺ Reiniciar câmera</button></div></header>
<section class="painel">{grafico}</section>
<p style="padding:10px 4px">Arraste o gráfico para girar • Role para aproximar • Use o botão para rotação automática • Dados da simulação isolada, sem F10.</p>
</main>
<script>
(() => {{
  const gd = document.getElementById('{div_id}');
  const botao = document.getElementById('rodar');
  const resetar = document.getElementById('resetar');
  let ativo = false;
  let angulo = 0.6;
  let handle = null;
  const raio = 2.05;
  const altura = 0.88;
  function parar() {{
    ativo = false;
    botao.textContent = '▶ Rotação automática';
    if (handle !== null) {{clearInterval(handle);handle = null;}}
  }}
  function girar() {{
    angulo += 0.026;
    Plotly.relayout(gd, {{'scene.camera.eye': {{
      x: raio * Math.cos(angulo), y: raio * Math.sin(angulo), z: altura
    }}}});
  }}
  botao.addEventListener('click', () => {{
    if (ativo) {{parar();return;}}
    ativo = true;
    botao.textContent = '⏸ Pausar rotação';
    handle = setInterval(girar, 65);
  }});
  resetar.addEventListener('click', () => {{
    parar();
    angulo = 0.6;
    Plotly.relayout(gd, {{'scene.camera.eye': {{x: 1.7, y: 1.6, z: 1.0}}}});
  }});
  gd.addEventListener('pointerdown', parar);
  window.addEventListener('pagehide', parar);
}})();
</script>
</body></html>'''
    destino.write_text(conteudo, encoding='utf-8')


def base_layout(fig: go.Figure, titulo: str, col_x: str='PC1', col_y: str='PC2', col_z: str='PC3') -> None:
    fig.update_layout(
        title={'text': titulo, 'x': .5, 'font': {'size': 19}},
        font={'family': 'Arial, sans-serif', 'size': 12},
        height=770,
        margin={'l': 8, 'r': 8, 'b': 8, 't': 75},
        legend={'orientation': 'h', 'y': 1.05, 'x': .04, 'bgcolor': 'rgba(255,255,255,.75)'},
        scene={
            'xaxis': {'title': col_x, 'backgroundcolor': '#f2f6fa', 'showbackground': True},
            'yaxis': {'title': col_y, 'backgroundcolor': '#f2f6fa', 'showbackground': True},
            'zaxis': {'title': col_z, 'backgroundcolor': '#f2f6fa', 'showbackground': True},
            'aspectmode': 'data',
            'dragmode': 'orbit',
            'camera': {'eye': {'x': 1.7, 'y': 1.6, 'z': 1}},
        },
    )


def desenhar_classes(fig: go.Figure, df: pd.DataFrame, prefixo: str = '') -> None:
    for y, rotulo, cor, simbolo, tam, opacity in (
        (0, 'Legítimas', '#2374b8', 'circle', 3.7, .50),
        (1, 'Fraudes', '#e34445', 'diamond', 5.2, .90),
    ):
        grupo = df.loc[df['classe_real'].astype(int) == y]
        if grupo.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=grupo['PC1'], y=grupo['PC2'], z=grupo['PC3'],
            mode='markers', name=prefixo + rotulo,
            marker={'size': tam, 'color': cor, 'symbol': simbolo, 'opacity': opacity},
            customdata=grupo.index.astype(str),
            hovertemplate=(
                '<b>' + rotulo + '</b><br>'
                'PC1: %{x:.7f}<br>PC2: %{y:.7f}<br>PC3: %{z:.7f}'
                '<extra></extra>'
            ),
        ))


def projetar_pca3d(csv: Path) -> Path:
    df = pd.read_csv(csv)
    exigidas = {'PC1', 'PC2', 'PC3', 'classe_real'}
    if not exigidas.issubset(df.columns):
        raise ValueError(f'CSV de projecao invalido: faltam {exigidas - set(df.columns)}')
    # Reproduz somente os pontos marcados como exibidos no PNG, se a coluna existir.
    if 'exibido_na_imagem' in df.columns:
        flag = df['exibido_na_imagem']
        df = df.loc[flag.eq(True) | flag.astype(str).str.lower().eq('true')]
    fig = go.Figure()
    desenhar_classes(fig, df)
    variancia = []
    for i in (1, 2, 3):
        col = f'variancia_explicada_PC{i}'
        variancia.append(float(df[col].iloc[0]) if col in df.columns else None)
    labels = [f'PC{i} ({variancia[i-1]*100:.1f}%)' if variancia[i-1] is not None else f'PC{i}' for i in (1,2,3)]
    base_layout(fig, 'PROJEÇÃO PCA 3D — PC1 × PC2 × PC3', *labels)
    destino = csv.with_name('PCA12_projecao_3D_INTERATIVA.html')
    html_plot(fig, destino, 'PCA12 — Projeção 3D interativa', f'{len(df)} observações exibidas • Girar com o mouse ou automaticamente')
    return destino


def projetar_biplot(csv_scores: Path) -> Path:
    df = pd.read_csv(csv_scores)
    csv_corr = csv_scores.with_name('PCA_BIPLOT_3D_correlacoes.csv')
    if not csv_corr.is_file():
        raise FileNotFoundError(f'CSV de correlacoes ausente: {csv_corr}')
    corr = pd.read_csv(csv_corr)
    exigidas = {'variavel_preprocessada','corr_PC1','corr_PC2','corr_PC3'}
    if not exigidas.issubset(corr.columns):
        raise ValueError(f'CSV de correlacoes invalido: faltam {exigidas-set(corr.columns)}')
    fig = go.Figure()
    desenhar_classes(fig, df)
    # Correlacoes [-1,1] dimensionadas para sobrepor aos scores; isto e APENAS graficamente.
    dimensoes = np.max(np.abs(df[['PC1','PC2','PC3']].to_numpy()), axis=0)
    escala = 0.55 * float(np.min(dimensoes))
    corr = corr.copy()
    corr['norma'] = np.linalg.norm(corr[['corr_PC1','corr_PC2','corr_PC3']].to_numpy(), axis=1)
    corr = corr.sort_values('norma',ascending=False).reset_index(drop=True)
    for j, r in corr.iterrows():
        vx, vy, vz = [escala * float(r[c]) for c in ('corr_PC1','corr_PC2','corr_PC3')]
        top6 = j < 6
        fig.add_trace(go.Scatter3d(
            x=[0,vx], y=[0,vy], z=[0,vz],
            mode='lines+markers+text' if top6 else 'lines+markers',
            marker={'size': [1, 4] if top6 else [1, 3], 'color': '#7049bf'},
            line={'color': '#7049bf' if top6 else '#94a3b8', 'width': 6 if top6 else 3},
            text=[None,str(r['variavel_preprocessada'])] if top6 else None,
            textposition='top center',
            visible=True if top6 else False,
            showlegend=False,
            hovertemplate=(f"<b>{escape(str(r['variavel_preprocessada']))}</b><br>"
                           f"corr(PC1)={float(r['corr_PC1']):.7f}<br>"
                           f"corr(PC2)={float(r['corr_PC2']):.7f}<br>"
                           f"corr(PC3)={float(r['corr_PC3']):.7f}<extra></extra>"),
        ))
    total = len(fig.data)
    num_vars = len(corr)
    fig.update_layout(updatemenus=[{
        'type': 'buttons', 'direction': 'right', 'x': .98,'y': 1.13,
        'xanchor':'right','yanchor':'top','showactive':True,
        'buttons':[
            {'label':'6 vetores','method':'update','args':[{'visible':[True,True]+[i<6 for i in range(num_vars)]}]},
            {'label':'39 vetores (todos)','method':'update','args':[{'visible':[True]*total}]},
        ]
    }])
    base_layout(fig, 'BIPLOT PCA 3D — OBSERVAÇÕES E VETORES')
    destino = csv_scores.with_name('PCA_BIPLOT_3D_INTERATIVO.html')
    html_plot(fig, destino, 'Biplot PCA 3D interativo', f'{len(df)} observações • {len(corr)} variáveis • Exibir 6 ou todas as setas')
    return destino


def main():
    raiz = Path.cwd()
    if not (raiz / 'src').is_dir() or not (raiz / 'outputs').is_dir():
        raise SystemExit('Execute este script a partir da raiz do TCC_EXPERIMENTAL_V3.')
    resultados=[]
    origens = (
        ('validacao_pca3d_01_projecao_', '12_pca_projecao_PC1_PC2_PC3.csv', projetar_pca3d),
        ('validacao_pca3d_02_biplot_', 'PCA_BIPLOT_3D_scores_amostra.csv', projetar_biplot),
    )
    for prefixo, arquivo, funcao in origens:
        try:
            caminho = ultimo_csv(raiz, prefixo, arquivo)
        except FileNotFoundError as exc:
            print('NAO ENCONTRADO:', exc)
            continue
        destino = funcao(caminho)
        print('HTML GERADO:', destino.resolve())
        resultados.append(destino)
    if not resultados:
        raise SystemExit('Nenhum CSV de auditoria PCA3D encontrado; execute as validacoes 01/02.')
    print('PRIMEIRO_HTML=' + str(resultados[0].resolve()))
    print('YAML preservado, nenhuma modelagem executada, F10 nao utilizado.')


if __name__ == '__main__':
    main()
