# -*- coding: utf-8 -*-
"""PRE01 F10 PCA 1D — diagnóstico descritivo, espelho visual do PRE01 V2.

Executar da raiz TCC_EXPERIMENTAL_V3:
    python scripts/execucao/GERAR_PRE01_F10_PCA1D.py --verificar
    python scripts/execucao/GERAR_PRE01_F10_PCA1D.py --gerar

APENAS usa CSV/JSON/Npy já salvos e valida hashes. Nunca lê Parquet,
TARGET bruto ou modelos joblib. Não executa fit, predict ou métricas de fraude.
Não altera o PRE01 OOF, os 6+ gráficos F10 anteriores nem a galeria principal.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

ENSAIO_REL = Path('outputs/ensaio_pca_gmm_20261029_173123')
MODELO_REL = Path('outputs/treinamento_final_pca_10pct_20261029_174241/PCA_1D')
PRE01_ARQ = 'PRE01_F10_diagnostico_padronizacao_completo.html'
TABELA_ARQ = 'PRE01_F10_todas_variaveis.csv'
HIST_ARQ = 'PRE01_F10_PC1_histograma.csv'
MANIFESTO_ARQ = 'PRE01_F10_manifesto.json'


def exigir(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def ler_json(path: Path) -> dict:
    exigir(path.is_file(), f'Arquivo obrigatório ausente: {path}')
    obj = json.loads(path.read_text(encoding='utf-8-sig'))
    exigir(isinstance(obj, dict), f'JSON inválido: {path}')
    return obj


def sha256_arquivo(path: Path) -> str:
    exigir(path.is_file(), f'Arquivo obrigatório ausente: {path}')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def caminhos(root: Path) -> dict:
    ensaio = root / ENSAIO_REL
    teste = ensaio / 'vencedor/teste_final'
    geo = teste / 'dados_geometria'
    output = teste / 'diagnosticos/PRE01_F10'
    return {'raiz': root, 'ensaio': ensaio, 'teste': teste, 'geo': geo,
            'modelo': root / MODELO_REL, 'saida': output}


def validar_tabelas(resumo: pd.DataFrame, coords: pd.DataFrame,
                    pred_ids: pd.DataFrame, n: int) -> None:
    schema = [
        'feature', 'tipo', 'n_f10', 'media_treino_f1_f9_codificada',
        'escala_treino_f1_f9', 'media_z_f10', 'desvio_z_f10',
        'variancia_z_f10', 'drift_media_em_escala_treino',
        'percentual_ausentes_brutos_f10',
    ]
    exigir(list(resumo.columns) == schema, 'Schema do resumo F10 inesperado.')
    exigir(len(resumo) > 0 and resumo['feature'].notna().all() and
           resumo['feature'].is_unique, 'Nomes padronizados ausentes ou repetidos.')
    exigir(set(resumo['tipo'].unique()) <= {'FEWF', 'numerica'},
           'Tipo inesperado no diagnóstico de padronização.')
    exigir((resumo['n_f10'] == n).all(), 'N de F10 divergente no resumo.')
    numericos = schema[3:]
    for c in numericos:
        exigir(np.isfinite(resumo[c].to_numpy(dtype=float)).all(),
               f'Coluna não finita: {c}')
    exigir((resumo['escala_treino_f1_f9'] > 0).all() and
           (resumo['desvio_z_f10'] >= 0).all() and
           (resumo['variancia_z_f10'] >= 0).all(),
           'Escalas/variâncias inválidas.')
    exigir(np.allclose(resumo['desvio_z_f10'].to_numpy(dtype=float) ** 2,
                          resumo['variancia_z_f10'].to_numpy(dtype=float),
                          rtol=1e-7, atol=1e-10), 'Desvio e variância F10 inconsistentes.')
    exigir(np.allclose(resumo['media_z_f10'],
                          resumo['drift_media_em_escala_treino'],
                          atol=1e-10, rtol=1e-9), 'Drift/média z F10 inconsistentes.')
    exigir(resumo['percentual_ausentes_brutos_f10'].between(0, 100).all(),
           'Percentual de ausentes fora de 0–100.')
    exigir(list(coords.columns) == ['linha_original', 'PC1'], 'Schema PC1 inesperado.')
    exigir(list(pred_ids.columns) == ['linha_original'], 'Schema de IDs inesperado.')
    exigir(len(coords) == len(pred_ids) == n, 'Quantidade de linhas F10 divergente.')
    ids = coords['linha_original'].to_numpy(dtype=np.int64)
    exigir(np.all(ids >= 0) and np.all(np.diff(ids) > 0),
           'IDs PC1 duplicados, negativos ou desordenados.')
    exigir(np.array_equal(ids, pred_ids['linha_original'].to_numpy(dtype=np.int64)),
           'Alinhamento dos mesmos registros F10 falhou.')
    exigir(np.isfinite(coords['PC1'].to_numpy(dtype=float)).all(),
           'PC1 inválido.')


def enriquecer(resumo: pd.DataFrame) -> pd.DataFrame:
    """Reconstrói apenas estatísticas DESCRITIVAS no espaço codificado/imputado.

    Se z=(x-mu_treino)/escala_treino, então mu_f10=mu_treino+E[z]*escala_treino
    e var_f10=Var(z)*escala_treino². Nenhum scaler é ajustado no F10.
    """
    df = resumo.copy(deep=True)
    escala = df['escala_treino_f1_f9'].to_numpy(dtype=float)
    varz = df['variancia_z_f10'].to_numpy(dtype=float)
    zmean = df['media_z_f10'].to_numpy(dtype=float)
    zstd = df['desvio_z_f10'].to_numpy(dtype=float)
    df['media_f10_codificada_imputada'] = (
        df['media_treino_f1_f9_codificada'].to_numpy(dtype=float) + escala*zmean)
    df['variancia_f10_codificada_imputada'] = varz * escala ** 2
    df['desvio_f10_codificado_imputado'] = zstd * escala
    df['razao_dispersao_f10_treino'] = zstd
    # log definido somente se a variabilidade no F10 for positiva.
    df['drift_log_dispersao'] = np.where(zstd > 0, np.log(np.maximum(zstd, 1e-300)), np.nan)
    df['drift_media_padronizada'] = zmean
    return df


def histograma_pc1(coords: pd.DataFrame, bins: int = 60) -> pd.DataFrame:
    x = coords['PC1'].to_numpy(dtype=np.float64)
    exigir(x.ndim == 1 and len(x)>0 and np.isfinite(x).all(), 'PC1 inválido.')
    exigir(2 <= bins <= 250, 'Bins inválidos.')
    if np.ptp(x) == 0:
        edges = np.linspace(x[0] - .5, x[0] + .5, bins + 1)
    else:
        edges = np.linspace(float(x.min()), float(x.max()), bins + 1)
    counts, edges = np.histogram(x, bins=edges)
    exigir(int(counts.sum()) == len(x), 'Histograma PC1 não conserva n.')
    return pd.DataFrame({'pc1_inicio': edges[:-1], 'pc1_fim': edges[1:],
                         'pc1_centro': (edges[1:] + edges[:-1])/2,
                         'contagem_f10': counts})


def conferir(root: Path = RAIZ, *, exige_destino_livre: bool = False) -> dict:
    """Verifica dados já extraídos; não lê Parquet, TARGET bruto ou joblib."""
    paths = caminhos(Path(root).resolve())
    geomet = ler_json(paths['geo'] / 'manifesto_geometria_f10.json')
    mod = ler_json(paths['modelo'] / 'manifesto_modelo_final.json')
    escolha = ler_json(paths['ensaio'] / 'vencedor/metricas/melhor_configuracao.json')
    espelho = ler_json(paths['teste'] / 'manifesto_espelho_graficos_f10.json')
    metricas = ler_json(paths['teste'] / 'metricas_f10.json')
    exig = exigir
    exig(geomet.get('dimensao') == mod.get('dimensao_pca') == 1,
         'Referência não é PCA 1D.')
    exig(geomet.get('tipo') == 'EXTRACAO_DESCRITIVA_F10_JA_AVALIADO',
         'Origem da geometria não reconhecida.')
    exig(all(geomet.get(k) is False for k in ('parquet_target_lido', 'novo_treinamento',
                                             'nova_inferencia_probabilidades', 'nova_avaliacao_f10')),
         'Geometria não respeita o isolamento F10.')
    n = int(geomet['n'])
    exig(n > 0 and n == int(espelho['n']) == int(metricas['n'])
         and mod['n_treino_original'] == 166716,
         'Tamanho do desenvolvimento/F10 incompatível.')
    tau = float(escolha['limiar_mcc_oof'])
    exig(np.isfinite(tau) and tau == float(geomet['limiar_mcc_oof_congelado'])
         == float(mod['limiar_mcc_oof_congelado'])
         == float(espelho['limiar_mcc_congelado_oof']), 'Limiar OOF mudou.')
    exig(escolha['id_candidato'] == mod['id_candidato_oof'],
         'ID do candidato/modelo divergente.')
    exig(sha256_arquivo(paths['modelo'] / 'modelo_final_f1_f9.joblib') ==
         mod['sha256_modelo_final'] == geomet['sha256_modelo_final'],
         'SHA-256 do modelo final divergente.')
    for f in ('coordenadas_PC1_f10.csv', 'resumo_padronizacao_f10.csv'):
        exig(sha256_arquivo(paths['geo'] / f) == geomet['arquivos'][f],
             f'Arquivo geométrico F10 alterado: {f}')
    exig(sha256_arquivo(paths['teste'] / 'predicoes_f10.csv') ==
         espelho['predicoes_sha256'], 'Predições F10 divergentes do manifesto.')
    exig(sha256_arquivo(paths['teste'] / 'metricas_f10.json') ==
         espelho['metricas_sha256'], 'Métricas F10 divergentes do manifesto.')
    resumo = pd.read_csv(paths['geo'] / 'resumo_padronizacao_f10.csv')
    coords = pd.read_csv(paths['geo'] / 'coordenadas_PC1_f10.csv')
    pred_ids = pd.read_csv(paths['teste'] / 'predicoes_f10.csv', usecols=['linha_original'])
    validar_tabelas(resumo, coords, pred_ids, n)
    exig(len(resumo) > 0, 'Nenhuma variável padronizada.')
    if exige_destino_livre:
        exig(not paths['saida'].exists(),
             'PRE01 F10 já existe. Pasta preservada, geração bloqueada.')
    print('=== PRE01 PCA 1D — F10 DESCRITIVO ===', flush=True)
    print(f'[APROVADO] {n} linhas F10, {len(resumo)} variáveis padronizadas, PC1 e IDs alinhados.', flush=True)
    print(f'[APROVADO] Hash do GMM+PCA e tau*={tau:.17g} permanecem congelados.', flush=True)
    print('[APROVADO] Sem Parquet, desserialização joblib, TARGET bruto, novos ajustes ou avaliação F10.', flush=True)
    return dict(paths=paths, geometria_meta=geomet, modelo_meta=mod,
                escolha=escolha, espelho_meta=espelho,
                resumo=resumo, coords=coords, n=n, tau=tau)


def _json_limpo(obj):
    def conv(v):
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, list):
            return [conv(x) for x in v]
        if isinstance(v, (np.integer, np.floating)):
            return conv(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    return conv(obj)


# Cores, tipografia e layout principais do PRE01 V2 fornecido pelo autor.
CSS = r'''
:root{--ink:#20324a;--blue:#285b93;--edge:#dce4ee;--muted:#60738d;--bg:#f4f7fb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);font:14px/1.5 system-ui,Arial,sans-serif;color:var(--ink)}
main{max-width:1670px;margin:auto;padding:24px 22px 55px}h1{font-size:28px;margin:0 0 5px}h2{font-size:19px;margin:0 0 12px}h3{font-size:16px;margin:0 0 8px}
p{margin:8px 0}small,.muted{color:var(--muted)}.lead{font-size:15px;color:#536a85;max-width:1400px}
.banner{background:#eaf0f7;border-left:4px solid var(--blue);padding:12px 15px;margin:18px 0;border-radius:5px}
.card,.section{background:white;border:1px solid var(--edge);border-radius:12px;padding:18px;box-shadow:0 2px 9px #20324a08}
.section{margin-top:20px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px;margin:19px 0}
.card .n{font-size:24px;font-weight:750;color:#204a78;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.card .k{font-size:12px;color:#53647b}
.cards .card{padding:13px 15px}.nav{display:flex;flex-wrap:wrap;gap:9px;margin:14px 0}
.nav a,.download a{border:1px solid #c6d6e9;border-radius:7px;padding:7px 11px;text-decoration:none;color:#245386;background:#fff}
.nav a:hover,.download a:hover{background:#eff6ff}.plot{width:100%;min-height:360px}
.twocol{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:12px}.scroll{overflow-x:auto;border:1px solid var(--edge);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{padding:8px 10px;border-bottom:1px solid #e7edf5;text-align:right;font-variant-numeric:tabular-nums}
th{position:sticky;top:0;background:#edf3fa;color:#274765;cursor:pointer;user-select:none}th:first-child,td:first-child{text-align:left}tbody tr:nth-child(even){background:#f8fafc}tbody tr:hover{background:#eef5ff}
.controls{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0;align-items:center}.controls input,.controls select{border:1px solid #bfcde1;border-radius:6px;padding:7px 9px;font:inherit;min-width:135px}
.hint{font-size:12px;color:#596d84;margin-top:9px}.download{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}
.badge{font-size:12px;color:#385977;background:#edf3fa;border:1px solid #d4e0ee;padding:4px 8px;border-radius:5px}
@media(max-width:760px){main{padding:15px 10px}.twocol{grid-template-columns:1fr}h1{font-size:24px}}
'''


def html_pre01(df: pd.DataFrame, hist: pd.DataFrame, pc1: np.ndarray,
               meta: dict, *, link_oof: str | None = None) -> str:
    """Gera página autônoma e offline, visual PRE01 V2 com informação F10 correta."""
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as ex:
        raise RuntimeError('Plotly não instalado na venv. Execute: python -m pip install plotly') from ex
    exigir(len(df) > 0 and len(hist) > 0, 'Sem dados para HTML.')
    payload = {
        'rows': df.to_dict(orient='records'),
        'hist': hist.to_dict(orient='records'),
        'pc1': {
            'media': float(pc1.mean()), 'desvio': float(pc1.std(ddof=0)),
            'minimo': float(pc1.min()), 'maximo': float(pc1.max()),
            'p01': float(np.quantile(pc1, .01)), 'p50': float(np.median(pc1)),
            'p99': float(np.quantile(pc1, .99)),
        },
        'meta': meta,
    }
    serial = json.dumps(_json_limpo(payload), ensure_ascii=False,
                        allow_nan=False, separators=(',', ':')).replace('</', '<\\/')
    plotly = get_plotlyjs().replace('</', '<\\/')
    ref = (f'<a href="{html.escape(link_oof, quote=True)}" target="_blank" '
           f'rel="noopener">Abrir PRE01 original dos folds F1–F9</a>' if link_oof
           else 'Referência OOF interativa não encontrada; os dados F10 permanecem disponíveis.')
    layout = r'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRE01 F10 — Diagnóstico completo da padronização</title><style>__CSS__</style></head>
<body><main><h1>PRE01 — Diagnóstico completo da padronização (F10)</h1>
<p class="lead">Espelho visual do PRE01 V2: avaliação descritiva do pré-processamento da amostra F10 já examinada, usando somente o encoder, imputador, StandardScaler e PCA congelados em F1–F9. Mostra todos os critérios padronizados sem refazer nenhum ajuste.</p>
<div class="banner"><strong>F10 amostral exploratório, já avaliado.</strong> Não existe novo treinamento, novo limiar ou nova avaliação preditiva neste painel. Os parâmetros de referência pertencem ao ajuste F1–F9. FEWF pode refletir variação das frequências na população de teste. Em F10 não há “ajustes 8F”; suas estatísticas são observadas após aplicar o mesmo scaler treinado. Os limites de drift são apenas descritivos.</div>
<p id="origem" class="muted"></p>
<nav class="nav"><a href="#resumo">Indicadores</a><a href="#variaveis">Todas as variáveis</a><a href="#folds">Referência OOF</a><a href="#pos">Após padronização</a><a href="#pca">PCA diagnóstico</a><a href="#dados">Todos os valores</a><a href="#metodo">Método</a></nav>
<section class="section" id="resumo"><h2>1. Indicadores gerais</h2><div class="cards" id="cards"></div><p class="hint">Todos os indicadores são descritivos de features. Não são AP, MCC ou métricas de fraude.</p></section>
<section class="section" id="variaveis"><h2>2. Parâmetros de todas as variáveis</h2>
<p>μ e σ de referência provêm exclusivamente do encoder, imputação e StandardScaler de F1–F9. μ e σ F10 são reconstruídos dos z-scores já salvos, no espaço codificado e imputado, sem reajustar o scaler. A escala F10 é <em>apenas uma estatística descritiva</em>.</p>
<div class="twocol"><div class="card"><h3>Drift de média F10 × F1–F9</h3><div id="plot-mu" class="plot"></div></div><div class="card"><h3>Drift de dispersão F10 × F1–F9</h3><div id="plot-esc" class="plot"></div></div></div>
<div class="controls"><label>Buscar variável <input id="search-summary" placeholder="Nome da feature"></label><label>Tipo <select id="tipo-summary"><option value="">Todos</option><option value="numerica">Numérica</option><option value="FEWF">FEWF</option></select></label><span class="badge" id="count-summary"></span></div>
<div class="scroll"><table id="table-summary"></table></div></section>
<section class="section" id="folds"><h2>3. Referência de estabilidade entre os nove folds</h2>
<p>O estudo original 8F × 9F pertence à validação OOF e <strong>não pode ser repetido no F10</strong>. Os parâmetros do treinamento permanecem fixos. Use o painel OOF original para consultar a estabilidade histórica.</p>
<div class="download">__OOF_LINK__</div></section>
<section class="section" id="pos"><h2>4. Diagnóstico após padronização no F10</h2>
<p>O scaler F1–F9 é aplicado sem ajuste. Os z-scores F10 podem ter média diferente de zero e desvio diferente de um; isso revela diferenças entre as distribuições.</p>
<div class="twocol"><div class="card"><h3>|Média z| por variável (F10)</h3><div id="plot-zmu" class="plot"></div></div><div class="card"><h3>|Desvio z − 1| por variável (F10)</h3><div id="plot-zsd" class="plot"></div></div></div></section>
<section class="section" id="pca"><h2>5. PCA diagnóstico — PC1 no F10</h2>
<p>Distribuição PC1 das mesmas observações F10, projetadas por meio do PCA 1D congelado. Não foi feito novo ajuste PCA. Não se apresentam PC2/PC3, pois este modelo é PCA 1D.</p>
<div class="cards" id="pca-cards"></div><div id="plot-pca" class="plot"></div></section>
<section class="section" id="dados"><h2>6. Tabela completa: todas as variáveis padronizadas</h2>
<p>Todas as variáveis disponíveis no resumo preservado do F10, sem seleção Top 10. Valores originais e calculados são explicitamente diferenciados no manifesto.</p>
<div class="controls"><label>Buscar variável <input id="search-details" placeholder="Nome da feature"></label><label>Tipo <select id="tipo-details"><option value="">Todos</option><option value="numerica">Numérica</option><option value="FEWF">FEWF</option></select></label><span class="badge" id="count-details"></span></div><div class="scroll"><table id="table-details"></table></div></section>
<section class="section" id="metodo"><h2>7. Fórmulas, limitações e rastreabilidade</h2>
<p><strong>Média F10 no espaço codificado/imputado:</strong> μ<sub>F10</sub> = μ<sub>F1–F9</sub> + escala<sub>F1–F9</sub> × média(z<sub>F10</sub>).</p>
<p><strong>Variância F10 no mesmo espaço:</strong> Var<sub>F10</sub> = escala<sub>F1–F9</sub><sup>2</sup> × Var(z<sub>F10</sub>).</p>
<p><strong>Drift padronizado de média:</strong> média(z<sub>F10</sub>); <strong>razão de dispersões:</strong> σ(z<sub>F10</sub>).</p>
<p><strong>Drift log de dispersão:</strong> ln(σ(z<sub>F10</sub>)), indefinido se σ=0. Não equivale ao drift entre scalers reestimados.</p>
<p><strong>Escopo:</strong> inclui somente variáveis padronizadas pelo StandardScaler original (numéricas e FEWF). Variáveis binárias, OHE ou cíclicas não padronizadas ficam fora desta análise de escala, como no PRE01 original. A amostra F10 é exploratória de 10% e não foi reavaliada.</p>
<p><strong>Proveniência:</strong> hashes dos modelos e arquivos F10 no manifesto. As coordenadas PC1 e os z-scores resumidos vieram da extração anterior com transformadores congelados.</p>
<div class="download"><a href="__CSV__">CSV — todas as variáveis</a><a href="__HIST__">CSV — histograma PC1</a><a href="__META__">Manifesto SHA-256</a></div>
</section></main><script>__PLOTLY__</script><script>const PRE01_F10 = __DATA__;</script><script>__JS__</script></body></html>'''
    javascript = r'''(() => {
'use strict'; const D=PRE01_F10; const rows=D.rows;
const $=id=>document.getElementById(id);
const num=x=>typeof x==='number'&&Number.isFinite(x);
const fmt=x=>x===null||x===undefined||(typeof x==='number'&&!Number.isFinite(x))?'—':typeof x==='number'?x.toFixed(7):String(x);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const mean=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:0;
$('origem').textContent=`PCA 1D | F10 amostral | ${D.meta.n_f10.toLocaleString('pt-BR')} observações | ${rows.length} variáveis padronizadas | τ*=${D.meta.tau_star_oof.toFixed(7)} (OOF, sem novo ajuste)`;
const cards=[
['Variáveis padronizadas',rows.length,'Numéricas e FEWF'],
['Observações F10',D.meta.n_f10.toLocaleString('pt-BR'),'Amostra já avaliada'],
['Drift médio |Δμ|',fmt(mean(rows.map(r=>Math.abs(r.drift_media_padronizada)))),'F10 vs parâmetros congelados F1–F9'],
['Drift máximo |Δμ|',fmt(Math.max(...rows.map(r=>Math.abs(r.drift_media_padronizada)))),'Maior mudança padronizada'],
['Razão média σF10/escala treino',fmt(mean(rows.map(r=>r.razao_dispersao_f10_treino))),'Apenas descrição de dispersão'],
['|Média z| ≤ 0,10',fmt(100*rows.filter(r=>Math.abs(r.media_z_f10)<=.1).length/rows.length)+'%','Proporção das features'],
['σz entre 0,90 e 1,10',fmt(100*rows.filter(r=>r.desvio_z_f10>=.9&&r.desvio_z_f10<=1.1).length/rows.length)+'%','Proporção das features'],
['Ausência bruta média (%)',fmt(mean(rows.map(r=>r.percentual_ausentes_brutos_f10))),'Por variável']];
$('cards').innerHTML=cards.map(([k,v,n])=>`<div class="card"><div class="k">${esc(k)}</div><div class="n">${esc(v)}</div><div class="hint">${esc(n)}</div></div>`).join('');
function card(k,v,n){return `<div class="card"><div class="k">${esc(k)}</div><div class="n">${esc(fmt(v))}</div><div class="hint">${esc(n)}</div></div>`}
$('pca-cards').innerHTML=card('Média PC1',D.pc1.media,'F10 no PCA congelado')+card('Desvio PC1',D.pc1.desvio,'Populacional, ddof=0')+card('Mediana PC1',D.pc1.p50,'F10')+card('Intervalo 1%–99%',`${fmt(D.pc1.p01)} a ${fmt(D.pc1.p99)}`,'Percentis F10');
const cols=[['feature','Variável'],['tipo','Tipo'],['n_f10','N F10'],['media_treino_f1_f9_codificada','μ treino'],['escala_treino_f1_f9','Escala treino'],['media_f10_codificada_imputada','μ F10'],['variancia_f10_codificada_imputada','Var F10'],['desvio_f10_codificado_imputado','σ F10'],['media_z_f10','μz F10'],['desvio_z_f10','σz F10'],['variancia_z_f10','Var z F10'],['drift_media_padronizada','Drift μ'],['razao_dispersao_f10_treino','Razão σ'],['drift_log_dispersao','ln razão σ'],['percentual_ausentes_brutos_f10','% ausentes F10']];
function table(id, data, input, select, count){let key='feature',rev=false; const el=$(id);
 function render(){const s=$(input).value.toLocaleLowerCase('pt-BR'),tp=$(select).value;let a=data.filter(r=>r.feature.toLocaleLowerCase('pt-BR').includes(s)&&(!tp||r.tipo===tp));a.sort((x,y)=>{const v=x[key],w=y[key];return ((num(v)&&num(w))?v-w:String(v).localeCompare(String(w),'pt-BR'))*(rev?-1:1)});
 el.innerHTML='<thead><tr>'+cols.map(([k,l])=>`<th data-key="${k}">${esc(l)}${k===key?(rev?' ▼':' ▲'):''}</th>`).join('')+'</tr></thead><tbody>'+a.map(r=>'<tr>'+cols.map(([k])=>`<td>${esc(fmt(r[k]))}</td>`).join('')+'</tr>').join('')+'</tbody>';
 el.querySelectorAll('th').forEach(t=>t.onclick=()=>{let k=t.dataset.key;rev=(key===k?!rev:false);key=k;render()});$(count).textContent=`${a.length} de ${data.length} linhas`;}
 $(input).oninput=render;$(select).onchange=render;render();}
table('table-summary',rows,'search-summary','tipo-summary','count-summary');table('table-details',rows,'search-details','tipo-details','count-details');
const sorted=[...rows].sort((a,b)=>a.feature.localeCompare(b.feature,'pt-BR'));const names=sorted.map(r=>r.feature);
const opt={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}};
const layout=(t)=>({paper_bgcolor:'#fff',plot_bgcolor:'#fff',margin:{l:190,r:24,t:28,b:48},font:{family:'Arial,sans-serif',size:11,color:'#344b65'},xaxis:{title:t,gridcolor:'#e5ebf2',zeroline:true,zerolinecolor:'#9cacc0'},yaxis:{autorange:'reversed',automargin:true},height:Math.max(400,names.length*28+110)});
function bars(id,key,title,color,abs=false){Plotly.newPlot(id,[{type:'bar',orientation:'h',x:sorted.map(r=>abs?Math.abs(r[key]):r[key]),y:names,marker:{color},hovertemplate:'%{y}<br>%{x:.7f}<extra></extra>'}],layout(title),opt)}
bars('plot-mu','drift_media_padronizada','|(μF10−μtreino)/escala treino|','#3975a7',true);
bars('plot-esc','razao_dispersao_f10_treino','σF10 / escala treino','#d27a3b');
bars('plot-zmu','media_z_f10','|média z F10|','#438b7d',true);
Plotly.newPlot('plot-zsd',[{type:'bar',orientation:'h',x:sorted.map(r=>Math.abs(r.desvio_z_f10-1)),y:names,marker:{color:'#9871ba'},hovertemplate:'%{y}<br>%{x:.7f}<extra></extra>'}],layout('|σz F10−1|'),opt);
Plotly.newPlot('plot-pca',[{type:'bar',x:D.hist.map(r=>r.pc1_centro),y:D.hist.map(r=>r.contagem_f10),marker:{color:'#3b6b9e'},hovertemplate:'PC1=%{x:.7f}<br>Contagem=%{y}<extra></extra>'}],{paper_bgcolor:'#fff',plot_bgcolor:'#fff',height:390,margin:{l:65,r:25,t:15,b:55},font:{family:'Arial,sans-serif',color:'#344b65'},xaxis:{title:'PC1 • PCA congelado F1–F9'},yaxis:{title:'Frequência de observações F10'}},opt);
})();'''
    return (layout.replace('__CSS__', CSS)
                  .replace('__OOF_LINK__', ref)
                  .replace('__CSV__', TABELA_ARQ)
                  .replace('__HIST__', HIST_ARQ)
                  .replace('__META__', MANIFESTO_ARQ)
                  .replace('__PLOTLY__', plotly)
                  .replace('__DATA__', serial)
                  .replace('__JS__', javascript))


def preparar(estado: dict):
    tabela = enriquecer(estado['resumo'])
    hist = histograma_pc1(estado['coords'])
    meta = {'versao': 'PRE01_F10_PCA1D_V1', 'tipo': 'DIAGNOSTICO_DESCRITIVO_SEM_AJUSTES',
            'n_f10': estado['n'], 'n_treino_f1_f9': 166716,
            'pca_dimensao': 1, 'variaveis_padronizadas': int(len(tabela)),
            'tau_star_oof': estado['tau'],
            'carater': 'AMOSTRAL_EXPLORATORIO',
            'parquet_lido': False, 'modelo_joblib_desserializado': False,
            'treinamento_realizado': False, 'previsao_realizada': False,
            'avaliacao_preditiva_realizada': False,
            'novo_limiar_otimizado': False,
            'metodologia': 'Reconstrução descritiva a partir das médias e variâncias z F10 previamente salvas; transformadores F1–F9 congelados.'}
    return tabela, hist, meta


def gerar(estado: dict) -> Path:
    paths = estado['paths']
    saida = paths['saida']
    exigir(not saida.exists(), f'Destino existente, nenhuma sobrescrita: {saida}')
    tabela, hist, meta = preparar(estado)
    # Link relativo apenas quando o PRE01 OOF já está efetivamente disponível.
    oof = (paths['ensaio'] / 'vencedor/visualizacao/galeria/oof/PRE01_diagnostico_padronizacao_completo.html')
    link_oof = os.path.relpath(oof, saida).replace('\\','/') if oof.is_file() else None
    teste = paths['teste']
    teste.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.pre01_f10_pca1d_parcial_', dir=teste))
    try:
        arq_tab = temp / TABELA_ARQ
        arq_hist = temp / HIST_ARQ
        tabela.to_csv(arq_tab, index=False, float_format='%.17g', encoding='utf-8-sig')
        hist.to_csv(arq_hist, index=False, float_format='%.17g', encoding='utf-8-sig')
        fonte = {
            'resumo_padronizacao_f10.csv': sha256_arquivo(paths['geo']/'resumo_padronizacao_f10.csv'),
            'coordenadas_PC1_f10.csv': sha256_arquivo(paths['geo']/'coordenadas_PC1_f10.csv'),
            'predicoes_f10.csv': sha256_arquivo(paths['teste']/'predicoes_f10.csv'),
            'metricas_f10.json': sha256_arquivo(paths['teste']/'metricas_f10.json'),
            'modelo_final_f1_f9.joblib': estado['modelo_meta']['sha256_modelo_final'],
        }
        manifest = dict(meta, origem_experimento=ENSAIO_REL.as_posix(),
                        origem_f10=estado['geometria_meta']['original_f10'],
                        fontes_sha256=fonte,
                        conteudos_derivados={'tabela_variaveis':TABELA_ARQ,
                                            'histograma_pc1':HIST_ARQ},
                        arquivos_sha256={TABELA_ARQ:sha256_arquivo(arq_tab),
                                         HIST_ARQ:sha256_arquivo(arq_hist)},
                        observacao='PC1 histórico do desenvolvimento não é reestimado nem copiado para a amostra F10.')
        arq_man = temp / MANIFESTO_ARQ
        arq_man.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        arq_html = temp / PRE01_ARQ
        arq_html.write_text(html_pre01(tabela, hist,
                                      estado['coords']['PC1'].to_numpy(dtype=float),
                                      meta, link_oof=link_oof), encoding='utf-8')
        exigir(arq_html.stat().st_size > 0 and all(f.is_file() for f in
               (arq_tab,arq_hist,arq_man)), 'Saída incompleta.')
        # Não usar shutil.move nem remoção implícita: destino deve ser inédito.
        saida.parent.mkdir(parents=True, exist_ok=True)
        exigir(not saida.exists(), 'Destino passou a existir, nada publicado.')
        os.replace(temp, saida)
        print('[CONCLUIDO] PRE01 F10 completo, HTML interativo e 2 CSVs; sem alterações OOF.', flush=True)
        print('[HTML]', saida / PRE01_ARQ, flush=True)
        print('[MANIFESTO]', saida / MANIFESTO_ARQ, flush=True)
        return saida / PRE01_ARQ
    except Exception:
        print('[AVISO] Pasta temporária preservada para diagnóstico:', temp, flush=True)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    modos=p.add_mutually_exclusive_group(required=True)
    modos.add_argument('--verificar', action='store_true')
    modos.add_argument('--gerar', action='store_true')
    args=p.parse_args()
    estado = conferir(exige_destino_livre=args.gerar)
    if args.gerar:
        gerar(estado)
    else:
        print('[PREFLIGHT APROVADO] Nada gravado. Use --gerar após os testes.', flush=True)


if __name__ == '__main__':
    main()
