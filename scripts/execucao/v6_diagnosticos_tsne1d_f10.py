# -*- coding: utf-8 -*-
"""V6: TSNE13, TSNE14 e PRE01 do t-SNE 1D F10 exploratorio.

--planejar: somente leitura de V1..V5; sem ler dados F10 brutos nem carregar modelo.
--executar: usa OOS TSNE1/scores V3, GMM V2 e transformadores congelados.
PRE01 reabre SOMENTE os mesmos 18523 indices F10 para descrever o pre-processamento,
sem fit(), sem TSNE transform() e sem gerar novas predicoes ou otimizar limiar.

Saida transacional nova; se falhar, preservar pasta parcial e parar.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

from scripts.execucao import v5_geometria_f10_tsne1d as v5
from src.treinamento_final_tsne import carregar_pacote_confiavel

ROOT = Path(__file__).resolve().parents[2]
DIR_NAME = 'diagnosticos_tsne_v6'
LOCK = ROOT / 'outputs' / '.diagnosticos_tsne_f10_v6_em_andamento.lock'
FILES = (
    '13_tsne_mapa_TSNE1.png',
    '13_tsne_mapa_TSNE1.csv',
    '14_tsne_diagnostico.png',
    '14_tsne_diagnostico.csv',
    'PRE01_F10/PRE01_F10_diagnostico_padronizacao_completo.html',
    'PRE01_F10/PRE01_F10_todas_variaveis.csv',
    'PRE01_F10/PRE01_F10_TSNE1_histograma.csv',
)


def exigir(cond, mensagem):
    if not cond:
        raise RuntimeError(mensagem)


def sha(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def ler_json(path):
    with Path(path).open('r', encoding='utf-8-sig') as f:
        res = json.load(f)
    exigir(isinstance(res, dict), 'JSON precisa ser objeto: ' + str(path))
    return res


def arquivos_entrada(st):
    dirs = st['info']['modelo_dir']
    return [
        dirs / 'modelo_final_f1_f9.joblib',
        dirs / 'manifesto_modelo_final.json',
        *[st['v3'] / p for p in ('predicoes_f10.csv', 'coordenadas_TSNE1_f10.csv',
                                 'indices_f10_amostra.npy', 'metricas_f10.json', 'manifesto_v3.json')],
        st['v4'] / 'manifesto_v4.json',
        st['out'] / 'manifesto_v5.json',
    ]


def preflight():
    st = v5.preflight()
    v5.verificar_saida(st)
    dest = st['out'].parent / DIR_NAME
    res = {'st': st, 'out': dest, 'stage': dest.parent / ('.' + DIR_NAME + '_incompleta')}
    for f in arquivos_entrada(st):
        exigir(f.is_file(), 'Fonte V2-V5 ausente: ' + str(f))
    exigir(st['q']['v3_meta']['n_f10'] == 18523 and
           st['escolha']['configuracao']['reducao']['n_components'] == 1,
           'V6 exige o mesmo F10 pareado TSNE 1D da V3.')
    exigir(st['q']['v3_meta']['nao_reotimizou_limiar_f10'] is True,
           'Limiar V3 deve estar congelado.')
    return res


def imprimir_plano(info):
    st = info['st']
    print('=== V6 - DIAGNOSTICOS TSNE 1D / F10 ===', flush=True)
    print('EXPERIMENTO_TSNE=' + str(st['info']['info']['pasta']), flush=True)
    print('FONTE_V3=' + str(st['v3']), flush=True)
    print('FONTE_V4=' + str(st['v4']), flush=True)
    print('FONTE_V5=' + str(st['out']), flush=True)
    print('CANDIDATO_OOF=' + st['escolha']['id_candidato'], flush=True)
    print('TAU_OOF_CONGELADO=' + format(st['escolha']['limiar_mcc_oof'], '.17g'), flush=True)
    print('N_F10_PAREADO=' + str(st['q']['v3_meta']['n_f10']), flush=True)
    print('DESTINO_V6=' + str(info['out']), flush=True)
    print('GRAFICOS_PLANEJADOS=13_tsne_mapa_TSNE1.png,14_tsne_diagnostico.png', flush=True)
    print('PRE01_INTERATIVO=SIM', flush=True)
    print('TEMPO_OOS_V3=NAO_REGISTRADO (nao estimar)', flush=True)
    print('F10_DADOS_BRUTOS_LIDOS_NO_PLANEJAMENTO=NAO', flush=True)
    print('OOF_REEXECUTADO=NAO', flush=True)
    print('NOVAS_PREDICOES_F10=NAO', flush=True)
    print('LIMIAR_REOTIMIZADO_F10=NAO', flush=True)


def conferir_saida(info):
    st = info['st']
    dest = info['out']
    exigir((dest / 'manifesto_v6.json').is_file(), 'V6 existe mas manifesto ausente: preservar.')
    meta = ler_json(dest / 'manifesto_v6.json')
    exigir(meta.get('status') == 'APROVADA' and meta.get('n_f10') == 18523 and
           meta.get('id_candidato_oof') == st['escolha']['id_candidato'] and
           meta.get('limiar_mcc_oof_congelado') == st['escolha']['limiar_mcc_oof'] and
           meta.get('sha256_modelo_v2') == st['info']['modelo_hash'],
           'Identidade/parametros da V6 anterior divergentes.')
    arquivos = meta.get('sha256_arquivos', {})
    exigir(isinstance(arquivos, dict) and set(arquivos) == set(FILES),
           'Lista de arquivos V6 incorreta.')
    for name, digest in arquivos.items():
        file = dest / name
        exigir(file.is_file() and sha(file) == digest, f'Arquivo V6 ausente/alterado: {name}')
    sources = meta.get('sha256_entradas', {})
    exigir(sources == {str(x.relative_to(ROOT)): sha(x) for x in arquivos_entrada(st)},
           'Fontes V2-V5 foram alteradas desde a V6.')
    return meta


def dados_v3(info):
    st = info['st']
    z, y, p, metricas = v5.carregar_artefatos(st)
    require_ids = st['info']['ids']
    coords = pd.read_csv(st['v3'] / 'coordenadas_TSNE1_f10.csv')
    exigir(np.array_equal(coords['linha_original'].to_numpy(dtype=np.int64), require_ids),
           'IDs TSNE1 F10 divergentes.')
    return z['TSNE1'].to_numpy(dtype=np.float64), y, p, metricas, require_ids


def desenhar_tsne13(dest, coords, y, scores, ids, gamma, cfg, tau):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from matplotlib.ticker import MaxNLocator
    opc = cfg['visualizacao']['graficos']['tsne_mapa']
    bins = int(opc.get('bins_histograma_1d', 36))
    exigir(5 <= bins <= 200, 'Numero de bins t-SNE13 invalido.')
    lo, hi = float(coords.min()), float(coords.max())
    if hi == lo:
        lo, hi = lo - .5, hi + .5
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    fig, ax = plt.subplots(figsize=(11.8, 6.9))
    styles = cfg['visualizacao']['classes']
    for target, name, color in ((0, 'Legítimas', styles['legitima'].get('cor','#478acc')),
                                 (1, 'Fraudes', styles['fraude'].get('cor','#ef8734'))):
        sel = coords[y == target]
        dens, _ = np.histogram(sel, bins=edges, density=True)
        ax.bar(centers, dens, width=widths*.93, color=color, alpha=.32,
               edgecolor=color, linewidth=.5, label=f'{name} (n={len(sel):,})')
        if len(sel) >= 3 and opc.get('curvas_suavizadas_1d', True):
            ax.plot(centers, gaussian_filter1d(dens, 1.15, mode='nearest'),
                    color=color, lw=2.0)
    ax.set(title='t-SNE 1D | Histograma do embedding no F10',
           xlabel='TSNE1 — coordenada projetada fora da amostra (OOS)',
           ylabel='Densidade por classe (área = 1)')
    ax.grid(axis='y', alpha=.14)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(loc='best')
    fig.subplots_adjust(left=.1, right=.975, bottom=.15, top=.90)
    fig.savefig(dest / '13_tsne_mapa_TSNE1.png', dpi=int(cfg['visualizacao']['padrao'].get('dpi',180)), bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame({
        'linha_original': ids, 'TSNE1': coords, 'classe_real': y,
        'prob_fraude_v3': scores,
        'predicao_tau_oof': (scores >= tau).astype(np.uint8),
        'gaussiana_dominante_gmm': np.argmax(gamma, axis=1)+1,
    }).to_csv(dest / '13_tsne_mapa_TSNE1.csv', index=False, float_format='%.17g')
    print('[V6 GRAFICO] 13_tsne_mapa_TSNE1.png', flush=True)


def desenhar_tsne14(dest, modelo, n_f10):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    kl_raw = getattr(modelo.tsne, 'kl_divergence_', None)
    try:
        kl = float(kl_raw) if kl_raw is not None else None
    except (TypeError, ValueError):
        kl = None
    if kl is not None and (not np.isfinite(kl) or kl < 0):
        kl = None
    fig, ax = plt.subplots(1, 2, figsize=(11.8,4.7))
    fig.patch.set_facecolor('#f8fafc')
    for a in ax:
        a.set_facecolor('#f1f5f9')
        a.set_xticks([]); a.set_yticks([])
        for spine in a.spines.values():
            spine.set_color('#cbd5e1')
    ax[0].text(.06, .82, 'DIVERGÊNCIA KL DO AJUSTE F1–F9', fontweight='bold',
               color='#334155', fontsize=11, transform=ax[0].transAxes)
    ax[0].text(.06, .50, f'{kl:.7f}' if kl is not None else 'Não disponível',
               color='#1d4ed8', fontsize=23, fontweight='bold', transform=ax[0].transAxes)
    ax[0].text(.06, .21, 'Diagnóstico interno do t-SNE do treino.\nNão é métrica de classificação.',
               color='#475569', fontsize=9.1, transform=ax[0].transAxes)
    ax[1].text(.06, .82, 'TEMPO DE INFERÊNCIA OOS NO F10', fontweight='bold',
               color='#334155', fontsize=11, transform=ax[1].transAxes)
    ax[1].text(.06, .50, 'Não registrado', color='#047857',
               fontsize=22, fontweight='bold', transform=ax[1].transAxes)
    ax[1].text(.06, .21, f'V3 projetou {n_f10:,} registros em lotes.\nNão criar duração estimada.',
               color='#475569', fontsize=9.1, transform=ax[1].transAxes)
    fig.suptitle('t-SNE | Diagnóstico KL e tempo', fontsize=15, weight='bold', y=.975)
    fig.subplots_adjust(left=.055, right=.945, top=.81, bottom=.13, wspace=.09)
    fig.savefig(dest / '14_tsne_diagnostico.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame([{
        'kl_final_ajuste_f1_f9':kl, 'tempo_inferencia_oos_f10_segundos':None,
        'tempo_documentado_na_v3':False,
        'origem_kl':'modelo_final_tsne_F1_F9_congelado',
        'origem_tempo':'nao_registrado',
        'n_f10':int(n_f10),
        'nenhum_ajuste_no_f10':True,
    }]).to_csv(dest / '14_tsne_diagnostico.csv', index=False)
    print('[V6 GRAFICO] 14_tsne_diagnostico.png', flush=True)
    return kl


def resumo_transformacao_f10(info, pacote, ids, y_v3):
    from src import orquestracao_oof as orq
    st = info['st']
    cfg = st['cfg']
    parq = ROOT / cfg['dados']['arquivo']
    print('[V6] Conferindo hash real do Parquet e fold mestre.', flush=True)
    folds, arq_folds, hash_parq = orq._identificar_folds(ROOT, cfg, parq)
    m = st['info']['info']['manifesto_oof']
    exigir(hash_parq == m['sha256_parquet'] and orq.hash_arquivo(arq_folds) == m['sha256_folds'],
           'Parquet/fold mestre mudou desde o OOF.')
    exigir(ids[-1] < len(folds) and np.all(folds[ids] == 10),
           'V6 recebeu indices fora do F10 reservado.')
    print('[V6] Lendo exclusivamente as 18523 linhas F10 ja selecionadas.', flush=True)
    df = orq._ler_desenvolvimento(parq, ids)
    coluna = cfg['dados']['coluna_alvo']
    exigir(len(df) == len(ids) and np.array_equal(df[coluna].to_numpy(dtype=np.uint8), y_v3),
           'Rótulos F10 divergem das previsões V3.')
    excluidas = [coluna, cfg['dados'].get('coluna_id'), *cfg['dados'].get('colunas_excluir',[])]
    x = df.drop(columns=[p for p in excluidas if p], errors='ignore').copy()
    exigir(list(x.columns) == pacote.encoder.colunas_, 'Colunas/ordem brutas diferentes do modelo F1-F9.')
    # Somente transform() com transformadores ajustados em F1-F9; nunca fit().
    enc = pacote.encoder.transform(x)
    pre = pacote.imputador.transform(enc)
    exigir(len(pre) == len(ids) and list(pre.columns) == pacote.imputador.colunas_,
           'PRE01: matriz transformada difere do modelo F1-F9.')
    z = pre.to_numpy(dtype=np.float64)
    exigir(np.isfinite(z).all(), 'PRE01 F10 recebeu NaN/infinito.')
    scaler = pacote.imputador.scaler_
    standardized = list(pacote.imputador.padronizadas_)
    pos = {name: j for j,name in enumerate(standardized)}
    table = []
    for col in pre.columns:
        vals = pre[col].to_numpy(dtype=np.float64)
        typ = 'padronizada' if col in pos else 'preservada_sem_padronizar'
        mu = float(vals.mean()); sigma = float(vals.std(ddof=0)); variance = float(vals.var(ddof=0))
        raw_na = float(x[col].isna().mean()*100) if col in x.columns else None
        if col in pos:
            idx = pos[col]
            mean_training = float(scaler.mean_[idx]); scale_training = float(scaler.scale_[idx])
            mean_encoded_f10 = mean_training + mu*scale_training
            std_encoded_f10 = sigma*scale_training
            drift = mu
        else:
            mean_training = None; scale_training = None
            mean_encoded_f10 = None; std_encoded_f10 = None; drift = None
        table.append({
            'feature':str(col), 'tipo':typ, 'n_f10':int(len(ids)),
            'media_treino_f1_f9_codificada':mean_training,
            'escala_treino_f1_f9':scale_training,
            'media_transformada_f10':mu,
            'desvio_transformado_f10':sigma,
            'variancia_transformada_f10':variance,
            'media_f10_codificada_imputada':mean_encoded_f10,
            'desvio_f10_codificado_imputado':std_encoded_f10,
            'drift_media_padronizada':drift,
            'percentual_ausentes_brutos_f10':raw_na,
        })
    del z, pre, enc, x, df
    print(f'[V6 PRE01] {len(table)} variaveis transformadas, {len(standardized)} padronizadas.', flush=True)
    return pd.DataFrame(table), len(standardized)


def histograma_f10(coords, bins=60):
    x = np.asarray(coords, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        lo, hi = lo-.5, hi+.5
    counts, edges = np.histogram(x, bins=np.linspace(lo, hi, bins+1))
    exigir(int(counts.sum()) == len(x), 'Histograma TSNE1 nao conserva quantidade.')
    return pd.DataFrame({'tsne1_inicio':edges[:-1], 'tsne1_fim':edges[1:],
                         'tsne1_centro':(edges[:-1]+edges[1:])/2, 'contagem_f10':counts})


def html_pre01(df, hist, meta):
    from plotly.offline import get_plotlyjs
    def json_safe(v):
        if isinstance(v, dict):
            return {str(k):json_safe(x) for k,x in v.items()}
        if isinstance(v, list):
            return [json_safe(x) for x in v]
        if isinstance(v, (np.floating, np.integer)):
            return json_safe(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    records = json_safe(df.astype(object).where(pd.notna(df), None).to_dict(orient='records'))
    h = json_safe(hist.to_dict(orient='records'))
    manifest = json_safe(meta)
    data = json.dumps({'rows':records, 'hist':h, 'meta':manifest},
                      ensure_ascii=False, allow_nan=False).replace('</','<\\/')
    script = get_plotlyjs().replace('</','<\\/')
    # HTML autonomo offline: Plotly embutido, sem CDN ou requisicoes externas.
    # Todas as colunas sao exclusivamente descritivas do preprocessamento congelado.
    return '''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PRE01 F10 — t-SNE 1D</title>
<style>body{background:#f4f7fb;color:#19324a;font:14px system-ui,Arial,sans-serif;margin:0}
main{max-width:1480px;margin:0 auto;padding:24px}h1{font-size:27px;margin:0 0 4px}h2{font-size:18px}
.lead{color:#5a728e;margin:6px 0 18px}.box{background:#fff;border:1px solid #dbe4ee;border-radius:11px;padding:18px;margin:14px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:15px}
.banner{border-left:4px solid #31699c;background:#e9f1f8;padding:14px;margin:14px 0}
.scroll{max-height:500px;overflow:auto;border:1px solid #dae4ee}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}
th,td{padding:7px 10px;border-bottom:1px solid #e7edf5;text-align:right}th:first-child,td:first-child{text-align:left}
th{position:sticky;top:0;background:#eaf1f8}input{padding:8px;border:1px solid #b6c5d5;border-radius:7px;width:min(450px,100%)}
.small{font-size:12px;color:#64758a}#chart1,#chart2,#chart3{min-height:345px}</style></head><body>
<main><h1>PRE01 — Diagnóstico da padronização e projeção TSNE1</h1>
<p class="lead">F10 exploratório reutilizado | modelo t-SNE/GMM ajustado exclusivamente em F1–F9</p>
<div class="banner"><b>Sem reestimativa:</b> encoder, imputador e scaler são os de F1–F9. A média z em F10 e a escala F10 descrevem apenas o deslocamento dos dados de teste. Variáveis binárias, cíclicas e one-hot mantêm seus valores transformados sem padronização. O t-SNE não possui coeficientes lineares de contribuição.</div>
<div class="grid"><section class="box"><h2>Média padronizada em F10</h2><div id="chart1"></div></section>
<section class="box"><h2>Desvio-padrão padronizado em F10</h2><div id="chart2"></div></section></div>
<section class="box"><h2>Distribuição das coordenadas TSNE1 no F10</h2><div id="chart3"></div></section>
<section class="box"><h2>Tabela de variáveis</h2><input type="search" id="busca" placeholder="Filtrar variável ou tipo de transformação..."><p class="small">Os valores ausentes (—) indicam parâmetros não aplicáveis a variáveis não padronizadas.</p>
<div class="scroll"><table><thead><tr><th>Variável</th><th>Tipo</th><th>Média transformada F10</th><th>DP transformado F10</th><th>Média treino codificada</th><th>Escala treino</th><th>Ausentes brutos %</th></tr></thead><tbody id="tabela"></tbody></table></div></section>
<p class="small">Relatório descritivo; métricas de detecção de fraude permanecem na avaliação V3 e gráficos V4. Nenhum limiar foi otimizado no F10.</p></main>
<script>''' + script + '''</script><script>const D=''' + data + ''';
const rows=D.rows, pad=rows.filter(r=>r.tipo==='padronizada');
const labels=pad.map(r=>r.feature), mus=pad.map(r=>r.media_transformada_f10),
sds=pad.map(r=>r.desvio_transformado_f10);
const base={margin:{l:165,r:25,b:60,t:24},paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'system-ui,Arial',size:11}};
const common={responsive:true,displaylogo:false};
Plotly.newPlot('chart1',[{type:'bar',orientation:'h',x:mus,y:labels,marker:{color:'#3977b7'}}],
{...base,height:Math.max(380,25*labels.length),xaxis:{title:'Média z em F10'},yaxis:{automargin:true,autorange:'reversed'},shapes:[{type:'line',xref:'x',x0:0,x1:0,yref:'paper',y0:0,y1:1,line:{dash:'dot',color:'#9b6072'}}]},common);
Plotly.newPlot('chart2',[{type:'bar',orientation:'h',x:sds,y:labels,marker:{color:'#358d75'}}],
{...base,height:Math.max(380,25*labels.length),xaxis:{title:'Desvio z em F10'},yaxis:{automargin:true,autorange:'reversed'},shapes:[{type:'line',xref:'x',x0:1,x1:1,yref:'paper',y0:0,y1:1,line:{dash:'dot',color:'#9b6072'}}]},common);
Plotly.newPlot('chart3',[{type:'bar',x:D.hist.map(r=>r.tsne1_centro),y:D.hist.map(r=>r.contagem_f10),marker:{color:'#4d85c0'}}],
{...base,height:400,xaxis:{title:'TSNE1 OOS'},yaxis:{title:'Contagem F10'}},common);
const cell=(value)=>value===null||value===undefined?'—':(typeof value==='number'?value.toFixed(7):String(value));
const esc=s=>String(s).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));
function draw(){const q=document.getElementById('busca').value.toLowerCase();
const items=rows.filter(r=>(r.feature+' '+r.tipo).toLowerCase().includes(q));
document.getElementById('tabela').innerHTML=items.map(r=>'<tr>'+['feature','tipo','media_transformada_f10','desvio_transformado_f10','media_treino_f1_f9_codificada','escala_treino_f1_f9','percentual_ausentes_brutos_f10'].map(k=>'<td>'+esc(cell(r[k]))+'</td>').join('')+'</tr>').join('');}
document.getElementById('busca').addEventListener('input',draw);draw();
</script></body></html>'''


def executar(info):
    st = info['st']
    dest = info['out']; stage = info['stage']
    imprimir_plano(info)
    if dest.exists():
        conferir_saida(info)
        print('V6_STATUS=JA_APROVADA_SAIDA_VALIDADA', flush=True)
        return 0
    exigir(not stage.exists(), 'Pasta V6 parcial presente. Preserve e investigue antes de repetir.')
    exigir(not LOCK.exists(), 'Lock V6 existente. Conferir se Python esta ativo antes de retomar.')
    with LOCK.open('x', encoding='utf-8') as f:
        f.write('V6 diagnosticos TSNE F10, sem fit nem novas predicoes.\n')
    try:
        origens = {str(x.relative_to(ROOT)):sha(x) for x in arquivos_entrada(st)}
        coords, y, scores, metricas, ids = dados_v3(info)
        modelo = carregar_pacote_confiavel(st['info']['modelo_dir'])
        exigir(modelo.limiar_oof == st['escolha']['limiar_mcc_oof'] and
               modelo.escolha_oof['id_candidato'] == st['escolha']['id_candidato'] and
               int(modelo.tsne.parametros['n_components']) == 1,
               'Modelo congelado nao corresponde ao vencedor TSNE1D.')
        exigir(len(coords) == len(y) == len(scores) == len(ids) == 18523,
               'Quantidade de coordenadas F10 divergente.')
        # Atribuicao de gaussianas, sem ajuste ou nova previsao do modelo.
        coord_frame = pd.DataFrame({'TSNE1':coords})
        gamma = modelo.gmm.responsabilidades(coord_frame).to_numpy()
        exigir(gamma.shape == (len(ids),4) and np.isfinite(gamma).all(),
               'Responsabilidades GMM invalidas em V6.')
        stage.mkdir(parents=False, exist_ok=False)
        print('[V6] Gerando histograma do embedding F10 a partir de V3.', flush=True)
        desenhar_tsne13(stage, coords, y, scores, ids, gamma, st['cfg'], modelo.limiar_oof)
        print('[V6] Registrando KL do ajuste F1-F9 e ausencia de tempo V3.', flush=True)
        kl = desenhar_tsne14(stage, modelo, len(ids))
        print('[V6] Gerando PRE01 F10 com encoder/scaler congelados, sem fit.', flush=True)
        tabela, n_pad = resumo_transformacao_f10(info, modelo, ids, y)
        hist = histograma_f10(coords)
        sub = stage / 'PRE01_F10'
        sub.mkdir(exist_ok=False)
        tabela.to_csv(sub / 'PRE01_F10_todas_variaveis.csv', index=False, float_format='%.17g')
        hist.to_csv(sub / 'PRE01_F10_TSNE1_histograma.csv', index=False, float_format='%.17g')
        premeta = {'n_f10':len(ids), 'n_variaveis':len(tabela), 'n_padronizadas':n_pad,
                   'origem':'F10_exploratorio_reutilizado', 'ajuste_scaler':'F1_F9_apenas',
                   'kl_ajuste_f1_f9':kl, 'tempo_oos_v3_segundos':None,
                   'id_candidato_oof':st['escolha']['id_candidato']}
        pagina = html_pre01(tabela, hist, premeta)
        (sub / 'PRE01_F10_diagnostico_padronizacao_completo.html').write_text(pagina,encoding='utf-8')
        for name in FILES:
            path = stage / name
            exigir(path.is_file() and path.stat().st_size > 0, 'Arquivo V6 nao gerado: '+name)
            if name.endswith('.png'):
                exigir(path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n', 'PNG invalido: '+name)
        exigir('plotly' in pagina.lower() and 'TSNE1' in pagina and len(pagina)>100000,
               'PRE01 HTML interativo nao parece autonomo/offline.')
        exigir(origens == {str(x.relative_to(ROOT)):sha(x) for x in arquivos_entrada(st)},
               'Uma fonte V2-V5 foi alterada durante a V6.')
        hashes = {name: sha(stage/name) for name in FILES}
        meta = {
            'versao':'V6_DIAGNOSTICOS_F10_TSNE1D_V1', 'status':'APROVADA',
            'data_geracao_utc':datetime.now(timezone.utc).isoformat(),
            'n_f10':int(len(ids)), 'n_fraudes_f10':int(np.sum(y)),
            'id_candidato_oof':st['escolha']['id_candidato'],
            'limiar_mcc_oof_congelado':float(modelo.limiar_oof),
            'sha256_modelo_v2':st['info']['modelo_hash'],
            'sha256_entradas':origens, 'sha256_arquivos':hashes,
            'kl_final_tsne_f1_f9':kl, 'tempo_inferencia_oos_v3_segundos':None,
            'n_variaveis_pre01':int(len(tabela)), 'n_padronizadas_pre01':int(n_pad),
            'modelo_reajustado_no_f10':False,
            'previsoes_recalculadas_no_f10':False,
            'limiar_otimizado_no_f10':False,
            'avaliacao_f10_exploratoria_reutilizada':True,
        }
        (stage/'manifesto_v6.json').write_text(json.dumps(meta, ensure_ascii=False,
                               indent=2, allow_nan=False)+'\n',encoding='utf-8')
        exigir(not dest.exists(), 'Destino V6 apareceu durante geracao.')
        os.replace(stage,dest)
        conferir_saida(info)
        print('V6_TSNE13=GERADO', flush=True)
        print('V6_TSNE14=GERADO', flush=True)
        print('V6_PRE01_INTERATIVO=GERADO', flush=True)
        print('V6_N_VARIAVEIS_PRE01='+str(len(tabela)), flush=True)
        print('V6_TEMPO_OOS_V3=NAO_REGISTRADO', flush=True)
        print('V6_STATUS=APROVADA', flush=True)
        print('OOF_REEXECUTADO=NAO', flush=True)
        print('NOVAS_PREDICOES_F10=NAO', flush=True)
        return 0
    finally:
        LOCK.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument('--planejar', action='store_true')
    grupo.add_argument('--executar', action='store_true')
    args = parser.parse_args(argv)
    info = preflight()
    imprimir_plano(info) if args.planejar else None
    if args.planejar:
        if info['out'].exists():
            conferir_saida(info)
            print('V6_STATUS=JA_APROVADA_SAIDA_VALIDADA', flush=True)
        else:
            print('V6_STATUS=PLANEJAMENTO_APROVADO_SEM_F10_DADOS', flush=True)
        return 0
    return executar(info)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('V6_STATUS=BLOQUEADA',file=sys.stderr,flush=True)
        print('V6_MOTIVO='+str(error),file=sys.stderr,flush=True)
        traceback.print_exc()
        sys.exit(1)
