# -*- coding: utf-8 -*-
"""Fase 3B.2B: TSNE13/14, PRE01 e galeria offline para F10 t-SNE 1D.

Sem treinamento, sem nova inferencia t-SNE/GMM e sem recalibracao em F10.
PRE01 apenas transforma as linhas F10 previamente avaliadas usando encoder,
imputador e scaler congelados; seus resultados sao diagnosticos.
"""
from __future__ import annotations
from html import escape
import json
import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from src.folds import hash_arquivo
from src.amostragem_f10 import fracao_amostra_f10
from src.visualizacao_final_tsne1d import exigir, ler_json, PNG_FILES, _entradas
from src.visualizacao_geometrica_final_tsne1d import PNG_GEOMETRIA, _fontes
from src.treinamento_final_tsne import carregar_pacote_confiavel

DIAGNOSTICOS = ('13_tsne_mapa_TSNE1.png','13_tsne_mapa_TSNE1.csv',
                '14_tsne_diagnostico.png','14_tsne_diagnostico.csv',
                'PRE01_F10/PRE01_F10_diagnostico_padronizacao_completo.html',
                'PRE01_F10/PRE01_F10_todas_variaveis.csv',
                'PRE01_F10/PRE01_F10_TSNE1_histograma.csv')
PRE01 = 'PRE01_F10/PRE01_F10_diagnostico_padronizacao_completo.html'

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
        'prob_fraude_f10': scores,
        'predicao_tau_oof': (scores >= tau).astype(np.uint8),
        'gaussiana_dominante_gmm': np.argmax(gamma, axis=1)+1,
    }).to_csv(dest / '13_tsne_mapa_TSNE1.csv', index=False, float_format='%.17g')
    print('[F10 TSNE DIAGNOSTICO] 13_tsne_mapa_TSNE1.png', flush=True)


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
    ax[1].text(.06, .21, f'Modelo projetou {n_f10:,} registros em lotes.\nNão criar duração estimada.',
               color='#475569', fontsize=9.1, transform=ax[1].transAxes)
    fig.suptitle('t-SNE | Diagnóstico KL e tempo', fontsize=15, weight='bold', y=.975)
    fig.subplots_adjust(left=.055, right=.945, top=.81, bottom=.13, wspace=.09)
    fig.savefig(dest / '14_tsne_diagnostico.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame([{
        'kl_final_ajuste_f1_f9':kl, 'tempo_inferencia_oos_f10_segundos':None,
        'tempo_oos_registrado':False,
        'origem_kl':'modelo_final_tsne_F1_F9_congelado',
        'origem_tempo':'nao_registrado',
        'n_f10':int(n_f10),
        'nenhum_ajuste_no_f10':True,
    }]).to_csv(dest / '14_tsne_diagnostico.csv', index=False)
    print('[F10 TSNE DIAGNOSTICO] 14_tsne_diagnostico.png', flush=True)
    return kl


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
<p class="lead">F10 exploratório — modelo F1–F9 congelado | modelo t-SNE/GMM ajustado exclusivamente em F1–F9</p>
<div class="banner"><b>Sem reestimativa:</b> encoder, imputador e scaler são os de F1–F9. A média z em F10 e a escala F10 descrevem apenas o deslocamento dos dados de teste. Variáveis binárias, cíclicas e one-hot mantêm seus valores transformados sem padronização. O t-SNE não possui coeficientes lineares de contribuição.</div>
<div class="grid"><section class="box"><h2>Média padronizada em F10</h2><div id="chart1"></div></section>
<section class="box"><h2>Desvio-padrão padronizado em F10</h2><div id="chart2"></div></section></div>
<section class="box"><h2>Distribuição das coordenadas TSNE1 no F10</h2><div id="chart3"></div></section>
<section class="box"><h2>Tabela de variáveis</h2><input type="search" id="busca" placeholder="Filtrar variável ou tipo de transformação..."><p class="small">Os valores ausentes (—) indicam parâmetros não aplicáveis a variáveis não padronizadas.</p>
<div class="scroll"><table><thead><tr><th>Variável</th><th>Tipo</th><th>Média transformada F10</th><th>DP transformado F10</th><th>Média treino codificada</th><th>Escala treino</th><th>Ausentes brutos %</th></tr></thead><tbody id="tabela"></tbody></table></div></section>
<p class="small">Relatório descritivo; métricas de detecção de fraude permanecem na avaliação F10 congelada e gráficos preditivos. Nenhum limiar foi otimizado no F10.</p></main>
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


def _resumo_pre01(pasta, pacote, ids, y_verdadeiro, config):
    """Somente leitura posicional do F10 congelado, SEM fit nem OOS."""
    from src import orquestracao_oof as orq
    raiz = Path(__file__).resolve().parents[1]
    parquet = raiz / config['dados']['arquivo']
    m = ler_json(Path(pasta)/'manifesto.json')
    folds, arq_folds, digest = orq._identificar_folds(raiz, config, parquet)
    exigir(digest == m['sha256_parquet'] and hash_arquivo(arq_folds) == m['sha256_folds'],
           'PRE01: base ou particoes divergem da origem OOF.')
    exigir(len(ids)>0 and np.all(folds[ids] == 10), 'PRE01: indices fora do F10.')
    df = orq._ler_desenvolvimento(parquet, ids)
    alvo = config['dados']['coluna_alvo']
    exigir(np.array_equal(df[alvo].to_numpy(dtype=np.uint8),y_verdadeiro),
           'PRE01: rotulos divergentes da avaliacao F10 congelada.')
    drop = [alvo,config['dados'].get('coluna_id'),*config['dados'].get('colunas_excluir',[])]
    x = df.drop(columns=[c for c in drop if c],errors='ignore')
    exigir(list(x.columns) == pacote.encoder.colunas_,
           'PRE01: features brutas divergentes do treino.')
    enc = pacote.encoder.transform(x)
    pre = pacote.imputador.transform(enc)
    exigir(len(pre)==len(ids) and list(pre.columns)==pacote.imputador.colunas_
           and np.isfinite(pre.to_numpy(dtype=np.float64)).all(),
           'PRE01: preprocessamento congelado inconsistente.')
    scaler = pacote.imputador.scaler_
    pad = list(pacote.imputador.padronizadas_)
    pos = {c:i for i,c in enumerate(pad)}
    linhas = []
    for col in pre.columns:
        valores = pre[col].to_numpy(dtype=np.float64)
        standard = col in pos
        i = pos.get(col)
        mean = float(scaler.mean_[i]) if standard else None
        scale = float(scaler.scale_[i]) if standard else None
        mu = float(valores.mean()); sd = float(valores.std(ddof=0))
        linhas.append({
           'feature':str(col), 'tipo':'padronizada' if standard else 'preservada_sem_padronizar',
           'n_f10':int(len(ids)), 'media_treino_f1_f9_codificada':mean,
           'escala_treino_f1_f9':scale,
           'media_transformada_f10':mu, 'desvio_transformado_f10':sd,
           'variancia_transformada_f10':float(valores.var(ddof=0)),
           'media_f10_codificada_imputada':mean+mu*scale if standard else None,
           'desvio_f10_codificado_imputado':sd*scale if standard else None,
           'drift_media_padronizada':mu if standard else None,
           'percentual_ausentes_brutos_f10':float(x[col].isna().mean()*100) if col in x else None
        })
    exigir(linhas and len(pad)<=len(linhas), 'PRE01: nenhuma feature transformada.')
    return pd.DataFrame(linhas),len(pad)


def _fontes_imutaveis(pasta):
    teste,cfg,met,marco,z,y,p,tau = _fontes(pasta)
    ids = np.load(teste/'indices_f10_amostra.npy',allow_pickle=False)
    exigir(len(ids)==len(y)==len(z), 'Diagnosticos: dimensoes incoerentes.')
    modelo=carregar_pacote_confiavel(teste/'modelo_final_f1_f9')
    exigir(modelo.configuracao==cfg and modelo.escolha_oof['id_candidato']==marco['id_candidato_oof']
           and np.isclose(modelo.limiar_oof,tau) and int(modelo.tsne.parametros['n_components'])==1,
           'Diagnosticos: modelo congelado nao corresponde ao F10.')
    gamma=np.asarray(modelo.gmm.responsabilidades(z),dtype=np.float64)
    exigir(gamma.shape == (len(y),int(modelo.gmm.modelo_.n_components))
           and np.isfinite(gamma).all(), 'Diagnosticos: responsabilidades GMM invalidas.')
    return teste,cfg,met,marco,z['TSNE1'].to_numpy(dtype=float),y,p,tau,ids,modelo,gamma


def _validar_diagnosticos(dest,marco):
    man=ler_json(dest/'manifesto_diagnosticos_f10.json')
    exigir(man.get('estado')=='DIAGNOSTICOS_F10_COMPLETOS' and
           man.get('sha256_modelo')==marco['sha256_modelo'] and
           man.get('sha256_coordenadas')==marco['sha256_coordenadas'] and
           man.get('sha256_predicoes')==marco['sha256_predicoes'] and
           set(man.get('sha256_arquivos',{}))==set(DIAGNOSTICOS),
           'Diagnosticos existentes nao correspondem ao F10 congelado.')
    for nome,digest in man['sha256_arquivos'].items():
        arq=dest/nome
        exigir(arq.is_file() and hash_arquivo(arq)==digest,
               'Diagnostico F10 ausente/alterado: '+nome)
    return dest


def gerar_diagnosticos(pasta):
    """Gera TSNE13/14 e PRE01 em unica pasta transacional."""
    teste,cfg,met,marco,coords,y,p,tau,ids,modelo,gamma=_fontes_imutaveis(pasta)
    dest=teste/'diagnosticos_tsne'
    if dest.exists():return _validar_diagnosticos(dest,marco)
    stage=teste/'.diagnosticos_tsne_incompletos'
    exigir(not stage.exists(),'Staging diagnostico pendente; preservar.')
    lock=teste/'.diagnosticos_tsne.lock'
    try:
        with lock.open('x',encoding='utf-8') as f: f.write('Geracao de diagnosticos F10.\n')
    except FileExistsError as exc:raise RuntimeError('Diagnosticos ja em execucao.') from exc
    try:
        stage.mkdir(exist_ok=False)
        desenhar_tsne13(stage,coords,y,p,ids,gamma,cfg,tau)
        kl=desenhar_tsne14(stage,modelo,len(ids))
        tabela,n_pad=_resumo_pre01(pasta,modelo,ids,y,cfg)
        hist=histograma_f10(coords)
        sub=stage/'PRE01_F10';sub.mkdir()
        tabela.to_csv(sub/'PRE01_F10_todas_variaveis.csv',index=False,float_format='%.17g')
        hist.to_csv(sub/'PRE01_F10_TSNE1_histograma.csv',index=False,float_format='%.17g')
        premeta={'n_f10':int(len(ids)), 'n_variaveis':len(tabela),
                 'n_padronizadas':n_pad, 'origem':'F10_exploratorio',
                 'ajuste_scaler':'F1_F9_apenas', 'kl_ajuste_f1_f9':kl,
                 'tempo_oos_f10_segundos':None, 'id_candidato_oof':marco['id_candidato_oof']}
        pagina=html_pre01(tabela,hist,premeta)
        exigir('plotly' in pagina.lower() and 'TSNE1' in pagina and len(pagina)>100000,
               'PRE01 nao parece offline/completo.')
        (stage/PRE01).write_text(pagina,encoding='utf-8')
        arquivos={}
        for nome in DIAGNOSTICOS:
            arq=stage/nome
            exigir(arq.is_file() and arq.stat().st_size>0,'Diagnostico nao gerado: '+nome)
            if nome.endswith('.png'):
                exigir(arq.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n', 'PNG invalido: '+nome)
            arquivos[nome]=hash_arquivo(arq)
        info={'estado':'DIAGNOSTICOS_F10_COMPLETOS',
              'sha256_modelo':marco['sha256_modelo'],
              'sha256_predicoes':marco['sha256_predicoes'],
              'sha256_coordenadas':marco['sha256_coordenadas'],
              'n_f10':len(ids), 'n_variaveis_pre01':len(tabela),
              'n_padronizadas_pre01':n_pad, 'sha256_arquivos':arquivos,
              'sem_fit_no_f10':True, 'sem_novas_predicoes':True}
        (stage/'manifesto_diagnosticos_f10.json').write_text(
           json.dumps(info,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        exigir(not dest.exists(),'Destino de diagnosticos surgiu durante criacao.')
        os.replace(stage,dest)
        return _validar_diagnosticos(dest,marco)
    finally:
        lock.unlink(missing_ok=True)


TITULOS = ('Painel inicial','Curva Precision–Recall','MCC por limiar',
           'Matriz de confusão τ=0,50','Matriz de confusão τ* OOF',
           'Distribuição de probabilidades','GMM01 — Densidades',
           'GMM04 — Dominância','GMM07 — Sensibilidade da fronteira',
           'GMM08 — Risco e geometria 95%','TSNE13 — Histograma TSNE1',
           'TSNE14 — Diagnóstico KL / tempo')


def _conferir_galeria(dest,marco):
    man=ler_json(dest/'manifesto_galeria_f10.json')
    exigir(man.get('estado')=='GALERIA_F10_TSNE1D_COMPLETA' and
           man.get('sha256_predicoes')==marco['sha256_predicoes'] and
           man.get('sha256_coordenadas')==marco['sha256_coordenadas'] and
           man.get('sha256_modelo')==marco['sha256_modelo'] and
           len(man.get('sha256_arquivos',{}))==14,
           'Galeria nao corresponde a origem F10.')
    for nome,h in man['sha256_arquivos'].items():
        f=dest/nome
        exigir(f.is_file() and hash_arquivo(f)==h,'Galeria alterada/ausente: '+nome)
    return dest


def gerar_galeria(pasta):
    """Exige todos os renderizadores verificados; nunca refaz inferencia F10."""
    from src.visualizacao_final_tsne1d import gerar as preditivos
    from src.visualizacao_geometrica_final_tsne1d import gerar as geometria
    # Primeiro verificar preexistentes com seus manifestos/hash, depois completar ausentes.
    teste,cfg,met,marco,coords,y,p,tau,ids,modelo,gamma=_fontes_imutaveis(pasta)
    pg=preditivos(pasta)
    gg=geometria(pasta)
    dg=gerar_diagnosticos(pasta)
    dest=teste/'galeria'
    if dest.exists():return _conferir_galeria(dest,marco)
    stage=teste/'.galeria_tsne_incompleta'
    exigir(not stage.exists(),'Galeria parcial anterior: preservar.')
    lock=teste/'.galeria_tsne.lock'
    try:
        with lock.open('x',encoding='utf-8') as f:f.write('Galeria F10 em execucao.\n')
    except FileExistsError as exc:raise RuntimeError('Galeria F10 ja em execucao.') from exc
    try:
        stage.mkdir(exist_ok=False)
        figs=stage/'figuras';figs.mkdir()
        fontes=[*(pg/nome for nome in PNG_FILES),*(gg/nome for nome in PNG_GEOMETRIA),
                dg/'13_tsne_mapa_TSNE1.png',dg/'14_tsne_diagnostico.png']
        exigir(len(fontes)==12,'Galeria exige exatamente 12 graficos.')
        arquivos={}
        cards=[]
        for nome,titulo,origem in zip([f.name for f in fontes],TITULOS,fontes):
            alvo=figs/nome
            shutil.copyfile(origem,alvo)
            exigir(hash_arquivo(origem)==hash_arquivo(alvo),'Copia diverge: '+nome)
            arquivos['figuras/'+nome]=hash_arquivo(alvo)
            cards.append(f'<a class="card" href="figuras/{escape(nome,quote=True)}">'
                         f'<img src="figuras/{escape(nome,quote=True)}" alt="{escape(titulo,quote=True)}">'
                         f'<span>{escape(titulo)}</span></a>')
        novo=stage/'PRE01_F10';novo.mkdir()
        origem_pre=dg/PRE01; alvo_pre=stage/PRE01
        shutil.copyfile(origem_pre,alvo_pre)
        exigir(hash_arquivo(origem_pre)==hash_arquivo(alvo_pre),'PRE01 copiado diverge.')
        arquivos[PRE01]=hash_arquivo(alvo_pre)
        link_oof='../../visualizacao/galeria/index.html'
        html=('<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
              '<meta name="viewport" content="width=device-width,initial-scale=1">'
              '<title>t-SNE 1D + GMM — F10 exploratório</title>'
              '<style>body{font:15px system-ui;background:#f4f7fb;color:#1b2b45;max-width:1400px;margin:auto;padding:22px}'
              '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}'
              '.card{display:block;background:white;padding:12px;border:1px solid #ccd6e1;border-radius:10px;color:#134e87;text-decoration:none}'
              'img{width:100%;height:auto}span{display:block;padding:6px}</style></head><body>'
              '<h1>t-SNE 1D + GMM — F10 exploratório</h1>'
              '<p>Modelo, encoder, scaler e t-SNE ajustados exclusivamente em F1–F9; '
              'teste F10 não independente inédito. Nenhum ajuste ou novo limiar em F10.</p>'
              f'<p>Desenvolvimento: {float(cfg["dados"]["fracao_base"]):.2%}; '
              f'F10: {fracao_amostra_f10(cfg):.2%} do fold; '
              f'n={len(ids)}; τ* OOF={tau:.7f}; AP F10={met["ap"]:.7f}; '
              f'MCC F10 em τ*={met["mcc_limiar_oof"]:.7f}; '
              f'LogLoss F10={met["logloss"]:.7f}</p>'
              f'<p><a href="{link_oof}">Abrir galeria OOF</a> | '
              f'<a href="{PRE01}">Abrir PRE01 interativo offline</a></p>'
              '<div class="grid">'+''.join(cards)+'</div></body></html>')
        (stage/'index.html').write_text(html,encoding='utf-8')
        arquivos['index.html']=hash_arquivo(stage/'index.html')
        exigir(len(arquivos)==14,'Galeria final exige 12 PNGs + PRE01 + index.')
        info={'estado':'GALERIA_F10_TSNE1D_COMPLETA',
              'n_f10':len(ids), 'fracao_desenvolvimento':float(cfg['dados']['fracao_base']),
              'fracao_f10_dentro_do_fold':float(marco['fracao_f10_dentro_do_fold']),
              'sha256_modelo':marco['sha256_modelo'],
              'sha256_predicoes':marco['sha256_predicoes'],
              'sha256_coordenadas':marco['sha256_coordenadas'],
              'sha256_metricas':marco['sha256_metricas'],
              'sha256_arquivos':arquivos, 'nenhum_refit_f10':True,
              'limiar_f10_otimizado':False}
        (stage/'manifesto_galeria_f10.json').write_text(
            json.dumps(info,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        exigir(not dest.exists(),'Pasta de galeria surgiu durante geracao.')
        os.replace(stage,dest)
        return _conferir_galeria(dest,marco)
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experimento',type=Path,required=True)
    args=p.parse_args(argv)
    destino=gerar_galeria(args.experimento)
    print('[TSNE1D F10] Galeria completa verificada:',destino/'index.html')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
