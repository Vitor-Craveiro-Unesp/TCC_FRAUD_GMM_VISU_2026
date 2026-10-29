# -*- coding: utf-8 -*-
"""PRE01 V2 — painel interativo completo e auditável do pré-processamento.

Não altera modelos/previsões/limiares. Ajustes diagnósticos de encoder,
imputador, scaler e PCA exclusivamente em F1..F9. F10 nunca é lido.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

from src import orquestracao_oof as orq
from src import preprocessing as prep

FILENAME = 'PRE01_diagnostico_padronizacao_completo.html'
CSV_SUM = 'PRE01_V2_todas_variaveis_resumo.csv'
CSV_DETAIL = 'PRE01_V2_todas_variaveis_por_fold.csv'
CSV_FOLDS = 'PRE01_V2_resumo_por_fold.csv'
CSV_PCA = 'PRE01_V2_estabilidade_PCA_por_fold.csv'
META = 'PRE01_V2_manifesto.json'
OLD_MARKER = '<!-- PRE01_DIAGNOSTICO_PADRONIZACAO_FOLDS -->'
NEW_MARKER = '<!-- PRE01_V2_DIAGNOSTICO_INTERATIVO -->'
OLD_PNG = 'PRE01_estabilidade_padronizacao_folds.png'


def exig(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def dim_one(v):
    if isinstance(v, list):
        if len(v) != 1: return None
        v = v[0]
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def comparable(cfg):
    cfg = deepcopy(cfg)
    cfg['pca']['n_components'] = '__DIM__'
    return json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)


def candidate(p, k, n_init, tipo):
    try:
        cfg = yaml.safe_load((p/'configuracao.yaml').read_text(encoding='utf-8-sig'))
        m = json.loads((p/'manifesto.json').read_text(encoding='utf-8-sig'))
        if not isinstance(cfg,dict) or not isinstance(m,dict): return None
        if m.get('status')!='concluido_06_03' or m.get('visualizacoes')!='concluidas' or m.get('tipo')!=tipo:
            return None
        if cfg['reducao']['metodo'].upper()!='PCA' or int(cfg['divisao']['qual_fold_teste'])!=10:
            return None
        if dim_one(cfg['gmm']['n_components'])!=k or int(cfg['gmm']['n_init'])!=n_init:
            return None
        d = dim_one(cfg['pca']['n_components'])
        if d not in (1,2,3): return None
        npz = p/'vencedor'/'oof'/'predicoes.npz'
        gallery = p/'vencedor'/'visualizacao'/'galeria'/'index.html'
        if not (npz.is_file() and gallery.is_file()):return None
        return dict(folder=p,dim=d,cfg=cfg,meta=m,npz=npz,gallery=gallery,key=comparable(cfg))
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError):
        return None


def trio_find(root,k,n_init,tipo):
    prefix = 'ensaio' if tipo=='ensaio' else 'experimento'
    matches=[]
    for p in (root/'outputs').glob(f'{prefix}_pca_gmm_*'):
        if p.is_dir():
            c=candidate(p,k,n_init,tipo)
            if c:matches.append(c)
    groups={}
    for c in matches:groups.setdefault(c['key'],[]).append(c)
    trios=[]
    for g in groups.values():
        dims={}
        for c in sorted(g,key=lambda o:o['folder'].stat().st_mtime,reverse=True):
            dims.setdefault(c['dim'],c)
        if all(d in dims for d in (1,2,3)):
            trios.append([dims[d] for d in (1,2,3)])
    exig(trios,f'Nao encontrei trio equivalente PCA 1D/2D/3D, K={k}, n_init={n_init}, tipo={tipo}.')
    trio=max(trios,key=lambda t:max(v['folder'].stat().st_mtime for v in t))
    return trio


def check_indices(trio):
    base=None
    for t in trio:
        with np.load(t['npz'],allow_pickle=False) as z:
            ids=np.asarray(z['linha_original'],dtype=np.int64)
            folds=np.asarray(z['fold'],dtype=np.uint8)
            exig(int(z['fold_esperado'].item())==0,'Esperado OOF consolidado.')
        exig(len(ids)>0 and np.all(np.diff(ids)>0),'IDs ausentes/repetidos/desordenados.')
        exig(set(np.unique(folds).tolist())==set(range(1,10)),'Folds OOF nao sao F1..F9.')
        if base is None:base=(ids,folds)
        else:exig(np.array_equal(ids,base[0]) and np.array_equal(folds,base[1]),
                  'As tres dimensoes nao possuem as mesmas observacoes e folds.')
        for hash_key in ('sha256_parquet','sha256_folds'):
            exig(t['meta'].get(hash_key) and t['meta'][hash_key]==trio[0]['meta'].get(hash_key),
                  f'Manifesto com {hash_key} divergente.')
        print(f"[OK] PCA {t['dim']}D: {len(ids):,} predicoes OOF conferidas",flush=True)
    return base


def fit_pre(X,cfg):
    enc=prep.EncodersPorSufixo(cfg).fit(X)
    XE=enc.transform(X)
    imp=prep.ImputadorPadronizador(cfg,enc).fit(XE)
    names=list(imp.padronizadas_)
    exig(imp.scaler_ is not None and len(names)>0,'Sem colunas StandardScaler.')
    return enc,imp,names


def arr(a):
    return np.asarray(a,dtype=np.float64)


def calc(X,folds,cfg):
    # Os encoders FEWF sao reajustados com cada treino, como no pipeline oficial.
    enc0,imp0,names=fit_pre(X,cfg)
    X0=imp0.transform(enc0.transform(X))
    ref_mean=arr(imp0.scaler_.mean_)
    ref_var=arr(imp0.scaler_.var_)
    ref_scale=arr(imp0.scaler_.scale_)
    med_ref=np.array([float(imp0.mediana_.get(n,np.nan)) for n in names])
    jidx=[X0.columns.get_loc(n) for n in names]
    ref_pca=PCA(n_components=3,svd_solver='auto',random_state=int(cfg['experimento']['semente']))
    exig(X0.shape[1]>=3 and len(X0)>=4,'PCA diagnostico necessita no minimo 3 features e 4 registros.')
    ref_pca.fit(X0.to_numpy(dtype=np.float64))

    details=[]; pca_details=[];folds_records=[]
    suffix=cfg['preprocessamento']['encodings']['frequency']['sufixo']
    for k in range(1,10):
        train=(folds!=k);valid=(folds==k)
        Xt=X.loc[train];Xv=X.loc[valid]
        enc,imp,ns=fit_pre(Xt,cfg)
        exig(ns==names,f'Colunas padronizadas diferentes no F{k}.')
        te=enc.transform(Xt);ve=enc.transform(Xv)
        vt=imp.transform(ve)
        ttrans=imp.transform(te)
        tr_mean=arr(imp.scaler_.mean_);tr_var=arr(imp.scaler_.var_);tr_scale=arr(imp.scaler_.scale_)
        med_tr=np.array([float(imp.mediana_.get(n,np.nan)) for n in names])
        Z=vt.iloc[:,jidx].to_numpy(dtype=np.float64)
        vmu=np.mean(Z,axis=0);vvar=np.var(Z,axis=0);vstd=np.sqrt(vvar)
        # FEWF pode ter distribuicao codificada alterada: delta tambem reflete encoder.
        dmu=(tr_mean-ref_mean)/np.where(ref_scale>0,ref_scale,1)
        dlog=np.log(tr_scale/ref_scale)
        # So existe razao de variancias informativa quando ambas sao positivas.
        vardelta=np.where(ref_var>0,(tr_var-ref_var)/np.where(ref_var>0,ref_var,1),np.nan)
        missing_tr=np.array([float(Xt[n].isna().mean()) if n in Xt.columns else np.nan for n in names])
        missing_va=np.array([float(Xv[n].isna().mean()) if n in Xv.columns else np.nan for n in names])
        for j,name in enumerate(names):
            details.append(dict(fold_validacao=k,feature=name,tipo='FEWF' if name.endswith(suffix) else 'numerica',
                n_treino=int(train.sum()),n_validacao=int(valid.sum()),
                media_8f=tr_mean[j],media_9f=ref_mean[j],
                variancia_8f=tr_var[j],variancia_9f=ref_var[j],
                desvio_8f=np.sqrt(tr_var[j]),desvio_9f=np.sqrt(ref_var[j]),
                escala_8f=tr_scale[j],escala_9f=ref_scale[j],
                mediana_imputacao_8f=med_tr[j],mediana_imputacao_9f=med_ref[j],
                delta_media_padronizada=dmu[j],delta_log_escala=dlog[j],
                delta_variancia_relativa=vardelta[j],
                media_z_validacao=vmu[j],variancia_z_validacao=vvar[j],desvio_z_validacao=vstd[j],
                faltantes_treino_percentual=missing_tr[j]*100,faltantes_validacao_percentual=missing_va[j]*100))
        folds_records.append(dict(fold=k,n_treino=int(train.sum()),n_validacao=int(valid.sum()),
            drift_media_abs_medio=np.mean(np.abs(dmu)),drift_media_abs_max=np.max(np.abs(dmu)),
            drift_log_escala_abs_medio=np.mean(np.abs(dlog)),drift_log_escala_abs_max=np.max(np.abs(dlog)),
            percentual_drift_media_ate_0_1=np.mean(np.abs(dmu)<=.10)*100,
            percentual_escala_entre_0_95_e_1_05=np.mean((tr_scale/ref_scale>=.95)&(tr_scale/ref_scale<=1.05))*100,
            media_abs_z_validacao=np.mean(np.abs(vmu)),media_abs_desvio_z_menos_1=np.mean(np.abs(vstd-1))))
        fold_pca=PCA(n_components=3,svd_solver='auto',random_state=int(cfg['experimento']['semente']))
        fold_pca.fit(ttrans.to_numpy(dtype=np.float64))
        global_axes=arr(ref_pca.components_)
        # OHE pode ter categorias ausentes em algum treino 8F; alinhar
        # vetores dos componentes por nome de coluna, nao por posicao.
        fold_axes=np.zeros_like(global_axes)
        pos=X0.columns.get_indexer(ttrans.columns)
        exig(np.all(pos>=0),f'PCA F{k}: feature nova nao presente em F1-F9.')
        fold_axes[:,pos]=arr(fold_pca.components_)
        # Similaridade do subespaco ignora troca de sinal/rotacao interna.
        for pc in (1,2,3):
            cos=np.abs(float(np.dot(global_axes[pc-1],fold_axes[pc-1])))
            similarity=float(np.mean(np.linalg.svd(global_axes[:pc]@fold_axes[:pc].T,compute_uv=False)))
            pca_details.append(dict(fold=k,componente=f'PC{pc}',
                var_explicada_ref_9f=float(ref_pca.explained_variance_ratio_[pc-1]),
                var_explicada_8f=float(fold_pca.explained_variance_ratio_[pc-1]),
                delta_var_explicada=float(fold_pca.explained_variance_ratio_[pc-1]-ref_pca.explained_variance_ratio_[pc-1]),
                cosseno_abs_direcao=float(np.clip(cos,0,1)),
                similaridade_subespaco_media=float(np.clip(similarity,0,1))))
        print(f'[PRE01 V2] F{k}: scaler, validacao transformada e PCA diagnostico calculados.',flush=True)

    df=pd.DataFrame.from_records(details)
    pca=pd.DataFrame.from_records(pca_details)
    folds_df=pd.DataFrame.from_records(folds_records)
    summary=[]
    for name,g in df.groupby('feature',sort=False):
        m=g.iloc[0]
        summary.append(dict(feature=name,tipo=m['tipo'],
            media_global_9f=m['media_9f'],variancia_global_9f=m['variancia_9f'],
            desvio_global_9f=m['desvio_9f'],escala_global_9f=m['escala_9f'],
            mediana_global_9f=m['mediana_imputacao_9f'],
            media_dos_9_ajustes_8f=g['media_8f'].mean(),
            media_8f_min=g['media_8f'].min(),media_8f_max=g['media_8f'].max(),
            variancia_dos_9_ajustes_8f=g['variancia_8f'].mean(),
            variancia_8f_min=g['variancia_8f'].min(),variancia_8f_max=g['variancia_8f'].max(),
            escala_dos_9_ajustes_8f=g['escala_8f'].mean(),
            escala_8f_min=g['escala_8f'].min(),escala_8f_max=g['escala_8f'].max(),
            mediana_dos_9_ajustes_8f=g['mediana_imputacao_8f'].mean(),
            drift_media_abs_medio=g['delta_media_padronizada'].abs().mean(),
            drift_media_abs_max=g['delta_media_padronizada'].abs().max(),
            drift_escala_log_abs_medio=g['delta_log_escala'].abs().mean(),
            drift_escala_log_abs_max=g['delta_log_escala'].abs().max(),
            razao_escala_min=(g['escala_8f']/g['escala_9f']).min(),
            razao_escala_max=(g['escala_8f']/g['escala_9f']).max(),
            percentual_folds_media_ate_0_1=np.mean(g['delta_media_padronizada'].abs()<=.1)*100,
            percentual_folds_escala_0_95_a_1_05=np.mean(((g['escala_8f']/g['escala_9f'])>=.95)&((g['escala_8f']/g['escala_9f'])<=1.05))*100,
            media_abs_z_validacao=g['media_z_validacao'].abs().mean(),
            media_abs_desvio_z_menos_1=g['desvio_z_validacao'].sub(1).abs().mean(),
            desvio_z_validacao_min=g['desvio_z_validacao'].min(),
            desvio_z_validacao_max=g['desvio_z_validacao'].max(),
            percentual_folds_z_media_ate_0_1=np.mean(g['media_z_validacao'].abs()<=.1)*100,
            percentual_folds_z_desvio_0_9_a_1_1=np.mean(g['desvio_z_validacao'].between(.9,1.1))*100,
            percentual_medio_faltantes_validacao=g['faltantes_validacao_percentual'].mean()))
    summary=pd.DataFrame.from_records(summary)
    return summary,df,folds_df,pca


def _json_clean(data):
    def convert(v):
        if isinstance(v,np.generic):return convert(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        if isinstance(v,dict):return {str(k):convert(y) for k,y in v.items()}
        if isinstance(v,list):return [convert(y) for y in v]
        return v
    return convert(data)


def _get_plotly_js():
    try:
        from plotly.offline import get_plotlyjs
        return get_plotlyjs()
    except ImportError as ex:
        raise RuntimeError('Plotly nao instalado na venv. Execute: .venv\\Scripts\\python.exe -m pip install plotly') from ex


def build_html(summary,detail,folds_df,pca,meta):
    payload=_json_clean({'summary':summary.to_dict('records'),'detail':detail.to_dict('records'),
                         'folds':folds_df.to_dict('records'),'pca':pca.to_dict('records'),'meta':meta})
    serial=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('</','<\\/')
    # JS plotly incorporado integralmente; nenhuma conexao externa necessaria.
    plotly_js=_get_plotly_js().replace('</','<\\/')
    css=r'''
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
legend{font-weight:600}.hint{font-size:12px;color:#596d84;margin-top:9px}.download{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}
.badge{font-size:12px;color:#385977;background:#edf3fa;border:1px solid #d4e0ee;padding:4px 8px;border-radius:5px}
@media(max-width:760px){main{padding:15px 10px}.twocol{grid-template-columns:1fr}h1{font-size:24px}}
'''
    pre_html=r'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PRE01 V2 — Diagnóstico completo da padronização</title>
<style>__CSS__</style></head><body><main>
<h1>PRE01 — Diagnóstico completo da padronização</h1>
<p class="lead">Comparação entre encoders, imputação e StandardScaler ajustados nos 8 folds de cada treino OOF e a referência de desenvolvimento F1–F9. Inclui as variáveis numéricas e FEWF, estatísticas por fold, comportamento dos folds de validação transformados e estabilidade diagnóstica do PCA.</p>
<div class="banner"><strong>Somente análise de desenvolvimento F1–F9.</strong> A referência de 9 folds não é usada para prever os folds OOF. F10 permanece reservado. Valores de μ/variância/escala são obtidos <em>após encoding e imputação</em>, antes do StandardScaler. Para FEWF, as diferenças também podem refletir mudanças nas frequências do encoder. Os limites de estabilidade apresentados são critérios exploratórios, não garantias de qualidade preditiva.</div>
<p id="origem" class="muted"></p>
<nav class="nav"><a href="#resumo">Indicadores</a><a href="#variaveis">Todas as variáveis</a><a href="#folds">Todos os folds</a><a href="#pos">Após padronização</a><a href="#pca">PCA diagnóstico</a><a href="#dados">Todos os valores</a><a href="#metodo">Método</a></nav>
<section class="section" id="resumo"><h2>1. Indicadores gerais</h2><div class="cards" id="cards"></div><p class="hint">Indicadores calculados sobre todos os pares (fold, variável) padronizados; não são métricas de fraude ou de previsão.</p></section>
<section class="section" id="variaveis"><h2>2. Parâmetros de todas as variáveis</h2><p>μ: média; σ: desvio-padrão; variância = σ². A <em>escala</em> é o denominador efetivo usado pelo StandardScaler (pode ser 1 para variância zero). Toda variável aparece na tabela; é possível buscar e ordenar sem limitar linhas.</p><div class="twocol"><div class="card"><h3>Drift médio absoluto de μ por variável</h3><div id="plot-mu" class="plot"></div></div><div class="card"><h3>Drift médio absoluto logarítmico da escala</h3><div id="plot-esc" class="plot"></div></div></div>
<div class="controls"><label>Buscar variável <input id="search-summary" placeholder="Nome da feature"></label><label>Tipo <select id="tipo-summary"><option value="">Todos</option><option value="numerica">Numérica</option><option value="FEWF">FEWF</option></select></label><span class="badge" id="count-summary"></span></div><div class="scroll"><table id="table-summary"></table></div></section>
<section class="section" id="folds"><h2>3. Estabilidade entre os 9 folds</h2><p>Valores agregados por fold de validação. Uma rodada aprende os parâmetros nos outros oito folds.</p><div id="plot-folds" class="plot"></div><div class="scroll"><table id="table-folds"></table></div></section>
<section class="section" id="pos"><h2>4. Diagnóstico após padronizar os folds de validação</h2><p>Depois de transformar cada fold usando exclusivamente seu treino 8F, avaliamos a média dos z-scores (referência visual: 0) e seu desvio-padrão (referência visual: 1). Desvios não provam defeito no scaler; indicam variação de distribuição entre treino e validação.</p><div class="twocol"><div class="card"><h3>Média de |μ dos z-scores|, por variável</h3><div id="plot-zmu" class="plot"></div></div><div class="card"><h3>Média de |σ dos z-scores − 1|, por variável</h3><div id="plot-zsd" class="plot"></div></div></div>
<p class="hint">O StandardScaler garante média próxima de zero e variância unitária no <strong>treino transformado</strong>, salvo casos degenerados; isso não é garantido na validação.</p></section>
<section class="section" id="pca"><h2>5. Impacto diagnóstico no PCA (PC1–PC3)</h2><p>PCAs auxiliares são ajustados novamente apenas para <strong>diagnóstico</strong> (F1–F9 global versus cada treino 8F). Sem ajuste de GMM. A comparação de direções usa o valor absoluto do cosseno (ignora troca de sinal) e a similaridade dos subespaços. Para frequências FEWF aprendidas separadamente, o espaço das features pode mudar; interprete os cossenos como descritivos.</p><div id="plot-pca" class="plot"></div><div class="scroll"><table id="table-pca"></table></div></section>
<section class="section" id="dados"><h2>6. Tabela completa: todas as variáveis × 9 folds</h2><p>Contém médias, variâncias, desvios, escalas, medianas de imputação, diferenças e resultados da validação. Não existe corte Top 10: todas as linhas podem ser exibidas e exportadas.</p>
<div class="controls"><label>Buscar variável <input id="search-details" placeholder="Nome da feature"></label><label>Fold <select id="fold-details"><option value="">F1 a F9</option></select></label><label>Tipo <select id="tipo-details"><option value="">Todos</option><option value="numerica">Numérica</option><option value="FEWF">FEWF</option></select></label><span class="badge" id="count-details"></span></div><div class="scroll"><table id="table-details"></table></div></section>
<section class="section" id="metodo"><h2>7. Fórmulas, limitações e rastreabilidade</h2>
<p><strong>Drift da média:</strong> (μ<sub>8F</sub> − μ<sub>9F</sub>) / escala<sub>9F</sub>.</p>
<p><strong>Drift logarítmico da escala:</strong> ln(escala<sub>8F</sub> / escala<sub>9F</sub>).</p>
<p><strong>Variância relativa:</strong> (var<sub>8F</sub> − var<sub>9F</sub>) / var<sub>9F</sub>, indefinida se var<sub>9F</sub> = 0.</p>
<p><strong>PCA:</strong> variância explicada nos dois ajustes; |cosseno| entre direções homólogas PC1/PC2/PC3; similaridade média dos subespaços via valores singulares dos vetores de componentes.</p>
<p><strong>Alcance:</strong> só variáveis efetivamente padronizadas pelo StandardScaler (numéricas + FEWF). Colunas binárias, OHE e cíclicas não entram no diagnóstico das escalas porque são preservadas sem padronização. O encoding e a imputação são aprendidos em cada treino. A referência global é descritiva e não substitui o preprocessamento OOF.</p>
<p><strong>Amostra:</strong> este documento usa exatamente os registros F1–F9 selecionados no experimento de origem. Em ensaio amostral, os resultados não equivalem a uma análise no conjunto integral.</p>
<div class="download"><a href="__CSV_SUM__">CSV — todas as variáveis (resumo)</a><a href="__CSV_DETAIL__">CSV — todos os folds × variáveis</a><a href="__CSV_FOLDS__">CSV — folds</a><a href="__CSV_PCA__">CSV — PCA</a><a href="__META__">Manifesto de auditoria</a></div>
</section>
</main><script>__PLOTLY__</script><script>const PRE01_DATA = __DATA__;</script><script>__JS__</script></body></html>'''
    js=r'''
(() => {
'use strict';
const D=PRE01_DATA; const fmt=x=>x===null||x===undefined||(typeof x==='number'&&!Number.isFinite(x))?'—':typeof x==='number'?x.toFixed(7):String(x);
const num=x=>typeof x==='number'&&Number.isFinite(x);
const $=id=>document.getElementById(id);
$('origem').textContent=`${D.meta.tipo.toUpperCase()} | K=${D.meta.k} | n_init=${D.meta.n_init} | ${D.meta.total_observacoes.toLocaleString('pt-BR')} registros | ${D.meta.total_features} variáveis padronizadas | F10 excluído`;
const det=D.detail, sum=D.summary;
const mean=a=>a.reduce((s,v)=>s+v,0)/(a.length||1);
const metrics=[
 ['Variáveis padronizadas',D.meta.total_features.toString(),'Numéricas e FEWF'],
 ['Observações F1–F9',D.meta.total_observacoes.toLocaleString('pt-BR'),'Amostra do experimento'],
 ['Drift médio |Δμ|',fmt(mean(det.map(r=>Math.abs(r.delta_media_padronizada)))),'Média de todos os folds × variáveis'],
 ['Drift máximo |Δμ|',fmt(Math.max(...det.map(r=>Math.abs(r.delta_media_padronizada)))),'Maior diferença observada'],
 ['Drift médio |ln(escala)|',fmt(mean(det.map(r=>Math.abs(r.delta_log_escala)))),'Média dos módulos'],
 ['Drift máximo |ln(escala)|',fmt(Math.max(...det.map(r=>Math.abs(r.delta_log_escala)))),'Maior diferença observada'],
 ['Pares com |Δμ| ≤ 0,10',fmt(100*det.filter(r=>Math.abs(r.delta_media_padronizada)<=.1).length/det.length)+'%','Critério exploratório'],
 ['Escala 8F / 9F em [0,95; 1,05]',fmt(100*det.filter(r=>Math.exp(r.delta_log_escala)>=.95&&Math.exp(r.delta_log_escala)<=1.05).length/det.length)+'%','Critério exploratório'],
 ['Validação com |média z| ≤ 0,10',fmt(100*det.filter(r=>Math.abs(r.media_z_validacao)<=.1).length/det.length)+'%','Distribuição do fold validado'],
 ['Validação com σz em [0,90; 1,10]',fmt(100*det.filter(r=>r.desvio_z_validacao>=.9&&r.desvio_z_validacao<=1.1).length/det.length)+'%','Distribuição do fold validado']
];
$('cards').innerHTML=metrics.map(([k,v,n])=>`<div class="card"><div class="k">${k}</div><div class="n">${v}</div><div class="hint">${n}</div></div>`).join('');
const configs={
 summary:[['feature','Variável'],['tipo','Tipo'],['media_global_9f','μ 9F'],['variancia_global_9f','Var 9F'],['desvio_global_9f','σ 9F'],['escala_global_9f','Escala 9F'],['mediana_global_9f','Mediana 9F'],['media_dos_9_ajustes_8f','Média μ (8F)'],['media_8f_min','Mín μ (8F)'],['media_8f_max','Máx μ (8F)'],['variancia_dos_9_ajustes_8f','Média Var (8F)'],['variancia_8f_min','Mín Var (8F)'],['variancia_8f_max','Máx Var (8F)'],['escala_dos_9_ajustes_8f','Média escala 8F'],['escala_8f_min','Mín escala 8F'],['escala_8f_max','Máx escala 8F'],['mediana_dos_9_ajustes_8f','Média mediana 8F'],['drift_media_abs_medio','|Δμ| médio'],['drift_media_abs_max','|Δμ| máx'],['drift_escala_log_abs_medio','|Δlnσ| médio'],['drift_escala_log_abs_max','|Δlnσ| máx'],['razao_escala_min','Mín razão escalas'],['razao_escala_max','Máx razão escalas'],['percentual_folds_media_ate_0_1','% folds |Δμ|≤0,1'],['percentual_folds_escala_0_95_a_1_05','% folds escala 5%'],['media_abs_z_validacao','Média |μz| validação'],['media_abs_desvio_z_menos_1','Média |σz−1|'],['desvio_z_validacao_min','Mín σz'],['desvio_z_validacao_max','Máx σz'],['percentual_folds_z_media_ate_0_1','% folds |μz|≤0,1'],['percentual_folds_z_desvio_0_9_a_1_1','% folds σz±10%'],['percentual_medio_faltantes_validacao','% ausentes validação']],
 details:[['fold_validacao','Fold'],['feature','Variável'],['tipo','Tipo'],['n_treino','N treino'],['n_validacao','N validação'],['media_8f','μ 8F'],['media_9f','μ 9F'],['variancia_8f','Var 8F'],['variancia_9f','Var 9F'],['desvio_8f','σ 8F'],['desvio_9f','σ 9F'],['escala_8f','Escala 8F'],['escala_9f','Escala 9F'],['mediana_imputacao_8f','Mediana 8F'],['mediana_imputacao_9f','Mediana 9F'],['delta_media_padronizada','Δμ / escala 9F'],['delta_log_escala','Δln escala'],['delta_variancia_relativa','ΔVar relativa'],['media_z_validacao','μz validação'],['variancia_z_validacao','Var z validação'],['desvio_z_validacao','σz validação'],['faltantes_treino_percentual','% missing treino'],['faltantes_validacao_percentual','% missing validação']],
 folds:[['fold','Fold'],['n_treino','N treino'],['n_validacao','N validação'],['drift_media_abs_medio','Média |Δμ|'],['drift_media_abs_max','Máx |Δμ|'],['drift_log_escala_abs_medio','Média |Δlnσ|'],['drift_log_escala_abs_max','Máx |Δlnσ|'],['percentual_drift_media_ate_0_1','% |Δμ|≤0,1'],['percentual_escala_entre_0_95_e_1_05','% escala ±5%'],['media_abs_z_validacao','Média |μz|'],['media_abs_desvio_z_menos_1','Média |σz−1|']],
 pca:[['fold','Fold'],['componente','Componente'],['var_explicada_ref_9f','Var explicada 9F'],['var_explicada_8f','Var explicada 8F'],['delta_var_explicada','Δ var explicada'],['cosseno_abs_direcao','|Cos direção|'],['similaridade_subespaco_media','Similaridade subespaço']]
};
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function table(id,data,columns,filters=()=>true,countId=null){
 let key=columns[0][0],reverse=false;
 const t=$(id);
 function render(){let filtered=data.filter(filters),sorted=[...filtered].sort((a,b)=>{let x=a[key],y=b[key];return (num(x)&&num(y)?x-y:String(x).localeCompare(String(y),'pt-BR'))*(reverse?-1:1)});
  t.innerHTML='<thead><tr>'+columns.map(([k,l])=>`<th data-key="${k}">${esc(l)}${k===key?(reverse?' ▼':' ▲'):''}</th>`).join('')+'</tr></thead><tbody>'+sorted.map(r=>'<tr>'+columns.map(([k])=>`<td>${esc(fmt(r[k]))}</td>`).join('')+'</tr>').join('')+'</tbody>';
  t.querySelectorAll('th').forEach(el=>el.onclick=()=>{const k=el.dataset.key;reverse=(key===k?!reverse:false);key=k;render()});
  if(countId)$(countId).textContent=`${filtered.length} de ${data.length} linhas`;
 }
 render();return render;
}
const sfilter=()=>sum.filter(r=>r.feature.toLocaleLowerCase('pt-BR').includes($('search-summary').value.toLocaleLowerCase('pt-BR'))&&(!$('tipo-summary').value||r.tipo===$('tipo-summary').value));
const sRender=table('table-summary',sum,configs.summary, r=>sfilter().includes(r),'count-summary');
$('search-summary').oninput=sRender;$('tipo-summary').onchange=sRender;
for(let k=1;k<=9;k++){const op=document.createElement('option');op.value=String(k);op.textContent=`F${k}`;$('fold-details').append(op)}
const dfilter=()=>det.filter(r=>r.feature.toLocaleLowerCase('pt-BR').includes($('search-details').value.toLocaleLowerCase('pt-BR'))&&(!$('fold-details').value||r.fold_validacao===Number($('fold-details').value))&&(!$('tipo-details').value||r.tipo===$('tipo-details').value));
const dRender=table('table-details',det,configs.details,r=>dfilter().includes(r),'count-details');
$('search-details').oninput=dRender;$('fold-details').onchange=dRender;$('tipo-details').onchange=dRender;
table('table-folds',D.folds,configs.folds);table('table-pca',D.pca,configs.pca);
const cfg={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}};
const layout=(title,margin={l:190,r:24,t:22,b:42})=>({paper_bgcolor:'#fff',plot_bgcolor:'#fff',margin,font:{family:'Arial,sans-serif',size:11,color:'#344b65'},xaxis:{title:title,zeroline:true,zerolinecolor:'#9cacc0',gridcolor:'#e5ebf2'},yaxis:{automargin:true,gridcolor:'#f5f7fa'}});
const sorted=[...sum].sort((a,b)=>a.feature.localeCompare(b.feature,'pt-BR'));const names=sorted.map(r=>r.feature);
function bars(id,key,title,color){return Plotly.newPlot(id,[{type:'bar',orientation:'h',x:sorted.map(r=>r[key]),y:names,marker:{color},hovertemplate:'%{y}<br>%{x:.7f}<extra></extra>'}],{...layout(title),height:Math.max(400,names.length*28+110),yaxis:{...layout('').yaxis,autorange:'reversed'}},cfg)}
bars('plot-mu','drift_media_abs_medio','Média |(μ8F−μ9F)/escala9F|','#3975a7');
bars('plot-esc','drift_escala_log_abs_medio','Média |ln(escala8F/escala9F)|','#d27a3b');
bars('plot-zmu','media_abs_z_validacao','Média de |μz| nos 9 folds','#438b7d');
bars('plot-zsd','media_abs_desvio_z_menos_1','Média de |σz−1| nos 9 folds','#9871ba');
const fvals=D.folds.map(r=>`F${r.fold}`);
Plotly.newPlot('plot-folds',[
 {type:'scatter',mode:'lines+markers',x:fvals,y:D.folds.map(r=>r.drift_media_abs_medio),name:'Drift médio |Δμ|',line:{color:'#2f6299'}},
 {type:'scatter',mode:'lines+markers',x:fvals,y:D.folds.map(r=>r.drift_log_escala_abs_medio),name:'Drift médio |Δlnσ|',line:{color:'#d37a43'}}
 ],{...layout('Instabilidade média por fold',{l:64,r:32,t:30,b:55}),height:340,legend:{orientation:'h',y:1.2},xaxis:{title:'Fold de validação'}},cfg);
const pcColors=['#2d5f99','#d47b41','#4a9075'];
let pt=[];for(let c=1;c<=3;c++){const rows=D.pca.filter(r=>r.componente===`PC${c}`);pt.push({type:'scatter',mode:'lines+markers',x:rows.map(r=>`F${r.fold}`),y:rows.map(r=>r.var_explicada_8f),name:`PC${c} — 8F`,line:{color:pcColors[c-1]}});pt.push({type:'scatter',mode:'lines',x:rows.map(r=>`F${r.fold}`),y:rows.map(r=>r.var_explicada_ref_9f),name:`PC${c} — 9F`,line:{color:pcColors[c-1],dash:'dash'}})}
Plotly.newPlot('plot-pca',pt,{...layout('Proporção de variância explicada',{l:70,r:20,t:45,b:45}),height:450,legend:{orientation:'h',y:1.26},xaxis:{title:'Fold de validação'}},cfg);
})();
'''
    html_txt=pre_html.replace('__CSS__',css).replace('__PLOTLY__',plotly_js).replace('__DATA__',serial).replace('__JS__',js)
    for a,b in [('__CSV_SUM__',CSV_SUM),('__CSV_DETAIL__',CSV_DETAIL),('__CSV_FOLDS__',CSV_FOLDS),('__CSV_PCA__',CSV_PCA),('__META__',META)]:html_txt=html_txt.replace(a,b)
    return html_txt


def new_gallery_index(text):
    # Substituir apenas a secao PRE01 legada, preservando as outras secoes.
    old_pattern=r'<!-- PRE01_DIAGNOSTICO_PADRONIZACAO_FOLDS -->.*?</section>\s*'
    text=re.sub(old_pattern,'',text,flags=re.S)
    # Caso galeria tenha sido regenerada automaticamente e contenha cartao PNG.
    png_article=re.compile(r'<article>(?:(?!</article>).)*?PRE01_estabilidade_padronizacao_folds\.png(?:(?!</article>).)*?</article>',re.S)
    text=png_article.sub('',text)
    new_pattern=r'<!-- PRE01_V2_DIAGNOSTICO_INTERATIVO -->.*?</section>\s*'
    text=re.sub(new_pattern,'',text,flags=re.S)
    exig('</body></html>' in text,'Indice da galeria em formato nao reconhecido; nenhuma alteracao feita.')
    card=(NEW_MARKER+'\n<section><h2>PRE01 — Diagnóstico completo da padronização (HTML interativo)</h2>'
      '<p>Parâmetros de todas as variáveis, folds, pós-padronização e estabilidade PCA; apenas desenvolvimento F1–F9.</p>'
      '<div class="grade"><article><a href="oof/'+FILENAME+'" target="_blank" rel="noopener">'
      '<div class="sem-previa">Abrir painel interativo de diagnóstico PRE01</div>'
      '<div>PRE01 — Médias, variâncias, escalas, folds e PCA</div></a></article></div></section>\n')
    return text.replace('</body></html>',card+'</body></html>')


def backup_one(dst,root,backup):
    rel=dst.relative_to(root)
    target=backup/rel
    if dst.is_file():
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(dst,target)
        return target
    return None


def write_atomic(dst,raw):
    dst.parent.mkdir(parents=True,exist_ok=True)
    tmp=dst.with_name(dst.name+'.pre01v2_tmp')
    try:
        with tmp.open('wb') as f:f.write(raw)
        os.replace(tmp,dst)
    finally:
        tmp.unlink(missing_ok=True)


def future_integration(root):
    """Integra PRE01, inclusive em projetos que ja receberam o PRE01 V1."""
    import ast
    p = root / 'src' / 'visualizacao_integrada.py'
    texto = p.read_text(encoding='utf-8')
    padrao = re.compile(
        r'(?m)^(?P<indent>[ \t]*)permitido\s*=\s*arq\.stem\s+in\s+'
        r'\(oof\.ARQUIVOS\[n\]\s+for\s+n\s+in\s+ativos_oof\)[^\n]*$'
    )
    resultados = list(padrao.finditer(texto))
    exig(len(resultados) == 1, 'Filtro OOF inesperado; nenhuma alteracao aplicada.')
    m = resultados[0]
    if ('PRE01_V2_IGNORAR_PNG' in texto and FILENAME in m.group(0)
            and 'permitido = permitido and arq.name != ' + repr(OLD_PNG) in texto):
        ast.parse(texto)
        return p, texto.encode('utf-8')
    espaco = m.group('indent')
    novo = (
        espaco + '# PRE01_V2_IGNORAR_PNG\n'
        + espaco + "permitido=arq.stem in (oof.ARQUIVOS[n] for n in ativos_oof) or arq.name == " + repr(FILENAME) + '\n'
        + espaco + 'permitido = permitido and arq.name != ' + repr(OLD_PNG)
    )
    resultado = texto[:m.start()] + novo + texto[m.end():]
    ast.parse(resultado)
    return p, resultado.encode('utf-8')


def run(root,k,n_init,tipo):
    exig((root/'src'/'orquestracao_oof.py').is_file(),'Raiz do TCC incorreta.')
    os.chdir(root)
    trio=trio_find(root,k,n_init,tipo)
    for t in trio:print(f"[TRIO] PCA {t['dim']}D: {t['folder'].name}",flush=True)
    ids,folds=check_indices(trio)
    cfg=trio[0]['cfg']
    exig(np.all((folds>=1)&(folds<=9)),'F10 detectado indevidamente.')
    parquet=root/cfg['dados']['arquivo']
    print('[DADOS] Auditando Parquet, folds originais e IDs (F10 excluido).',flush=True)
    orig_folds,p_folds,hash_data=orq._identificar_folds(root,cfg,parquet)
    exig(hash_data==trio[0]['meta']['sha256_parquet'],'Hash Parquet diferente da execucao OOF.')
    exig(orq.hash_arquivo(p_folds)==trio[0]['meta']['sha256_folds'],'Hash dos folds mudou.')
    exig(np.array_equal(orig_folds[ids],folds),'IDs/folds originais divergentes.')
    df=orq._ler_desenvolvimento(parquet,ids)
    exig(len(df)==len(ids),'Dados lidos incompletos.')
    X=prep.selecionar_features(df,cfg)
    del df
    print('[PRE01 V2] Calculando todas as variaveis e PCA apenas para diagnostico.',flush=True)
    summary,detail,folds_df,pca=calc(X,folds,cfg)
    meta={'tipo':tipo,'k':k,'n_init':n_init,'total_observacoes':int(len(ids)),
          'total_features':int(len(summary)),'f10_usado':False,
          'mesmos_registros_e_folds':True,
          'fons':'predicoes OOF, folds originais, Parquet, config YAML congelado',
          'experimentos':{f"{t['dim']}D":str(t['folder']) for t in trio},
          'sha256_parquet':hash_data,'sha256_folds':trio[0]['meta']['sha256_folds'],
          'medida_de_desempenho':False,'pca_ajuste_qualificacao':'PCA auxiliar; nao o GMM/OOF',
          'nota_fewf':'FEWF e estimado separadamente em cada treino. Diferenças nos parametros incluem o encoder.',
          'limites_estabilidade':'criterios exploratorios, sem validacao estatistica universal'}
    backup=root/'outputs'/('validacao_PRE01_V2_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True,exist_ok=False)
    htmltxt=build_html(summary,detail,folds_df,pca,meta)
    blobs={FILENAME:htmltxt.encode('utf-8'),
           CSV_SUM:summary.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
           CSV_DETAIL:detail.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
           CSV_FOLDS:folds_df.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
           CSV_PCA:pca.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
           META:(json.dumps(meta,ensure_ascii=False,indent=2)+'\n').encode('utf-8')}
    for name,data in blobs.items():write_atomic(backup/name,data)
    for t in trio:
        _ = new_gallery_index(t['gallery'].read_text(encoding='utf-8'))
    src,source_bytes=future_integration(root)
    edits={src:source_bytes}
    for t in trio:
        view=t['folder']/'vencedor'/'visualizacao'
        for directory in (view/'oof',view/'galeria'/'oof'):
            for name,data in blobs.items():edits[directory/name]=data
        edits[t['gallery']]=new_gallery_index(t['gallery'].read_text(encoding='utf-8')).encode('utf-8')
    # Preparar rollback inclusive de novos arquivos; backups preservados.
    done=[]
    try:
        for dest,data in edits.items():
            had=backup_one(dest,root,backup)
            write_atomic(dest,data)
            done.append((dest,had))
        for t in trio:
            view=t['folder']/'vencedor'/'visualizacao'
            exig((view/'galeria'/'oof'/FILENAME).is_file(),'HTML PRE01 ausente da galeria.')
            exig(NEW_MARKER in t['gallery'].read_text(encoding='utf-8'),'Link PRE01 nao incluido.')
            print(f"[CONCLUIDO] PCA {t['dim']}D: link PRE01 V2 incluido; todas as variaveis.",flush=True)
    except Exception:
        for dest,old in reversed(done):
            if old:shutil.copy2(old,dest)
            else:dest.unlink(missing_ok=True)
        raise
    # Painel central leve; cada HTML e autonomo, sem internet.
    links=''.join('<li><a href="'+html.escape((t['folder']/'vencedor'/'visualizacao'/'galeria'/'oof'/FILENAME).relative_to(root/'outputs').as_posix(),quote=True)+'">PCA '+str(t['dim'])+'D — Abrir diagnóstico PRE01 completo</a></li>' for t in trio)
    panel=root/'outputs'/'ABRIR_PRE01_V2_TRES_GALERIAS.html'
    existing=backup_one(panel,root,backup)
    try:write_atomic(panel,('''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>PRE01 V2 — PCA 1D / 2D / 3D</title><style>body{background:#f5f7fb;font:17px system-ui;color:#20324a;max-width:900px;margin:60px auto}li{margin:20px}a{color:#245c98}</style><h1>PRE01 V2 — Diagnóstico completo</h1><p>Todos os parâmetros, folds e variáveis disponíveis em páginas interativas locais. F1–F9; F10 reservado.</p><ul>'''+links+'</ul></html>').encode('utf-8'))
    except Exception:
        if existing:shutil.copy2(existing,panel)
        else:panel.unlink(missing_ok=True)
        raise
    print('[SUCESSO] PRE01 V2 incluido nas tres galerias; grafico PNG antigo retirado da pagina, arquivo preservado.',flush=True)
    print('PAINEL='+str(panel),flush=True)
    print('BACKUP='+str(backup),flush=True)
    print('F10 intocado; GMM/OOF nao treinados; PCA apenas diagnostico.',flush=True)
    return panel


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raiz',required=True,type=Path)
    ap.add_argument('--k',type=int,default=4)
    ap.add_argument('--n-init',type=int,default=5)
    ap.add_argument('--tipo',choices=['ensaio','oficial'],default='ensaio')
    a=ap.parse_args()
    panel=run(a.raiz.resolve(),a.k,a.n_init,a.tipo)
    return panel

if __name__=='__main__':main()
