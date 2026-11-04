# -*- coding: utf-8 -*-
"""Camada visual F10 3D a partir de scores e embeddings persistidos.

Nenhum metodo de ajuste, OOF ou transform do F10 e chamado aqui.
PCA3D preserva os graficos oficiais interativos. TSNE3D usa apenas
renderizadores compatíveis com TSNE1/TSNE2/TSNE3.
"""
from __future__ import annotations

from hashlib import sha256
from html import escape
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd


def nomes(metodo):
    return ['PC1','PC2','PC3'] if metodo=='pca' else ['TSNE1','TSNE2','TSNE3']


def nome_projecao(metodo):
    return ('12_pca_projecao_' if metodo=='pca' else '13_tsne_mapa_')+'_'.join(nomes(metodo))+'.png'


def graficar_projecao_3d(cfg, dados, destino, metodo, dpi):
    import matplotlib.pyplot as plt
    x=dados['coord'].to_numpy(dtype=float)
    y=np.asarray(dados['y'],dtype=int)
    assert x.ndim==2 and x.shape[1]==3 and len(x)==len(y)
    fig=plt.figure(figsize=(11.2,8.0))
    ax=fig.add_subplot(111,projection='3d')
    rng=np.random.default_rng(42)
    for cl,name,cor,marker in [(0,'Legitimas','#1f77b4','o'),(1,'Fraudes','#d62728','*')]:
        ids=np.flatnonzero(y==cl)
        if len(ids)>3500: ids=np.sort(rng.choice(ids,3500,replace=False))
        ax.scatter(x[ids,0],x[ids,1],x[ids,2],label=f'{name} (n={int((y==cl).sum())})',
                   s=9 if cl==0 else 45,c=cor,marker=marker,
                   alpha=.3 if cl==0 else .82,depthshade=False,rasterized=True)
    for n in range(3):
        if n==0:ax.set_xlabel(nomes(metodo)[n],labelpad=12)
        elif n==1:ax.set_ylabel(nomes(metodo)[n],labelpad=12)
        else:ax.set_zlabel(nomes(metodo)[n],labelpad=12)
    ax.view_init(elev=23,azim=-59)
    ax.legend(loc='upper left',fontsize=9)
    fig.suptitle(f'{metodo.upper()} 3D — Projecao das transacoes F10 (3% real)',
                 fontsize=15,weight='bold',y=.97)
    fig.text(.5,.045,'Projecao congelada em F1–F9; rotulos F10 usados somente para colorir.',
             ha='center',fontsize=9)
    fig.subplots_adjust(left=.02,right=.91,top=.91,bottom=.10)
    file=Path(destino)/nome_projecao(metodo)
    fig.savefig(file,dpi=dpi,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return file


def gerar_graficos_f10_3d(pasta,cfg,info,metodo,model_dir,f10,*,base):
    destino=pasta/'vencedor'/'teste_final'/base.SAIDA_GRAFICOS
    if destino.is_dir():
        base.validar_graficos(pasta,metodo,f10)
        print(f'[{metodo.upper()}] Graficos 3D existentes validados; sem recalcular F10.',flush=True)
        return destino
    base.exigir(not destino.exists(),'Destino de graficos ja existe. Proibido sobrescrever.')
    import matplotlib
    matplotlib.use('Agg')
    from src import visualizacao_oof as vo
    from src import visualizacao_geometrica as geo
    from src import visualizacao_gmm01 as g01
    from src import visualizacao_gmm04 as g04
    from src import visualizacao_gmm07 as g07
    mod=base.modulo_modelo(metodo)
    pacote=mod.carregar_pacote_confiavel(model_dir)
    pred=pd.read_csv(f10/'predicoes_f10.csv')
    cols=nomes(metodo)
    tabela=pd.read_csv(f10/('coordenadas_'+'_'.join(cols)+'_f10.csv'))
    base.exigir(np.array_equal(pred['linha_original'].to_numpy(dtype=np.int64),
                               tabela['linha_original'].to_numpy(dtype=np.int64)),
                'IDs das coordenadas F10 divergentes das previsoes.')
    y=pred['y_true'].to_numpy(dtype=np.uint8)
    p=pred['prob_fraude'].to_numpy(dtype=np.float64)
    x=pd.DataFrame({c:tabela[c].to_numpy(dtype=float) for c in cols})
    tau=float(pacote.limiar_oof)
    met=base.ler_json(f10/'metricas_f10.json')
    v=pd.Series({'ap':met['ap'],'mcc':met['mcc_limiar_oof'],
                 'logloss':met['logloss'],'score':met['score_composto_limiar_oof'],
                 'prevalencia':met['prevalencia']})
    escolha={'limiar_mcc_oof':tau}
    gamma=pacote.gmm.responsabilidades(x)
    from src.probabilidade import calcular_probabilidades
    conferido=calcular_probabilidades(gamma,pacote.risco.riscos_).to_numpy(dtype=np.float64)
    base.exigir(np.max(np.abs(conferido-p))<=1e-6 and
                np.array_equal(conferido>=tau,p>=tau) and
                np.array_equal(conferido>=.5,p>=.5),
                'Scores F10 nao coincidem com as responsabilidades do modelo salvo.')
    data={'coord':x,'y':y,'scores':p,'gamma':gamma,'gmm':pacote.gmm,
          'rh':pacote.risco,'redutor':pacote.pca if metodo=='pca' else pacote.tsne,
          'limiar_oof':tau}
    stage=destino.with_name('.graficos_f10_3d_incompletos_'+uuid.uuid4().hex[:12])
    stage.mkdir(exist_ok=False)
    dpi=int(cfg['visualizacao']['padrao'].get('dpi',180))
    print(f'[{metodo.upper()} 3D] 6 figuras preditivas F10.',flush=True)
    with base.rotulos_preditivos_f10():
        vo._plot_painel_00(cfg,y,p,v,escolha,stage,'png',dpi,True)
        vo._plot_pr(cfg,y,p,v,escolha,stage,'png',dpi,True)
        vo._plot_mcc(cfg,y,p,v,escolha,stage,'png',dpi,True)
        vo._matriz(cfg,y,p,.5,'matriz_confusao_05',stage,'png',dpi,True)
        vo._matriz(cfg,y,p,tau,'matriz_confusao_mcc',stage,'png',dpi,True)
        vo._distribuicao(cfg,y,p,escolha,stage,'png',dpi,True)
    print(f'[{metodo.upper()} 3D] Geometria FULL 95% / projecao F10.',flush=True)
    import contextlib
    with (base.proteger_visualizacoes_tsne_3d() if metodo=='tsne' else contextlib.nullcontext()):
        g01.gerar(cfg,data,stage,'png',dpi,True)
        g04.gerar(cfg,data,stage,'png',dpi,True)
        with base.rotulos_geometria_f10(tau,y,p,metodo):
            avaliacao={'origem':'f10_real_3pct'}
            if metodo=='pca':
                from src import visualizacao_gmm08_3d as g08_3d
                original=g08_3d._origem_mcc
                g08_3d._origem_mcc=lambda av: (
                    ('MCC F10','F10 real (3%); limiar selecionado no OOF')
                    if av.get('origem')=='f10_real_3pct' else original(av))
                try:
                    g07.gerar(cfg,data,escolha,stage,'png',dpi,True,avaliacao=avaliacao)
                    geo._fronteira(cfg,data,escolha,stage,'png',dpi,True,avaliacao=avaliacao)
                finally:
                    g08_3d._origem_mcc=original
            else:
                g07.gerar(cfg,data,escolha,stage,'png',dpi,True,avaliacao=avaliacao)
                geo._fronteira(cfg,data,escolha,stage,'png',dpi,True,avaliacao=avaliacao)
    graficar_projecao_3d(cfg,data,stage,metodo,dpi)
    files=sorted(stage.iterdir())
    pngs=sorted(stage.glob('*.png'))
    expected=set(base.IMAGENS_PREDITIVAS)|{
        f'GMM01_densidade_marginal_3d_{c}.png' for c in cols}|{
        'GMM04_dominancia_3d.png','GMM07_sensibilidade_3d_comparativo.png',
        'GMM08_mapa_risco_3d_comparativo.png',nome_projecao(metodo)}
    base.exigir(expected.issubset({p.name for p in pngs}),
                'Figuras 3D obrigatorias ausentes: '+str(expected-{p.name for p in pngs}))
    base.exigir(len(pngs)>=13,'Quantidade de figuras F10 3D insuficiente.')
    for img in pngs:
        base.exigir(img.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n',
                    'PNG 3D invalido: '+img.name)
    fs={f.name:base.hash_arquivo(f) for f in files if f.is_file()}
    base.salvar_json(stage/'manifesto_graficos_f10_3pct.json',{
        'status':'APROVADA','metodo':metodo.upper(),'n_imagens':len(pngs),
        'imagens':[p.name for p in pngs],
        'sha256_predicoes_f10':base.hash_arquivo(f10/'predicoes_f10.csv'),
        'sha256_arquivos':fs,'sem_fit':True,'limiar_f10_reotimizado':False})
    base.exigir(not destino.exists(),'Destino de graficos apareceu durante escrita.')
    stage.rename(destino)
    base.validar_graficos(pasta,metodo,f10)
    print(f'[{metodo.upper()} 3D] FIGURAS_F10={len(pngs)} (minimo=13)',flush=True)
    return destino


def gerar_galeria_f10_3d(pasta,cfg,metodo,f10,graficos,*,base):
    dest=pasta/'vencedor'/'teste_final'/base.SAIDA_GALERIA
    if dest.is_dir():
        m=base.ler_json(dest/'manifesto_galeria_3pct.json')
        base.exigir(m.get('metodo')==metodo.upper() and
                    m.get('sha256_predicoes_f10')==base.hash_arquivo(f10/'predicoes_f10.csv')
                    and (dest/'index.html').is_file()
                    and base.hash_arquivo(dest/'index.html')==m.get('sha256_html'),
                    'Galeria F10 existente divergente.')
        for nome,digest in m['hashes_png'].items():
            base.exigir((dest/nome).is_file() and base.hash_arquivo(dest/nome)==digest,
                        'Imagem da galeria alterada: '+nome)
        return dest
    base.exigir(not dest.exists(),'Destino galeria existe; nao sobrescrever.')
    stage=dest.with_name('.galeria_f10_3d_incompleta_'+uuid.uuid4().hex[:12])
    stage.mkdir(exist_ok=False)
    imgs=sorted(graficos.glob('*.png'))
    extra_html=sorted(graficos.glob('*.html'))
    for f in imgs+extra_html:
        shutil.copy2(f,stage/f.name)
        base.exigir(base.hash_arquivo(f)==base.hash_arquivo(stage/f.name),
                    'Copia nao identica: '+f.name)
    met=base.ler_json(f10/'metricas_f10.json')
    cards=''.join(f'<div class="card"><strong>{escape(t)}</strong><div>{float(v):.7f}</div></div>'
                  for t,v in [('AP',met['ap']),('MCC tau* OOF',met['mcc_limiar_oof']),
                              ('LogLoss',met['logloss']),('Score composto',met['score_composto_limiar_oof'])])
    parts=[]
    for caption,prefix in [('Indicadores preditivos','0'),('Geometria GMM 3D','GMM'),
                           ('Projecao real 3D','12_' if metodo=='pca' else '13_')]:
        selected=[f for f in imgs if (f.name[0]=='0' and f.name[:2].isdigit()
                  and int(f.name[:2])<=5) ] if prefix=='0' else [f for f in imgs if f.name.startswith(prefix)]
        if not selected:continue
        figure='\n'.join(f'<figure><a href="{escape(f.name,quote=True)}" target="_blank"><img loading="lazy" src="{escape(f.name,quote=True)}" alt="{escape(f.name)}"></a><figcaption>{escape(f.name)}</figcaption></figure>' for f in selected)
        parts.append(f'<section><h2>{caption}</h2><div class="figs">{figure}</div></section>')
    interactive=''.join(f'<li><a href="{escape(f.name,quote=True)}" target="_blank" rel="noopener">{escape(f.name)}</a></li>' for f in extra_html)
    links=(f'<section><h2>Visualizacoes 3D interativas</h2><ul>{interactive}</ul></section>' if extra_html else '')
    html=f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>F10 {metodo.upper()} GMM 3D — 3% real</title>
<style>body{{background:#f5f7fb;color:#1a2e46;font:16px system-ui,sans-serif;margin:0}}main{{max-width:1450px;margin:auto;padding:24px}}
.header{{padding:22px;background:#12345a;color:white;border-radius:12px}}.cards,.figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}}
.cards{{grid-template-columns:repeat(auto-fit,minmax(200px,1fr));margin:16px 0}}.card,section{{background:white;border:1px solid #d4dfea;border-radius:10px;padding:17px;margin:16px 0}}
.card strong{{font-size:13px;display:block}}.card div{{font-size:24px;font-weight:700}}figure{{margin:0;border:1px solid #dce2ea;border-radius:8px;overflow:hidden}}
figure img{{width:100%;object-fit:contain}}figcaption{{font-size:12px;padding:10px;word-break:break-word}}a{{color:#2563a6}}</style></head><body><main>
<div class="header"><h1>{metodo.upper()} + GMM 3D — F10 real (3%)</h1><p>Mesmo F10 estratificado; avaliacao exploratoria e limiar congelado no OOF.</p></div>
<div class="cards">{cards}</div><p><b>F10:</b> {met['n']} observacoes, {met['n_fraudes']} fraudes. <b>tau*:</b> {float(met['limiar_mcc_oof_congelado']):.7f}</p>
<p><a href="../../visualizacao/galeria/index.html">Abrir galeria OOF F1–F9</a> · <a href="../{base.SAIDA_F10}/metricas_f10.json">Metricas F10 JSON</a> · <a href="../{base.SAIDA_F10}/predicoes_f10.csv">Predicoes F10 CSV</a></p>
{''.join(parts)}{links}
<section><h2>PRE01 — Graficos e descricao da padronizacao</h2><p>Diagnostico F1–F9. Nao ajusta nem recalcula o F10; mesmo conjunto de observacoes para PCA e t-SNE.</p><a href="../../visualizacao/galeria/oof/PRE01_diagnostico_padronizacao_completo.html" target="_blank">Abrir diagnostico PRE01 interativo</a></section>
</main></body></html>'''
    (stage/'index.html').write_text(html,encoding='utf-8')
    base.salvar_json(stage/'manifesto_galeria_3pct.json',{
        'status':'APROVADA','metodo':metodo.upper(),
        'sha256_predicoes_f10':base.hash_arquivo(f10/'predicoes_f10.csv'),
        'sha256_html':base.hash_arquivo(stage/'index.html'),
        'hashes_png':{f.name:base.hash_arquivo(f) for f in stage.glob('*.png')},
        'hashes_html_3d':{f.name:base.hash_arquivo(f) for f in extra_html}})
    base.exigir(not dest.exists(),'Galeria final apareceu inesperadamente.')
    stage.rename(dest)
    print(f'[{metodo.upper()} 3D] GALERIA_F10={dest/"index.html"}',flush=True)
    return dest
