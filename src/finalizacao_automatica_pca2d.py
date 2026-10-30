# -*- coding: utf-8 -*-
"""Finalizacao PCA 2D para o comando original OOF; depende do nucleo PCA 1D aprovado.

Reutiliza apenas funcoes independentes de dimensionalidade do PCA 1D. Nao reotimiza
limiar no F10. F10 previamente acessado: execucoes repetidas EXPLORATORIAS.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
import yaml

from src import finalizacao_automatica_pca1d as comum

ROOT = comum.ROOT
exigir = comum.exigir
ler_json = comum.ler_json
sha = comum.sha
salvar_json_novo = comum.salvar_json_novo


def validar_automacao(config):
    exigir(comum.parametros_ok(config),
           'PCA 2D exige fracao=.1, GMM K4, n_init5, FULL, RUS16, seed42 e F10 reservado.')
    exigir(config.get('pca', {}).get('n_components') == [2],
           'Automacao PCA 2D exige pca.n_components: [2].')
    for arq in ('src/finalizacao_automatica_pca1d.py',
                'src/treinamento_final_pca.py', 'src/avaliacao_final_pca.py',
                'src/pre01_v2_diagnostico.py', 'src/visualizacao_geometrica.py',
                'src/visualizacao_gmm01.py', 'src/visualizacao_gmm04.py',
                'src/visualizacao_gmm07.py', 'scripts/execucao/GERAR_PRE01_F10_PCA1D.py',
                'scripts/execucao/EXTRAIR_COORDENADAS_PCA1D_F10.py'):
        exigir((ROOT/arq).is_file(), f'Dependencia ausente: {arq}')
    exigir((ROOT/config['dados']['arquivo']).is_file(), 'Parquet nao encontrado.')
    return True


def conferir_oof(folder):
    from src import orquestracao_oof as orq
    from src.folds import hash_arquivo
    m=ler_json(folder/'manifesto.json')
    esc=ler_json(folder/'vencedor/metricas/melhor_configuracao.json')
    c=yaml.safe_load((folder/'configuracao.yaml').read_text(encoding='utf-8-sig'))
    validar_automacao(c)
    exigir(m.get('status')=='concluido_06_03' and int(m.get('registros_desenvolvimento',-1))>0,
           'OOF PCA 2D nao concluido.')
    e=esc['configuracao']
    exigir(e['reducao']['n_components']==2 and e['gmm']['n_init']==5
           and e['gmm']['n_components']==4 and esc['teste_final_usado'] is False,
           'Vencedor OOF nao corresponde ao PCA 2D/K4/n_init5.')
    with np.load(folder/'vencedor/oof/predicoes.npz',allow_pickle=False) as z:
        ids=np.asarray(z['linha_original'], dtype=np.int64)
        folds=np.asarray(z['fold'],dtype=np.uint8)
        y=np.asarray(z['y_true'],dtype=np.uint8)
    exigir(len(ids)==int(m['registros_desenvolvimento']) and np.all(np.diff(ids)>0)
           and set(np.unique(folds))==set(range(1,10)), 'OOF ids/folds invalidos.')
    parquet=ROOT/c['dados']['arquivo']
    orig, arq_folds, h=orq._identificar_folds(ROOT,c,parquet)
    exigir(h==m['sha256_parquet'] and hash_arquivo(arq_folds)==m['sha256_folds'],
           'Base ou folds diferentes do OOF.')
    exigir(np.array_equal(orig[ids],folds),'Folds OOF desalinhados.')
    return {'config':c,'escolha':esc,'manifesto':m,'ids':ids,'folds':folds,
            'y':y,'orig_folds':orig,'parquet':parquet}


def gerar_pre01_oof(folder,state):
    from src import orquestracao_oof as orq
    from src import preprocessing as prep
    from src import pre01_v2_diagnostico as pre
    vis=folder/'vencedor/visualizacao'
    destino=vis/'oof'
    index=vis/'galeria/index.html'
    if (destino/pre.FILENAME).is_file():
        print('[OOF PRE01] Ja existe; preservado.',flush=True)
        return
    exigir(index.is_file(), 'Galeria OOF ausente.')
    df=orq._ler_desenvolvimento(state['parquet'],state['ids'])
    X=prep.selecionar_features(df,state['config'])
    summary,detail,folds_df,pca=pre.calc(X,state['folds'],state['config'])
    meta={'tipo':'ensaio','k':4,'n_init':5,'total_observacoes':int(len(X)),
          'total_features':len(summary),'f10_usado':False,
          'mesmos_registros_e_folds':True,
          'experimentos':{'2D':folder.relative_to(ROOT).as_posix()},
          'sha256_parquet':state['manifesto']['sha256_parquet'],
          'sha256_folds':state['manifesto']['sha256_folds'],
          'medida_de_desempenho':False,
          'pca_ajuste_qualificacao':'PCA auxiliar diagnostico F1-F9, nao modelo F10'}
    blobs={pre.FILENAME:pre.build_html(summary,detail,folds_df,pca,meta).encode('utf-8'),
           pre.CSV_SUM:summary.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
           pre.CSV_DETAIL:detail.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
           pre.CSV_FOLDS:folds_df.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
           pre.CSV_PCA:pca.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
           pre.META:(json.dumps(meta,ensure_ascii=False,indent=2)+'\n').encode('utf-8')}
    for pasta in (vis/'oof',vis/'galeria/oof'):
        pasta.mkdir(parents=True,exist_ok=True)
        for nome,blob in blobs.items():
            with (pasta/nome).open('xb') as f: f.write(blob)
    original=index.read_text(encoding='utf-8')
    novo=pre.new_gallery_index(original)
    exigir(novo!=original,'PRE01 OOF nao inserido na galeria.')
    index.write_text(novo,encoding='utf-8')
    print('[OOF PRE01] Diagnostico 2D completo.',flush=True)


def treinar_final(folder,state):
    from src import orquestracao_oof as orq
    from src.treinamento_final_pca import ajustar_final_f1_f9, salvar_pacote
    dest=folder/'vencedor/teste_final/modelo_final_f1_f9'
    exigir(not dest.exists(),'Modelo final existente; nao sobrescrever.')
    df=orq._ler_desenvolvimento(state['parquet'],state['ids'])
    exigir(np.array_equal(df[state['config']['dados']['coluna_alvo']].to_numpy(),state['y']),
           'TARGET desenvolvimento divergente.')
    m=state['manifesto']
    prov={'sha256_parquet':m['sha256_parquet'],'sha256_folds':m['sha256_folds'],
          'sha256_yaml':m['sha256_yaml'],
          'experimento_oof':folder.relative_to(ROOT).as_posix(),
          'sha256_predicoes_oof':sha(folder/'vencedor/oof/predicoes.npz')}
    pacote=ajustar_final_f1_f9(df,state['folds'],state['config'],state['escolha'],prov,
                               registrar_historico_em=True)
    exigir(pacote.pca.modelo_.n_components_==2, 'Modelo final nao tem 2 componentes.')
    salvar_pacote(pacote,dest)
    print('[FINAL F1-F9] PCA 2D congelado e salvo.',flush=True)
    return pacote


def preparar_geometria(folder,state,pacote,df,ids):
    from scripts.execucao.EXTRAIR_COORDENADAS_PCA1D_F10 import resumo_preprocessamento
    dest=folder/'vencedor/teste_final/dados_geometria'
    al=state['config']['dados']['coluna_alvo']
    excl=[state['config']['dados'].get('coluna_id'),
          *state['config']['dados'].get('colunas_excluir',[])]
    x=df.drop(columns=[al,*[n for n in excl if n]],errors='ignore').copy()
    exigir(list(x.columns)==list(pacote.encoder.colunas_), 'Features F10 divergentes das do modelo.')
    z=pacote.imputador.transform(pacote.encoder.transform(x))
    pc=pacote.pca.transform(z)
    exigir(list(pc.columns)==['PC1','PC2'] and len(pc)==len(ids),
           'Modelo congelado nao projetou 2 PCs.')
    exigir(np.isfinite(pc.to_numpy(dtype=float)).all(), 'PCs contem valores invalidos.')
    resumo=resumo_preprocessamento(pacote,x,z)
    coords=pd.DataFrame({'linha_original':ids,'PC1':pc['PC1'].to_numpy(dtype=float),
                         'PC2':pc['PC2'].to_numpy(dtype=float)})
    dest.mkdir(parents=True,exist_ok=False)
    coords.to_csv(dest/'coordenadas_PC1_PC2_f10.csv',index=False,float_format='%.17g')
    resumo.to_csv(dest/'resumo_padronizacao_f10.csv',index=False,float_format='%.17g')
    salvar_json_novo(dest/'manifesto_geometria_f10.json',{
        'dimensao':2,'n':len(ids),'tipo':'F10_EXPLORATORIO_REPETIDO_NOVO_MODELO',
        'limiar_mcc_oof_congelado':pacote.limiar_oof,
        'sha256_coordenadas':sha(dest/'coordenadas_PC1_PC2_f10.csv'),
        'sha256_resumo':sha(dest/'resumo_padronizacao_f10.csv'),
        'sem_ajuste_pca_f10':True,'sem_ajuste_scaler_f10':True})
    return coords,resumo


def dados_geometria(pacote,tab,coords):
    c=coords[['PC1','PC2']].copy()
    gamma=pacote.gmm.modelo_.predict_proba(c.to_numpy(dtype=float))
    return {'coord':c,'y':tab['y_true'].to_numpy(dtype=np.uint8),
            'scores':tab['prob_fraude'].to_numpy(dtype=float),
            'gamma':pd.DataFrame(gamma,columns=list(pacote.gmm.saidas_)),
            'gmm':pacote.gmm,'rh':pacote.risco,
            'redutor':pacote.pca,'limiar_oof':pacote.limiar_oof}


def desenhar_geometria(folder,state,pacote,tab,coords,met):
    from src import visualizacao_geometrica as geo
    from src import visualizacao_gmm01 as gmm01
    from src import visualizacao_gmm04 as gmm04
    from src import visualizacao_gmm07 as gmm07
    from src import visualizacao_oof as vo
    from matplotlib import pyplot as plt
    from matplotlib.text import Text
    cfg=state['config'];dpi=int(cfg['visualizacao']['padrao'].get('dpi',180))
    figs=folder/'vencedor/teste_final/graficos_gerados'
    dados=dados_geometria(pacote,tab,coords)
    ids=coords['linha_original'].to_numpy(dtype=np.int64)
    geo._pca_projecao_2d3d(cfg,dados,figs,'png',dpi,True,
                          cfg['visualizacao']['graficos']['pca_projecoes'])
    csv=figs/'12_pca_projecao_PC1_PC2.csv'
    if csv.is_file():
        t=pd.read_csv(csv)
        if 'risco_estimado_treino' in t:
            t=t.rename(columns={'risco_estimado_treino':'prob_fraude_f10'})
        t.insert(0,'linha_original_f10',ids)
        t.to_csv(csv,index=False,float_format='%.17g')
    # Contraparte do grafico PCA_OBSERVACOES_PC1 do 1D, usando os mesmos estilos.
    idx=geo._amostra_plot(len(ids),dados['y'],cfg,
                         int(cfg['visualizacao']['geometria'].get('pontos_maximos_amostra',20000)))
    fig,ax=plt.subplots(figsize=(11.5,7.5))
    geo._scatter(ax,dados['coord'].to_numpy(dtype=float),dados['y'],idx,cfg,(0,1))
    v=pacote.pca.modelo_.explained_variance_ratio_
    ax.set(xlabel=f'PC1 (variancia {v[0]:.1%})',
           ylabel=f'PC2 (variancia {v[1]:.1%})',title='OBSERVACOES F10 — PCA 2D')
    ax.grid(alpha=.16,linestyle=':')
    fig.tight_layout()
    vo._salvar(fig,figs/'PCA_OBSERVACOES_PC1_PC2.png',dpi=dpi)
    pd.DataFrame({'linha_original_f10':ids[idx],
                  'PC1':coords['PC1'].to_numpy()[idx],
                  'PC2':coords['PC2'].to_numpy()[idx],
                  'classe_real':dados['y'][idx]}).to_csv(
                      figs/'PCA_OBSERVACOES_PC1_PC2.csv',index=False,float_format='%.17g')
    # Metricas ja salvas na avaliacao F10; nenhuma escolha nova de tau.
    mc=met['comparacao_limiares']
    mcc05=float(mc['tau_0_50']['mcc'])
    mccstar=float(mc['tau_mcc_oof']['mcc'])
    exigir(mccstar==float(met['mcc_limiar_oof']) and
           float(mc['tau_mcc_oof']['limiar'])==float(pacote.limiar_oof),
           'MCC e tau F10 nao conferem com resultados congelados.')
    avaliacao={'origem':'f10_amostral_congelado'}
    salvar=geo._guardar
    rotulo=gmm07._rotulo_mcc
    avaliar=geo._avaliar_cortes_oof
    def cortes(obj,tau):
        exigir(tau==pacote.limiar_oof and obj.get('origem') in
               ('f10_amostral_congelado','oof'), 'Origem ou limiar geometricos invalidos.')
        return mcc05,mccstar
    def rotulo_f10(obj):
        exigir(obj.get('origem')=='f10_amostral_congelado','MCC GMM07 sem origem F10.')
        return 'MCC F10','MCC do F10 exploratorio; tau* definido exclusivamente em OOF.'
    def guardar(fig,destino,nome,ext,dpi):
        for objeto in fig.findobj(match=Text):
            t=objeto.get_text()
            novo=(t.replace('MCC OOF','MCC F10')
                   .replace('LIMIAR ÓTIMO DO MCC','LIMIAR τ* CONGELADO (OOF)')
                   .replace('Limiar ótimo','Limiar τ* congelado')
                   .replace('F1–F9 (desenvolvimento); F10 não utilizado',
                            'GMM congelado F1–F9; observações F10')
                   .replace('F10 não utilizado','F10 exploratório repetido'))
            if t!=novo:objeto.set_text(novo)
        return salvar(fig,destino,nome,ext,dpi)
    geo._guardar=guardar
    geo._avaliar_cortes_oof=cortes
    gmm07._rotulo_mcc=rotulo_f10
    try:
        gmm01.gerar(cfg,dados,figs,'png',dpi,True)
        gmm04.gerar(cfg,dados,figs,'png',dpi,True)
        escolha={'limiar_mcc_oof':pacote.limiar_oof}
        gmm07.gerar(cfg,dados,escolha,figs,'png',dpi,True,avaliacao=avaliacao)
        # A rotina GMM08 original so aceita origem oof/simulado; os rotulos e
        # valores sao corrigidos antes de salvar, sem mudar o codigo do 1D.
        geo._fronteira(cfg,dados,escolha,figs,'png',dpi,True,
                      avaliacao={'origem':'oof'})
    finally:
        geo._guardar=salvar
        geo._avaliar_cortes_oof=avaliar
        gmm07._rotulo_mcc=rotulo
    print('[F10] Geometria 2D completa; limiar congelado em OOF.',flush=True)


def histograma_pc2(coords,bins=60):
    x=coords['PC2'].to_numpy(dtype=float)
    exigir(len(x)>0 and np.isfinite(x).all(), 'PC2 invalida.')
    mi,ma=float(x.min()),float(x.max())
    e=np.linspace(mi-.5,ma+.5,bins+1) if mi==ma else np.linspace(mi,ma,bins+1)
    freq,ed=np.histogram(x,bins=e)
    exigir(int(freq.sum())==len(x),'Histograma PC2 nao conserva observacoes.')
    return pd.DataFrame({'pc2_inicio':ed[:-1], 'pc2_fim':ed[1:],
                         'pc2_centro':(ed[:-1]+ed[1:])/2,
                         'contagem_f10':freq})


def gerar_pre01(folder,state,resumo,coords,pacote,ids):
    from scripts.execucao import GERAR_PRE01_F10_PCA1D as pre
    rel=folder/'vencedor/teste_final/diagnosticos/PRE01_F10'
    exigir(not rel.exists(),'PRE01 ja existente.')
    tab=pre.enriquecer(resumo)
    hist1=pre.histograma_pc1(coords)
    hist2=histograma_pc2(coords)
    meta={'versao':'PRE01_F10_PCA2D_TESTE_COMPLETO_V1',
          'n_f10':len(ids),'n_treino_f1_f9':len(state['ids']),
          'pca_dimensao':2,'variaveis_padronizadas':len(tab),
          'tau_star_oof':pacote.limiar_oof,'carater':'AMOSTRAL_EXPLORATORIO_REPETIDO',
          'sem_ajustes_preprocessamento_f10':True}
    rel.mkdir(parents=True,exist_ok=False)
    tab.to_csv(rel/pre.TABELA_ARQ,index=False,float_format='%.17g')
    hist1.to_csv(rel/pre.HIST_ARQ,index=False,float_format='%.17g')
    hist2.to_csv(rel/'PRE01_F10_PC2_histograma.csv',index=False,float_format='%.17g')
    oof=folder/'vencedor/visualizacao/galeria/oof/PRE01_diagnostico_padronizacao_completo.html'
    link=os.path.relpath(oof,rel).replace('\\','/') if oof.is_file() else None
    pagina=pre.html_pre01(tab,hist1,coords['PC1'].to_numpy(dtype=float),meta,link_oof=link)
    # O PRE01 aprovado e reutilizado; adiciona-se apenas o painel interativo PC2.
    import json as _json
    payload=_json.dumps({'x':hist2['pc2_centro'].to_list(),
                         'y':hist2['contagem_f10'].to_list()},ensure_ascii=False)
    sec='''<section class="section"><h2>PC2 — Distribuição F10 com PCA congelado</h2>
<p>A segunda componente foi projetada com o PCA ajustado em F1–F9, sem refit no F10.</p>
<div id="pre01-f10-pc2" class="plot"></div></section>'''
    js='''<script>const P2=__PC2_DATA__; Plotly.newPlot('pre01-f10-pc2',
[{type:'bar',x:P2.x,y:P2.y,marker:{color:'#438b7d'}}],
{paper_bgcolor:'#fff',plot_bgcolor:'#fff',height:390,xaxis:{title:'PC2 (PCA F1–F9 congelado)'},
yaxis:{title:'Frequência F10'},margin:{l:65,r:25,t:25,b:55}},
{responsive:true,displaylogo:false});</script>'''.replace('__PC2_DATA__',payload)
    exigir(pagina.count('</main>')==1 and pagina.count('</body>')==1,
           'PRE01 original alterado: revisao necessaria.')
    pagina=pagina.replace('</main>',sec+'</main>')
    pagina=pagina.replace('</body>',js+'</body>')
    pagina=pagina.replace('PCA 1D | F10 amostral','PCA 2D | F10 amostral')
    (rel/pre.PRE01_ARQ).write_text(pagina,encoding='utf-8')
    salvar_json_novo(rel/pre.MANIFESTO_ARQ,{
        **meta,'origem_oof':folder.relative_to(ROOT).as_posix(),
        'csv_padronizacao':pre.TABELA_ARQ,
        'histograma_pc1':pre.HIST_ARQ,
        'histograma_pc2':'PRE01_F10_PC2_histograma.csv',
        'sha256_html':sha(rel/pre.PRE01_ARQ),
        'sha256_csv':sha(rel/pre.TABELA_ARQ),
        'sha256_pc1':sha(rel/pre.HIST_ARQ),
        'sha256_pc2':sha(rel/'PRE01_F10_PC2_histograma.csv')})
    print('[F10 PRE01] PCA 2D com PC1 e PC2.',flush=True)


GRAFICOS = (
    *comum.GRAFICOS[:6],
    ('12_pca_projecao_PC1_PC2.png','12 — Projecao PCA PC1 x PC2'),
    ('PCA_OBSERVACOES_PC1_PC2.png','PCA — Observacoes PC1 x PC2'),
    ('GMM01_densidade_marginal_2d_PC1.png','GMM01 — Densidade marginal PC1'),
    ('GMM01_densidade_marginal_2d_PC2.png','GMM01 — Densidade marginal PC2'),
    ('GMM01_densidade_conjunta_2d_PC1_PC2.png','GMM01 — Densidade conjunta PC1 x PC2'),
    ('GMM01_densidade_conjunta_superficie_PCA2D.png','GMM01 — Superficie conjunta 3D'),
    ('GMM04_dominancia_2d.png','GMM04 — Dominancia gaussiana 2D'),
    ('GMM07_sensibilidade_2d_comparativo.png','GMM07 — Sensibilidade 2D'),
    ('GMM08_mapa_risco_2d_comparativo.png','GMM08 — Risco e elipses 95%'),
)


def galeria_final(folder,ids,pacote,met):
    dest=folder/'vencedor/teste_final'
    figs=dest/'graficos_gerados';gal=dest/'galeria'
    for nome,_ in GRAFICOS:
        exigir((figs/nome).is_file(),f'Figura obrigatoria faltando: {nome}')
    pre='diagnosticos/PRE01_F10/PRE01_F10_diagnostico_padronizacao_completo.html'
    exigir((dest/pre).is_file(),'PRE01 F10 2D nao encontrado.')
    cards=''.join(f'<article><a href="../graficos_gerados/{escape(n,quote=True)}" target="_blank">'
                  f'<img src="../graficos_gerados/{escape(n,quote=True)}" loading="lazy" alt="{escape(label)}">'
                  f'<p>{escape(label)}</p></a></article>' for n,label in GRAFICOS)
    arquivos=['../predicoes_f10.csv','../metricas_f10.json',
              '../dados_geometria/coordenadas_PC1_PC2_f10.csv',
              '../dados_geometria/resumo_padronizacao_f10.csv',
              '../modelo_final_f1_f9/manifesto_modelo_final.json',
              '../'+pre]
    links=''.join(f'<a href="{escape(a,quote=True)}">{escape(Path(a).name)}</a>' for a in arquivos)
    html=f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PCA 2D F10 — galeria completa</title><style>body{{font:15px system-ui,sans-serif;background:#f5f7fa;color:#172236;max-width:1400px;margin:auto;padding:20px}}
.grade{{display:grid;grid-template-columns:repeat(auto-fit,minmax(315px,1fr));gap:16px}}article{{background:#fff;border:1px solid #dae1e9;border-radius:10px;padding:12px;box-shadow:0 2px 8px #0001}}article img{{width:100%;aspect-ratio:4/3;object-fit:contain}}
a{{color:#285783;text-decoration:none}}article p{{font-weight:600}}.aviso{{background:#e8edf4;border-left:4px solid #355877;padding:14px}}.links{{display:flex;gap:16px;flex-wrap:wrap}}</style></head><body>
<h1>PCA 2D / GMM K4 n_init5 — Teste final F10</h1><p class="aviso">Amostra F10 de {len(ids)} observações, 10%, EXPLORATÓRIO repetido. F10 não é nova validação independente. τ* = {pacote.limiar_oof:.17g}, escolhido em OOF F1–F9. Modelo, encoder, scaler e PCA congelados.</p>
<p>F10: AP={met['ap']:.7f} | MCC em τ*={met['mcc_limiar_oof']:.7f} | LogLoss={met['logloss']:.7f} | Score={met['score_composto_limiar_oof']:.7f}</p>
<p><a href="../../visualizacao/galeria/index.html">Galeria OOF</a> · <a href="../{pre}">PRE01 F10 interativo</a></p>
<div class="grade">{cards}</div><h2>Resultados e proveniência</h2><div class="links">{links}</div></body></html>'''
    gal.mkdir(parents=True,exist_ok=True)
    with (gal/'index.html').open('x',encoding='utf-8') as f:f.write(html)
    salvar_json_novo(dest/'manifesto_teste_completo_pca2d_f10.json',{
        'tipo':'TESTE_COMPLETO_PCA2D_10PCT_EXPLORATORIO_REPETIDO',
        'n_f10':len(ids),'n_dev':pacote.n_treino_original,
        'limiar_congelado_oof':pacote.limiar_oof,
        'limiar_otimizado_no_f10':False,
        'modelo_sha256':sha(dest/'modelo_final_f1_f9/modelo_final_f1_f9.joblib'),
        'predicoes_sha256':sha(dest/'predicoes_f10.csv'),
        'metricas_sha256':sha(dest/'metricas_f10.json'),
        'graficos_sha256':{n:sha(figs/n) for n,_ in GRAFICOS},
        'pre01':pre,'galeria':'galeria/index.html'})
    print('[GALERIA PCA 2D]',gal/'index.html',flush=True)


def concluir_ensaio(folder):
    from src.treinamento_final_pca import carregar_pacote_confiavel
    with comum.bloqueio():
        folder=Path(folder).resolve()
        exigir(folder.is_relative_to((ROOT/'outputs').resolve()),
               'O ensaio deve estar dentro de outputs.')
        state=conferir_oof(folder)
        gerar_pre01_oof(folder,state)
        dest=folder/'vencedor/teste_final'
        modelo_dir=dest/'modelo_final_f1_f9'
        if (modelo_dir/'modelo_final_f1_f9.joblib').is_file():
            exigir((modelo_dir/'manifesto_modelo_final.json').is_file(),
                   'Modelo parcial; preservar e auditar.')
            pacote=carregar_pacote_confiavel(modelo_dir)
            exigir(pacote.pca.modelo_.n_components_==2,'Retomada encontrou PCA de outra dimensao.')
            print('[RETOMADA] Modelo PCA 2D existente.',flush=True)
        else:
            pacote=treinar_final(folder,state)
        ids=comum.selecionar_amostra_f10(state,folder)
        if (dest/'predicoes_f10.csv').is_file():
            exigir((dest/'metricas_f10.json').is_file(),
                   'Predicoes parciais; F10 nao pode ser reavaliado.')
            tab=pd.read_csv(dest/'predicoes_f10.csv')
            exigir(np.array_equal(tab['linha_original'].to_numpy(dtype=np.int64),ids),
                   'F10 de outra amostra.')
            met=ler_json(dest/'metricas_f10.json')
            df=comum.ler_linhas_f10(state['parquet'],ids)
        else:
            df,tab,met=comum.avaliar_f10(folder,state,pacote,ids)
        exigir(float(met['limiar_mcc_oof_congelado'])==float(state['escolha']['limiar_mcc_oof'])
               and met['limiar_otimizado_no_f10'] is False,
               'F10 alterou o limiar OOF.')
        geo=dest/'dados_geometria'
        coord=geo/'coordenadas_PC1_PC2_f10.csv'
        if coord.is_file():
            coords=pd.read_csv(coord)
            resumo=pd.read_csv(geo/'resumo_padronizacao_f10.csv')
            exigir(np.array_equal(coords['linha_original'].to_numpy(dtype=np.int64),ids),
                   'Coordenadas PCA 2D de outra amostra.')
        else:
            exigir(not geo.exists(),'Geometria parcial; revisar antes de retomar.')
            coords,resumo=preparar_geometria(folder,state,pacote,df,ids)
        figs=dest/'graficos_gerados'
        if figs.exists():
            for nome,_ in GRAFICOS:
                exigir((figs/nome).is_file(),f'Figuras parciais; revisar: {nome}')
            print('[RETOMADA] Figuras ja existentes.',flush=True)
        else:
            comum.desenhar_preditivos(folder,state,tab,met)
            desenhar_geometria(folder,state,pacote,tab,coords,met)
        pre=dest/'diagnosticos/PRE01_F10'
        if not pre.exists():
            gerar_pre01(folder,state,resumo,coords,pacote,ids)
        else:
            exigir((pre/'PRE01_F10_diagnostico_padronizacao_completo.html').is_file(),
                   'PRE01 F10 parcial.')
        if not (dest/'galeria/index.html').exists():
            galeria_final(folder,ids,pacote,met)
        else:
            print('[RETOMADA] Galeria presente; nao sobrescrever.',flush=True)
        from src import orquestracao_oof as orq
        m=ler_json(folder/'manifesto.json')
        m['pendente_em_outros_passos']=[]
        m['teste_final']={'status':'concluido','dimensao_pca':2,
                          'carater':'AMOSTRAL_EXPLORATORIO_F10_JA_UTILIZADO',
                          'n_f10':len(ids),'sem_reotimizar_tau':True,
                          'galeria':'vencedor/teste_final/galeria/index.html',
                          'manifesto':'vencedor/teste_final/manifesto_teste_completo_pca2d_f10.json'}
        orq._salvar_json(folder/'manifesto.json',m)
        print('[CONCLUIDO] PCA 2D OOF + F10 em',folder,flush=True)
        print('[ABRIR]',dest/'galeria/index.html',flush=True)
