# -*- coding: utf-8 -*-
"""Finalizacao PCA 3D para o comando OOF, preservando modulos 1D e 2D.

Somente teste F10 exploratorio; modelo/limiar congelados em F1-F9.
Usa os modulos 3D oficiais para PNGs e HTMLs interativos.
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
           'PCA 3D exige fracao=.1, GMM K4, n_init5, FULL, RUS16, seed42 e F10 reservado.')
    exigir(config.get('pca', {}).get('n_components') == [3],
           'Automacao PCA 3D exige pca.n_components: [3].')
    for arq in ('src/finalizacao_automatica_pca1d.py',
                'src/treinamento_final_pca.py', 'src/avaliacao_final_pca.py',
                'src/pre01_v2_diagnostico.py', 'src/visualizacao_geometrica.py',
                'src/visualizacao_gmm01.py', 'src/visualizacao_gmm04.py',
                'src/visualizacao_gmm07.py',
                'src/visualizacao_gmm01_3d.py', 'src/visualizacao_gmm04_3d.py',
                'src/visualizacao_gmm07_3d.py', 'src/visualizacao_gmm08_3d.py', 'scripts/execucao/GERAR_PRE01_F10_PCA1D.py',
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
           'OOF PCA 3D nao concluido.')
    e=esc['configuracao']
    exigir(e['reducao']['n_components']==3 and e['gmm']['n_init']==5
           and e['gmm']['n_components']==4 and esc['teste_final_usado'] is False,
           'Vencedor OOF nao corresponde ao PCA 3D/K4/n_init5.')
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
          'experimentos':{'3D':folder.relative_to(ROOT).as_posix()},
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
    print('[OOF PRE01] Diagnostico 3D completo.',flush=True)


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
    exigir(pacote.pca.modelo_.n_components_==3, 'Modelo final nao tem 3 componentes.')
    salvar_pacote(pacote,dest)
    print('[FINAL F1-F9] PCA 3D congelado e salvo.',flush=True)
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
    exigir(list(pc.columns)==['PC1','PC2','PC3'] and len(pc)==len(ids),
           'Modelo congelado nao projetou 3 PCs.')
    exigir(np.isfinite(pc.to_numpy(dtype=float)).all(), 'PCs contem valores invalidos.')
    resumo=resumo_preprocessamento(pacote,x,z)
    coords=pd.DataFrame({'linha_original':ids,'PC1':pc['PC1'].to_numpy(dtype=float),
                         'PC2':pc['PC2'].to_numpy(dtype=float),
                         'PC3':pc['PC3'].to_numpy(dtype=float)})
    dest.mkdir(parents=True,exist_ok=False)
    coords.to_csv(dest/'coordenadas_PC1_PC2_PC3_f10.csv',index=False,float_format='%.17g')
    resumo.to_csv(dest/'resumo_padronizacao_f10.csv',index=False,float_format='%.17g')
    salvar_json_novo(dest/'manifesto_geometria_f10.json',{
        'dimensao':3,'n':len(ids),'tipo':'F10_EXPLORATORIO_REPETIDO_NOVO_MODELO',
        'limiar_mcc_oof_congelado':pacote.limiar_oof,
        'sha256_coordenadas':sha(dest/'coordenadas_PC1_PC2_PC3_f10.csv'),
        'sha256_resumo':sha(dest/'resumo_padronizacao_f10.csv'),
        'sem_ajuste_pca_f10':True,'sem_ajuste_scaler_f10':True})
    return coords,resumo


def dados_geometria(pacote,tab,coords):
    c=coords[['PC1','PC2','PC3']].copy()
    gamma=pacote.gmm.modelo_.predict_proba(c.to_numpy(dtype=float))
    return {'coord':c,'y':tab['y_true'].to_numpy(dtype=np.uint8),
            'scores':tab['prob_fraude'].to_numpy(dtype=float),
            'gamma':pd.DataFrame(gamma,columns=list(pacote.gmm.saidas_)),
            'gmm':pacote.gmm,'rh':pacote.risco,
            'redutor':pacote.pca,'limiar_oof':pacote.limiar_oof}


def _normalizar_rotulos_figura(fig):
    """Corrige somente legendas de contexto, sem modificar dados, eixos ou calculos."""
    from matplotlib.text import Text
    trocas = (
        ('F10 não utilizado', 'F10 exploratório (modelo F1–F9 congelado)'),
        ('F10 nao utilizado', 'F10 exploratório (modelo F1–F9 congelado)'),
        ('Observações do desenvolvimento', 'Observações F10'),
        ('Observações da simulação', 'Observações F10'),
        ('Planos fixados na média do desenvolvimento',
         'Planos fixados na média das projeções F10'),
        ('média do desenvolvimento', 'média das projeções F10'),
    )
    for objeto in fig.findobj(match=Text):
        titulo=objeto.get_text()
        novo=titulo
        for antigo,sub in trocas:
            novo=novo.replace(antigo,sub)
        if novo!=titulo:
            objeto.set_text(novo)


def _normalizar_html_interativo(dest):
    """HTML 3D original possui textos de treinamento; aqui descreve o F10."""
    for path in sorted(dest.glob('GMM*_INTERATIVA.html')):
        html=path.read_text(encoding='utf-8-sig')
        changes=(
            ('Observações do desenvolvimento','Observações F10'),
            ('Observações da simulação','Observações F10'),
            ('F10 não utilizado.','F10 exploratório repetido, modelo F1–F9 congelado.'),
            ('F10 não utilizado','F10 exploratório repetido, modelo F1–F9 congelado'),
            ('F10 nao utilizado','F10 exploratório repetido, modelo F1–F9 congelado'),
            ('média do desenvolvimento','média das projeções F10'),
            ('Rotulos de desenvolvimento','Rótulos F10'),
        )
        for antes,depois in changes:
            html=html.replace(antes,depois)
        if html!=path.read_text(encoding='utf-8-sig'):
            path.write_text(html,encoding='utf-8')


def desenhar_geometria(folder,state,pacote,tab,coords,met):
    """Geometria 3D com F10 apenas como projeção/avaliação; sem refit."""
    from src import visualizacao_geometrica as geo
    from src import visualizacao_gmm01 as gmm01
    from src import visualizacao_gmm04 as gmm04
    from src import visualizacao_gmm07 as gmm07
    from src import visualizacao_gmm08_3d as gmm08_3d
    from src import visualizacao_oof as vo
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    cfg=state['config']
    dpi=int(cfg['visualizacao']['padrao'].get('dpi',180))
    figs=folder/'vencedor/teste_final/graficos_gerados'
    dados=dados_geometria(pacote,tab,coords)
    ids=coords['linha_original'].to_numpy(dtype=np.int64)
    exigir(list(coords.columns)==['linha_original','PC1','PC2','PC3'],
           'Colunas e ordenacao PCA 3D invalidas.')
    exigir(np.array_equal(ids,tab['linha_original'].to_numpy(dtype=np.int64)),
           'Coordenadas e classificacoes F10 desalinhadas.')
    modelo=pacote.gmm.modelo_
    exigir(modelo.covariance_type=='full' and modelo.means_.shape==(4,3),
           'GMM final nao e FULL/K4/PCA3D.')
    mc=met['comparacao_limiares']
    mcc05=float(mc['tau_0_50']['mcc'])
    mccstar=float(mc['tau_mcc_oof']['mcc'])
    exigir(mccstar==float(met['mcc_limiar_oof']) and
           float(mc['tau_mcc_oof']['limiar'])==float(pacote.limiar_oof),
           'Valores F10 divergem do limiar e MCC congelados.')
    # O arquivo de scores ja foi salvo por avaliar_f10; NUNCA recalcular F10.
    geo._pca_projecao_2d3d(cfg,dados,figs,'png',dpi,True,
        cfg['visualizacao']['graficos']['pca_projecoes'])
    csv=figs/'12_pca_projecao_PC1_PC2_PC3.csv'
    if csv.is_file():
        t=pd.read_csv(csv)
        exigir(len(t)==len(ids),'CSV da projecao F10 desalinhado.')
        if 'risco_estimado_treino' in t:
            t=t.rename(columns={'risco_estimado_treino':'prob_fraude_f10'})
        t.insert(0,'linha_original_f10',ids)
        t.to_csv(csv,index=False,float_format='%.17g')

    # Contraparte das observacoes 1D/2D: dispersao tridimensional descritiva.
    lim=int(cfg['visualizacao']['geometria'].get('pontos_maximos_amostra',20000))
    iy=geo._amostra_plot(len(ids),dados['y'],cfg,lim)
    fig=plt.figure(figsize=(11.5,8.0))
    ax=fig.add_subplot(111,projection='3d')
    z=coords[['PC1','PC2','PC3']].to_numpy(dtype=float)
    cores=cfg['visualizacao']['classes']
    for classe,chave in ((0,'legitima'),(1,'fraude')):
        sel=iy[dados['y'][iy]==classe]
        if not len(sel):continue
        estilo=cores[chave]
        ax.scatter(z[sel,0],z[sel,1],z[sel,2],
                   c=estilo.get('cor','#4389c2'),s=float(estilo.get('tamanho',12)),
                   alpha=float(estilo.get('alpha',.55)),
                   marker=estilo.get('marcador','o'),
                   label=f'{estilo.get("nome",chave)} (n={len(sel)})',rasterized=True)
    ratios=pacote.pca.modelo_.explained_variance_ratio_
    for i,n in enumerate(('PC1','PC2','PC3')):
        getattr(ax,'set_'+('xlabel','ylabel','zlabel')[i])(
            f'{n} (variância {ratios[i]:.1%})')
    ax.view_init(elev=22,azim=-64)
    ax.set_box_aspect((1,1,.85))
    ax.set_title('OBSERVAÇÕES F10 — PCA 3D')
    ax.legend(loc='upper left')
    fig.subplots_adjust(left=.06,right=.94,bottom=.08,top=.91)
    vo._salvar(fig,figs/'PCA_OBSERVACOES_PC1_PC2_PC3.png',dpi=dpi)
    pd.DataFrame({'linha_original_f10':ids[iy],
                  'PC1':z[iy,0],'PC2':z[iy,1],'PC3':z[iy,2],
                  'classe_real':dados['y'][iy]}).to_csv(
                  figs/'PCA_OBSERVACOES_PC1_PC2_PC3.csv',index=False,float_format='%.17g')

    avaliacao={'origem':'f10_amostral_congelado'}
    guardar=geo._guardar
    avaliar_cortes=geo._avaliar_cortes_oof
    rotulo07=gmm07._rotulo_mcc
    rotulo08=gmm08_3d._origem_mcc
    salvar_figura=Figure.savefig

    def cortes(obj,tau):
        exigir(tau==pacote.limiar_oof and
               obj.get('origem')=='f10_amostral_congelado',
               'Origem ou limiar F10 incorretos na geometria.')
        return mcc05,mccstar

    def rotulo_f10(obj):
        exigir(obj.get('origem')=='f10_amostral_congelado',
               'Rotulo MCC sem origem F10.')
        return ('MCC F10','MCC F10 exploratório; τ* selecionado exclusivamente em OOF F1–F9.')

    def fig_salvar(fig,*args,**kwargs):
        _normalizar_rotulos_figura(fig)
        return salvar_figura(fig,*args,**kwargs)

    def geo_guardar(fig,*args,**kwargs):
        _normalizar_rotulos_figura(fig)
        return guardar(fig,*args,**kwargs)

    geo._guardar=geo_guardar
    geo._avaliar_cortes_oof=cortes
    gmm07._rotulo_mcc=rotulo_f10
    gmm08_3d._origem_mcc=rotulo_f10
    Figure.savefig=fig_salvar
    try:
        gmm01.gerar(cfg,dados,figs,'png',dpi,True)
        gmm04.gerar(cfg,dados,figs,'png',dpi,True)
        escolha={'limiar_mcc_oof':pacote.limiar_oof}
        gmm07.gerar(cfg,dados,escolha,figs,'png',dpi,True,avaliacao=avaliacao)
        geo._fronteira(cfg,dados,escolha,figs,'png',dpi,True,avaliacao=avaliacao)
    finally:
        Figure.savefig=salvar_figura
        geo._guardar=guardar
        geo._avaliar_cortes_oof=avaliar_cortes
        gmm07._rotulo_mcc=rotulo07
        gmm08_3d._origem_mcc=rotulo08
    _normalizar_html_interativo(figs)
    print('[F10] Geometria 3D, elipsoides 95% e HTMLs interativos concluídos.',flush=True)

def histograma_pc(coords,nome,bins=60):
    exigir(nome in ('PC1','PC2','PC3'),'Componente PCA invalida.')
    x=coords[nome].to_numpy(dtype=float)
    exigir(len(x)>0 and np.isfinite(x).all(),f'{nome} invalida.')
    mi,ma=float(x.min()),float(x.max())
    e=np.linspace(mi-.5,ma+.5,bins+1) if mi==ma else np.linspace(mi,ma,bins+1)
    freq,ed=np.histogram(x,bins=e)
    exigir(int(freq.sum())==len(x),f'Histograma {nome} perdeu observacoes.')
    prefixo=nome.lower()
    return pd.DataFrame({f'{prefixo}_inicio':ed[:-1],f'{prefixo}_fim':ed[1:],
                         f'{prefixo}_centro':(ed[:-1]+ed[1:])/2,
                         'contagem_f10':freq})

def gerar_pre01(folder,state,resumo,coords,pacote,ids):
    # PRE01 F10 com PC1, PC2 e PC3 sem ajustar transformadores.
    from scripts.execucao import GERAR_PRE01_F10_PCA1D as pre
    rel=folder/'vencedor/teste_final/diagnosticos/PRE01_F10'
    exigir(not rel.exists(),'PRE01 F10 ja existente; preservar.')
    tab=pre.enriquecer(resumo)
    hist1=pre.histograma_pc1(coords)
    hist2=histograma_pc(coords,'PC2')
    hist3=histograma_pc(coords,'PC3')
    meta={'versao':'PRE01_F10_PCA3D_TESTE_COMPLETO_V1',
          'n_f10':len(ids),'n_treino_f1_f9':len(state['ids']),
          'pca_dimensao':3,'variaveis_padronizadas':len(tab),
          'tau_star_oof':pacote.limiar_oof,
          'carater':'AMOSTRAL_EXPLORATORIO_REPETIDO',
          'sem_ajustes_preprocessamento_f10':True}
    oof=folder/'vencedor/visualizacao/galeria/oof/PRE01_diagnostico_padronizacao_completo.html'
    link=os.path.relpath(oof,rel).replace('\\','/') if oof.is_file() else None
    pagina=pre.html_pre01(tab,hist1,coords['PC1'].to_numpy(dtype=float),meta,link_oof=link)
    exigir(pagina.count('</main>')==1 and pagina.count('</body>')==1,
           'PRE01 original mudou; conferir antes de editar HTML.')
    import json as _json
    secoes=[];scripts=[]
    for nome,hist in (('PC2',hist2),('PC3',hist3)):
        ident=nome.lower()
        secoes.append(f'''<section class="section"><h2>{nome} — Distribuição F10 com PCA congelado</h2>
<p>{nome} projetada com PCA ajustado em F1–F9, sem refit no F10.</p>
<div id="pre01-f10-{ident}" class="plot"></div></section>''')
        payload=_json.dumps({'x':hist[f'{ident}_centro'].to_list(),
                             'y':hist['contagem_f10'].to_list()},ensure_ascii=False)
        scripts.append(f'''<script>const DATA_{nome}={payload};
Plotly.newPlot('pre01-f10-{ident}',[{{type:'bar',x:DATA_{nome}.x,y:DATA_{nome}.y,
marker:{{color:'#438b7d'}}}}],
{{paper_bgcolor:'#fff',plot_bgcolor:'#fff',height:390,
xaxis:{{title:'{nome} (PCA F1–F9 congelado)'}},
yaxis:{{title:'Frequência F10'}},margin:{{l:65,r:25,t:25,b:55}}}},
{{responsive:true,displaylogo:false}});</script>''')
    pagina=pagina.replace('</main>',''.join(secoes)+'</main>')
    pagina=pagina.replace('</body>',''.join(scripts)+'</body>')
    pagina=pagina.replace('PCA 1D | F10 amostral','PCA 3D | F10 amostral')
    rel.mkdir(parents=True,exist_ok=False)
    tab.to_csv(rel/pre.TABELA_ARQ,index=False,float_format='%.17g')
    hist1.to_csv(rel/pre.HIST_ARQ,index=False,float_format='%.17g')
    hist2.to_csv(rel/'PRE01_F10_PC2_histograma.csv',index=False,float_format='%.17g')
    hist3.to_csv(rel/'PRE01_F10_PC3_histograma.csv',index=False,float_format='%.17g')
    (rel/pre.PRE01_ARQ).write_text(pagina,encoding='utf-8')
    salvar_json_novo(rel/pre.MANIFESTO_ARQ,{
        **meta,'origem_oof':folder.relative_to(ROOT).as_posix(),
        'csv_padronizacao':pre.TABELA_ARQ,
        'histograma_pc1':pre.HIST_ARQ,
        'histograma_pc2':'PRE01_F10_PC2_histograma.csv',
        'histograma_pc3':'PRE01_F10_PC3_histograma.csv',
        'sha256_html':sha(rel/pre.PRE01_ARQ),
        'sha256_csv':sha(rel/pre.TABELA_ARQ),
        'sha256_pc1':sha(rel/pre.HIST_ARQ),
        'sha256_pc2':sha(rel/'PRE01_F10_PC2_histograma.csv'),
        'sha256_pc3':sha(rel/'PRE01_F10_PC3_histograma.csv')})
    print('[F10 PRE01] PCA 3D com PC1, PC2 e PC3.',flush=True)

GRAFICOS = (
    *comum.GRAFICOS[:6],
    ('12_pca_projecao_PC1_PC2_PC3.png','12 — Projeção PCA PC1 × PC2 × PC3'),
    ('PCA_OBSERVACOES_PC1_PC2_PC3.png','PCA — Observações F10 em 3D'),
    ('GMM01_densidade_marginal_3d_PC1.png','GMM01 — Densidade marginal PC1'),
    ('GMM01_densidade_marginal_3d_PC2.png','GMM01 — Densidade marginal PC2'),
    ('GMM01_densidade_marginal_3d_PC3.png','GMM01 — Densidade marginal PC3'),
    ('GMM01_conjunta_corte_PC1_PC2.png','GMM01 — Densidade conjunta corte PC1 × PC2'),
    ('GMM01_conjunta_corte_PC1_PC3.png','GMM01 — Densidade conjunta corte PC1 × PC3'),
    ('GMM01_conjunta_corte_PC2_PC3.png','GMM01 — Densidade conjunta corte PC2 × PC3'),
    ('GMM01_geometria_gaussianas_FULL_3d_95.png','GMM01 — Elipsoides gaussianos FULL 95%'),
    ('GMM04_dominancia_3d.png','GMM04 — Dominância 3D'),
    ('GMM07_sensibilidade_3d_comparativo.png','GMM07 — Sensibilidade 3D'),
    ('GMM08_mapa_risco_3d_comparativo.png','GMM08 — Risco e geometria 3D 95%'),
)

HTMLS_INTERATIVOS = (
    ('GMM01_densidade_conjunta_PCA3D_INTERATIVA.html','GMM01 — Densidade conjunta 3D interativa'),
    ('GMM01_geometria_gaussianas_FULL_3d_95_INTERATIVA.html','GMM01 — Elipsoides 95% interativos'),
    ('GMM04_dominancia_3d_INTERATIVA.html','GMM04 — Dominância 3D interativa'),
    ('GMM07_sensibilidade_3d_comparativo_INTERATIVA.html','GMM07 — Sensibilidade 3D interativa'),
    ('GMM08_mapa_risco_3d_comparativo_INTERATIVA.html','GMM08 — Mapa de risco 3D interativo'),
)


def galeria_final(folder,ids,pacote,met):
    dest=folder/'vencedor/teste_final'
    figs=dest/'graficos_gerados';gal=dest/'galeria'
    for nome,_ in GRAFICOS:
        exigir((figs/nome).is_file(),f'Figura obrigatoria faltando: {nome}')
    for nome,_ in HTMLS_INTERATIVOS:
        exigir((figs/nome).is_file(),f'HTML interativo obrigatorio faltando: {nome}')
    pre='diagnosticos/PRE01_F10/PRE01_F10_diagnostico_padronizacao_completo.html'
    exigir((dest/pre).is_file(),'PRE01 F10 3D nao encontrado.')
    cards=''.join(f'<article><a href="../graficos_gerados/{escape(n,quote=True)}" target="_blank">'
                  f'<img src="../graficos_gerados/{escape(n,quote=True)}" loading="lazy" alt="{escape(label)}">'
                  f'<p>{escape(label)}</p></a></article>' for n,label in GRAFICOS)
    interativos=''.join(f'<a href="../graficos_gerados/{escape(n,quote=True)}" target="_blank">'
                       f'{escape(label)}</a>' for n,label in HTMLS_INTERATIVOS)
    arquivos=['../predicoes_f10.csv','../metricas_f10.json',
              '../dados_geometria/coordenadas_PC1_PC2_PC3_f10.csv',
              '../dados_geometria/resumo_padronizacao_f10.csv',
              '../modelo_final_f1_f9/manifesto_modelo_final.json','../'+pre]
    links=''.join(f'<a href="{escape(a,quote=True)}">{escape(Path(a).name)}</a>' for a in arquivos)
    html=f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PCA 3D F10 — galeria completa</title><style>body{{font:15px system-ui,sans-serif;background:#f5f7fa;color:#172236;max-width:1400px;margin:auto;padding:20px}}
.grade{{display:grid;grid-template-columns:repeat(auto-fit,minmax(315px,1fr));gap:16px}}article{{background:#fff;border:1px solid #dae1e9;border-radius:10px;padding:12px;box-shadow:0 2px 8px #0001}}article img{{width:100%;aspect-ratio:4/3;object-fit:contain}}
a{{color:#285783;text-decoration:none}}article p{{font-weight:600}}.aviso{{background:#e8edf4;border-left:4px solid #355877;padding:14px}}.links{{display:flex;gap:16px;flex-wrap:wrap}}</style></head><body>
<h1>PCA 3D / GMM K4 n_init5 — Teste final F10</h1><p class="aviso">Amostra F10 de {len(ids)} observações, 10%, EXPLORATÓRIO repetido. F10 não é nova validação independente. τ* = {pacote.limiar_oof:.17g}, escolhido em OOF F1–F9. Modelo, encoder, scaler e PCA congelados.</p>
<p>F10: AP={met['ap']:.7f} | MCC em τ*={met['mcc_limiar_oof']:.7f} | LogLoss={met['logloss']:.7f} | Score={met['score_composto_limiar_oof']:.7f}</p>
<p><a href="../../visualizacao/galeria/index.html">Galeria OOF</a> · <a href="../{pre}">PRE01 F10 interativo</a></p>
<h2>Gráficos tridimensionais interativos</h2><div class="links">{interativos}</div>
<h2>Gráficos estáticos F10 ({len(GRAFICOS)})</h2><div class="grade">{cards}</div>
<h2>Resultados e proveniência</h2><div class="links">{links}</div></body></html>'''
    gal.mkdir(parents=True,exist_ok=True)
    with (gal/'index.html').open('x',encoding='utf-8') as f:f.write(html)
    salvar_json_novo(dest/'manifesto_teste_completo_pca3d_f10.json',{
        'tipo':'TESTE_COMPLETO_PCA3D_10PCT_EXPLORATORIO_REPETIDO',
        'n_f10':len(ids),'n_dev':pacote.n_treino_original,
        'limiar_congelado_oof':pacote.limiar_oof,
        'limiar_otimizado_no_f10':False,
        'modelo_sha256':sha(dest/'modelo_final_f1_f9/modelo_final_f1_f9.joblib'),
        'predicoes_sha256':sha(dest/'predicoes_f10.csv'),
        'metricas_sha256':sha(dest/'metricas_f10.json'),
        'graficos_sha256':{n:sha(figs/n) for n,_ in GRAFICOS},
        'interativos_sha256':{n:sha(figs/n) for n,_ in HTMLS_INTERATIVOS},
        'pre01':pre,'galeria':'galeria/index.html'})
    print('[GALERIA PCA 3D]',gal/'index.html',flush=True)

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
            exigir(pacote.pca.modelo_.n_components_==3,'Retomada encontrou PCA de outra dimensao.')
            print('[RETOMADA] Modelo PCA 3D existente.',flush=True)
        else:
            exigir(not modelo_dir.exists(),'Modelo parcial; nao sobrescrever.')
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
            exigir(not (dest/'metricas_f10.json').exists(),
                   'Metricas sem predicoes; nao reavaliar F10.')
            df,tab,met=comum.avaliar_f10(folder,state,pacote,ids)
        exigir(float(met['limiar_mcc_oof_congelado'])==float(state['escolha']['limiar_mcc_oof'])
               and met['limiar_otimizado_no_f10'] is False,
               'F10 alterou o limiar OOF.')
        geo=dest/'dados_geometria'
        coord=geo/'coordenadas_PC1_PC2_PC3_f10.csv'
        if coord.is_file():
            coords=pd.read_csv(coord)
            resumo=pd.read_csv(geo/'resumo_padronizacao_f10.csv')
            exigir(np.array_equal(coords['linha_original'].to_numpy(dtype=np.int64),ids),
                   'Coordenadas PCA 3D de outra amostra.')
        else:
            exigir(not geo.exists(),'Geometria parcial; revisar antes de retomar.')
            coords,resumo=preparar_geometria(folder,state,pacote,df,ids)
        figs=dest/'graficos_gerados'
        if figs.exists():
            for nome,_ in (*GRAFICOS,*HTMLS_INTERATIVOS):
                exigir((figs/nome).is_file(),f'Figuras/HTML parciais; revisar: {nome}')
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
        m['teste_final']={'status':'concluido','dimensao_pca':3,
                          'carater':'AMOSTRAL_EXPLORATORIO_F10_JA_UTILIZADO',
                          'n_f10':len(ids),'sem_reotimizar_tau':True,
                          'galeria':'vencedor/teste_final/galeria/index.html',
                          'manifesto':'vencedor/teste_final/manifesto_teste_completo_pca3d_f10.json'}
        orq._salvar_json(folder/'manifesto.json',m)
        print('[CONCLUIDO] PCA 3D OOF + F10 em',folder,flush=True)
        print('[ABRIR]',dest/'galeria/index.html',flush=True)
