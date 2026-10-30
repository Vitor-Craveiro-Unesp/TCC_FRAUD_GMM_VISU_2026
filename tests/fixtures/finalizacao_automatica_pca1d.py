# -*- coding: utf-8 -*-
"""Finalizacao automatica PCA1D para o CLI original `python -m src.orquestracao_oof --executar`.

Somente PCA 1D, K4, n_init5, RUS16, fracao desenvolvimento 10% e F10 10%.
F10 foi previamente acessado neste projeto: execucoes repetidas sao EXPLORATORIAS.
Nunca utiliza F10 para escolher candidato, ajustar PCA/GMM ou otimizar tau.
Nao e uma CLI; chamada exclusivamente pelo orquestrador original apos OOF + galeria.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import sys
from html import escape

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

LOCK = ROOT / 'outputs/.teste_completo_pca1d_em_andamento.lock'


def exigir(valor, mensagem):
    if not valor:
        raise RuntimeError(mensagem)


def ler_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    dig = sha256()
    with Path(path).open('rb') as arq:
        for bloco in iter(lambda: arq.read(1024*1024), b''):
            dig.update(bloco)
    return dig.hexdigest()


def salvar_json_novo(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as arq:
        json.dump(obj, arq, indent=2, ensure_ascii=False, allow_nan=False)
        arq.write('\n')


def parametros_ok(c):
    unico = lambda a: a[0] if isinstance(a, list) and len(a)==1 else a
    return (float(c['dados']['fracao_base']) == 0.1
            and c['reducao']['metodo'].upper() == 'PCA'
            and int(unico(c['gmm']['n_components'])) == 4
            and int(c['gmm']['n_init']) == 5
            and c['gmm']['covariance_type'].lower() == 'full'
            and c['amostragem']['modo'].upper() == 'RUS'
            and int(c['amostragem']['razao_negativos_positivos']) == 16
            and int(c['experimento']['semente']) == 42
            and int(c['divisao']['qual_fold_teste']) == 10
            and c['execucao']['proteger_resultados'] is True)




@contextmanager
def bloqueio():
    with LOCK.open('x', encoding='utf-8') as f:
        f.write('Execucao em curso; nao executar simultaneamente.\n')
    try:
        yield
    finally:
        LOCK.unlink(missing_ok=True)






def conferir_oof(folder):
    from src import orquestracao_oof as orq
    from src.folds import hash_arquivo
    m = ler_json(folder/'manifesto.json')
    esc = ler_json(folder/'vencedor/metricas/melhor_configuracao.json')
    c = yaml.safe_load((folder/'configuracao.yaml').read_text(encoding='utf-8-sig'))
    exigir(parametros_ok(c) and c['pca']['n_components']==[1], 'OOF nao corresponde a PCA1D K4 .1.')
    exigir(m.get('status')=='concluido_06_03' and int(m.get('registros_desenvolvimento',-1))>0,
           'OOF nao concluido com 166716 registros.')
    exigir(esc['configuracao']['reducao']['n_components']==1
           and esc['configuracao']['gmm']['n_init']==5
           and esc['configuracao']['gmm']['n_components']==4
           and esc['teste_final_usado'] is False,
           'Escolha OOF incompativel.')
    arq=folder/'vencedor/oof/predicoes.npz'
    with np.load(arq, allow_pickle=False) as z:
        ids=np.asarray(z['linha_original'],dtype=np.int64)
        folds=np.asarray(z['fold'],dtype=np.uint8)
        y=np.asarray(z['y_true'],dtype=np.uint8)
    exigir(len(ids)==int(m['registros_desenvolvimento']) and np.all(np.diff(ids)>0) and np.all(folds<=9)
           and set(np.unique(folds))==set(range(1,10)), 'Vetor OOF invalido.')
    parquet=ROOT/c['dados']['arquivo']
    orig_folds, arq_folds, hash_pq = orq._identificar_folds(ROOT,c,parquet)
    exigir(hash_pq==m['sha256_parquet'] and hash_arquivo(arq_folds)==m['sha256_folds'],
           'Parquet ou folds diferentes do OOF.')
    exigir(np.array_equal(orig_folds[ids], folds), 'Atribuicao dos folds do OOF mudou.')
    return {'config':c,'escolha':esc,'manifesto':m,'ids':ids,'folds':folds,
            'y':y,'orig_folds':orig_folds,'parquet':parquet}



def gerar_pre01_oof(folder, state):
    """PRE01 de 9 folds apenas para ESTE ensaio novo; nunca modifica trios antigos."""
    from src import orquestracao_oof as orq
    from src import preprocessing as prep
    from src import pre01_v2_diagnostico as pre
    vis=folder/'vencedor/visualizacao'
    index=vis/'galeria/index.html'
    destino=vis/'oof'
    html_path=destino/pre.FILENAME
    if html_path.is_file():
        print('[OOF PRE01] Ja presente, sem sobrescrever.',flush=True)
        return
    exigir(index.is_file(), 'Galeria OOF ainda nao existe; PRE01 bloqueado.')
    df=orq._ler_desenvolvimento(state['parquet'],state['ids'])
    X=prep.selecionar_features(df,state['config'])
    print('[OOF PRE01] Calculando diagnostico completo F1..F9...',flush=True)
    summary,detail,folds_df,pca=pre.calc(X,state['folds'],state['config'])
    meta={'tipo':'ensaio','k':4,'n_init':5,'total_observacoes':int(len(X)),
          'total_features':int(len(summary)),'f10_usado':False,
          'mesmos_registros_e_folds':True,
          'experimentos':{'1D':folder.relative_to(ROOT).as_posix()},
          'sha256_parquet':state['manifesto']['sha256_parquet'],
          'sha256_folds':state['manifesto']['sha256_folds'],
          'medida_de_desempenho':False,
          'pca_ajuste_qualificacao':'PCA auxiliar diagnostico F1–F9; nao modelo F10'}
    blobs={
        pre.FILENAME:pre.build_html(summary,detail,folds_df,pca,meta).encode('utf-8'),
        pre.CSV_SUM:summary.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
        pre.CSV_DETAIL:detail.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
        pre.CSV_FOLDS:folds_df.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
        pre.CSV_PCA:pca.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
        pre.META:(json.dumps(meta,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
    }
    for pasta in (vis/'oof',vis/'galeria/oof'):
        pasta.mkdir(parents=True,exist_ok=True)
        for nome,dados in blobs.items():
            with (pasta/nome).open('xb') as f:f.write(dados)
    original=index.read_text(encoding='utf-8')
    novo=pre.new_gallery_index(original)
    exigir(original!=novo, 'Link PRE01 nao inserido no indice OOF.')
    index.write_text(novo,encoding='utf-8')
    print('[OOF PRE01] HTML interativo e CSVs incluidos no novo ensaio.',flush=True)


def treinar_final(folder, state):
    from src import orquestracao_oof as orq
    from src.treinamento_final_pca import ajustar_final_f1_f9, salvar_pacote
    tdir = folder/'vencedor/teste_final'
    modelo_dir=tdir/'modelo_final_f1_f9'
    exigir(not modelo_dir.exists(), 'Modelo final ja existe. Nao sobrescrever.')
    df=orq._ler_desenvolvimento(state['parquet'],state['ids'])
    exigir(len(df)==len(state['ids']) and np.array_equal(
        df[state['config']['dados']['coluna_alvo']].to_numpy(),state['y']),
        'Rotulos de desenvolvimento divergentes.')
    m=state['manifesto']
    prov={'sha256_parquet':m['sha256_parquet'],'sha256_folds':m['sha256_folds'],
          'sha256_yaml':m['sha256_yaml'],
          'experimento_oof':folder.relative_to(ROOT).as_posix(),
          'sha256_predicoes_oof':sha(folder/'vencedor/oof/predicoes.npz')}
    print('[FINAL F1..F9] Ajustando unico GMM PCA 1D K4 n_init5...',flush=True)
    pacote=ajustar_final_f1_f9(df,state['folds'],state['config'],state['escolha'],prov,
                               registrar_historico_em=True)
    caminho, manifest=salvar_pacote(pacote, modelo_dir)
    print('[MODELO FINAL]', caminho,flush=True)
    return pacote


def ler_linhas_f10(parquet, ids):
    # Mesma estrategia em lotes da Etapa 8, implementacao sem acesso a outros scripts.
    import pyarrow.parquet as pq
    parts=[]; inicio=0; pf=pq.ParquetFile(parquet)
    exigir(int(ids[-1])<pf.metadata.num_rows,'Indices F10 ultrapassam Parquet.')
    for batch in pf.iter_batches(batch_size=20000):
        fim=inicio+batch.num_rows
        a=int(np.searchsorted(ids,inicio,'left'))
        b=int(np.searchsorted(ids,fim,'left'))
        if a<b:
            parts.append(batch.to_pandas().iloc[ids[a:b]-inicio].copy())
        inicio=fim
        if inicio>int(ids[-1]):break
    df=pd.concat(parts,ignore_index=True)
    exigir(len(df)==len(ids) and df.index.equals(pd.RangeIndex(len(df))),
           'Amostra F10 lida incorretamente.')
    return df


def avaliar_f10(folder, state, pacote, ids):
    from src.avaliacao_final_pca import avaliar_f10_em_memoria
    from src.folds import hash_arquivo
    dest=folder/'vencedor/teste_final'
    exigir(not (dest/'predicoes_f10.csv').exists() and not (dest/'metricas_f10.json').exists(),
           'F10 desta nova execucao ja foi salvo. Nao repetir.')
    exigir(np.all(state['orig_folds'][ids]==10),'Amostra inclui indices que nao sao F10.')
    print(f'[F10] Avaliacao EXPLORATORIA em {len(ids)} linhas F10 (fold 10; seed 42; 10% estratificado).',flush=True)
    df=ler_linhas_f10(state['parquet'],ids)
    tab,metricas=avaliar_f10_em_memoria(pacote,df,state['orig_folds'][ids],ids)
    exigir(metricas['limiar_otimizado_no_f10'] is False
           and float(metricas['limiar_mcc_oof_congelado'])==float(state['escolha']['limiar_mcc_oof']),
           'Reotimizacao de limiar ou troca de tau detectada.')
    dest.mkdir(exist_ok=True,parents=True)
    with (dest/'predicoes_f10.csv').open('x',encoding='utf-8') as f:
        tab.to_csv(f,index=False,float_format='%.17g')
    salvar_json_novo(dest/'metricas_f10.json',metricas)
    np.save(dest/'indices_f10_amostra.npy',ids,allow_pickle=False)
    salvar_json_novo(dest/'manifesto_amostra_f10.json', {'fracao':0.1,'semente':42,'n':len(ids),
        'sha256_indices':sha256(ids.tobytes()).hexdigest(),
        'origem':'F10 da base/folds congelados do proprio experimento',
        'classe':'AMOSTRAL_EXPLORATORIO_REPETIDO',
        'sem_ajuste_no_f10':True})
    print(f'[F10] AP={metricas["ap"]:.7f} MCC={metricas["mcc_limiar_oof"]:.7f} '
          f'LogLoss={metricas["logloss"]:.7f}; tau OOF={pacote.limiar_oof:.17g}',flush=True)
    return df,tab,metricas


def preparar_geometria(folder,state,pacote,df,ids):
    extr=importlib.import_module('scripts.execucao.EXTRAIR_COORDENADAS_PCA1D_F10')
    destino=folder/'vencedor/teste_final/dados_geometria'
    alvo=state['config']['dados']['coluna_alvo']
    bruto=df.drop(columns=[alvo]).copy()
    excl=[state['config']['dados'].get('coluna_id'),*state['config']['dados'].get('colunas_excluir',[])]
    x=bruto.drop(columns=[n for n in excl if n],errors='ignore')
    x_z,pc=extr.aplicar_transformadores_congelados(pacote,x)
    resumo=extr.resumo_preprocessamento(pacote,x,x_z)
    coords=pd.DataFrame({'linha_original':ids,'PC1':pc['PC1'].to_numpy(dtype=float)})
    destino.mkdir(parents=True,exist_ok=False)
    coords.to_csv(destino/'coordenadas_PC1_f10.csv',index=False,float_format='%.17g')
    resumo.to_csv(destino/'resumo_padronizacao_f10.csv',index=False,float_format='%.17g')
    salvar_json_novo(destino/'manifesto_geometria_f10.json',{
        'dimensao':1,'n':len(ids),'tipo':'F10_EXPLORATORIO_REPETIDO_NOVO_MODELO',
        'limiar_mcc_oof_congelado':pacote.limiar_oof,
        'sha256_coordenadas':sha(destino/'coordenadas_PC1_f10.csv'),
        'sha256_resumo':sha(destino/'resumo_padronizacao_f10.csv'),
        'sem_ajuste_pca_f10':True,'sem_ajuste_scaler_f10':True})
    return coords,resumo


@contextmanager
def rotulos_graficos_f10(figs=True):
    from src import visualizacao_oof as vo
    from matplotlib.text import Text
    old_save=vo._salvar
    old_csv=vo._csv_protegido
    pares=(
        ('Resultados da validação OOF','Resultados do teste F10 (τ* OOF congelado)'),
        ('Curva Precision–Recall do modelo selecionado em OOF','Curva Precision–Recall do modelo avaliado em F10'),
        ('AP calculada sobre as predições OOF.','AP calculada sobre as predições F10.'),
        ('MCC calculado no limiar ótimo das predições OOF.','MCC calculado em F10 no limiar τ* OOF.'),
        ('LogLoss calculado sobre as predições OOF.','LogLoss calculado sobre as predições F10.'),
        ('OOF F1–F9','F10 estratificado 10%'),
        ('F10 não utilizado','F10 exploratório repetido'),
        ('Limiar ótimo do MCC (τ*)','Limiar τ* congelado em OOF'),
        ('Limiar ótimo (τ*)','Limiar τ* OOF (fixo)'),
        ('MCC no limiar ótimo','MCC no limiar OOF'),
        ('Limiar ótimo τ*','Limiar OOF τ*'),
        ('τ* MCC','τ* OOF (fixo)'),
    )
    def relabel(fig):
        for text in fig.findobj(match=Text):
            label=text.get_text()
            if not label:continue
            for a,b in pares:label=label.replace(a,b)
            if label!=text.get_text():text.set_text(label)
        # Nenhuma assinatura decorativa adicionada no canto inferior.
    def saver(fig,dest,*,dpi):
        relabel(fig)
        return old_save(fig,dest,dpi=dpi)
    def saver_csv(tab,dest):
        tab=tab.copy()
        tab.rename(columns={'n_oof':'n_f10','prevalencia_oof':'prevalencia_f10'},inplace=True)
        return old_csv(tab,dest)
    vo._salvar=saver
    vo._csv_protegido=saver_csv
    try:yield relabel
    finally:
        vo._salvar=old_save
        vo._csv_protegido=old_csv


def desenhar_preditivos(folder,state,tab,met):
    from src import visualizacao_oof as vo
    fig=folder/'vencedor/teste_final/graficos_gerados'
    fig.mkdir(exist_ok=False,parents=True)
    cfg=state['config']; y=tab['y_true'].to_numpy(dtype=np.uint8); p=tab['prob_fraude'].to_numpy(dtype=float)
    e={'limiar_mcc_oof':float(state['escolha']['limiar_mcc_oof'])}
    vencedor=pd.Series({'ap':met['ap'],'mcc':met['mcc_limiar_oof'],
                       'logloss':met['logloss'],'score':met['score_composto_limiar_oof']})
    dpi=int(cfg['visualizacao']['padrao'].get('dpi',180))
    with rotulos_graficos_f10():
        funcs=[
            ('00_painel_score_composto',lambda:vo._plot_painel_00(cfg,y,p,vencedor,e,fig,'png',dpi,True)),
            ('01_curva_precision_recall',lambda:vo._plot_pr(cfg,y,p,vencedor,e,fig,'png',dpi,True)),
            ('02_mcc_por_limiar',lambda:vo._plot_mcc(cfg,y,p,vencedor,e,fig,'png',dpi,True)),
            ('03_matriz_confusao_05',lambda:vo._matriz(cfg,y,p,.5,'matriz_confusao_05',fig,'png',dpi,True)),
            ('04_matriz_confusao_mcc',lambda:vo._matriz(cfg,y,p,e['limiar_mcc_oof'],'matriz_confusao_mcc',fig,'png',dpi,True)),
            ('05_distribuicao_probabilidades',lambda:vo._distribuicao(cfg,y,p,e,fig,'png',dpi,True)),
        ]
        for name,fn in funcs:
            fn();exigir((fig/f'{name}.png').is_file(),f'Grafico F10 ausente: {name}')
            print('[F10 GRAFICO]',name,flush=True)
    return fig


def dados_geometria(state,pacote,tab,coords):
    c=pd.DataFrame({'PC1':coords['PC1'].to_numpy(dtype=float)})
    gam = pacote.gmm.modelo_.predict_proba(c.to_numpy(dtype=float))
    return {'coord':c,'y':tab['y_true'].to_numpy(dtype=np.uint8),
            'scores':tab['prob_fraude'].to_numpy(dtype=float),
            'gamma':pd.DataFrame(gam,columns=list(pacote.gmm.saidas_)),
            'gmm':pacote.gmm,'rh':pacote.risco,'redutor':pacote.pca,
            'limiar_oof':pacote.limiar_oof}


def desenhar_geometria(folder,state,pacote,tab,coords,met):
    from src import visualizacao_geometrica as geo
    from src import visualizacao_gmm01 as gmm01
    from src import visualizacao_gmm04 as gmm04
    from src import visualizacao_pca_biplot as biplot
    from src import visualizacao_oof as vo
    from src import visualizacao_gmm07 as gmm07
    from sklearn.metrics import matthews_corrcoef
    from matplotlib.text import Text
    cfg=state['config']; dpi=int(cfg['visualizacao']['padrao'].get('dpi',180))
    figs=folder/'vencedor/teste_final/graficos_gerados'
    dados=dados_geometria(state,pacote,tab,coords)
    ids=coords['linha_original'].to_numpy(dtype=np.int64)
    # As rotinas originais sao usadas com seus parametros atuais; nenhum .fit() aqui.
    geo._pca_histograma_1d(cfg,dados,figs,'png',dpi,True,
                          cfg['visualizacao']['graficos']['pca_projecoes'])
    csv=figs/'12_pca_projecao_PC1.csv'
    if csv.is_file():
        t=pd.read_csv(csv)
        if 'risco_estimado_treino' in t:
            t=t.rename(columns={'risco_estimado_treino':'prob_fraude_f10'})
        if 'linha_original_f10' not in t:
            t.insert(0,'linha_original_f10',ids)
        t.to_csv(csv,index=False,float_format='%.17g')
    lim=int(cfg['visualizacao']['geometria'].get('pontos_maximos_amostra',20000))
    idx=geo._amostra_plot(len(ids),dados['y'],cfg,lim)
    figura=biplot._fig_observacoes_pc1(dados['coord'].iloc[idx,0].to_numpy(),dados['y'][idx],cfg)
    vo._salvar(figura,figs/'PCA_OBSERVACOES_PC1.png',dpi=dpi)
    pd.DataFrame({'linha_original_f10':ids[idx],'PC1':dados['coord'].iloc[idx,0].to_numpy(),
                  'classe_real':dados['y'][idx]}).to_csv(figs/'PCA_OBSERVACOES_PC1.csv',index=False,float_format='%.17g')
    gmm01.gerar(cfg,dados,figs,'png',dpi,True)
    gmm04.gerar(cfg,dados,figs,'png',dpi,True)
    # GMM07/GMM08: mesmos campos e cortes, mas MCC F10 (ja calculado, nao novo tau).
    original_avaliar=geo._avaliar_cortes_oof
    original_guardar=geo._guardar
    original_rotulo=gmm07._rotulo_mcc
    mcc05=float(matthews_corrcoef(dados['y'],dados['scores']>=.5))
    mcc_oof=float(met['mcc_limiar_oof'])
    def avaliacao_f10(obj,tau):
        exigir(obj.get('origem')=='f10_amostral_congelado' and tau==pacote.limiar_oof,
               'GMM F10 recebeu origem de MCC incorreta.')
        return mcc05,mcc_oof
    def rotulo_f10(obj):
        exigir(obj.get('origem')=='f10_amostral_congelado','Origem de MCC GMM07 incorreta.')
        return 'MCC F10','MCC do teste F10 exploratorio; tau* fixo em OOF.'
    def save_geo(fig,dest,base,ext,dpi):
        for t in fig.findobj(match=Text):
            atual=t.get_text()
            novo=atual.replace('B — Corte ótimo MCC OOF','B — Limiar τ* congelado OOF')\
                      .replace('LIMIAR ÓTIMO DO MCC','LIMIAR τ* CONGELADO (OOF)')\
                      .replace('MCC OOF','MCC F10')
            if novo!=atual:t.set_text(novo)
        # Nao adicionar textos/selos nos cantos da figura.
        return original_guardar(fig,dest,base,ext,dpi)
    geo._avaliar_cortes_oof=avaliacao_f10
    geo._guardar=save_geo
    gmm07._rotulo_mcc=rotulo_f10
    try:
        e={'limiar_mcc_oof':pacote.limiar_oof}
        av={'origem':'f10_amostral_congelado'}
        gmm07.gerar(cfg,dados,e,figs,'png',dpi,True,avaliacao=av)
        geo._fronteira(cfg,dados,e,figs,'png',dpi,True,avaliacao=av)
    finally:
        geo._avaliar_cortes_oof=original_avaliar
        geo._guardar=original_guardar
        gmm07._rotulo_mcc=original_rotulo
    print('[F10] Geometria: PCA12, observacoes, GMM01, GMM04, GMM07 e GMM08.',flush=True)


def gerar_pre01(folder,state,resumo,coords,pacote,ids):
    modulo=importlib.import_module('scripts.execucao.GERAR_PRE01_F10_PCA1D')
    rel=folder/'vencedor/teste_final/diagnosticos/PRE01_F10'
    exigir(not rel.exists(),'PRE01 ja existente.')
    tab=modulo.enriquecer(resumo)
    hist=modulo.histograma_pc1(coords)
    meta={'versao':'PRE01_F10_PCA1D_TESTE_COMPLETO_V1',
          'n_f10':len(ids),'n_treino_f1_f9':len(state['ids']),'pca_dimensao':1,
          'variaveis_padronizadas':len(tab),'tau_star_oof':pacote.limiar_oof,
          'carater':'AMOSTRAL_EXPLORATORIO_REPETIDO',
          'sem_ajustes_preprocessamento_f10':True}
    rel.mkdir(parents=True,exist_ok=False)
    tab.to_csv(rel/modulo.TABELA_ARQ,index=False,float_format='%.17g')
    hist.to_csv(rel/modulo.HIST_ARQ,index=False,float_format='%.17g')
    oof=folder/'vencedor/visualizacao/galeria/oof/PRE01_diagnostico_padronizacao_completo.html'
    link=None
    if oof.is_file():link=os.path.relpath(oof,rel).replace('\\','/')
    pagina=modulo.html_pre01(tab,hist,coords['PC1'].to_numpy(dtype=float),meta,link_oof=link)
    (rel/modulo.PRE01_ARQ).write_text(pagina,encoding='utf-8')
    salvar_json_novo(rel/modulo.MANIFESTO_ARQ,{
        **meta,'origem_oof':folder.relative_to(ROOT).as_posix(),
        'arquivo_csv':modulo.TABELA_ARQ,'arquivo_histograma':modulo.HIST_ARQ,
        'sha256_csv':sha(rel/modulo.TABELA_ARQ),
        'sha256_html':sha(rel/modulo.PRE01_ARQ)})
    print('[F10 PRE01]',rel/modulo.PRE01_ARQ,flush=True)


GRAFICOS=(
    ('00_painel_score_composto.png','00 — Painel inicial'),
    ('01_curva_precision_recall.png','01 — Precision–Recall'),
    ('02_mcc_por_limiar.png','02 — MCC por limiar'),
    ('03_matriz_confusao_05.png','03 — Matriz τ=0,50'),
    ('04_matriz_confusao_mcc.png','04 — Matriz τ* OOF'),
    ('05_distribuicao_probabilidades.png','05 — Distribuicao das probabilidades'),
    ('12_pca_projecao_PC1.png','12 — Projecao PCA PC1'),
    ('PCA_OBSERVACOES_PC1.png','PCA — Observacoes PC1'),
    ('GMM01_densidade_marginal_1d_PC1.png','GMM01 — Densidade marginal'),
    ('GMM04_dominancia_1d.png','GMM04 — Dominancia gaussiana'),
    ('GMM07_sensibilidade_1d_comparativo.png','GMM07 — Sensibilidade da fronteira'),
    ('GMM08_mapa_risco_1d_comparativo.png','GMM08 — Mapa de risco e gaussianas 95%'),
)


def galeria_final(folder,ids,pacote,met):
    dest=folder/'vencedor/teste_final'
    figs=dest/'graficos_gerados';gal=dest/'galeria'
    for nome,_ in GRAFICOS:
        exigir((figs/nome).is_file(),f'Imagem incompleta: {nome}')
    pre='diagnosticos/PRE01_F10/PRE01_F10_diagnostico_padronizacao_completo.html'
    exigir((dest/pre).is_file(),'PRE01 ausente.')
    cards=''.join(
        f'<article><a href="../graficos_gerados/{escape(nome,quote=True)}" target="_blank">'
        f'<img src="../graficos_gerados/{escape(nome,quote=True)}" loading="lazy" alt="{escape(rotulo)}">'
        f'<p>{escape(rotulo)}</p></a></article>' for nome,rotulo in GRAFICOS)
    arquivos=['../predicoes_f10.csv','../metricas_f10.json','../dados_geometria/coordenadas_PC1_f10.csv',
              '../dados_geometria/resumo_padronizacao_f10.csv','../modelo_final_f1_f9/manifesto_modelo_final.json']
    links=''.join(f'<a href="{escape(arq,quote=True)}">{escape(Path(arq).name)}</a>' for arq in arquivos)
    html=f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Teste completo PCA 1D — F10 exploratorio</title><style>body{{font:15px system-ui,sans-serif;background:#f5f7fa;color:#172236;max-width:1400px;margin:auto;padding:20px}}
.grade{{display:grid;grid-template-columns:repeat(auto-fit,minmax(315px,1fr));gap:16px}}article{{background:white;border:1px solid #dae1e9;border-radius:10px;padding:12px;box-shadow:0 2px 8px #0001}}article img{{width:100%;aspect-ratio:4/3;object-fit:contain}}a{{color:#285783;text-decoration:none}}article p{{font-weight:600}}.aviso{{background:#e8edf4;border-left:4px solid #355877;padding:14px}}.links{{display:flex;gap:16px;flex-wrap:wrap}}</style></head><body>
<h1>TESTE COMPLETO — PCA 1D / GMM K4 n_init5</h1>
<p class="aviso">Execucao de 10% F1–F9 e 10% F10 (mesmos {len(ids)} indices de teste ja avaliados anteriormente). Resultados exploratorios: <b>F10 reutilizado, nao e teste independente inedito</b>. τ* OOF congelado = {pacote.limiar_oof:.17g}. As assinaturas decorativas foram omitidas dos novos PNGs; o carater exploratorio continua documentado aqui e nos manifestos.</p>
<p>F10: AP={met['ap']:.7f} | MCC em τ*={met['mcc_limiar_oof']:.7f} | LogLoss={met['logloss']:.7f} | Score={met['score_composto_limiar_oof']:.7f}</p>
<p><a href="../../visualizacao/galeria/index.html">Abrir galeria OOF completa</a> · <a href="../{pre}">Abrir PRE01 F10 interativo</a></p>
<div class="grade">{cards}</div><h2>Resultados e proveniencia</h2><div class="links">{links}</div>
</body></html>'''
    gal.mkdir(parents=True,exist_ok=True)
    with (gal/'index.html').open('x',encoding='utf-8') as f:f.write(html)
    hashes={f'graficos_gerados/{n}':sha(figs/n) for n,_ in GRAFICOS}
    salvar_json_novo(dest/'manifesto_teste_completo_pca1d_f10.json',{
        'tipo':'TESTE_COMPLETO_PCA1D_10PCT_EXPLORATORIO_REPETIDO',
        'amostra_f10_ja_vista_em_execucao_anterior':True,
        'nao_classificar_como_teste_independente_inedito':True,
        'n_dev':pacote.n_treino_original,'n_f10':len(ids),'tau_congelado_oof':pacote.limiar_oof,
        'codigo_executor_sha256':sha(Path(__file__).resolve()),
        'modelo_sha256':sha(dest/'modelo_final_f1_f9/modelo_final_f1_f9.joblib'),
        'predicoes_f10_sha256':sha(dest/'predicoes_f10.csv'),
        'metricas_f10_sha256':sha(dest/'metricas_f10.json'),
        'graficos_f10_sha256':hashes,
        'galeria':'galeria/index.html',
        'pre01':pre,
        'observacao':'F10 repetido deliberadamente para inspecao manual; nunca usado para escolher ou reajustar modelo.'})
    print('[GALERIA MANUAL]',gal/'index.html',flush=True)



def validar_automacao(config):
    """Falha antes do OOF quando a automacao solicitada nao e suportada."""
    exigir(parametros_ok(config),
           'Finalizacao automatica PCA1D exige fracao=.1, K4, n_init5, FULL, RUS16, seed42 e F10 reservado.')
    exigir(config.get('pca',{}).get('n_components') == [1],
           'Finalizacao automatica V1: apenas PCA 1D. Desative execucao.finalizacao_pca1d_automatica em 2D/3D.')
    for rel in ('src/treinamento_final_pca.py','src/avaliacao_final_pca.py',
                'src/pre01_v2_diagnostico.py','src/visualizacao_geometrica.py',
                'src/visualizacao_oof.py', 'src/visualizacao_gmm01.py',
                'src/visualizacao_gmm04.py','src/visualizacao_gmm07.py',
                'src/visualizacao_pca_biplot.py',
                'scripts/execucao/EXTRAIR_COORDENADAS_PCA1D_F10.py',
                'scripts/execucao/GERAR_PRE01_F10_PCA1D.py'):
        exigir((ROOT/rel).is_file(), f'Dependencia ausente para finalizacao automatica: {rel}')
    exigir((ROOT/config['dados']['arquivo']).is_file(), 'Parquet da configuracao nao encontrado.')
    return True


def selecionar_amostra_f10(state, folder):
    """Repete a regra de 10% por classe e semente 42 usada na Etapa 7.

    Os labels F10 sao usados somente para a amostragem predefinida e metricas.
    Se o F10 ja foi previsto neste ensaio, le SOMENTE seus indices salvos.
    """
    path=folder/'vencedor/teste_final/indices_f10_amostra.npy'
    folds=state['orig_folds']
    if path.exists():
        ids=np.load(path,allow_pickle=False)
        exigir(ids.ndim==1 and ids.size>0 and np.all(np.diff(ids)>0)
               and np.all(folds[ids]==10),'Indices F10 ja salvos invalidos.')
        return ids
    from src import orquestracao_oof as orq
    al=state['config']['dados']['coluna_alvo']
    y=orq._ler_alvo(state['parquet'],al)
    exigir(y.shape==folds.shape,'Folds e TARGET com dimensoes distintas.')
    f10=np.flatnonzero(folds==10)
    ys=y[f10]
    exigir(set(np.unique(ys).tolist())=={0,1},'F10 exige classes 0 e 1.')
    rng=np.random.default_rng(42)
    parts=[]
    for classe in (0,1):
        pos=f10[ys==classe]
        n=max(2,round(.1*len(pos)))
        exigir(n<=len(pos),'F10 sem observacoes suficientes de cada classe.')
        parts.append(rng.choice(pos,size=n,replace=False))
    ids=np.sort(np.concatenate(parts)).astype(np.int64)
    exigir(np.all(folds[ids]==10) and np.all(np.diff(ids)>0),'Amostragem F10 desalinhada.')
    print(f'[F10] Amostra estratificada fixa: {len(ids)} linhas; fraude={int(y[ids].sum())}; '
          'seed 42. F10 ja foi acessado anteriormente neste projeto.',flush=True)
    return ids


def concluir_ensaio(folder):
    """Completa um experimento OOF existente, usando seus arquivos congelados.

    O caminho `folder` vem diretamente do orquestrador, sem paths historicos.
    """
    with bloqueio():
        folder=Path(folder).resolve()
        estado=conferir_oof(folder)
        validar_automacao(estado['config'])
        return _concluir_ensaio(folder)


def _concluir_ensaio(folder):
    from src.treinamento_final_pca import carregar_pacote_confiavel
    folder=Path(folder).resolve()
    exigir(folder.is_relative_to((ROOT/'outputs').resolve()),'Ensaio deve estar em outputs.')
    state=conferir_oof(folder)
    gerar_pre01_oof(folder,state)
    dest=folder/'vencedor/teste_final'
    arq_modelo=dest/'modelo_final_f1_f9/modelo_final_f1_f9.joblib'
    if arq_modelo.is_file():
        exigir((dest/'modelo_final_f1_f9/manifesto_modelo_final.json').is_file(),
               'Modelo ja existe mas manifesto ausente: nao retomar automaticamente.')
        pacote=carregar_pacote_confiavel(arq_modelo.parent)
        print('[RETOMADA] Modelo final previamente salvo e verificado.',flush=True)
    else:
        exigir(not arq_modelo.parent.exists(),'Modelo incompleto: nao sobrescrever.')
        pacote=treinar_final(folder,state)
    # Acessar rotulos F10 somente DEPOIS do ajuste final F1-F9.
    ids=selecionar_amostra_f10(state,folder)
    if (dest/'predicoes_f10.csv').exists():
        exigir((dest/'metricas_f10.json').is_file(),'Predicoes F10 parciais; nao repetir.')
        tab=pd.read_csv(dest/'predicoes_f10.csv')
        exigir(np.array_equal(tab['linha_original'].to_numpy(dtype=np.int64),ids),'Indices F10 diferentes.')
        met=ler_json(dest/'metricas_f10.json')
        # Nesta fase os dados brutos sao lidos apenas para construir representacoes graficas.
        df=ler_linhas_f10(state['parquet'],ids)
        print('[RETOMADA] Avaliacao F10 ja salva. Sem refazer scores.',flush=True)
    else:
        df,tab,met=avaliar_f10(folder,state,pacote,ids)
    coord_arquivo=dest/'dados_geometria/coordenadas_PC1_f10.csv'
    if coord_arquivo.exists():
        coords=pd.read_csv(coord_arquivo)
        resumo=pd.read_csv(dest/'dados_geometria/resumo_padronizacao_f10.csv')
        exigir(np.array_equal(coords['linha_original'].to_numpy(dtype=np.int64),ids),
               'Coordenadas F10 desalinhadas.')
    else:
        exigir(not coord_arquivo.parent.exists(),'Dados geometricos incompletos; nao sobrescrever.')
        coords,resumo=preparar_geometria(folder,state,pacote,df,ids)
    figs=dest/'graficos_gerados'
    if figs.exists():
        # Nao sobreescrever figuras. Se ainda faltam arquivos, reportar a falha.
        for nome,_ in GRAFICOS:
            exigir((figs/nome).is_file(),
                   f'Pasta de graficos parcial. Preserve e informe o erro antes de continuar: {nome}')
        print('[RETOMADA] 12 figuras ja existentes, sem redesenhar.',flush=True)
    else:
        desenhar_preditivos(folder,state,tab,met)
        desenhar_geometria(folder,state,pacote,tab,coords,met)
    pre=dest/'diagnosticos/PRE01_F10'
    if not pre.exists():
        gerar_pre01(folder,state,resumo,coords,pacote,ids)
    else:
        exigir((pre/'PRE01_F10_diagnostico_padronizacao_completo.html').is_file(),
               'PRE01 incompleto.')
    if not (dest/'galeria/index.html').exists():
        galeria_final(folder,ids,pacote,met)
    else:
        print('[RETOMADA] Galeria final ja presente; nada sobrescrito.',flush=True)
    # Registrar conclusao no manifesto da mesma pasta de ensaio.
    from src import orquestracao_oof as orq
    mfinal=ler_json(folder/'manifesto.json')
    mfinal['pendente_em_outros_passos']=[]
    mfinal['teste_final']={
        'status':'concluido', 'dimensao_pca':1,
        'carater':'AMOSTRAL_EXPLORATORIO_F10_JA_UTILIZADO',
        'n_f10':int(len(ids)), 'sem_reotimizar_tau':True,
        'galeria':'vencedor/teste_final/galeria/index.html',
        'manifesto':'vencedor/teste_final/manifesto_teste_completo_pca1d_f10.json'
    }
    orq._salvar_json(folder/'manifesto.json',mfinal)
    print('\n[CONCLUIDO] TESTE COMPLETO PCA 1D, pasta:',folder,flush=True)
    print('[ABRIR] ',dest/'galeria/index.html',flush=True)
