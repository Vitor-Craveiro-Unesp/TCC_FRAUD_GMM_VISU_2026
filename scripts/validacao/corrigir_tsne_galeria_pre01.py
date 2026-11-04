# -*- coding: utf-8 -*-
"""Corrige TSNE02/13/14 e implementa Pearson/PRE01 no OOF t-SNE 1D real.
Nao toca F10, PCA, YAML ou modulos oficiais. Nao reexecuta OOF nem t-SNE.
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import yaml
from src import orquestracao_oof as orq
from src import preprocessing as prep
from src import pre01_v2_diagnostico as pre

ROOT=Path(__file__).resolve().parents[2]
NAMES=('TSNE02_trustworthiness_vs_perplexity','13_tsne_mapa_TSNE1','14_tsne_diagnostico')
PEARSON='TSNE_CORRELACAO_PEARSON_TODAS_FEATURES'

def require(x,msg):
    if not x:raise RuntimeError(msg)

def sha(path):
    h=sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save_atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tsne_v1_tmp')
    require(not temp.exists(),f'Temporario existente: {temp}')
    try:
        with temp.open('xb') as f:f.write(data)
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)

def locate(exp):
    if exp:return Path(exp).resolve()
    candidates=[]
    for p in (ROOT/'outputs').glob('ensaio_tsne_gmm_*'):
        try:
            c=yaml.safe_load((p/'configuracao.yaml').read_text(encoding='utf-8-sig'))
            m=json.loads((p/'manifesto.json').read_text(encoding='utf-8-sig'))
            if c['reducao']['metodo'].upper()=='TSNE' and m['status']=='concluido_06_03':
                candidates.append(p)
        except (OSError,ValueError,KeyError):pass
    require(candidates,'Nenhum ensaio t-SNE concluido encontrado.')
    return max(candidates,key=lambda x:x.stat().st_mtime).resolve()

def validar(exp):
    require(exp.is_relative_to((ROOT/'outputs').resolve()),'Pasta fora de outputs!')
    cfg=yaml.safe_load((exp/'configuracao.yaml').read_text(encoding='utf-8-sig'))
    m=json.loads((exp/'manifesto.json').read_text(encoding='utf-8-sig'))
    winner=json.loads((exp/'vencedor/metricas/melhor_configuracao.json').read_text(encoding='utf-8-sig'))
    require(cfg['reducao']['metodo'].upper()=='TSNE' and winner['configuracao']['metodo']=='TSNE' and winner['configuracao']['reducao']['n_components']==1,'Exige vencedor t-SNE 1D.')
    require(m['status']=='concluido_06_03' and m.get('visualizacoes')=='concluidas','Galeria OOF incompleta.')
    require(winner['teste_final_usado'] is False and cfg['divisao']['qual_fold_teste']==10,'F10 deve estar reservado.')
    with np.load(exp/'vencedor/oof/predicoes.npz',allow_pickle=False) as npz:
        ids=np.asarray(npz['linha_original'],dtype=np.int64)
        folds=np.asarray(npz['fold'],dtype=np.uint8)
        y=np.asarray(npz['y_true'],dtype=np.uint8)
        require(int(npz['fold_esperado'])==0,'Predicoes nao OOF')
    require(len(ids)>0 and np.all(np.diff(ids)>0) and set(folds)==set(range(1,10)),'IDs/folds OOF invalidos.')
    return cfg,m,winner,ids,folds,y

def clean02(df,winner,path):
    p=float(winner['configuracao']['reducao']['perplexity'])
    k=int(df['k_vizinhos'].iloc[0])
    fig,ax=plt.subplots(figsize=(10.5,6.2))
    ax.plot(df.perplexity,df.trustworthiness,'o-',lw=2.1,ms=6.5,color='#1565c0',label=f'Trustworthiness (k={k})')
    for a,b in zip(df.perplexity,df.trustworthiness):ax.annotate(f'{b:.3f}',(a,b),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8.5)
    ax.axvline(p,color='#d97706',ls='--',lw=1.4,label=f'Perplexity do vencedor OOF = {p:g}')
    ax.set_xticks(df.perplexity.to_list());ax.set_ylim(max(0.,df.trustworthiness.min()-.025),1.003)
    ax.set(xlabel='Perplexity',ylabel='Trustworthiness (0 a 1; eixo ampliado)')
    ax.set_title('Preservação de vizinhanças | Trustworthiness × Perplexity')
    ax.grid(alpha=.2);ax.legend(loc='best',fontsize=9);fig.tight_layout()
    fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)

def clean13(df,cfg,path):
    from scipy.ndimage import gaussian_filter1d
    op=cfg['visualizacao']['graficos']['tsne_mapa']
    require(op.get('colorir_por','classe_real')=='classe_real','TSNE13 requer colorir_por=classe_real')
    xx=df.TSNE1.to_numpy(float); yy=df.classe_real.to_numpy(int)
    require(np.isfinite(xx).all() and set(yy)=={0,1},'TSNE13 invalido')
    nb=int(op.get('bins_histograma_1d',36)); edges=np.linspace(xx.min(),xx.max(),nb+1)
    centers=.5*(edges[:-1]+edges[1:]);w=edges[1]-edges[0]
    fig,ax=plt.subplots(figsize=tuple(cfg['visualizacao']['padrao'].get('figsize',[9,6])))
    for cl,name in [(0,'legitima'),(1,'fraude')]:
        z=xx[yy==cl];e=cfg['visualizacao']['classes'][name];color=e.get('cor','#1f77b4' if cl==0 else '#d62728')
        d,_=np.histogram(z,bins=edges,density=True)
        ax.bar(centers,d,width=w*.94,color=color,alpha=.38,edgecolor=color,linewidth=.4,label=f"{e.get('nome',name)} (n={len(z):,})",zorder=2)
        if len(z)>=3 and op.get('curvas_suavizadas_1d',True):ax.plot(centers,gaussian_filter1d(d,1.15,mode='nearest'),color=color,lw=1.8,zorder=4)
    ax.set_title('t-SNE 1D | Histograma do embedding');ax.set_xlabel('t-SNE 1 (coordenada do embedding; sem variância explicada)')
    ax.set_ylabel('Densidade por grupo (área = 1)');ax.grid(axis='y',alpha=.18,linestyle=':');ax.legend(loc='best',fontsize=8)
    fig.tight_layout();fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)

def clean14(df,path):
    require(len(df)==1,'TSNE14 requer uma linha de diagnostico')
    r=df.iloc[0];fig,axes=plt.subplots(1,2,figsize=(11.8,4.7));fig.patch.set_facecolor('#f8fafc')
    for ax in axes:
        ax.set_facecolor('#f1f5f9');ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_color('#cbd5e1')
    a,b=axes
    a.text(.06,.88,'DIVERGÊNCIA KL FINAL',transform=a.transAxes,fontweight='bold',color='#334155',fontsize=12)
    kl=None if pd.isna(r.kl_final) else float(r.kl_final)
    a.text(.06,.52,f'{kl:.6g}' if kl is not None else 'Indisponível',transform=a.transAxes,color='#1d4ed8',fontsize=24,weight='bold')
    a.text(.06,.22,'Métrica de otimização interna do t-SNE.\nNão mede desempenho de detecção de fraude.',transform=a.transAxes,color='#475569',fontsize=9.5)
    b.text(.06,.88,'TEMPO REAL DA REDUÇÃO',transform=b.transAxes,fontweight='bold',color='#334155',fontsize=12)
    b.text(.06,.52,f'{r.tempo_reducao_s:.2f} s',transform=b.transAxes,color='#047857',fontsize=24,weight='bold')
    b.text(.06,.22,f'Embedding: {int(r.n_componentes)}D | perplexity={r.perplexity:g}\n{int(r.n_desenvolvimento):,} observações | métrica={r.metrica}',transform=b.transAxes,color='#475569',fontsize=9.5)
    fig.suptitle('t-SNE | Diagnóstico KL / otimização',fontsize=15,weight='bold',y=.985)
    fig.subplots_adjust(left=.055,right=.945,top=.83,bottom=.12,wspace=.085)
    fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)

def corr_figure(tab,path):
    labels=tab.variavel_preprocessada.to_list();n=len(labels);h=max(7,1.8+.40*n)
    fig,ax=plt.subplots(figsize=(14,h));ax.axvline(0,color='#78818e',lw=1)
    for i,val in enumerate(tab.corr_Pearson_TSNE1.to_numpy(float)):
        if not np.isfinite(val):
            ax.text(.045,i,'n/d (constante)',ha='left',va='center',fontsize=9,color='#777777');continue
        color='#138276' if val>=0 else '#8051a8'
        ax.annotate('',xy=(val,i),xytext=(0,i),arrowprops=dict(arrowstyle='-|>',lw=1.9,color=color,mutation_scale=12))
        v=0. if abs(val)<5e-8 else val
        ax.text(val+(.045 if val>=0 else -.045),i,f'{v:+.7f}',ha='left' if val>=0 else 'right',va='center',color=color,fontsize=9)
    ax.set_yticks(np.arange(n),labels=labels);ax.set_ylim(n-.85,-.85);ax.set_xlim(-1.4,1.4)
    ax.set_xlabel('Correlação de Pearson com TSNE1',fontsize=11,labelpad=12)
    ax.set_xticks([-1,-.5,0,.5,1]);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{(0. if abs(x)<5e-8 else x):.7f}'))
    ax.tick_params(axis='x',labelsize=9);ax.tick_params(axis='y',labelsize=9,length=0,pad=8)
    ax.set_axisbelow(True);ax.grid(axis='x',alpha=.2)
    fig.suptitle('CORRELAÇÃO DAS VARIÁVEIS COM TSNE1',fontsize=17,fontweight='bold',y=.995)
    fig.subplots_adjust(left=.30,right=.96,top=1-.66/h,bottom=min(.13,.98/h))
    fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)

def gallery_index(original):
    txt=pre.new_gallery_index(original)
    txt=txt.replace('pós-padronização e estabilidade PCA;','pós-padronização e PCA auxiliar;')
    txt=txt.replace('escalas, folds e PCA</div>','escalas, folds e PCA auxiliar</div>')
    marker='<!-- TSNE_PEARSON_TODAS_FEATURES_V1 -->'
    require(marker not in txt,'Grafico de Pearson ja vinculado nesta galeria.')
    card=(marker+'\n<section><h2>t-SNE 1D — Correlação de Pearson com todas as variáveis</h2>'
          '<p>Associação descritiva das features preprocessadas com TSNE1, não coeficientes aprendidos pelo t-SNE.</p>'
          '<div class="grade"><article><a href="geometria_desenvolvimento/'+PEARSON+'.png" target="_blank" rel="noopener">'
          '<img src="geometria_desenvolvimento/'+PEARSON+'.png" alt="Pearson entre todas as features e TSNE1">'
          '<div>Pearson — todas as features (TSNE1)</div></a></article></div></section>\n')
    require('</body></html>' in txt,'Galeria HTML nao reconhecida')
    return txt.replace('</body></html>',card+'</body></html>')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--planejar',action='store_true');actions.add_argument('--executar',action='store_true')
    parser.add_argument('--experimento',type=Path)
    a=parser.parse_args();os.chdir(ROOT)
    exp=locate(a.experimento)
    cfg,m,winner,ids,folds,y=validar(exp)
    view=exp/'vencedor/visualizacao';geom=view/'geometria_desenvolvimento'
    gal=view/'galeria';gal_geo=gal/'geometria_desenvolvimento';index=gal/'index.html'
    require(index.is_file(),'Galeria HTML OOF nao encontrada')
    for name in NAMES:
        require((geom/(name+'.csv')).is_file(),f'CSV original ausente: {name}')
        require((geom/(name+'.png')).is_file() and (gal_geo/(name+'.png')).is_file(),f'PNG ausente: {name}')
    require(not (geom/(PEARSON+'.png')).exists(),'Pearson ja integrado; nao sobrescrever')
    # PRE01 antigo, se existir, sera preservado no backup antes de qualquer substituicao.
    print('EXPERIMENTO_TSNE='+str(exp),flush=True)
    print('REGISTROS_F1_F9='+str(len(ids)),flush=True)
    if a.planejar:
        print('REMOVER_TEXTO_ROSA_TSNE02_13_14=SIM')
        print('GERAR_PEARSON_DE_TODAS_FEATURES=SIM')
        print('GERAR_PRE01_HTML_INTERATIVO_COMPLETO=SIM')
        print('ACESSAR_F10=NAO; TREINAR_TSNE_OOF=NAO; EDITAR_CODIGO_PCA=NAO')
        print('PLANEJAMENTO_EXIT_CODE=0')
        return 0

    # Todos os artefatos sao preparados ANTES de editar a galeria.
    path=ROOT/'outputs'/('validacao_tsne_pre01_galeria_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    stage=path/'NOVOS';stage.mkdir(parents=True,exist_ok=False)
    clean02(pd.read_csv(geom/(NAMES[0]+'.csv')),winner,stage/(NAMES[0]+'.png'))
    d13=pd.read_csv(geom/(NAMES[1]+'.csv'))
    clean13(d13,cfg,stage/(NAMES[1]+'.png'))
    clean14(pd.read_csv(geom/(NAMES[2]+'.csv')),stage/(NAMES[2]+'.png'))
    for name in NAMES:require((stage/(name+'.png')).stat().st_size>1000,'PNG de previa vazio')
    require(len(d13)==len(ids),'TSNE13 e OOF tem tamanhos diferentes')
    np.testing.assert_array_equal(d13.classe_real.to_numpy(dtype=np.uint8),y,err_msg='TSNE13 e OOF nao alinhados')
    print('[OK] Tres PNGs sem textos marcados preparados',flush=True)

    # F1-F9 autenticados: so leitura seletiva, NUNCA carregar F10 no DataFrame.
    parquet=ROOT/cfg['dados']['arquivo']; orig,fold_path,hash_pq=orq._identificar_folds(ROOT,cfg,parquet)
    require(np.array_equal(orig[ids],folds),'Folds F1-F9 nao correspondem ao OOF')
    require(hash_pq==m['sha256_parquet'] and orq.hash_arquivo(fold_path)==m['sha256_folds'],'Hashes Parquet/folds divergentes')
    df=orq._ler_desenvolvimento(parquet,ids)
    np.testing.assert_array_equal(df[cfg['dados']['coluna_alvo']].to_numpy(dtype=np.uint8),y)
    X=prep.selecionar_features(df,cfg);del df

    # MESMO embedding TSNE1 real do grafico 13, sem novo ajuste TSNE.
    enc=prep.EncodersPorSufixo(cfg).fit(X)
    values=enc.transform(X)
    transformed=prep.ImputadorPadronizador(cfg,enc).fit_transform(values)
    del values
    require(transformed.index.equals(pd.RangeIndex(len(ids))),'Ordem das features nao posicional')
    z=pd.Series(d13.TSNE1.to_numpy(dtype=float),index=transformed.index)
    correlations=transformed.corrwith(z,method='pearson')
    const=transformed.nunique(dropna=False).le(1)
    require(np.isfinite(correlations.loc[~const]).all(),'Pearson invalida nas colunas nao constantes')
    table=pd.DataFrame({'variavel_preprocessada':transformed.columns,
                        'corr_Pearson_TSNE1':correlations.to_numpy(),
                        'constante_na_amostra':const.to_numpy()})
    table['modulo']=table.corr_Pearson_TSNE1.abs()
    table=table.sort_values(['modulo','variavel_preprocessada'],ascending=[False,True],na_position='last',kind='stable').reset_index(drop=True)
    require(len(table)==transformed.shape[1] and table.variavel_preprocessada.is_unique,'Nem todas as features foram mantidas')
    table.to_csv(stage/(PEARSON+'.csv'),index=False,encoding='utf-8-sig',float_format='%.17g')
    corr_figure(table,stage/(PEARSON+'.png'));del transformed
    print('TODAS_FEATURES_CORRELACIONADAS='+str(len(table)),flush=True)

    # HTML original PRE01: estatisticas dos 9 folds + PCA puramente auxiliar.
    summary,detail,fold_rows,pca_diag=pre.calc(X,folds,cfg);del X
    require(len(fold_rows)==9 and len(summary)>=1 and len(detail)>=9*len(summary),'PRE01 incompleto')
    meta={'tipo':m['tipo'],'metodo_principal':'TSNE','dimensao_principal':1,
          'k':4,'n_init':5,'total_observacoes':int(len(ids)),
          'total_features':int(len(summary)),'f10_usado':False,'mesmos_registros_e_folds':True,
          'experimentos':{'TSNE1D':exp.relative_to(ROOT).as_posix()},
          'sha256_parquet':hash_pq,'sha256_folds':m['sha256_folds'],
          'medida_de_desempenho':False,
          'pca_ajuste_qualificacao':'PCA auxiliar do preprocessamento; nao e o redutor TSNE',
          'nota':'Pearson descritivo com TSNE1; nao existem coeficientes lineares t-SNE.'}
    html=pre.build_html(summary,detail,fold_rows,pca_diag,meta)
    html=html.replace('<h1>PRE01 — Diagnóstico completo da padronização</h1>',
                      '<h1>PRE01 — Diagnóstico completo da padronização | t-SNE 1D</h1>')
    html=html.replace('5. Impacto diagnóstico no PCA (PC1–PC3)',
                      '5. PCA auxiliar — estabilidade das entradas (PC1–PC3)')
    html=html.replace('PCA diagnóstico</a>','PCA auxiliar</a>')
    html=html.replace('<p class="lead">',
                      '<p class="lead"><strong>Modelo principal: t-SNE 1D. A seção 5 usa PCA auxiliar apenas para diagnóstico; não representa coeficientes t-SNE.</strong> ')
    require('PRE01_DATA' in html and 'table-details' in html and 'plot-pca' in html,'PRE01 interativo incompleto')
    blobs={pre.FILENAME:html.encode('utf-8'),
           pre.CSV_SUM:summary.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
           pre.CSV_DETAIL:detail.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
           pre.CSV_FOLDS:fold_rows.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
           pre.CSV_PCA:pca_diag.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
           pre.META:(json.dumps(meta,ensure_ascii=False,indent=2)+'\n').encode('utf-8')}
    for name,data in blobs.items():save_atomic(stage/name,data)
    updated=gallery_index(index.read_text(encoding='utf-8'))
    save_atomic(stage/'index.html',updated.encode('utf-8'))
    print('[OK] PRE01 completo e index da galeria preparados',flush=True)

    # Backup antes de qualquer troca. Regras: jamais sobrescrever PCA/YAML/src.
    dest=[]
    for name in NAMES:
        raw=(stage/(name+'.png')).read_bytes()
        for folder in (geom,gal_geo):dest.append((folder/(name+'.png'),raw))
    dest.append((geom/(PEARSON+'.png'),(stage/(PEARSON+'.png')).read_bytes()))
    dest.append((geom/(PEARSON+'.csv'),(stage/(PEARSON+'.csv')).read_bytes()))
    dest.append((gal_geo/(PEARSON+'.png'),(stage/(PEARSON+'.png')).read_bytes()))
    for folder in (view/'oof',gal/'oof'):
        for name,raw in blobs.items():dest.append((folder/name,raw))
    dest.append((index,updated.encode('utf-8')))
    snapshots=[]
    for target,raw in dest:
        require(not target.is_symlink(),'Destino symlink bloqueado: '+str(target))
        backup=None
        if target.exists():
            backup=path/'BACKUP'/target.relative_to(ROOT)
            backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(target,backup)
            require(sha(target)==sha(backup),'Backup invalido: '+str(target))
        snapshots.append((target,backup))
    manifest={'experimento_tsne':str(exp),'registros_oof':int(len(ids)),
              'n_features_exibidas':int(len(table)),
              'origem_dos_valores':'F1-F9: mesmas linhas OOF, embedding TSNE13, preprocessamento oficial',
              'pre01':'9 folds, PCA auxiliar, HTML interativo completo',
              'arquivos_atualizados':[str(p.relative_to(ROOT)) for p,_ in dest],
              'backup':str(path/'BACKUP'),'f10_usado':False,'novos_ajustes_tsne_gmm':0}
    save_atomic(path/'manifesto_reparo.json',(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    written=[]
    try:
        for target,raw in dest:
            save_atomic(target,raw);written.append(target)
    except Exception:
        for target,backup in reversed(snapshots):
            if target not in written:continue
            if backup is None:target.unlink(missing_ok=True)
            else:save_atomic(target,backup.read_bytes())
        raise
    require((gal/'oof'/pre.FILENAME).is_file(),'PRE01 nao presente')
    require((gal_geo/(PEARSON+'.png')).is_file(),'Pearson nao presente')
    require('TSNE_PEARSON_TODAS_FEATURES_V1' in index.read_text(encoding='utf-8'),'Indice nao referencia Pearson')
    print('=== GALERIA TSNE 1D ATUALIZADA ===',flush=True)
    print('TSNE02_TSNE13_TSNE14_TITULOS_LIMPOS=SIM')
    print(f'TSNE1_PEARSON_N_FEATURES={len(table)}')
    print('COEFICIENTES_LINEARES_DO_TSNE=INEXISTENTES')
    print('PRE01_INTERATIVO_COMPLETO=SIM')
    print('REAJUSTES_TSNE_GMM=0')
    print('F10_ACESSADO=NAO')
    print('CODIGO_PCA_YAML_ALTERADOS=NAO')
    print('BACKUP='+str(path/'BACKUP'))
    print('GALERIA='+str(index))
    print('TSNE_GALERIA_V1_EXIT_CODE=0')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
