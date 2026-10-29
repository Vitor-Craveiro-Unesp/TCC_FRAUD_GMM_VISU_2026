# -*- coding: utf-8 -*-
"""PRE01 — estabilidade do encoder, imputador e StandardScaler entre folds.

Calcula apenas parâmetros de pré-processamento sobre o desenvolvimento
OOF já finalizado, sem treinar PCA/GMM, sem avaliar F10 e sem alterar
escolha/score/limiar. A referência global F1..F9 é SOMENTE diagnóstica.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import html
import json
import os
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import yaml

from src import orquestracao_oof as orq
from src import preprocessing as prep

PNG = 'PRE01_estabilidade_padronizacao_folds.png'
CSV = 'PRE01_estabilidade_padronizacao_folds.csv'
META = 'PRE01_estabilidade_padronizacao_folds_manifesto.json'
MARCADOR = '<!-- PRE01_DIAGNOSTICO_PADRONIZACAO_FOLDS -->'


def _exigir(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def _dim_config(cfg):
    vals=cfg['pca']['n_components']
    if isinstance(vals, list):
        if len(vals) != 1: return None
        vals=vals[0]
    return int(vals)


def _k_config(cfg):
    vals=cfg['gmm']['n_components']
    if isinstance(vals, list):
        if len(vals) != 1: return None
        vals=vals[0]
    return int(vals)


def _chave_comparavel(cfg):
    c=deepcopy(cfg)
    c['pca']['n_components']=['UNICA_DIMENSAO_VARIAVEL']
    return json.dumps(c,ensure_ascii=False,sort_keys=True)


def _ler_candidato(pasta, k, n_init, tipo):
    try:
        cfg=yaml.safe_load((pasta/'configuracao.yaml').read_text(encoding='utf-8-sig'))
        m=json.loads((pasta/'manifesto.json').read_text(encoding='utf-8-sig'))
        if (m.get('status')!='concluido_06_03' or m.get('visualizacoes')!='concluidas'
            or cfg['reducao']['metodo'].upper()!='PCA'
            or cfg['divisao']['qual_fold_teste']!=10 or m.get('tipo')!=tipo
            or _k_config(cfg)!=k or int(cfg['gmm']['n_init'])!=n_init):
            return None
        d=_dim_config(cfg)
        if d not in (1,2,3):return None
        npz=pasta/'vencedor'/'oof'/'predicoes.npz'
        galeria=pasta/'vencedor'/'visualizacao'/'galeria'/'index.html'
        if not npz.is_file() or not galeria.is_file():return None
        return {'pasta':pasta,'cfg':cfg,'manifesto':m,'dim':d,'npz':npz,'galeria':galeria,
                'chave':_chave_comparavel(cfg)}
    except (OSError, KeyError, ValueError, json.JSONDecodeError, yaml.YAMLError):
        return None


def _selecionar_trio(raiz, k, n_init, tipo):
    todos=[]
    for pasta in (raiz/'outputs').glob(('experimento' if tipo=='oficial' else 'ensaio')+'_pca_gmm_*'):
        if pasta.is_dir():
            c=_ler_candidato(pasta,k,n_init,tipo)
            if c is not None:todos.append(c)
    grupos={}
    for item in todos:
        grupos.setdefault(item['chave'],[]).append(item)
    trios=[]
    for grupo in grupos.values():
        por_dim={}
        for item in sorted(grupo,key=lambda v:v['pasta'].stat().st_mtime,reverse=True):
            por_dim.setdefault(item['dim'],item)
        if set(por_dim)=={1,2,3}:
            trios.append([por_dim[d] for d in (1,2,3)])
    _exigir(trios,f'Nenhum trio PCA 1D/2D/3D do tipo {tipo} com K={k}, n_init={n_init} e YAML equivalente. Confira os experimentos em outputs/.')
    return max(trios,key=lambda t:max(v['pasta'].stat().st_mtime for v in t))


def _indices_iguais(trio):
    base=None
    for t in trio:
        with np.load(t['npz'],allow_pickle=False) as z:
            ids=np.asarray(z['linha_original'],dtype=np.int64)
            folds=np.asarray(z['fold'],dtype=np.uint8)
            _exigir(int(z['fold_esperado'].item())==0,'NPZ nao e o OOF consolidado.')
        _exigir(len(ids)>0 and np.all(np.diff(ids)>0),'Indices OOF ausentes, repetidos ou desordenados.')
        _exigir(set(np.unique(folds).tolist())==set(range(1,10)), 'OOF sem exatamente F1..F9.')
        if base is None:base=(ids,folds)
        else:
            _exigir(np.array_equal(ids,base[0]) and np.array_equal(folds,base[1]),
                    'As tres dimensoes usam registros/folds diferentes. Nao replicar o mesmo diagnostico.')
        _exigir(t['manifesto'].get('sha256_parquet')==trio[0]['manifesto'].get('sha256_parquet')
                and t['manifesto'].get('sha256_folds')==trio[0]['manifesto'].get('sha256_folds'),
                'Hashes originais de dados ou folds divergem.')
        print(f"[VALIDADO] PCA {t['dim']}D: {len(ids):,} registros OOF.",flush=True)
    return base


def _parametros_standardscaler(x, cfg):
    enc=prep.EncodersPorSufixo(cfg).fit(x)
    encoded=enc.transform(x)
    imp=prep.ImputadorPadronizador(cfg,enc).fit(encoded)
    nomes=list(imp.padronizadas_)
    _exigir(imp.scaler_ is not None and len(nomes)>0,'Nenhuma variavel numerica/FEWF padronizada.')
    medias=np.asarray(imp.scaler_.mean_,dtype=np.float64).copy()
    escalas=np.asarray(imp.scaler_.scale_,dtype=np.float64).copy()
    _exigir(len(nomes)==len(medias)==len(escalas),'Parametros scaler desalinhados.')
    _exigir((np.isfinite(medias)).all() and (np.isfinite(escalas)).all() and (escalas>0).all(),
             'Scaler com parametros invalidos.')
    return nomes,medias,escalas


def _calcular(df,folds,cfg):
    x=prep.selecionar_features(df,cfg)
    del df
    print('[PRE01] Ajustando somente encoder/imputador/scaler de referencia F1..F9.',flush=True)
    nomes,media_ref,escala_ref=_parametros_standardscaler(x,cfg)
    mm=np.empty((9,len(nomes)),dtype=np.float64)
    ss=np.empty_like(mm)
    counts=[]
    for k in range(1,10):
        mask=(folds!=k)
        print(f'[PRE01] F{k}: parametros ajustados exclusivamente em outros 8 folds: {int(mask.sum()):,} linhas.',flush=True)
        nomes_k,mm_k,ss_k=_parametros_standardscaler(x.loc[mask],cfg)
        _exigir(nomes_k==nomes,f'Colunas padronizadas diferentes no fold {k}.')
        mm[k-1]=mm_k
        ss[k-1]=ss_k
        counts.append(int(mask.sum()))
    delta_media=(mm-media_ref[None,:])/escala_ref[None,:]
    delta_log_escala=np.log(ss/escala_ref[None,:])
    _exigir(np.isfinite(delta_media).all() and np.isfinite(delta_log_escala).all(),
            'Diferencas de padronizacao nao finitas.')
    registros=[]
    for i,k in enumerate(range(1,10)):
        for j,n in enumerate(nomes):
            registros.append({'fold_validacao':k,'folds_ajuste':f'F1-F9 exceto F{k}',
                'n_treino_padronizador':counts[i],'feature':n,
                'tipo':'FEWF' if n.endswith(cfg['preprocessamento']['encodings']['frequency']['sufixo']) else 'numerica',
                'media_8_folds':mm[i,j], 'media_9_folds':media_ref[j],
                'escala_8_folds':ss[i,j], 'escala_9_folds':escala_ref[j],
                'delta_media_padronizada':delta_media[i,j],
                'delta_log_escala':delta_log_escala[i,j]})
    return pd.DataFrame.from_records(registros),nomes,delta_media,delta_log_escala


def _figura(nomes,dm,ds,meta):
    n=len(nomes)
    largura=max(19,min(34,13+.48*n))
    fig,ax=plt.subplots(1,2,figsize=(largura,9.4),facecolor='white')
    fig.subplots_adjust(left=.040,right=.965,bottom=.32,top=.80,wspace=.18)
    fig.suptitle('ESTABILIDADE DA PADRONIZACAO ENTRE FOLDS',fontsize=20,
                 fontweight='bold',color='#1d3048',y=.965)
    fig.text(.5,.91,'Ajustes OOF com 8 folds  versus  referencia global F1-F9 (somente diagnostica)',
             ha='center',fontsize=12,color='#455a73')
    fig.text(.5,.86,f"{meta['tipo']}  |  K={meta['k']}  |  n_init={meta['n_init']}  |  "
             f"{meta['total_registros']:,} observacoes de desenvolvimento  |  F10 nao utilizado",
             ha='center',fontsize=10.3,color='#4d6178')
    for a,arr,title,bar in zip(ax,[dm,ds],
        ['A — Deslocamento das medias','B — Mudanca relativa das escalas (log)'],
        ['(media 8F - media 9F) / escala 9F','ln(escala 8F / escala 9F)']):
        max_abs=max(float(np.max(np.abs(arr))),1e-8)
        norm=mcolors.TwoSlopeNorm(vmin=-max_abs,vcenter=0.,vmax=max_abs)
        im=a.imshow(arr,cmap='RdBu_r',norm=norm,interpolation='nearest',aspect='auto')
        a.set_title(title,fontsize=13,fontweight='bold',pad=12)
        a.set_yticks(np.arange(9),[f'F{k}' for k in range(1,10)])
        a.set_ylabel('Fold de validacao',fontsize=11)
        a.set_xticks(np.arange(n),nomes,rotation=70,ha='right',rotation_mode='anchor',fontsize=8.2)
        a.tick_params(axis='y',labelsize=9)
        a.set_xticks(np.arange(-.5,n,1),minor=True)
        a.set_yticks(np.arange(-.5,9,1),minor=True)
        a.grid(which='minor',color='white',alpha=.38,lw=.55)
        a.tick_params(which='minor',length=0)
        cb=fig.colorbar(im,ax=a,location='right',fraction=.025,pad=.013)
        cb.set_label(bar,fontsize=9)
        cb.ax.tick_params(labelsize=8)
    fig.text(.5,.058,'Somente variaveis numericas e FEWF recebem StandardScaler. '
             'Os encoders e medianas tambem sao aprendidos em cada conjunto de treino.',
             ha='center',fontsize=10,color='#455a73')
    fig.text(.5,.031,'Interpretacao: zero indica parametros iguais aos do desenvolvimento F1-F9. '
             'Este diagnostico nao modifica as previsoes OOF.',
             ha='center',fontsize=10,color='#455a73')
    return fig


def _backup_arquivo(pasta_backup,arquivo,raiz):
    if not arquivo.is_file():return
    try:rel=arquivo.relative_to(raiz)
    except ValueError:rel=Path(arquivo.name)
    dst=pasta_backup/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(arquivo,dst)


def _inserir_html(texto):
    if MARCADOR in texto:
        return texto
    _exigir('</body></html>' in texto,'Formato do index.html da galeria nao reconhecido.')
    ref='oof/'+PNG
    fragmento=(MARCADOR+'\n<section><h2>PRE01 — Estabilidade da padronização entre folds</h2>'
      '<p>Diagnóstico somente de desenvolvimento: parâmetros do pré-processamento '
      'OOF (oito folds de ajuste) comparados à referência F1–F9. Não é treino '
      'alternativo nem estimativa de desempenho. F10 não utilizado.</p>'
      '<div class="grade"><article><a href="'+html.escape(ref,quote=True)+'" '
      'target="_blank" rel="noopener">'
      '<img src="'+html.escape(ref,quote=True)+'" loading="lazy" '
      'alt="Estabilidade da padronização entre folds">'
      '<div>PRE01 — Estabilidade da padronização entre folds (8F vs 9F)</div>'
      '</a></article></div></section>\n')
    return texto.replace('</body></html>',fragmento+'</body></html>')


def _integracao_futura(raiz,backup):
    caminho=raiz/'src'/'visualizacao_integrada.py'
    texto=caminho.read_text(encoding='utf-8')
    alvo="permitido=arq.stem in (oof.ARQUIVOS[n] for n in ativos_oof)"
    novo=alvo+" or arq.name == '"+PNG+"'"
    if novo in texto:
        print('[FUTURO] Integracao da PRE01 ja habilitada.',flush=True)
        return
    _exigir(texto.count(alvo)==1,'Nao foi possivel localizar unicamente o filtro OOF da galeria; codigo preservado.')
    _backup_arquivo(backup,caminho,raiz)
    caminho.write_text(texto.replace(alvo,novo),encoding='utf-8')
    import ast
    try:ast.parse(caminho.read_text(encoding='utf-8'))
    except Exception:
        shutil.copy2(backup/caminho.relative_to(raiz),caminho)
        raise
    print('[FUTURO] Galerias regeneradas incluirao PRE01 automaticamente.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raiz',type=Path,required=True)
    parser.add_argument('--k',type=int,default=4)
    parser.add_argument('--n-init',type=int,default=5)
    parser.add_argument('--tipo',choices=['oficial','ensaio'],default='oficial')
    args=parser.parse_args()
    raiz=args.raiz.resolve()
    _exigir((raiz/'src'/'visualizacao_oof.py').is_file(),'Raiz de projeto invalida.')
    os.chdir(raiz)
    trio=_selecionar_trio(raiz,args.k,args.n_init,args.tipo)
    for t in trio:print(f"[TRIO] PCA {t['dim']}D: {t['pasta'].name}",flush=True)
    ids,fold_oof=_indices_iguais(trio)
    cfg=trio[0]['cfg']
    _exigir((fold_oof!=10).all() and (fold_oof>=1).all(), 'F10 presente nos indices OOF.')
    _exigir(trio[0]['manifesto'].get('sha256_parquet'),'Manifesto sem hash de dados.')
    _exigir(trio[0]['manifesto'].get('sha256_folds'),'Manifesto sem hash de folds.')
    base=raiz/cfg['dados']['arquivo']
    print('[VERIFICACAO] Conferindo hashes da base e atribuicao original dos folds.',flush=True)
    todos_folds,arquivo_fold,hash_base=orq._identificar_folds(raiz,cfg,base)
    _exigir(trio[0]['manifesto']['sha256_parquet']==hash_base,
            'Hash do Parquet nao corresponde ao experimento.')
    _exigir(trio[0]['manifesto']['sha256_folds']==orq.hash_arquivo(arquivo_fold),
            'Hash dos folds nao corresponde ao experimento.')
    _exigir(np.array_equal(todos_folds[ids],fold_oof) and (todos_folds[ids]!=10).all(),
            'F10 ou indices de fold inconsistentes. Analise interrompida.')
    print('[DADOS] Lendo apenas os registros OOF F1..F9 ja selecionados.',flush=True)
    df=orq._ler_desenvolvimento(base,ids)
    _exigir(len(df)==len(ids),'Registros de desenvolvimento incompletos.')
    resultado,nomes,dm,ds=_calcular(df,fold_oof,cfg)
    meta={'tipo':args.tipo,'k':args.k,'n_init':args.n_init,
          'total_registros':int(len(ids)),'n_features':len(nomes),
          'fonte':'predicoes.npz, folds.npy e dataset_raw.parquet; somente F1..F9',
          'comparacao':'OOF 8 folds vs referencia F1..F9; somente diagnostica',
          'teste_f10_usado':False,
          'experimentos':{str(t['dim'])+'D':str(t['pasta']) for t in trio},
          'nota_fewf':'Frequencias FEWF sao aprendidas em cada treino; diferencas incluem esse efeito.',
          'delta_media':'(media_8_folds-media_9_folds)/escala_9_folds',
          'delta_log_escala':'ln(escala_8_folds/escala_9_folds)'}
    carimbo=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup=raiz/'outputs'/('validacao_padronizacao_folds_'+carimbo)
    backup.mkdir(parents=True,exist_ok=False)
    print('[BACKUP] '+str(backup),flush=True)
    csv_temp=backup/CSV
    png_temp=backup/PNG
    resultado.to_csv(csv_temp,index=False,float_format='%.17g',encoding='utf-8-sig')
    fig=_figura(nomes,dm,ds,meta)
    fig.savefig(png_temp,dpi=160,facecolor='white')
    plt.close(fig)
    meta_temp=backup/META
    meta_temp.write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    # Verificar antecipadamente cada destino antes de alterar galerias.
    for t in trio:
        _exigir('</body></html>' in t['galeria'].read_text(encoding='utf-8'),
                'Galeria existente com formato desconhecido: '+str(t['galeria']))
    _integracao_futura(raiz,backup)
    for t in trio:
        pasta=t['pasta']/'vencedor'/'visualizacao'
        original=pasta/'oof'
        gal=pasta/'galeria'/'oof'
        original.mkdir(parents=True,exist_ok=True)
        gal.mkdir(parents=True,exist_ok=True)
        for arquivo,nova in ((original/PNG,png_temp),(original/CSV,csv_temp),(original/META,meta_temp),
                             (gal/PNG,png_temp)):
            _backup_arquivo(backup,arquivo,raiz)
            shutil.copy2(nova,arquivo)
        index=t['galeria']
        txt=index.read_text(encoding='utf-8')
        _backup_arquivo(backup,index,raiz)
        index.write_text(_inserir_html(txt),encoding='utf-8')
        _exigir((gal/PNG).is_file() and MARCADOR in index.read_text(encoding='utf-8'),
                'Inclusao na galeria nao verificada.')
        print(f"[ADICIONADO] PCA {t['dim']}D | {index}",flush=True)
    print('\nCONCLUIDO: 1 diagnostico de padronizacao em 3 galerias.',flush=True)
    print('Nenhum PCA, GMM ou fold OOF foi refeito. F10 preservado.',flush=True)
    print('PNG='+str(png_temp.resolve()),flush=True)
    print('BACKUP='+str(backup.resolve()),flush=True)
    for t in trio:print(f"GALERIA_{t['dim']}D="+str(t['galeria'].resolve()),flush=True)


if __name__=='__main__':
    main()
