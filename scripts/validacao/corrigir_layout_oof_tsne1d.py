# -*- coding: utf-8 -*-
"""Corrige exclusivamente o layout t-SNE 1D de TSNE02/TSNE13/TSNE14.

--planejar: somente inventario, nao altera arquivos.
--executar: ajusta fontes para futuras execucoes e re-renderiza OOF existente
             EXCLUSIVAMENTE dos CSVs ja persistidos, sem fit/refit/scores.
--validar: verifica formato PNG, sincronizacao da galeria e backups.

Nunca le Parquet, folds mestre, joblib ou predicoes F10. Sao graficos OOF F1-F9.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import textwrap

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
MARK = 'PATCH_LAYOUT_TSNE1D_OOF_20261103'
GRAPHS = (
    'TSNE02_trustworthiness_vs_perplexity',
    '13_tsne_mapa_TSNE1',
    '14_tsne_diagnostico',
)
SOURCES = ('src/visualizacao_tsne_pdf.py', 'src/visualizacao_geometrica.py')
BACKUPS = ROOT / 'backups' / 'layout_tsne1d_oof'


def exigir(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def salvar_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp_layout_tsne1d')
    exigir(not tmp.exists(), f'Arquivo temporario anterior exige revisao: {tmp}')
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        if tmp.exists(): tmp.unlink()


def fonte_patches(src: str, nome: str) -> str:
    """Substituicoes com ancoras exatas; em caso de divergencia, bloquear."""
    if MARK in src:
        exigir(src.count(MARK) == (2 if nome.endswith('visualizacao_tsne_pdf.py') else 1),
               f'Marcadores de hotfix suspeitos em {nome}')
        return src

    if nome.endswith('visualizacao_tsne_pdf.py'):
        a = "    ax.set_title('t-SNE 02 — Preservação de vizinhanças | Trustworthiness × Perplexity')"
        z = "    nome = 'TSNE02_trustworthiness_vs_perplexity'"
        exigir(src.count(a)==src.count(z)==1, 'Bloco TSNE02 nao encontrado exatamente uma vez.')
        i=src.index(a); j=src.index(z, i)
        bloco='''    # PATCH_LAYOUT_TSNE1D_OOF_20261103: faixa exclusiva para titulo e rodape.
    ax.set_title('')
    ax.grid(alpha=.2)
    ax.legend(loc='best', fontsize=9)
    aviso = 'Prévia sintética' if origem == 'simulado' else 'Desenvolvimento F1–F9'
    fig.suptitle('t-SNE 02 — Preservação de vizinhanças | Trustworthiness × Perplexity',
                 fontsize=13.4, fontweight='bold', y=.974)
    rodape = (f'{aviso}: amostra não supervisionada de {len(idx)} observações; '
              f'dimensão {params["n_components"]}D; k={ks}; F10 não utilizado. '
              'Diagnóstico auxiliar, não altera seleção OOF.')
    from textwrap import wrap
    fig.text(.5, .060, '\\n'.join(wrap(rodape, width=94)),
             ha='center', va='center', fontsize=8.4, color='#526077', linespacing=1.45)
    fig.tight_layout(rect=(.015, .145, .985, .91))
'''
        src=src[:i]+bloco+src[j:]
        a = "    fig.suptitle('14 — t-SNE | Diagnóstico KL / otimização'"
        z = "    nome = '14_tsne_diagnostico'"
        exigir(src.count(a)==src.count(z)==1, 'Bloco TSNE14 nao encontrado exatamente uma vez.')
        i=src.index(a); j=src.index(z,i)
        bloco='''    # PATCH_LAYOUT_TSNE1D_OOF_20261103: rodape fora das caixas.
    fig.suptitle('14 — t-SNE | Diagnóstico KL / otimização',
                 fontsize=15, weight='bold', y=.965)
    nota = ('Histórico de KL por iteração: indisponível neste redutor; '
            'não foi criada curva sintética.')
    sufixo = 'PRÉVIA SINTÉTICA' if origem == 'simulado' else 'Desenvolvimento F1–F9'
    fig.text(.5, .155, nota, fontsize=8.8, ha='center',
             va='center', color='#b45309')
    fig.text(.5, .102,
             f'{sufixo}; F10 não utilizado. KL final e tempo não validam AP/MCC.',
             ha='center', va='center', fontsize=8.6, color='#64748b')
    fig.subplots_adjust(left=.055, right=.945, top=.81, bottom=.30, wspace=.085)
'''
        src=src[:i]+bloco+src[j:]
    elif nome.endswith('visualizacao_geometrica.py'):
        a = "    ax.set_title('13 — t-SNE 1D | Histograma do embedding')"
        z = "    nome = '13_tsne_mapa_TSNE1'"
        exigir(src.count(a)==src.count(z)==1, 'Bloco TSNE13 nao encontrado exatamente uma vez.')
        i=src.index(a); j=src.index(z,i)
        bloco='''    # PATCH_LAYOUT_TSNE1D_OOF_20261103: faixa exclusiva para titulo, configuracao e rodape.
    ax.set_title('')
    ax.grid(axis='y', alpha=.18, linestyle=':')
    if colorir != 'risco':
        ax.legend(loc='best', fontsize=8)
    p = parametros
    subtitulo = (f"Configuração: perplexity={p.get('perplexity', 'n/d')} | "
                 f"metric={p.get('metric', 'n/d')} | 1 dimensão")
    fig.suptitle('13 — t-SNE 1D | Histograma do embedding',
                 fontsize=14.2, fontweight='bold', y=.969)
    fig.text(.5, .912, subtitulo, ha='center', fontsize=9.2, color='#374151')
    rodape = ('Leitura local: distâncias globais não devem ser superinterpretadas. '
              + nota + ' Desenvolvimento F1–F9; F10 não utilizado.')
    if origem == 'simulado':
        rodape = 'PRÉVIA SINTÉTICA: t-SNE ajustado em dados simulados. ' + rodape
    from textwrap import wrap
    fig.text(.5, .070, '\\n'.join(wrap(rodape, width=98)),
             ha='center', va='center', fontsize=8.1, color='#526077',
             linespacing=1.45)
    fig.subplots_adjust(left=.10, right=.96, top=.83, bottom=.24)
'''
        src=src[:i]+bloco+src[j:]
    else:
        raise RuntimeError('Arquivo de codigo fora do escopo do hotfix.')
    ast.parse(src,filename=nome)
    return src


def elegiveis(fracao):
    base = ROOT / 'outputs'
    exigir(base.is_dir(),f'Diretorio outputs ausente: {base}')
    found=[]
    for p in base.glob('ensaio_tsne_gmm_*'):
        f=p/'configuracao.yaml'
        if not f.is_file(): continue
        try:
            cfg=yaml.safe_load(f.read_text(encoding='utf-8-sig'))
            if cfg.get('reducao',{}).get('metodo','').upper()!='TSNE': continue
            if int(cfg.get('tsne',{}).get('n_components',-1))!=1: continue
            q=float(cfg['dados']['fracao_base'])
            if not (abs(q-.03)<1e-10 or abs(q-.1)<1e-10): continue
            if fracao != 'ambos' and abs(q-float(fracao))>=1e-10: continue
            found.append((p,q,cfg))
        except (OSError,ValueError,TypeError,KeyError,yaml.YAMLError):
            continue
    by_frac={}
    for record in sorted(found,key=lambda a:a[0].name):
        by_frac[round(record[1],3)]=record
    return sorted(by_frac.values(),key=lambda a:a[1])


def preparar(p):
    origem=p/'vencedor/visualizacao/geometria_desenvolvimento'
    galeria=p/'vencedor/visualizacao/galeria/geometria_desenvolvimento'
    exigir((p/'vencedor/visualizacao/galeria/index.html').is_file(), f'Galeria OOF ausente: {p}')
    exigir(origem.is_dir() and galeria.is_dir(),f'Geometria/galeria OOF ausentes: {p}')
    result={}
    for name in GRAPHS:
        cs=origem/(name+'.csv'); a=origem/(name+'.png'); b=galeria/(name+'.png')
        for path in (cs,a,b): exigir(path.is_file(),f'Falta artefato OOF: {path}')
        exigir(a.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n',f'PNG original invalido: {a}')
        result[name]=(cs,a,b)
    return result


def render02(df, outfile, dpi):
    for c in ['perplexity','trustworthiness','k_vizinhos','n_amostra','n_dim_reducao','perplexity_vencedora']:
        exigir(c in df, f'TSNE02 CSV sem {c}')
    p=df.perplexity.to_numpy(dtype=float); v=df.trustworthiness.to_numpy(dtype=float)
    exigir(len(p)>=2 and np.isfinite(p).all() and np.isfinite(v).all() and
           ((v>=0)&(v<=1)).all() and (np.diff(p)>0).all(), 'TSNE02 CSV invalido')
    k=int(df.k_vizinhos.iloc[0]); n=int(df.n_amostra.iloc[0]); dim=int(df.n_dim_reducao.iloc[0])
    star=float(df.perplexity_vencedora.iloc[0]);exigir(dim==1,'TSNE02 precisa ser 1D.')
    fig,ax=plt.subplots(figsize=(11.6,6.45))
    ax.plot(p,v,'o-',lw=2.2,ms=6.4,color='#1565c0',label=f'Trustworthiness (k={k})')
    for x,y in zip(p,v):ax.annotate(f'{y:.3f}',(x,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8.4)
    ax.axvline(star,color='#d97706',ls='--',lw=1.5,label=f'Perplexity do vencedor OOF = {star:g}')
    ax.set_xticks(p)
    ax.set_ylim(max(0,float(v.min())-.026),1.005)
    ax.set_xlabel('Perplexity');ax.set_ylabel('Trustworthiness (0 a 1; eixo ampliado)')
    ax.grid(alpha=.2);ax.legend(loc='best',fontsize=9)
    fig.suptitle('t-SNE 02 — Preservação de vizinhanças | Trustworthiness × Perplexity',fontsize=13.2,fontweight='bold',y=.974)
    note=f'Desenvolvimento F1–F9: amostra não supervisionada de {n} observações; dimensão 1D; k={k}; F10 não utilizado. Diagnóstico auxiliar, não altera a seleção OOF.'
    fig.text(.5,.065,'\n'.join(textwrap.wrap(note,width=103)),ha='center',va='center',fontsize=8.7,color='#526077',linespacing=1.5)
    fig.tight_layout(rect=(.015,.16,.985,.905))
    fig.savefig(outfile,dpi=dpi,facecolor=fig.get_facecolor());plt.close(fig)


def render13(df, outfile, dpi, cfg, diag):
    for c in ('TSNE1','classe_real'):
        exigir(c in df,f'TSNE13 CSV sem {c}')
    z=df.TSNE1.to_numpy(dtype=float);y=df.classe_real.to_numpy(dtype=np.int8)
    exigir(len(z)>1 and np.isfinite(z).all() and np.isin(y,[0,1]).all(), 'TSNE13 CSV invalido')
    nbin=36
    if isinstance(cfg.get('visualizacao',{}).get('graficos',{}).get('tsne_mapa'),dict):
        nbin=int(cfg['visualizacao']['graficos']['tsne_mapa'].get('bins_histograma_1d',36))
    exigir(5<=nbin<=200,'Bins invalidos')
    lo,hi=float(z.min()),float(z.max())
    if lo==hi:lo-=.5;hi+=.5
    edges=np.linspace(lo,hi,nbin+1);mid=.5*(edges[:-1]+edges[1:]);w=float(edges[1]-edges[0])
    fig,ax=plt.subplots(figsize=(11.6,7.0))
    for key,color,cls in (('Legítima','#1f77b4',0),('Fraude','#d62728',1)):
        vals=z[y==cls]
        if not len(vals):continue
        dens,_=np.histogram(vals,bins=edges,density=True)
        ax.bar(mid,dens,width=w*.94,color=color,alpha=.38,edgecolor=color,linewidth=.4,label=f'{key} (n={len(vals):,})')
        if len(vals)>=3:ax.plot(mid,gaussian_filter1d(dens,1.15,mode='nearest'),color=color,lw=2)
    ax.set_ylabel('Densidade por grupo (área = 1)')
    ax.set_xlabel('t-SNE 1 (coordenada do embedding; sem variância explicada)')
    ax.legend(loc='best',fontsize=9);ax.grid(axis='y',alpha=.18,ls=':')
    fig.suptitle('13 — t-SNE 1D | Histograma do embedding',fontsize=14.2,fontweight='bold',y=.969)
    p=diag.iloc[0]
    fig.text(.5,.914,f'Configuração: perplexity={p.perplexity:g} | metric={p.metrica} | 1 dimensão',ha='center',fontsize=9.2,color='#374151')
    note=('Leitura local: distâncias globais não devem ser superinterpretadas. '
          'Densidades normalizadas por grupo; alturas não são prevalência. '
          'Classes observadas, apenas sobrepostas ao t-SNE não supervisionado. '
          'Desenvolvimento F1–F9; F10 não utilizado.')
    fig.text(.5,.063,'\n'.join(textwrap.wrap(note,width=109)),ha='center',va='center',fontsize=8.4,color='#526077',linespacing=1.5)
    fig.subplots_adjust(left=.095,right=.96,top=.824,bottom=.225)
    fig.savefig(outfile,dpi=dpi,facecolor=fig.get_facecolor());plt.close(fig)


def render14(df,outfile,dpi):
    for c in ('kl_final','tempo_reducao_s','n_componentes','perplexity','metrica','n_desenvolvimento'):
        exigir(c in df,f'TSNE14 CSV sem {c}')
    d=df.iloc[0];exigir(int(d.n_componentes)==1,'TSNE14 precisa ser 1D')
    kl=float(d.kl_final) if pd.notna(d.kl_final) else np.nan
    t=float(d.tempo_reducao_s) if pd.notna(d.tempo_reducao_s) else np.nan
    fig,axs=plt.subplots(1,2,figsize=(11.8,5.35));fig.patch.set_facecolor('#f8fafc')
    for ax in axs:
        ax.set_facecolor('#f1f5f9');ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_color('#cbd5e1')
    axs[0].text(.06,.86,'DIVERGÊNCIA KL FINAL',transform=axs[0].transAxes,color='#334155',fontsize=12,fontweight='bold')
    axs[0].text(.06,.50,f'{kl:.7f}' if np.isfinite(kl) else 'Indisponível',transform=axs[0].transAxes,color='#1d4ed8',fontsize=23,fontweight='bold')
    axs[0].text(.06,.18,'Métrica de otimização interna do t-SNE.\nNão mede desempenho de detecção de fraude.',transform=axs[0].transAxes,color='#475569',fontsize=9.4)
    axs[1].text(.06,.86,'TEMPO REAL DA REDUÇÃO',transform=axs[1].transAxes,color='#334155',fontsize=12,fontweight='bold')
    axs[1].text(.06,.50,f'{t:.2f} s' if np.isfinite(t) else 'Não registrado',transform=axs[1].transAxes,color='#047857',fontsize=23,fontweight='bold')
    axs[1].text(.06,.18,f'Embedding: 1D | perplexity={float(d.perplexity):g}\n{int(d.n_desenvolvimento):,} observações | métrica={d.metrica}',transform=axs[1].transAxes,color='#475569',fontsize=9.4)
    fig.suptitle('14 — t-SNE | Diagnóstico KL / otimização',fontsize=15,fontweight='bold',y=.965)
    fig.text(.5,.157,'Histórico de KL por iteração: indisponível neste redutor; não foi criada curva sintética.',ha='center',va='center',fontsize=9,color='#b45309')
    fig.text(.5,.102,'Desenvolvimento F1–F9; F10 não utilizado. KL final e tempo não validam AP/MCC.',ha='center',va='center',fontsize=8.8,color='#64748b')
    fig.subplots_adjust(left=.055,right=.945,top=.80,bottom=.295,wspace=.085)
    fig.savefig(outfile,dpi=dpi,facecolor=fig.get_facecolor());plt.close(fig)


def valida_tabelas(tabelas):
    dados={n:pd.read_csv(paths[0]) for n,paths in tabelas.items()}
    n13=len(dados['13_tsne_mapa_TSNE1'])
    n14=int(dados['14_tsne_diagnostico'].n_desenvolvimento.iloc[0])
    exigir(n13==n14,f'Numero de observacoes TSNE13/14 inconsistente: {n13} != {n14}')
    exigir(int(dados['TSNE02_trustworthiness_vs_perplexity'].n_dim_reducao.iloc[0])==1,'Dimensao diagnostica divergente')
    return dados


def patch_fontes(backup):
    for rel in SOURCES:
        path=ROOT/rel
        exigir(path.is_file(),f'Fonte ausente: {path}')
        original_bytes=path.read_bytes()
        original=original_bytes.decode('utf-8-sig')
        novo=fonte_patches(original,rel)
        if novo != original:
            b=backup/'fontes'/rel
            b.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,b)
            prefix=b'\xef\xbb\xbf' if original_bytes.startswith(b'\xef\xbb\xbf') else b''
            nova_linha='\r\n' if b'\r\n' in original_bytes else '\n'
            salvar_atomic(path,prefix+novo.replace('\n',nova_linha).encode('utf-8'))
            print(f'[FONTE CORRIGIDA] {rel}')
        else:print(f'[FONTE JA CORRIGIDA] {rel}')


def execute(fracao):
    found=elegiveis(fracao)
    exigir(found,'Nenhum ensaio t-SNE 1D com 3%/10% encontrado.')
    for p,q,cfg in found:
        tab=preparar(p);tabelas=valida_tabelas(tab)
        print(f'[PLANEJADO] {p.name} | fração={q:.2f} | CSVs=3 | F10=NAO')
    for rel in SOURCES:
        p=ROOT/rel
        exigir(p.is_file(),f'Fonte ausente: {p}')
        fonte_patches(p.read_text(encoding='utf-8'),rel)
    print('DADOS_REAIS_CSV_OOF=SIM')
    print('REAJUSTE_MODELO=NAO')
    print('PREDICOES_RECALCULADAS=NAO')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    backup=BACKUPS/stamp
    exigir(not backup.exists(),f'Backup ja existe: {backup}')
    backup.mkdir(parents=True)
    patch_fontes(backup)
    report={'versao':MARK,'backup':str(backup),'experimentos':[],'arquivos_codigo':list(SOURCES)}
    for p,q,cfg in found:
        print(f'[RENDER] {p.name}: TSNE02 / TSNE13 / TSNE14')
        files=preparar(p);dfs=valida_tabelas(files)
        dpi=int(cfg['visualizacao']['padrao'].get('dpi',180))
        exp_backup=backup/'experimentos'/p.name
        artifacts=[]
        # Render tudo em memoria temporaria antes de copiar arquivos originais.
        with tempfile.TemporaryDirectory(prefix='tsne1d_layout_') as tmp:
            staged={}
            for name in GRAPHS:
                stage=Path(tmp)/(name+'.png')
                if name==GRAPHS[0]:render02(dfs[name],stage,dpi)
                elif name==GRAPHS[1]:render13(dfs[name],stage,dpi,cfg,dfs[GRAPHS[2]])
                else:render14(dfs[name],stage,dpi)
                exigir(stage.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n' and stage.stat().st_size>10000,
                       f'PNG renderizado invalido: {name}')
                staged[name]=stage
            # Preservar cada imagem antiga em backup separado antes da substituicao.
            for name,(csv,original,gallery) in files.items():
                hashes={'csv_oof_sha256':sha(csv),'original_anterior_sha256':sha(original),
                        'galeria_anterior_sha256':sha(gallery)}
                for current,suf in ((original,'origem'),(gallery,'galeria')):
                    b=exp_backup/suf/current.name
                    b.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(current,b)
                    exigir(sha(b)==sha(current),f'Backup incoerente: {current}')
                for target in (original,gallery):
                    salvar_atomic(target,staged[name].read_bytes())
                exigir(sha(original)==sha(gallery),'Copia geometria/galeria divergente')
                artifacts.append({'nome':name,**hashes,'png_novo_sha256':sha(original)})
                print(f'[OK] {name}.png -> origem e galeria')
        report['experimentos'].append({'pasta':str(p),'fracao':q,'imagens':artifacts})
    arquivo=backup/'manifesto_reparo.json'
    arquivo.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'BACKUP_VISUAL={backup}')
    print(f'REPARADOS_EXPERIMENTOS={len(found)}')
    print(f'REPARADOS_GRAFICOS={len(found)*3}')
    print('LAYOUT_TSNE1D_STATUS=APROVADO')
    return 0


def validar(fracao):
    found=elegiveis(fracao);exigir(found,'Nenhum experimento elegivel.')
    for rel in SOURCES:
        path=ROOT/rel;body=path.read_text(encoding='utf-8')
        exigir(MARK in body and fonte_patches(body,rel)==body, f'Hotfix nao aplicado ao codigo: {rel}')
        ast.parse(body,filename=rel)
    for p,q,_ in found:
        files=preparar(p);valida_tabelas(files)
        for name,(cs,a,b) in files.items():
            exigir(sha(a)==sha(b),f'Galeria desatualizada: {name}')
            exigir(a.stat().st_size>10000,f'Imagem invalida: {a}')
        print(f'[VALIDADO] {p.name}: 3/3')
    print(f'LAYOUT_TSNE1D_VALIDADO={len(found)*3}/{len(found)*3}')
    print('TREINAMENTO_REEXECUTADO=NAO')
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--planejar',action='store_true')
    m.add_argument('--executar',action='store_true')
    m.add_argument('--validar',action='store_true')
    p.add_argument('--fracao',choices=('ambos','0.03','0.1'),default='ambos')
    a=p.parse_args()
    if a.executar: return execute(a.fracao)
    if a.validar: return validar(a.fracao)
    found=elegiveis(a.fracao)
    for ex,f,_ in found:
        try:
            tab=preparar(ex);valida_tabelas(tab)
            print(f'[PRONTO] {ex.name} | fracao={f:.2f} | 3 CSV / 3 PNG / 3 copias')
        except Exception as err:
            print(f'[BLOQUEADO] {ex.name} | {type(err).__name__}: {err}')
    print(f'EXPERIMENTOS_ENCONTRADOS={len(found)}')
    print('PLANEJAMENTO_SEM_MODIFICACOES=SIM')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
