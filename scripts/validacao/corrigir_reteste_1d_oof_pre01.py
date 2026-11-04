# -*- coding: utf-8 -*-
"""Corrige PNGs OOF TSNE02/13/14 e adiciona PRE01 aos 4 HTMLs de PCA/TSNE 1D.

--planejar / --aplicar / --validar / --abrir
- Nao reexecuta treinamento, OOF, inferencia F10 nem graficos preditivos.
- Gera PRE01 apenas a partir do desenvolvimento autenticado F1..F9, se ausente.
- Atualiza o contrato de layout das fontes para que novos PNGs saiam limpos.
- Mantem backups, hashes de galerias F10, configuracoes, predicoes e modelos.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

NOMES = ('TSNE02_trustworthiness_vs_perplexity', '13_tsne_mapa_TSNE1', '14_tsne_diagnostico')
PRE01 = 'PRE01_diagnostico_padronizacao_completo.html'
MARCADOR = 'CORRECAO_1D_OOF_PRE01_20261103_V2'
FONTES = ('src/visualizacao_tsne_pdf.py', 'src/visualizacao_geometrica.py', 'src/garantia_layout_tsne.py')


def exigir(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    h = sha256()
    with Path(path).open('rb') as inp:
        for chunk in iter(lambda: inp.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def ler_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def salvar_atomico(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporario = path.with_name(path.name + '.tmp_correcao_1d_pre01')
    exigir(not temporario.exists(), 'Temporario existente: ' + str(temporario))
    try:
        with temporario.open('xb') as f:
            f.write(raw)
        temporario.replace(path)
    finally:
        if temporario.exists():
            temporario.unlink()


def selecionar(lote):
    from scripts.execucao import reteste_real_1d_3pct as ret
    ret.configurar(lote)
    _, pasta_pca, parcial_pca = ret.configurar_e_testar_metodo('pca')
    _, pasta_tsne, parcial_tsne = ret.configurar_e_testar_metodo('tsne')
    exigir(parcial_pca is None and parcial_tsne is None and pasta_pca and pasta_tsne,
           'O reteste 1D deste lote deve ter PCA e TSNE concluidos. Nao alterar pasta parcial.')
    validos = {}
    for metodo, pasta in (('pca', pasta_pca), ('tsne', pasta_tsne)):
        pasta = Path(pasta).resolve()
        exigir(pasta.parent == (RAIZ/'outputs').resolve() and
               pasta.name.startswith('ensaio_' + metodo + '_gmm_'),
               'Pasta de ensaio inesperada: ' + str(pasta))
        config = __import__('yaml').safe_load((pasta/'configuracao.yaml').read_text(encoding='utf-8-sig'))
        m = ler_json(pasta/'manifesto.json')
        exigir(m.get('status') == 'concluido_06_03' and m.get('visualizacoes') == 'concluidas',
               'OOF ainda nao concluiu: '+metodo)
        exigir(config['dados']['fracao_base']==.03 and
               config['reducao']['metodo'].upper()==metodo.upper() and
               config['divisao']['qual_fold_teste']==10 and
               config['experimento']['semente']==42,
               'Contrato do ensaio 1D divergente: '+metodo)
        selecionado = ler_json(pasta/'vencedor/metricas/melhor_configuracao.json')
        exigir(selecionado['configuracao']['reducao']['n_components']==1 and
               selecionado['teste_final_usado'] is False,
               'Modelo OOF inesperado: '+metodo)
        validos[metodo] = (pasta, config, m, selecionado)
    # Validacao rigorosa previamente aprovada do reteste; nao chama fit/inferencia.
    oof_ref, f10_ref = ret.conferir_baselines()
    ret.validar_tudo(oof_ref, f10_ref)
    return validos


def arquivos(validos):
    for metodo,(pasta,cfg,manifest,escolha) in validos.items():
        raiz = pasta/'vencedor'
        oof = raiz/'visualizacao/oof'
        geo = raiz/'visualizacao/geometria_desenvolvimento'
        galeria = raiz/'visualizacao/galeria'
        gal_geo = galeria/'geometria_desenvolvimento'
        f10_dir = raiz/'teste_final/galeria_3pct'
        for d in (oof,geo,galeria,gal_geo,f10_dir):
            exigir(d.is_dir(), 'Pasta inexistente: '+str(d))
        for h in (galeria/'index.html',f10_dir/'index.html',f10_dir/'manifesto_galeria_3pct.json'):
            exigir(h.is_file(), 'Galeria incompleta: '+str(h))
        if metodo == 'tsne':
            for nome in NOMES:
                exigir((geo/(nome+'.csv')).is_file(), 'CSV OOF TSNE ausente: '+nome)
                for d in (geo,gal_geo):
                    exigir((d/(nome+'.png')).is_file(),
                           'PNG TSNE ausente: '+str(d/(nome+'.png')))
        print(f'[OK] {metodo.upper()} 1D -> {pasta}', flush=True)


def patch_fontes_prepare():
    """Finaliza apenas o layout de TSNE02/13/14, com insercao AST no bloco correto.

    Diferentemente dos hotfixes antigos, nao substitui trechos de codigo
    anteriores nem depende do texto existente no rotulo do eixo.
    """
    fontes = {}
    patches = {
        'src/visualizacao_tsne_pdf.py': (
            ('trustworthiness_perplexity', 'TSNE02_trustworthiness_vs_perplexity',
             """    # CORRECAO_1D_OOF_PRE01_20261103_V2 / TSNE02
    for _txt_layout in list(fig.texts):
        if _txt_layout is not getattr(fig, '_suptitle', None):
            _txt_layout.remove()
    ax.set_title('')
    fig.suptitle('Preservação de vizinhanças | Trustworthiness × Perplexity',
                 fontsize=13.5, fontweight='bold', y=.975)
    fig.tight_layout(rect=(.015, .04, .985, .92))
"""),
            ('diagnostico_kl', '14_tsne_diagnostico',
             """    # CORRECAO_1D_OOF_PRE01_20261103_V2 / TSNE14
    for _txt_layout in list(fig.texts):
        if _txt_layout is not getattr(fig, '_suptitle', None):
            _txt_layout.remove()
    fig.suptitle('t-SNE | Diagnóstico KL / otimização',
                 fontsize=15, weight='bold', y=.965)
    fig.subplots_adjust(left=.055, right=.945, top=.83, bottom=.12, wspace=.085)
"""),
        ),
        'src/visualizacao_geometrica.py': (
            ('_tsne_histograma_1d', '13_tsne_mapa_TSNE1',
             """    # CORRECAO_1D_OOF_PRE01_20261103_V2 / TSNE13
    for _txt_layout in list(fig.texts):
        if _txt_layout is not getattr(fig, '_suptitle', None):
            _txt_layout.remove()
    if getattr(fig, '_suptitle', None) is not None:
        fig._suptitle.set_visible(False)
    ax.set_title('t-SNE 1D | Histograma do embedding', pad=12)
    ax.set_xlabel('t-SNE 1')
    fig.subplots_adjust(left=.10, right=.96, top=.91, bottom=.14)
"""),
        ),
    }
    for rel in FONTES:
        p = RAIZ / rel
        exigir(p.is_file(), 'Fonte ausente: ' + rel)
        raw = p.read_bytes()
        src = raw.decode('utf-8-sig').replace('\r\n', '\n')
        original = src
        if rel not in patches:
            # O guard ja instalado foi validado no reteste; nao modificar.
            fontes[p] = (raw, src, False)
            continue
        tree = ast.parse(src, filename=rel)
        linhas = src.splitlines(keepends=True)
        insercoes = []
        for funcao, nome, bloco in patches[rel]:
            candidatos = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                          and n.name == funcao]
            exigir(len(candidatos) == 1, 'Funcao de visualizacao inexistente/duplicada: ' + funcao)
            n = candidatos[0]
            # Encontrar a atribuicao top-level do nome do PNG dentro da funcao.
            pontos = [x for x in n.body if isinstance(x, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'nome' for t in x.targets)
                      and isinstance(x.value, ast.Constant) and x.value.value == nome]
            exigir(len(pontos) == 1, 'Ponto de salvamento desconhecido/duplicado: ' + nome)
            posicao = pontos[0].lineno - 1
            segmento = ''.join(linhas[n.lineno - 1:n.end_lineno])
            marca = 'CORRECAO_1D_OOF_PRE01_20261103_V2 / ' + (
                'TSNE02' if nome.startswith('TSNE02') else 'TSNE13' if nome.startswith('13_') else 'TSNE14')
            if marca in segmento:
                exigir(segmento.count(marca) == 1,
                       'Marcador visual duplicado na funcao: ' + funcao)
            else:
                inserir = bloco
                exigir(linhas[posicao].startswith('    nome = '),
                       'Salvamento do grafico com indentacao inesperada: ' + nome)
                insercoes.append((posicao, inserir))
        for posicao, inserir in sorted(insercoes, reverse=True):
            linhas.insert(posicao, inserir)
        src = ''.join(linhas)
        ast.parse(src, filename=rel)
        fontes[p] = (raw, src, src != original)
    return fontes


def preparar_pngs(validos):
    """Reconstroi as figuras a partir dos CSVs OOF reais existentes."""
    import io
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    pasta, cfg, _, escolha = validos['tsne']
    geo = pasta/'vencedor/visualizacao/geometria_desenvolvimento'
    data={nome:pd.read_csv(geo/(nome+'.csv')) for nome in NOMES}
    import numpy as np
    with np.load(pasta/'vencedor/oof/predicoes.npz',allow_pickle=False) as a:
        y=np.asarray(a['y_true'], dtype=np.uint8)
    exigir(np.array_equal(data[NOMES[1]]['classe_real'].to_numpy(dtype=np.uint8),y),
           'Histograma TSNE13 nao corresponde aos rotulos OOF reais.')
    exigir(len(data[NOMES[1]])==50013 and int(data[NOMES[2]]['n_desenvolvimento'].iloc[0])==50013,
           'Dimensao F1..F9 dos diagnosticos nao corresponde ao ensaio real 3%.')
    exigir(int(data[NOMES[0]]['n_dim_reducao'].iloc[0])==1,'TSNE02 nao e 1D')
    result={}
    def png(fig):
        with io.BytesIO() as f:
            fig.savefig(f,format='png',dpi=180,facecolor=fig.get_facecolor(),bbox_inches='tight')
            raw=f.getvalue()
        plt.close(fig)
        exigir(raw[:8]==b'\x89PNG\r\n\x1a\n' and len(raw)>4000,'PNG renderizado invalido')
        return raw

    d=data[NOMES[0]]
    pp=d.perplexity.to_numpy(float); vv=d.trustworthiness.to_numpy(float)
    exigir(len(pp)>=2 and np.isfinite(pp).all() and np.isfinite(vv).all() and np.all(np.diff(pp)>0),
           'Dados TSNE02 invalidos')
    k=int(d.k_vizinhos.iloc[0]); estrela=float(d.perplexity_vencedora.iloc[0])
    fig,ax=plt.subplots(figsize=(10.5,6.2))
    ax.plot(pp,vv,'o-',lw=2.1,ms=6.5,color='#1565c0',label=f'Trustworthiness (k={k})')
    for x,val in zip(pp,vv):
        ax.annotate(f'{val:.3f}',(x,val),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8.5)
    ax.axvline(estrela,color='#d97706',ls='--',lw=1.4,label=f'Perplexity do vencedor OOF = {estrela:g}')
    ax.set_xticks(pp);ax.set_ylim(max(0.,float(vv.min())-.025),1.003)
    ax.set(xlabel='Perplexity',ylabel='Trustworthiness (0 a 1; eixo ampliado)')
    ax.set_title('Preservação de vizinhanças | Trustworthiness × Perplexity',pad=12)
    ax.grid(alpha=.2);ax.legend(loc='best',fontsize=9)
    fig.tight_layout()
    result[NOMES[0]+'.png']=png(fig)

    d=data[NOMES[1]]
    xx=d.TSNE1.to_numpy(float);yy=d.classe_real.to_numpy(np.uint8)
    op=cfg['visualizacao']['graficos']['tsne_mapa'];n=int(op.get('bins_histograma_1d',36))
    exigir(5<=n<=200 and np.isfinite(xx).all() and set(yy)=={0,1},'TSNE13 invalido')
    lo,hi=float(xx.min()),float(xx.max())
    if lo==hi:lo-=.5;hi+=.5
    edges=np.linspace(lo,hi,n+1); mids=.5*(edges[:-1]+edges[1:]); w=edges[1]-edges[0]
    fig,ax=plt.subplots(figsize=(11.6,6.4))
    for classe,chave in ((0,'legitima'),(1,'fraude')):
        z=xx[yy==classe];item=cfg['visualizacao']['classes'][chave]
        cor=item.get('cor','#1f77b4' if classe==0 else '#d62728')
        hist,_=np.histogram(z,bins=edges,density=True)
        ax.bar(mids,hist,width=w*.94,color=cor,alpha=.38,edgecolor=cor,linewidth=.4,
               label=f"{item.get('nome',chave)} (n={len(z):,})")
        if len(z)>=3 and op.get('curvas_suavizadas_1d',True):
            ax.plot(mids,gaussian_filter1d(hist,1.15,mode='nearest'),color=cor,lw=2)
    ax.set(title='t-SNE 1D | Histograma do embedding',xlabel='t-SNE 1',
           ylabel='Densidade por grupo (área = 1)')
    ax.grid(axis='y',alpha=.18,ls=':');ax.legend(loc='best',fontsize=9)
    fig.tight_layout()
    result[NOMES[1]+'.png']=png(fig)

    d=data[NOMES[2]]
    exigir(len(d)==1 and int(d.n_componentes.iloc[0])==1,'TSNE14 invalido')
    row=d.iloc[0]
    kl=float(row.kl_final) if pd.notna(row.kl_final) else np.nan
    tempo=float(row.tempo_reducao_s) if pd.notna(row.tempo_reducao_s) else np.nan
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.7));fig.patch.set_facecolor('#f8fafc')
    for ax in axes:
        ax.set_facecolor('#f1f5f9');ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_color('#cbd5e1')
    a,b=axes
    a.text(.06,.86,'DIVERGÊNCIA KL FINAL',transform=a.transAxes,fontweight='bold',color='#334155',fontsize=12)
    a.text(.06,.52,f'{kl:.7f}' if np.isfinite(kl) else 'Indisponível',transform=a.transAxes,
           color='#1d4ed8',fontsize=24,weight='bold')
    a.text(.06,.22,'Métrica de otimização interna do t-SNE.\nNão mede desempenho de detecção de fraude.',
           transform=a.transAxes,color='#475569',fontsize=9.5)
    b.text(.06,.86,'TEMPO REAL DA REDUÇÃO',transform=b.transAxes,fontweight='bold',color='#334155',fontsize=12)
    b.text(.06,.52,f'{tempo:.2f} s' if np.isfinite(tempo) else 'Não registrado',
           transform=b.transAxes,color='#047857',fontsize=24,weight='bold')
    b.text(.06,.22,f'Embedding: 1D | perplexity={row.perplexity:g}\n{int(row.n_desenvolvimento):,} observações | métrica={row.metrica}',
           transform=b.transAxes,color='#475569',fontsize=9.5)
    fig.suptitle('t-SNE | Diagnóstico KL / otimização',fontsize=15,weight='bold',y=.975)
    fig.subplots_adjust(left=.055,right=.945,top=.83,bottom=.10,wspace=.085)
    result[NOMES[2]+'.png']=png(fig)
    return result


def gerar_pre01(validos):
    """Prepara o PRE01 sem fit de PCA/GMM/t-SNE do modelo principal."""
    from src import pre01_v2_diagnostico as pre
    from src import orquestracao_oof as orq
    from src import preprocessing as prep
    import numpy as np
    import yaml
    raw={}
    for metodo,(pasta,cfg,manifest,escolha) in validos.items():
        raiz=pasta/'vencedor/visualizacao'
        atual=raiz/'oof'/PRE01
        if not atual.is_file():
            atual=raiz/'galeria/oof'/PRE01
        if atual.is_file():
            htmltxt=atual.read_text(encoding='utf-8-sig')
            if all(x in htmltxt for x in ('PRE01_DATA','table-details','plot-pca')):
                raw[metodo]={PRE01:atual.read_bytes()}
                print(f'[PRE01] {metodo.upper()} existente autentico: reutilizar HTML.',flush=True)
                continue
        print(f'[PRE01] {metodo.upper()}: calculando diagnosticos somente em F1..F9.',flush=True)
        with np.load(pasta/'vencedor/oof/predicoes.npz',allow_pickle=False) as z:
            ids=np.asarray(z['linha_original'],dtype=np.int64)
            folds_oof=np.asarray(z['fold'],dtype=np.uint8)
            y=np.asarray(z['y_true'],dtype=np.uint8)
        exigir(len(ids)==50013 and np.array_equal(np.unique(folds_oof),np.arange(1,10)),
               'Folds desenvolvimento incompletos.')
        origem=RAIZ/cfg['dados']['arquivo']
        f, f_file, h = orq._identificar_folds(RAIZ,cfg,origem)
        exigir(h==manifest['sha256_parquet'] and orq.hash_arquivo(f_file)==manifest['sha256_folds'] and
               np.array_equal(f[ids],folds_oof),'Proveniencia PRE01 divergente.')
        dev=orq._ler_desenvolvimento(origem,ids)
        exigir(np.array_equal(dev[cfg['dados']['coluna_alvo']].to_numpy(dtype=np.uint8),y),
               'Rotulos PRE01 nao coincidem com OOF.')
        X=prep.selecionar_features(dev,cfg)
        del dev
        resumo,detalhe,porfold,pca_aux=pre.calc(X,folds_oof,cfg)
        del X
        exigir(len(porfold)==9 and len(detalhe)>=9*len(resumo) and len(resumo)>0,
               'Diagnostico PRE01 incompleto.')
        meta={
            'tipo':'ensaio','metodo_principal':metodo.upper(),
            'dimensao_principal':1,'total_observacoes':int(len(ids)),
            'total_features':int(len(resumo)),'f10_usado':False,
            'origem':'OOF real F1-F9','experimento':str(pasta.relative_to(RAIZ)),
            'sha256_parquet':h,'sha256_folds':manifest['sha256_folds'],
            'interpretacao':'PCA auxiliar e apenas diagnostico das entradas, nao a reducao principal do t-SNE.',
        }
        htm=pre.build_html(resumo,detalhe,porfold,pca_aux,meta)
        htm=htm.replace('<h1>PRE01 — Diagnóstico completo da padronização</h1>',
                        '<h1>PRE01 — Gráficos e descrição da padronização | '+metodo.upper()+' 1D</h1>')
        if metodo=='tsne':
            htm=htm.replace('<p class="lead">',
                '<p class="lead"><strong>Redução principal: t-SNE 1D; a PCA desta página é auxiliar, apenas para inspeção das entradas.</strong> ')
        exigir(all(x in htm for x in ('PRE01_DATA','table-details','plot-pca')),
               'PRE01 nao contem tabelas e graficos interativos esperados.')
        raw[metodo]={
            PRE01:htm.encode('utf-8'),
            pre.CSV_SUM:resumo.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
            pre.CSV_DETAIL:detalhe.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
            pre.CSV_FOLDS:porfold.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
            pre.CSV_PCA:pca_aux.to_csv(index=False,float_format='%.17g').encode('utf-8-sig'),
            pre.META:(json.dumps(meta,ensure_ascii=False,indent=2)+'\n').encode('utf-8'),
        }
    return raw


def atualizar_html(texto,metodo,local):
    exigir('</body></html>' in texto,'Fechamento de galeria HTML desconhecido.')
    padrao=r'<!-- '+MARCADOR+r' INICIO -->[\s\S]*?<!-- '+MARCADOR+r' FIM -->'
    texto=re.sub(padrao,'',texto)
    caminho='oof/'+PRE01 if local=='oof' else '../../visualizacao/galeria/oof/'+PRE01
    cor='#245b91'
    bloco=(f'<!-- {MARCADOR} INICIO -->\n'
           '<section id="pre01-padronizacao" style="margin:32px 0;padding:22px;border:1px solid #d7e1ec;border-radius:12px;background:#fff">'
           f'<h2>PRE01 — Gráficos e descrição da padronização | {metodo.upper()} 1D</h2>'
           '<p>Diagnóstico interativo de todas as variáveis: médias, variâncias, escalas, comportamento por fold e PCA auxiliar. '
           'Parâmetros calculados exclusivamente no desenvolvimento F1–F9; o F10 usa o pré-processamento congelado, sem novo ajuste.</p>'
           f'<p><a href="{html.escape(caminho,quote=True)}" target="_blank" rel="noopener" '
           f'style="display:inline-block;padding:12px 18px;background:{cor};color:white;border-radius:8px;text-decoration:none;font-weight:650">'
           'Abrir gráficos e descrição da padronização (PRE01)</a></p>'
           '</section>\n'
           f'<!-- {MARCADOR} FIM -->\n')
    return texto.replace('</body></html>',bloco+'</body></html>')


def alvos_mutacao(validos,imgs,pre01s,fontes):
    changes={}
    for metodo,(pasta,cfg,manifest,escolha) in validos.items():
        raiz=pasta/'vencedor'
        oof=raiz/'visualizacao/oof'
        gal=raiz/'visualizacao/galeria'
        f10=raiz/'teste_final/galeria_3pct'
        for k,raw in pre01s[metodo].items():
            changes[oof/k]=raw
            changes[gal/'oof'/k]=raw
        idx=gal/'index.html'
        changes[idx]=atualizar_html(idx.read_text(encoding='utf-8'),metodo,'oof').encode('utf-8')
        idxf=f10/'index.html'
        newhtml=atualizar_html(idxf.read_text(encoding='utf-8'),metodo,'f10').encode('utf-8')
        changes[idxf]=newhtml
        man=f10/'manifesto_galeria_3pct.json'
        m=ler_json(man);m['sha256_html']=sha256(newhtml).hexdigest()
        changes[man]=(json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
        if metodo=='tsne':
            for nome,blob in imgs.items():
                changes[raiz/'visualizacao/geometria_desenvolvimento'/nome]=blob
                changes[gal/'geometria_desenvolvimento'/nome]=blob
    for p,(raw,src,alterado) in fontes.items():
        if alterado:
            sufixo='\r\n' if b'\r\n' in raw else '\n'
            prefixo=b'\xef\xbb\xbf' if raw.startswith(b'\xef\xbb\xbf') else b''
            changes[p]=prefixo+src.replace('\n',sufixo).encode('utf-8')
    return changes


def conservar_hashes(validos):
    result={}
    for metodo,(pasta,cfg,manifest,escolha) in validos.items():
        paths=[pasta/'manifesto.json',pasta/'configuracao.yaml',
               pasta/'vencedor/oof/predicoes.npz',
               pasta/'vencedor/teste_final/modelo_final_f1_f9/modelo_final_f1_f9.joblib',
               pasta/'vencedor/teste_final/avaliacao_f10_3pct/predicoes_f10.csv',
               pasta/'vencedor/teste_final/avaliacao_f10_3pct/metricas_f10.json']
        result[metodo]={str(p):sha(p) for p in paths}
    return result


def aplicar(validos,fontes):
    originais=conservar_hashes(validos)
    # Calcular/renderizar tudo antes de editar qualquer arquivo.
    novos_png=preparar_pngs(validos)
    blobs=gerar_pre01(validos)
    edits=alvos_mutacao(validos,novos_png,blobs,fontes)
    altered={p:v for p,v in edits.items() if not p.exists() or p.read_bytes()!=v}
    backup=RAIZ/'backups'/'retificacao_1d_oof_pre01'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    exigir(not backup.exists(),'Backup ja existe.')
    saved={}
    for path in altered:
        if path.exists():
            bk=backup/path.relative_to(RAIZ)
            bk.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,bk)
            exigir(sha(path)==sha(bk),'Backup divergente: '+str(path))
            saved[path]=bk
        else:
            saved[path]=None
    escritos=[]
    try:
        for dest,blob in altered.items():
            salvar_atomico(dest,blob)
            escritos.append(dest)
        from src.garantia_layout_tsne import verificar_layout_tsne
        verificar_layout_tsne(validos['tsne'][1],fase='correcao_permanente_reteste_1d')
        for metodo,pairs in originais.items():
            for path,h in pairs.items():
                exigir(sha(path)==h,'Arquivo cientifico alterado inesperadamente: '+path)
        # Conferir IDs sem treinar e hashes HTML/PNG atualizados.
        from scripts.execucao import reteste_real_1d_3pct as ret
        refs=ret.conferir_baselines()
        ret.validar_tudo(*refs)
        validar_saidas(validos)
    except Exception:
        for dest in reversed(escritos):
            old=saved[dest]
            if old is None:dest.unlink(missing_ok=True)
            else:shutil.copy2(old,dest)
        raise
    resumo={'status':'APROVADO','timestamp':datetime.now().isoformat(),
            'lote':validos['pca'][1]['experimento']['nome'],
            'alterados':[str(p.relative_to(RAIZ)) for p in altered],
            'hashes_intactos':originais,
            'hashes_finais':{str(p.relative_to(RAIZ)):sha(p) for p in altered},
            'backup':str(backup)}
    salvar_atomico(backup/'MANIFESTO_CORRECAO.json',
                   (json.dumps(resumo,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print('BACKUP_CORRECAO='+str(backup),flush=True)
    return 0


def validar_saidas(validos):
    for metodo,(pasta,cfg,manifest,escolha) in validos.items():
        raiz=pasta/'vencedor'
        gal=raiz/'visualizacao/galeria';f10=raiz/'teste_final/galeria_3pct'
        for index in (gal/'index.html',f10/'index.html'):
            exigir(MARCADOR in index.read_text(encoding='utf-8') and
                   'Abrir gráficos e descrição da padronização (PRE01)' in index.read_text(encoding='utf-8'),
                   'Link PRE01 ausente: '+str(index))
        for h in (raiz/'visualizacao/oof'/PRE01,gal/'oof'/PRE01):
            exigir(h.is_file() and h.stat().st_size>1000 and
                   all(x in h.read_text(encoding='utf-8-sig') for x in ('PRE01_DATA','plot-pca')),
                   'PRE01 nao esta completo: '+str(h))
        m=ler_json(f10/'manifesto_galeria_3pct.json')
        exigir(m['sha256_html']==sha(f10/'index.html'),'SHA galeria F10 divergente')
        exigir(len(m['hashes_png'])==11 and
               all(sha(f10/k)==v for k,v in m['hashes_png'].items()),
               'Hashes dos 11 PNG F10 alterados')
    pasta=validos['tsne'][0]/'vencedor/visualizacao'
    for nome in NOMES:
        p=pasta/'geometria_desenvolvimento'/(nome+'.png')
        g=pasta/'galeria/geometria_desenvolvimento'/(nome+'.png')
        exigir(p.is_file() and sha(p)==sha(g),'PNG TSNE OOF nao sincronizado: '+nome)
    print('TSNE02_13_14_OOF_CORRIGIDOS=3/3',flush=True)
    print('PRE01_HTML_OOF=2/2',flush=True)
    print('PRE01_LINKS_OOF=2/2',flush=True)
    print('PRE01_LINKS_F10=2/2',flush=True)
    print('F10_GRAFICOS_PRESERVADOS=22/22',flush=True)
    print('TREINAMENTO_OOF_REEXECUTADO=NAO',flush=True)
    print('MODELOS_F1_F9_REAJUSTADOS=NAO',flush=True)
    print('INFERENCIA_F10_REPETIDA=NAO',flush=True)
    print('CORRECAO_1D_STATUS=APROVADA',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lote',default='RETESTE_20261103_1D_V2')
    group=parser.add_mutually_exclusive_group(required=True)
    for mode in ('planejar','aplicar','validar','abrir'):
        group.add_argument('--'+mode,action='store_true')
    args=parser.parse_args(argv)
    try:
        validos=selecionar(args.lote)
        arquivos(validos)
        if args.planejar:
            fontes=patch_fontes_prepare()
            for p,(_,_,alterado) in fontes.items():
                print('FONTE='+str(p.relative_to(RAIZ))+' | AJUSTAR='+str(alterado),flush=True)
            for metodo,(pasta,cfg,manifest,escolha) in validos.items():
                encontrado=(pasta/'vencedor/visualizacao/oof'/PRE01).is_file()
                print('PRE01_'+metodo.upper()+'_EXISTE='+str(encontrado),flush=True)
            print('PLANEJAMENTO_SEM_EDICAO=SIM',flush=True)
        elif args.aplicar:
            aplicar(validos,patch_fontes_prepare())
        else:
            validar_saidas(validos)
            if args.abrir:
                exigir(os.name=='nt','Abertura automatica somente no Windows.')
                for metodo,(pasta,_,_,_) in validos.items():
                    gal=pasta/'vencedor/visualizacao/galeria/index.html'
                    f10=pasta/'vencedor/teste_final/galeria_3pct/index.html'
                    for a in (gal,f10):
                        os.startfile(str(a))
                        print('HTML_ABERTO='+str(a),flush=True)
        print('EXIT_CODE_CORRECAO_1D=0',flush=True)
        return 0
    except Exception as e:
        print('CORRECAO_1D_STATUS=BLOQUEADO',file=sys.stderr,flush=True)
        print('CORRECAO_1D_MOTIVO='+str(e),file=sys.stderr,flush=True)
        traceback.print_exc()
        return 1


if __name__=='__main__':
    raise SystemExit(main())
