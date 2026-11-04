# -*- coding: utf-8 -*-
"""V3: homologacao com dados SINTETICOS, sem fit/OOF/F10/Parquet.

    python -B -m scripts.validacao.homologar_v3_2d --planejar
    python -B -m scripts.validacao.homologar_v3_2d --executar
    python -B -m scripts.validacao.homologar_v3_2d --validar

Gera apenas figuras de teste em outputs/validacao/v3_previa_sintetica_2d.
NAO cria HOMOLOGACAO_V3_APROVADA.json: liberacao requer revisao humana.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[2]
DESTINO = RAIZ / 'outputs' / 'validacao' / 'v3_previa_sintetica_2d'
ARQUIVO_INDICE = DESTINO / 'index.html'
MARCADOR = 'V3_PREVIA_SINTETICA_2D_20261103'
SHA_CONFIG = {
    'pca': '29b88f4d7f1de6c5ad86bbff0421e20f14ba635599aba559464dcb12a24a7604',
    'tsne': '64f105eb35e6928cc0843a27be187ded9951e6cb84191a875ebeb835974440c4',
}
PREVIA_GERADA = {
    'pca': ('GMM01_densidade_marginal_2d_PC1.png',
            'GMM01_densidade_marginal_2d_PC2.png',
            'GMM01_densidade_conjunta_2d_PC1_PC2.png',
            'GMM01_densidade_conjunta_superficie_PCA2D.png',
            'GMM04_dominancia_2d.png',
            'GMM07_sensibilidade_2d_comparativo.png',
            'GMM08_mapa_risco_2d_comparativo.png',
            '12_pca_projecao_PC1_PC2.png'),
    'tsne': ('GMM01_densidade_marginal_2d_TSNE1.png',
             'GMM01_densidade_marginal_2d_TSNE2.png',
             'GMM01_densidade_conjunta_2d_TSNE1_TSNE2.png',
             'GMM01_densidade_conjunta_superficie_TSNE2D.png',
             'GMM04_dominancia_2d.png',
             'GMM07_sensibilidade_2d_comparativo.png',
             'GMM08_mapa_risco_2d_comparativo.png',
             '13_tsne_mapa_TSNE1_TSNE2.png',
             '14_tsne_diagnostico.png'),
}


def exigir(teste, mensagem):
    if not teste:
        raise RuntimeError(mensagem)


def digest(arquivo):
    h = sha256()
    with Path(arquivo).open('rb') as f:
        for pedaco in iter(lambda: f.read(1024*1024), b''):
            h.update(pedaco)
    return h.hexdigest()


def inspecionar_v2():
    """Somente leitura. Depende da instalacao V2 ja aprovada."""
    from scripts.validacao import instalar_v2_2d as instalador
    from scripts.execucao import experimentos_reais_3pct_2d as execucao
    for rel in instalador.FONTE:
        instalador.testar_fonte_modificada(rel)
    instalador.conferir_arquivos_novos()
    configs = {}
    for metodo in ('pca', 'tsne'):
        caminho = execucao.YAMLS_CONGELADOS / f'{metodo}_gmm_2d_3pct.yaml'
        exigir(caminho.is_file() and digest(caminho) == SHA_CONFIG[metodo],
               f'YAML congelado {metodo.upper()} foi modificado ou mudou o SHA. Nao continuar.')
        cfg, grade = execucao.construir_config(metodo)
        exigir(cfg['dados']['fracao_base'] == .03, 'Fracao de amostragem diferente de 3%.')
        exigir(all(c['reducao']['n_components'] == 2 for c in grade),
               f'Grade {metodo} contem dimensao distinta de 2.')
        exigir(len(grade) == (1 if metodo == 'pca' else 2),
               f'Numero de candidatos inesperado: {metodo}.')
        configs[metodo] = cfg
        print(f'[OK] YAML {metodo.upper()} 2D congelado | SHA256={SHA_CONFIG[metodo]} | candidatos={len(grade)}', flush=True)
    from src.garantia_layout_tsne import verificar_layout_tsne
    verificar_layout_tsne(configs['tsne'], fase='homologacao_v3_2d')
    # Validar que o motor 2D nao alterou o 1D, nem tem comandos de treino aqui.
    modulo = RAIZ/'src'/'visualizacao_tsne_2d.py'
    texto = modulo.read_text(encoding='utf-8-sig')
    arvore = ast.parse(texto, filename=str(modulo))
    func = next((n for n in arvore.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'gerar_projecao_2d'), None)
    exigir(func is not None and "'f10'" in texto and "'desenvolvimento'" in texto,
           'Renderizador TSNE13/PCA12 2D incompleto.')
    from src import visualizacao_integrada as integ
    codigo_integ = (RAIZ/'src'/'visualizacao_integrada.py').read_text(encoding='utf-8-sig')
    exigir('GMM01_densidade_conjunta_2d_TSNE1_TSNE2' in codigo_integ and
           'GMM01_densidade_conjunta_superficie_TSNE2D' in codigo_integ,
           'Galeria OOF nao inclui as duas densidades conjuntas TSNE2D.')
    for metodo in ('pca','tsne'):
        from scripts.execucao.experimentos_reais_3pct_2d import (imagens_geometricas, imagem_projecao, IMAGENS_PREDITIVAS)
        imgs = set(IMAGENS_PREDITIVAS) | set(imagens_geometricas(metodo)) | {imagem_projecao(metodo)}
        exigir(len(imgs) == 14 and all(n.endswith('.png') for n in imgs),
               f'Contrato F10 {metodo.upper()} nao possui 14 figuras distintas.')
        print(f'[OK] CONTRATO_F10_{metodo.upper()}=14_PNG',flush=True)
    from scripts.execucao.experimentos_reais_3pct_2d import LIBERACAO_V3
    exigir(not LIBERACAO_V3.exists(),
           'Marcador V3 ja existe. Revise a autorizacao antes de repetir homologacao.')
    # Gate deve falhar ANTES do treinamento.
    try:
        execucao.verificar_autorizacao_v3()
    except RuntimeError as exc:
        exigir('BLOQUEADOS NA V2' in str(exc), 'Motivo de bloqueio de V3 inesperado.')
    else:
        raise RuntimeError('Falha de seguranca: OOF/F10 nao estao bloqueados.')
    print('[OK] OOF_F10_2D=BLOQUEADOS_ATE_APROVACAO_VISUAL', flush=True)
    print('PARQUET_LIDO=NAO | F10_LIDO=NAO | TREINO_REAL=NAO',flush=True)
    return configs, instalador


class _MisturaSintetica:
    """Mistura 2D especificada manualmente. NUNCA chama sklearn.fit."""
    covariance_type = 'full'

    def __init__(self):
        self.means_ = np.array([[-3.5,-2.0], [2.8,-2.3],[-2.7,3.2],[3.4,3.1]],dtype=float)
        self.weights_ = np.array([.38,.26,.21,.15],dtype=float)
        self.covariances_ = np.array([[[1.6,.35],[.35,.8]],
                                      [[1.2,-.25],[-.25,1.0]],
                                      [[.9,.19],[.19,1.3]],
                                      [[1.1,.40],[.40,1.35]]], dtype=float)

    def predict_proba(self, x):
        from scipy.stats import multivariate_normal
        a = np.asarray(x, dtype=float)
        un = np.column_stack([self.weights_[k]*multivariate_normal.pdf(
            a,mean=self.means_[k],cov=self.covariances_[k]) for k in range(4)])
        den=un.sum(axis=1,keepdims=True)
        exigir(np.all(den > 0),'Mistura sintetica fora da faixa suportada.')
        return un/den


class _GmmSintetico:
    def __init__(self, nomes):
        self.modelo_ = _MisturaSintetica()
        self.saidas_ = ('G1','G2','G3','G4')
        self.colunas_ = list(nomes)

    def responsabilidades(self,x):
        a=np.asarray(x,dtype=float)
        return pd.DataFrame(self.modelo_.predict_proba(a),columns=self.saidas_)


def fixture(metodo, cfg):
    """Sem ler dados reais; y e p sao ilustrativos e nunca reportados como metricas."""
    from types import SimpleNamespace
    from numpy.random import default_rng
    rng = default_rng(42)
    nomes = ('PC1','PC2') if metodo == 'pca' else ('TSNE1','TSNE2')
    gm=_GmmSintetico(nomes)
    n=320
    ks=np.concatenate([np.full(c,k,dtype=np.uint8) for k,c in enumerate((120,85,70,45))])
    x=np.vstack([rng.multivariate_normal(gm.modelo_.means_[k],gm.modelo_.covariances_[k],size=(ks==k).sum())
                 for k in range(4)])
    perm=rng.permutation(n)
    x=x[perm]
    risco=np.array([.025,.12,.42,.81])
    gam=gm.responsabilidades(x)
    p=gam.to_numpy()@risco
    y=(rng.random(n) < (.04 + p*.42)).astype(np.int8)
    if y.sum()==0: y[:5]=1
    rh=SimpleNamespace(
        riscos_=pd.Series(risco,index=gm.saidas_),
        prevalencia_treino_=float(y.mean()),
        massas_=pd.Series(np.ones(4)*80,index=gm.saidas_),
        fraudes_ponderadas_=pd.Series(np.ones(4)*12,index=gm.saidas_),
    )
    redutor=SimpleNamespace(parametros={'n_components':2,'perplexity':10,
                      'metric':'euclidean','initialization':'pca'},kl_divergence_=1.234567)
    dados={'coord':pd.DataFrame(x,columns=nomes),'y':y,'scores':p,'gmm':gm,'rh':rh,
           'gamma':gam,'redutor':redutor,'tempo_reducao':0.0,'limiar_oof':.25}
    cfg=deepcopy(cfg)
    cfg['visualizacao']['graficos']['densidades_marginais']['pontos_curva']=180
    cfg['visualizacao']['graficos']['densidades_marginais']['pontos_grade_2d']=90
    cfg['visualizacao']['graficos']['regioes_dominancia']['resolucao_2d']=60
    cfg['visualizacao']['graficos']['sensibilidade_fronteira']['resolucao_2d']=60
    cfg['visualizacao']['graficos']['mapa_risco_fronteira']['resolucao_2d']=60
    cfg['visualizacao']['geometria']['pontos_maximos_amostra']=250
    return cfg,dados


def gerar_previa(configs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src import visualizacao_gmm01 as g01
    from src import visualizacao_gmm04 as g04
    from src import visualizacao_gmm07 as g07
    from src import visualizacao_geometrica as geo
    from src import visualizacao_tsne_pdf as tsne_pdf
    from src.visualizacao_tsne_2d import gerar_projecao_2d
    if DESTINO.exists():
        validar_previa(configs)
        print('PREVIAS_JA_EXISTENTES=VALIDADAS_SEM_SOBRESCRITA',flush=True)
        return
    # Pasta externa a qualquer ensaio real; uma previa interrompida fica separada.
    import uuid
    stage=DESTINO.with_name('.v3_previa_sintetica_incompleta_'+uuid.uuid4().hex[:8])
    stage.mkdir(parents=True,exist_ok=False)
    try:
        for metodo,cfg in configs.items():
            pasta=stage/metodo
            pasta.mkdir(exist_ok=False)
            c,d=fixture(metodo,cfg)
            print(f'[SINTETICO] {metodo.upper()} 2D: GMM01, GMM04, GMM07, GMM08, projecao.',flush=True)
            g01.gerar(c,d,pasta,'png',85,False)
            g04.gerar(c,d,pasta,'png',85,False)
            av={'origem':'simulado','y':d['y'],'p':d['scores']}
            es={'limiar_mcc_oof':.25}
            g07.gerar(c,d,es,pasta,'png',85,False,avaliacao=av)
            geo._fronteira(c,d,es,pasta,'png',85,False,avaliacao=av)
            gerar_projecao_2d(c,d,pasta,'png',85,False,
                              origem='desenvolvimento',metodo=metodo.upper())
            if metodo=='tsne':
                tsne_pdf.diagnostico_kl(c,d,pasta,'png',85,False,origem='simulado')
            plt.close('all')
            encontrado={p.name for p in pasta.glob('*.png')}
            exigir(encontrado==set(PREVIA_GERADA[metodo]),
                   f'Previa {metodo}: arquivos faltando/extras {encontrado ^ set(PREVIA_GERADA[metodo])}')
            print(f'[OK] {metodo.upper()}_PREVIAS={len(encontrado)}',flush=True)
        grupos=[]
        for metodo,nomes in PREVIA_GERADA.items():
            figs='\n'.join(f'<figure><a href="{metodo}/{escape(n)}" target="_blank"><img loading="lazy" src="{metodo}/{escape(n)}"></a><figcaption>{escape(n)}</figcaption></figure>' for n in nomes)
            grupos.append(f'<section><h2>{metodo.upper()} 2D — prévias sintéticas</h2><div class="figs">{figs}</div></section>')
        html=f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>V3 — Prévia sintética 2D</title>
<style>body{{font-family:system-ui;background:#f3f6fa;color:#182f4b;margin:0}}main{{max-width:1280px;margin:auto;padding:20px}}
.banner{{background:#102c50;color:white;padding:20px;border-radius:12px}}
.figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:14px}}
figure{{border:1px solid #dde6f0;background:white;margin:0;border-radius:12px;padding:8px}}
img{{width:100%;height:auto}}figcaption{{font-size:12px;word-break:break-all;padding:8px}}</style></head>
<body><main><div class="banner"><h1>V3 — Homologação visual 2D</h1><p>PRÉVIAS SINTÉTICAS. Não são resultados do TCC: não houve fit, OOF, F10 ou leitura do Parquet.</p></div>
{''.join(grupos)}<p>O TSNE02 (trustworthiness) só será gerado com desenvolvimento real, pois pressupõe ajustes t-SNE. Sua integridade de layout já está protegida.</p></main></body></html>'''
        (stage/'index.html').write_text(html,encoding='utf-8')
        hashes={str(p.relative_to(stage)).replace('\\','/'):digest(p) for p in stage.rglob('*.png')}
        hashes['index.html']=digest(stage/'index.html')
        manifest={'tipo':MARCADOR,'status':'PREVIAS_GERADAS_PENDENTE_REVISAO_VISUAL',
                  'origem':'SINTETICA_SEM_FIT','sha256_yaml':SHA_CONFIG,
                  'sha256_artefatos':hashes,'n_png':len(hashes)-1,
                  'f10_utilizado':False,'oof_utilizado':False,'treino_executado':False,
                  'gerado_em_utc':datetime.now(timezone.utc).isoformat()}
        with (stage/'manifesto_previa_v3.json').open('x',encoding='utf-8') as f:
            json.dump(manifest,f,indent=2,ensure_ascii=False)
            f.write('\n')
        exigir(not DESTINO.exists(),'Outra previa V3 surgiu durante a escrita.')
        stage.rename(DESTINO)
    except Exception:
        import shutil
        shutil.rmtree(stage,ignore_errors=True)
        raise
    validar_previa(configs)
    print('GALERIA_PREVIA='+str(ARQUIVO_INDICE),flush=True)


def validar_previa(configs):
    mpath=DESTINO/'manifesto_previa_v3.json'
    exigir(mpath.is_file(),'Previa V3 ausente. Execute --executar.')
    obj=json.loads(mpath.read_text(encoding='utf-8-sig'))
    exigir(obj['tipo']==MARCADOR and obj['status']=='PREVIAS_GERADAS_PENDENTE_REVISAO_VISUAL'
           and obj['sha256_yaml']==SHA_CONFIG and obj['n_png']==17,
           'Manifesto V3 invalido: revisar.')
    esperados=set()
    for metodo, nomes in PREVIA_GERADA.items():
        for nome in nomes: esperados.add(f'{metodo}/{nome}')
    esperados.add('index.html')
    exigir(set(obj['sha256_artefatos'])==esperados,'Conjunto de arquivos V3 inesperado.')
    for rel, digest_esperado in obj['sha256_artefatos'].items():
        p=DESTINO/rel
        exigir(p.is_file() and digest(p)==digest_esperado,
               'Imagem V3 ausente ou modificada: '+rel)
        if p.suffix.lower()=='.png':
            exigir(p.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n', 'PNG V3 corrompido: '+rel)
    print('V3_PREVIAS_SINTETICAS=17/17',flush=True)
    print('V3_GALERIA_HTML='+str(ARQUIVO_INDICE),flush=True)
    print('V3_STATUS=HOMOLOGACAO_TECNICA_OK_AGUARDANDO_REVISAO_VISUAL',flush=True)
    print('OOF_F10_2D_AINDA_BLOQUEADOS=SIM',flush=True)
    print('TREINAMENTO_REEXECUTADO=NAO',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    grupo=p.add_mutually_exclusive_group(required=True)
    grupo.add_argument('--planejar',action='store_true')
    grupo.add_argument('--executar',action='store_true')
    grupo.add_argument('--validar',action='store_true')
    args=p.parse_args(argv)
    try:
        configs,_=inspecionar_v2()
        if args.planejar:
            print('V3_PLANEJAMENTO=APROVADO_PARA_PREVIAS_SINTETICAS',flush=True)
            print('V3_DESTINO='+str(ARQUIVO_INDICE),flush=True)
        elif args.executar:
            gerar_previa(configs)
        else:
            validar_previa(configs)
        return 0
    except Exception as e:
        print('V3_STATUS=BLOQUEADO_SEM_TREINAMENTO',file=sys.stderr,flush=True)
        print('MOTIVO='+str(e),file=sys.stderr,flush=True)
        traceback.print_exc()
        return 1

if __name__=='__main__':
    raise SystemExit(main())
