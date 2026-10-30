# -*- coding: utf-8 -*-
"""Instalador unico do fluxo PCA1D automatico; nao e executado durante os experimentos."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sys

import yaml

RAIZ = Path(__file__).resolve().parent
MOD_SRC = RAIZ / 'src' / 'orquestracao_oof.py'
INTERNAL_SRC = RAIZ / 'src' / 'finalizacao_automatica_pca1d.py'
CFG = RAIZ / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
TEMPLATE_ORQ = RAIZ / 'PACOTE_INSTALACAO' / 'orquestracao_oof.py'
TEMPLATE_INTERNAL = RAIZ / 'PACOTE_INSTALACAO' / 'finalizacao_automatica_pca1d.py'
ORIGINAL_HASH = '407be67574d0d19881fc45aa50ab2316ace51a4354d446e3f750dc1d93a17caf'


def exigir(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def sha(raw):
    return sha256(raw).hexdigest()


def corrigir_yaml(original):
    raw=original.decode('utf-8-sig')
    eol='\r\n' if '\r\n' in raw else '\n'
    pca=re.compile(r'(?m)^(pca:[ \t]*\r?\n[ \t]+n_components:[ \t]*\r?\n[ \t]*-[ \t]*)([123])([ \t]*(?:\r?\n|$))')
    matches=list(pca.finditer(raw))
    exigir(len(matches)==1,'YAML tem formato PCA diferente do esperado; nenhuma alteracao automatica segura.')
    raw=pca.sub(lambda m:m.group(1)+'1'+m.group(3),raw,count=1)
    exe=re.compile(r'(?m)^execucao:[ \t]*\r?$')
    exigir(len(list(exe.finditer(raw)))==1,'Secao execucao YAML ausente/ambigua.')
    # O bloco de execucao acaba no proximo cabecalho sem recuo.
    m=exe.search(raw)
    limite=re.search(r'(?m)^[^ \t\r\n#][^\r\n]*:',raw[m.end():])
    corpo=raw[m.end():m.end()+limite.start()] if limite else raw[m.end():]
    exigir('finalizacao_pca1d_automatica:' not in corpo,'YAML ja tem chave de automacao; revisar manualmente.')
    raw=raw[:m.end()]+eol+'  finalizacao_pca1d_automatica: true'+raw[m.end():]
    conf=yaml.safe_load(raw)
    exigir(conf['pca']['n_components']==[1] and conf['execucao']['finalizacao_pca1d_automatica'] is True,
           'Falha ao configurar PCA1D automatico.')
    exigir(float(conf['dados']['fracao_base'])==.1 and conf['gmm']['n_components']==[4]
           and int(conf['gmm']['n_init'])==5 and conf['gmm']['covariance_type']=='full'
           and conf['amostragem']['modo']=='RUS'
           and int(conf['amostragem']['razao_negativos_positivos'])==16,
           'YAML atual nao corresponde aos parametros PCA1D K4 n_init5 FULL RUS16 .1.')
    return raw.encode('utf-8')


def verificar_preflight():
    for p in (MOD_SRC,CFG,TEMPLATE_ORQ,TEMPLATE_INTERNAL):
        exigir(p.is_file(),f'Arquivo ausente: {p}')
    original=MOD_SRC.read_bytes()
    novo=TEMPLATE_ORQ.read_bytes()
    interno=TEMPLATE_INTERNAL.read_bytes()
    exigir(sha(original) in {ORIGINAL_HASH,sha(novo)},
           'Seu orquestracao_oof.py difere da versao analisada. NAO substituir. Envie seu arquivo atual.')
    exigir(not INTERNAL_SRC.exists() or INTERNAL_SRC.read_bytes()==interno,
           'Arquivo interno com mesmo nome ja existe e e diferente; nao sobrescrever.')
    ast.parse(novo.decode('utf-8'))
    ast.parse(interno.decode('utf-8'))
    return corrigir_yaml(CFG.read_bytes()),novo,interno


def instalar():
    alterado,novo,interno=verificar_preflight()
    backup=RAIZ/'backups'/('integracao_comando_original_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    exigir(not backup.exists(),'Backup ja existe.')
    backup.mkdir(parents=True)
    shutil.copy2(MOD_SRC, backup/'orquestracao_oof.py')
    shutil.copy2(CFG,backup/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml')
    atual_orq=MOD_SRC.read_bytes()
    atual_cfg=CFG.read_bytes()
    atual_internal=INTERNAL_SRC.read_bytes() if INTERNAL_SRC.is_file() else None
    try:
        INTERNAL_SRC.write_bytes(interno)
        MOD_SRC.write_bytes(novo)
        CFG.write_bytes(alterado)
        exigir(yaml.safe_load(CFG.read_text(encoding='utf-8-sig'))['execucao']['finalizacao_pca1d_automatica'] is True,
               'Falha ao instalar flag.')
        manifest={'backup':str(backup),'orquestrador_original_sha256':sha(atual_orq),
                  'orquestrador_novo_sha256':sha(novo),'yaml_original_sha256':sha(atual_cfg),
                  'yaml_novo_sha256':sha(alterado),'integracao_sha256':sha(interno)}
        (backup/'registro.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    except Exception:
        MOD_SRC.write_bytes(atual_orq)
        CFG.write_bytes(atual_cfg)
        if atual_internal is None: INTERNAL_SRC.unlink(missing_ok=True)
        else: INTERNAL_SRC.write_bytes(atual_internal)
        raise
    print('[INSTALADO] Comando original integrado para PCA1D e F10 exploratorio.')
    print('[BACKUP]',backup)
    print('[PARAMETROS] PCA1D, K4, n_init5, FULL, RUS16, 10%; F10 10%.')
    print('[PROXIMO] python -m src.orquestracao_oof --planejar')
    print('[EXECUCAO NORMAL] python -m src.orquestracao_oof --executar')


def main():
    p=argparse.ArgumentParser()
    modos=p.add_mutually_exclusive_group(required=True)
    modos.add_argument('--verificar',action='store_true')
    modos.add_argument('--instalar',action='store_true')
    args=p.parse_args()
    if args.verificar:
        verificar_preflight()
        print('[OK] Arquivos do projeto compativeis. Nenhuma alteracao ou F10 lido.')
    else:
        instalar()


if __name__=='__main__':
    main()
