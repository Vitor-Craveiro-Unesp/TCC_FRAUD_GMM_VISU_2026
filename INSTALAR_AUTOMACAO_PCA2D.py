# -*- coding: utf-8 -*-
"""Instalador seguro da automacao PCA 2D. Nao executa treinamento nem F10.

python INSTALAR_AUTOMACAO_PCA2D.py --verificar
python INSTALAR_AUTOMACAO_PCA2D.py --instalar
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
PACOTE=ROOT/'PACOTE_PCA2D/finalizacao_automatica_pca2d.py'
ORQ=ROOT/'src/orquestracao_oof.py'
YAML=ROOT/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
DEST=ROOT/'src/finalizacao_automatica_pca2d.py'
BRANCH='feature/pca2d-automacao-oof-f10'

ANTIGO_CHECK='''    if automatizar_f10:
        from src.finalizacao_automatica_pca1d import validar_automacao
        validar_automacao(config)'''
NOVO_CHECK='''    if automatizar_f10:
        dim=config.get('pca', {}).get('n_components')
        if dim == [1]:
            from src.finalizacao_automatica_pca1d import validar_automacao
        elif dim == [2]:
            from src.finalizacao_automatica_pca2d import validar_automacao
        else:
            raise RuntimeError('Finalizacao automatica disponivel apenas para PCA 1D ou 2D.')
        validar_automacao(config)'''
ANTIGO_RUN='''        from src.finalizacao_automatica_pca1d import concluir_ensaio
        concluir_ensaio(final)'''
NOVO_RUN='''        if config.get('pca', {}).get('n_components') == [1]:
            from src.finalizacao_automatica_pca1d import concluir_ensaio
        else:
            from src.finalizacao_automatica_pca2d import concluir_ensaio
        concluir_ensaio(final)'''
ANTIGO_PRINT='Iniciando finalizacao automatica PCA1D (F1-F9 + F10 exploratorio)'
NOVO_PRINT='Iniciando finalizacao automatica PCA (F1-F9 + F10 exploratorio)'
YAML_ANTIGO='pca:\n  n_components:\n  - 1'
YAML_NOVO='pca:\n  n_components:\n  - 2'


def exigir(c,m):
    if not c: raise RuntimeError(m)


def preparar():
    for p in (PACOTE,ORQ,YAML,ROOT/'src/finalizacao_automatica_pca1d.py'):
        exigir(p.is_file(),f'Arquivo necessario ausente: {p}')
    exigir(not DEST.exists(), 'Modulo PCA2D ja existe: preservar e verificar instalacao.')
    branch=subprocess.run(['git','branch','--show-current'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    exigir(branch==BRANCH,f'Branch atual {branch!r}; exige {BRANCH!r}.')
    orig=ORQ.read_text(encoding='utf-8-sig')
    cfg=YAML.read_text(encoding='utf-8-sig')
    exigir(orig.count(ANTIGO_CHECK)==1,'Ponto de validacao do orquestrador mudou.')
    exigir(orig.count(ANTIGO_RUN)==1,'Ponto de finalizacao do orquestrador mudou.')
    exigir(orig.count(ANTIGO_PRINT)==1,'Log do orquestrador mudou.')
    exigir(cfg.count(YAML_ANTIGO)==1,'PCA da configuracao nao esta em [1] ou YAML mudou.')
    novo_orq=orig.replace(ANTIGO_CHECK,NOVO_CHECK).replace(ANTIGO_RUN,NOVO_RUN).replace(ANTIGO_PRINT,NOVO_PRINT)
    novo_cfg=cfg.replace(YAML_ANTIGO,YAML_NOVO)
    compile(novo_orq,str(ORQ),'exec')
    compile(PACOTE.read_text(encoding='utf-8'),str(PACOTE),'exec')
    import yaml
    antes=yaml.safe_load(cfg);depois=yaml.safe_load(novo_cfg)
    exigir(antes['pca']['n_components']==[1] and depois['pca']['n_components']==[2],
           'Dimensoes PCA nao conferem.')
    depois['pca']['n_components']=[1]
    exigir(antes==depois,'Configuracao de outros parametros foi alterada.')
    print('[OK] Branch PCA2D correta; orquestrador e YAML compativeis.',flush=True)
    print('[OK] OOF, PCA1D e funcoes estatisticas nao serao alterados.',flush=True)
    return orig,cfg,novo_orq,novo_cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modos=parser.add_mutually_exclusive_group(required=True)
    modos.add_argument('--verificar',action='store_true')
    modos.add_argument('--instalar',action='store_true')
    a=parser.parse_args()
    orig,cfg,novo_orq,novo_cfg=preparar()
    if a.verificar:
        print('[VERIFICACAO] Nenhum arquivo alterado. Use --instalar.',flush=True)
        return
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    back=ROOT/'backups'/f'antes_automacao_pca2d_{stamp}'
    back.mkdir(parents=True,exist_ok=False)
    shutil.copy2(ORQ,back/ORQ.name)
    shutil.copy2(YAML,back/YAML.name)
    alterados=[]
    try:
        DEST.write_bytes(PACOTE.read_bytes())
        alterados.append(DEST)
        ORQ.write_text(novo_orq,encoding='utf-8')
        alterados.append(ORQ)
        YAML.write_text(novo_cfg,encoding='utf-8')
        alterados.append(YAML)
        py_compile.compile(str(ORQ),doraise=True)
        py_compile.compile(str(DEST),doraise=True)
    except Exception:
        shutil.copy2(back/ORQ.name,ORQ)
        shutil.copy2(back/YAML.name,YAML)
        if DEST in alterados:DEST.unlink(missing_ok=True)
        raise
    print('[INSTALADO] Automacao PCA 2D vinculada ao comando original.',flush=True)
    print('[BACKUP]',back,flush=True)
    print('[CONFIGURACAO] PCA n_components=[2]; GMM e demais parametros preservados.',flush=True)
    print('[PROXIMO] python -m src.orquestracao_oof --planejar',flush=True)


if __name__=='__main__':
    main()
