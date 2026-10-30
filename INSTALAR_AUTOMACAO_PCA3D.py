# -*- coding: utf-8 -*-
"""Instalador protegido da automacao PCA3D; NAO treina nem usa F10.

python INSTALAR_AUTOMACAO_PCA3D.py --verificar
python INSTALAR_AUTOMACAO_PCA3D.py --instalar
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import py_compile
import shutil
import subprocess
import yaml

ROOT=Path(__file__).resolve().parent
ORQ=ROOT/'src/orquestracao_oof.py'
YAML=ROOT/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
PACOTE=ROOT/'PACOTE_PCA3D/finalizacao_automatica_pca3d.py'
DEST=ROOT/'src/finalizacao_automatica_pca3d.py'
BRANCH='feature/pca3d-automacao-oof-f10'

ANTES_CHECK='''        elif dim == [2]:
            from src.finalizacao_automatica_pca2d import validar_automacao
        else:
            raise RuntimeError('Finalizacao automatica disponivel apenas para PCA 1D ou 2D.')'''
DEPOIS_CHECK='''        elif dim == [2]:
            from src.finalizacao_automatica_pca2d import validar_automacao
        elif dim == [3]:
            from src.finalizacao_automatica_pca3d import validar_automacao
        else:
            raise RuntimeError('Finalizacao automatica disponivel apenas para PCA 1D, 2D ou 3D.')'''
ANTES_RUN='''        if config.get('pca', {}).get('n_components') == [1]:
            from src.finalizacao_automatica_pca1d import concluir_ensaio
        else:
            from src.finalizacao_automatica_pca2d import concluir_ensaio
        concluir_ensaio(final)'''
DEPOIS_RUN='''        dim=config.get('pca', {}).get('n_components')
        if dim == [1]:
            from src.finalizacao_automatica_pca1d import concluir_ensaio
        elif dim == [2]:
            from src.finalizacao_automatica_pca2d import concluir_ensaio
        elif dim == [3]:
            from src.finalizacao_automatica_pca3d import concluir_ensaio
        else:
            raise RuntimeError('Dimensionalidade PCA final invalida.')
        concluir_ensaio(final)'''
ANTES_YAML='pca:\n  n_components:\n  - 2'
DEPOIS_YAML='pca:\n  n_components:\n  - 3'
# Em modo retomada, ler do experimento, nao do YAML global.
ANTES_RETOMADA='''        from src.finalizacao_automatica_pca1d import concluir_ensaio
        alvo = raiz/a.experimento if not a.experimento.is_absolute() else a.experimento
        concluir_ensaio(alvo)'''
DEPOIS_RETOMADA='''        alvo = raiz/a.experimento if not a.experimento.is_absolute() else a.experimento
        config_salva=alvo/'configuracao.yaml'
        if not config_salva.is_file():
            raise RuntimeError('Configuração do experimento de retomada ausente.')
        dim=yaml.safe_load(config_salva.read_text(encoding='utf-8-sig'))['pca']['n_components']
        if dim == [1]:
            from src.finalizacao_automatica_pca1d import concluir_ensaio
        elif dim == [2]:
            from src.finalizacao_automatica_pca2d import concluir_ensaio
        elif dim == [3]:
            from src.finalizacao_automatica_pca3d import concluir_ensaio
        else:
            raise RuntimeError('Dimensão PCA de retomada não suportada.')
        concluir_ensaio(alvo)'''


def exigir(c,m):
    if not c: raise RuntimeError(m)


def preparar():
    for arq in (ORQ,YAML,PACOTE,ROOT/'src/finalizacao_automatica_pca1d.py',
                ROOT/'src/finalizacao_automatica_pca2d.py'):
        exigir(arq.is_file(),f'Arquivo obrigatorio ausente: {arq}')
    exigir(not DEST.exists(),'PCA3D ja instalado; nao sobrescrever.')
    branch=subprocess.run(['git','branch','--show-current'],cwd=ROOT,
                          text=True,capture_output=True,check=True).stdout.strip()
    exigir(branch==BRANCH,f'Branch atual={branch!r}; exige={BRANCH!r}.')
    original=ORQ.read_text(encoding='utf-8-sig')
    cfg=YAML.read_text(encoding='utf-8-sig')
    for chave,trecho in (('validacao',ANTES_CHECK),('conclusao',ANTES_RUN),
                        ('retomada',ANTES_RETOMADA)):
        exigir(original.count(trecho)==1,f'Ancora do orquestrador mudou: {chave}.')
    exigir(cfg.count(ANTES_YAML)==1,'YAML nao esta em PCA [2] ou estrutura mudou.')
    novo=original.replace(ANTES_CHECK,DEPOIS_CHECK).replace(ANTES_RUN,DEPOIS_RUN)
    novo=novo.replace(ANTES_RETOMADA,DEPOIS_RETOMADA)
    novocfg=cfg.replace(ANTES_YAML,DEPOIS_YAML)
    compile(novo,str(ORQ),'exec')
    compile(PACOTE.read_text(encoding='utf-8'),str(PACOTE),'exec')
    antes=yaml.safe_load(cfg); depois=yaml.safe_load(novocfg)
    exigir(antes['pca']['n_components']==[2] and depois['pca']['n_components']==[3],
           'PCA dimensao inesperada.')
    depois['pca']['n_components']=[2]
    exigir(antes==depois,'Outros parametros YAML foram alterados.')
    print('[OK] Branch PCA3D correta; codigo e configuracao compativeis.')
    print('[OK] PCA1D e PCA2D preservados; F10 nao acessado.')
    return original,cfg,novo,novocfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--verificar',action='store_true')
    g.add_argument('--instalar',action='store_true')
    args=p.parse_args()
    original,cfg,novo,novocfg=preparar()
    if args.verificar:
        print('[VERIFICACAO] Nenhuma escrita; use --instalar.')
        return
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    backup=ROOT/'backups'/f'antes_automacao_pca3d_{stamp}'
    backup.mkdir(parents=True,exist_ok=False)
    shutil.copy2(ORQ,backup/ORQ.name)
    shutil.copy2(YAML,backup/YAML.name)
    try:
        with DEST.open('xb') as f:f.write(PACOTE.read_bytes())
        ORQ.write_text(novo,encoding='utf-8')
        YAML.write_text(novocfg,encoding='utf-8')
        py_compile.compile(str(ORQ),doraise=True)
        py_compile.compile(str(DEST),doraise=True)
    except BaseException:
        shutil.copy2(backup/ORQ.name,ORQ)
        shutil.copy2(backup/YAML.name,YAML)
        DEST.unlink(missing_ok=True)
        raise
    print('[INSTALADO] PCA3D no comando original: python -m src.orquestracao_oof --executar')
    print('[BACKUP]',backup)
    print('[YAML] PCA n_components=[3], demais parametros preservados.')


if __name__=='__main__':main()
