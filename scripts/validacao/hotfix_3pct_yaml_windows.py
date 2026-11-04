# -*- coding: utf-8 -*-
"""Hotfix pontual para sha256_yaml OOF 3% em Windows LF/CRLF.

NAO reexecuta OOF, nao consulta Parquet/F10, nao altera dados de saida.
Faz backup integral do script original antes de patch atomico.

python -B -m scripts.validacao.hotfix_3pct_yaml_windows --planejar
python -B -m scripts.validacao.hotfix_3pct_yaml_windows --aplicar
python -B -m scripts.validacao.hotfix_3pct_yaml_windows --validar
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import yaml

RAIZ = Path(__file__).resolve().parents[2]
SCRIPT_REL = Path('scripts/execucao/experimentos_reais_3pct_1d.py')
OLD = """    exigir(man.get('status') == 'concluido_06_03' and
           man.get('tipo') == 'ensaio' and
           man.get('sha256_yaml') == sha256(yaml_texto(cfg).encode('utf-8')).hexdigest(),
           'OOF nao tem status/tipo/hash YAML correto.')"""
NEW = """    # Windows pode serializar o YAML temporario com CRLF em vez de LF.
    # Nunca ignorar o hash: validar o conteudo E os dois hashes possiveis.
    config_registrada = yaml.safe_load(
        (concluido / 'configuracao.yaml').read_text(encoding='utf-8-sig')
    )
    exigir(config_registrada == cfg,
           'Configuracao do OOF diverge do planejamento atual.')
    texto_yaml = yaml_texto(cfg)
    sha_yaml_validos = {
        sha256(texto_yaml.encode('utf-8')).hexdigest(),
        sha256(texto_yaml.replace('\\n', '\\r\\n').encode('utf-8')).hexdigest(),
    }
    exigir(man.get('status') == 'concluido_06_03' and
           man.get('tipo') == 'ensaio' and
           man.get('sha256_yaml') in sha_yaml_validos,
           'OOF nao tem status/tipo/hash YAML correto.')
    print('[OOF] YAML estrutural e SHA LF/CRLF conferidos; sem refazer OOF.', flush=True)"""


def preparar(p: Path):
    if not p.is_file():
        raise RuntimeError(f'Script de 3% nao existe: {p}')
    texto = p.read_text(encoding='utf-8-sig')
    if 'MARCADOR = \'TCC_EXPERIMENTAL_V3_3PCT_1D_REAL_V1\'' not in texto:
        raise RuntimeError('Versao inesperada do script de 3%; nao modificar.')
    if texto.count(NEW) == 1 and OLD not in texto:
        return texto, 'JA_INSTALADO'
    if texto.count(OLD) != 1 or NEW in texto:
        raise RuntimeError('Assinatura original do codigo nao reconhecida; abortado sem alterar nada.')
    alterado = texto.replace(OLD, NEW)
    ast.parse(alterado, filename=str(p))
    return alterado, 'AINDA_NAO_INSTALADO'


def aplicar(p: Path, raiz: Path):
    texto, status = preparar(p)
    if status == 'JA_INSTALADO':
        print('HOTFIX_STATUS=JA_INSTALADO', flush=True)
        return
    carimbo = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    backup_dir = raiz / 'outputs' / ('backup_hotfix_3pct_yaml_' + carimbo)
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup = backup_dir / SCRIPT_REL.name
    shutil.copy2(p, backup)
    if sha256(backup.read_bytes()).digest() != sha256(p.read_bytes()).digest():
        raise RuntimeError('Backup nao confere com script original; abortado.')
    temp = p.with_name('.' + p.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('x', encoding='utf-8', newline='\n') as f:
            f.write(texto)
            f.flush()
            os.fsync(f.fileno())
        ast.parse(temp.read_text(encoding='utf-8'))
        os.replace(temp, p)
    finally:
        temp.unlink(missing_ok=True)
    _, status_novo = preparar(p)
    if status_novo != 'JA_INSTALADO':
        raise RuntimeError('Verificacao do hotfix instalado falhou.')
    print('BACKUP_SCRIPT_ORIGINAL=' + str(backup), flush=True)
    print('HOTFIX_STATUS=INSTALADO', flush=True)


def validar(p: Path):
    _, estado = preparar(p)
    if estado != 'JA_INSTALADO':
        raise RuntimeError('Hotfix ainda nao foi instalado.')
    # Apenas YAML e manifestos ja existentes, nenhum arquivo de dados/F10.
    modulo = importlib.import_module('scripts.execucao.experimentos_reais_3pct_1d')
    for metodo in ('pca', 'tsne'):
        cfg, _ = modulo.construir_config(metodo)
        concluido, parcial = modulo.localizar_experimento(cfg, metodo)
        if concluido is None:
            print(metodo.upper() + '_OOF_EXISTENTE=NAO', flush=True)
            if parcial is not None:
                print(metodo.upper() + '_OOF_PARCIAL=' + str(parcial), flush=True)
            continue
        man = modulo.ler_json(concluido / 'manifesto.json')
        cfg_saved = yaml.safe_load((concluido / 'configuracao.yaml').read_text(encoding='utf-8-sig'))
        s = modulo.yaml_texto(cfg)
        variantes = {
            'LF': sha256(s.encode('utf-8')).hexdigest(),
            'CRLF': sha256(s.replace('\n','\r\n').encode('utf-8')).hexdigest(),
        }
        hits = [k for k, v in variantes.items() if v == man.get('sha256_yaml')]
        if man.get('status') != 'concluido_06_03' or man.get('tipo') != 'ensaio' or cfg_saved != cfg or not hits:
            raise RuntimeError(metodo.upper() + ': OOF concluido nao passa verificacao YAML/status; nao prosseguir.')
        print(metodo.upper() + '_OOF_REUTILIZAVEL=SIM', flush=True)
        print(metodo.upper() + '_SHA_FORMATO=' + '/'.join(hits), flush=True)
        print(metodo.upper() + '_OOF_PASTA=' + str(concluido), flush=True)
        print(metodo.upper() + '_OOF_VISUALIZACOES=' + str(man.get('visualizacoes')), flush=True)
    print('HOTFIX_STATUS=VALIDADO_SEM_TREINAR', flush=True)
    print('F10_ACESSADO_PELO_HOTFIX=NAO', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('planejar', 'aplicar', 'validar'):
        mode.add_argument('--' + name, action='store_true')
    args = parser.parse_args(argv)
    p = RAIZ / SCRIPT_REL
    try:
        if args.planejar:
            _, estado = preparar(p)
            print('SCRIPT_3PCT=' + str(p), flush=True)
            print('CODIGO=' + estado, flush=True)
            print('TREINAMENTO_EXECUTADO=NAO', flush=True)
            print('HOTFIX_STATUS=PLANEJAMENTO_SEM_ESCRITA', flush=True)
        elif args.aplicar:
            aplicar(p, RAIZ)
        else:
            validar(p)
        return 0
    except Exception as e:
        print('HOTFIX_STATUS=BLOQUEADO', file=sys.stderr, flush=True)
        print('MOTIVO=' + str(e), file=sys.stderr, flush=True)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
