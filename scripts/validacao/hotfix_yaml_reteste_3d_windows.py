# -*- coding: utf-8 -*-
"""Correcao estrita da verificacao LF/CRLF para reteste PCA/TSNE 3D 3%.

Sem treinamento, inferencia F10, Parquet ou modificacao de resultados. O hotfix
modifica somente scripts/execucao/experimentos_reais_3pct_3d.py com backup.
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

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'scripts' / 'execucao' / 'experimentos_reais_3pct_3d.py'
REL_SOURCE = SOURCE.relative_to(ROOT)
OLD = """    exigir(man.get('status') == 'concluido_06_03' and
           man.get('tipo') == 'ensaio' and
           man.get('sha256_yaml') == sha256(yaml_texto(cfg).encode('utf-8')).hexdigest(),
           'OOF nao tem status/tipo/hash YAML correto.')"""
NEW = """    # Windows: Path.write_text() pode converter LF em CRLF no YAML temporario.
    # Exigir a mesma configuracao, mantendo SHA-256 obrigatorio (LF ou CRLF).
    configuracao_do_ensaio = yaml.safe_load(
        (concluido / 'configuracao.yaml').read_text(encoding='utf-8-sig')
    )
    exigir(configuracao_do_ensaio == cfg,
           'OOF possui configuracao diferente do lote 3D atual.')
    texto_yaml = yaml_texto(cfg)
    hashes_yaml_permitidos = {
        sha256(texto_yaml.encode('utf-8')).hexdigest(),
        sha256(texto_yaml.replace('\\n', '\\r\\n').encode('utf-8')).hexdigest(),
    }
    exigir(man.get('status') == 'concluido_06_03' and
           man.get('tipo') == 'ensaio' and
           man.get('sha256_yaml') in hashes_yaml_permitidos,
           'OOF nao tem status/tipo/hash YAML LF/CRLF correto.')
    print('[OOF 3D] YAML estrutural e SHA LF/CRLF validos; sem refazer OOF.', flush=True)"""


def fail(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def source_state():
    fail(SOURCE.is_file(), 'Executor 3D ausente: ' + str(SOURCE))
    content = SOURCE.read_text(encoding='utf-8-sig')
    fail('MARCADOR = \'TCC_EXPERIMENTAL_V3_3PCT_3D_REAL_BASE_V1\'' in content,
         'Versao do executor 3D nao reconhecida; nao modificar.')
    if content.count(NEW) == 1 and OLD not in content:
        ast.parse(content, filename=str(SOURCE))
        return 'INSTALADO', content, content
    fail(content.count(OLD) == 1 and NEW not in content,
         'Bloco LF/CRLF do executor 3D nao corresponde a versao aprovada; nenhum arquivo alterado.')
    patched = content.replace(OLD, NEW, 1)
    ast.parse(patched, filename=str(SOURCE))
    return 'PENDENTE', content, patched


def check_existing(lote):
    """Somente manifesto/configuracao: nao carrega Parquet, F10 ou joblib."""
    runner = importlib.import_module('scripts.execucao.reteste_real_3d_3pct')
    runner.configurar(lote)
    baseline = importlib.import_module('scripts.execucao.experimentos_reais_3pct_3d')
    results = {}
    for method in ('pca', 'tsne'):
        cfg, _ = baseline.construir_config(method)
        completed, partial = baseline.localizar_experimento(cfg, method)
        fail(not(completed and partial), 'Encontradas pastas completa e parcial simultaneamente.')
        if completed is None:
            print(method.upper() + '_OOF_CONCLUIDO=NAO', flush=True)
            if partial:
                print(method.upper() + '_OOF_EM_ANDAMENTO=' + str(partial), flush=True)
            results[method] = None
            continue
        mf = completed / 'manifesto.json'
        cf = completed / 'configuracao.yaml'
        fail(mf.is_file() and cf.is_file(), 'Faltam manifesto/configuracao: ' + str(completed))
        man = json.loads(mf.read_text(encoding='utf-8-sig'))
        saved = yaml.safe_load(cf.read_text(encoding='utf-8-sig'))
        fail(saved == cfg, method.upper() + ': YAML do OOF nao corresponde ao ensaio esperado.')
        text = baseline.yaml_texto(cfg)
        variants = {
            'LF': sha(text.encode('utf-8')),
            'CRLF': sha(text.replace('\n', '\r\n').encode('utf-8')),
        }
        hits = [name for name, digest in variants.items() if digest == man.get('sha256_yaml')]
        fail(man.get('status') == 'concluido_06_03' and man.get('tipo') == 'ensaio' and hits,
             method.upper() + ': status/tipo/hash YAML desconhecido, nao prosseguir.')
        print(method.upper() + '_OOF_CONCLUIDO=SIM', flush=True)
        print(method.upper() + '_SHA_FORMATO=' + '/'.join(hits), flush=True)
        print(method.upper() + '_OOF_GALERIA=' + str(completed / 'vencedor' / 'visualizacao' / 'galeria' / 'index.html'), flush=True)
        print(method.upper() + '_OOF_VISUALIZACOES=' + str(man.get('visualizacoes')), flush=True)
        print(method.upper() + '_EXPERIMENTO=' + str(completed), flush=True)
        results[method] = hits
    return results


def install():
    state, _, patched = source_state()
    if state == 'INSTALADO':
        print('HOTFIX_YAML_3D=JA_INSTALADO', flush=True)
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_UTC')
    backup = ROOT / 'backups' / 'hotfix_yaml_reteste_3d' / stamp / REL_SOURCE
    backup.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SOURCE, backup)
    fail(sha(backup.read_bytes()) == sha(SOURCE.read_bytes()), 'Backup SHA256 divergente; nenhuma substituicao realizada.')
    temp = SOURCE.with_name('.' + SOURCE.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('x', encoding='utf-8', newline='\n') as f:
            f.write(patched)
            f.flush()
            os.fsync(f.fileno())
        ast.parse(temp.read_text(encoding='utf-8'), filename=str(temp))
        os.replace(temp, SOURCE)
    finally:
        temp.unlink(missing_ok=True)
    fail(source_state()[0] == 'INSTALADO', 'O script nao apresentou a assinatura esperada apos a instalacao.')
    print('BACKUP_EXECUTOR_3D=' + str(backup), flush=True)
    print('SHA256_EXECUTOR_3D_ANTES=' + sha(backup.read_bytes()), flush=True)
    print('SHA256_EXECUTOR_3D_DEPOIS=' + sha(SOURCE.read_bytes()), flush=True)
    print('HOTFIX_YAML_3D=APLICADO', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lote', default='RETESTE_20261103_3D_V1')
    group = parser.add_mutually_exclusive_group(required=True)
    for s in ('planejar', 'aplicar', 'validar'):
        group.add_argument('--' + s, action='store_true')
    args = parser.parse_args(argv)
    try:
        mode = next(k for k in ('planejar', 'aplicar', 'validar') if getattr(args, k))
        state, _, _ = source_state()
        print('HOTFIX_YAML_3D_MODO=' + mode, flush=True)
        print('HOTFIX_YAML_3D_ESTADO_ANTES=' + state, flush=True)
        observed = check_existing(args.lote)
        if mode == 'planejar':
            print('HOTFIX_YAML_3D_PLANEJAMENTO=VALIDADO_SEM_ALTERACOES', flush=True)
        elif mode == 'aplicar':
            # O problema descrito so foi observado apos PCA concluir OOF.
            fail(observed.get('pca') is not None,
                 'OOF PCA 3D ainda nao concluido. Nao aplicar hotfix sem evidencia.')
            install()
        else:
            fail(state == 'INSTALADO', 'Hotfix ainda nao instalado.')
            print('HOTFIX_YAML_3D=VALIDADO', flush=True)
        print('TREINAMENTO_REEXECUTADO=NAO', flush=True)
        print('INFERENCIA_F10_REPETIDA=NAO', flush=True)
        print('PARQUET_LIDO=NAO', flush=True)
        print('EXIT_CODE_HOTFIX_YAML_3D=0', flush=True)
        return 0
    except Exception as err:
        print('HOTFIX_YAML_3D=BLOQUEADO', file=sys.stderr, flush=True)
        print('HOTFIX_MOTIVO=' + str(err), file=sys.stderr, flush=True)
        print('EXIT_CODE_HOTFIX_YAML_3D=1', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
