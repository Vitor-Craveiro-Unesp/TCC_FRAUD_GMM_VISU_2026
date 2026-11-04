# -*- coding: utf-8 -*-
"""Corrige SOMENTE o falso bloqueio da V4 por diretorio teste_final vazio.

NAO treina, NAO importa os modelos e NAO altera outputs. Verificacao final
chama exclusivamente --validar do executor V4, nunca --executar.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[2]
FONTE = ROOT / 'scripts' / 'execucao' / 'v4_oof_real_tsne2d_3pct.py'
YAML = ROOT / 'configs' / 'experimentos_reais_3pct_2d' / 'tsne_gmm_2d_3pct.yaml'

ANTIGO = ("        exigir(not pasta_f10.exists(),\n"
          "               'Pasta teste_final inesperada em experimento TSNE 2D. '\n"
          "               'A V4 nao pode continuar: ' + str(pasta_f10))\n")
NOVO = ("        # O orquestrador OOF cria antecipadamente teste_final/ VAZIA.\n"
        "        # A existencia da pasta NAO significa que F10 tenha sido utilizado.\n"
        "        # Recusar qualquer arquivo/subpasta: V4 so pode validar F1..F9.\n"
        "        if pasta_f10.exists():\n"
        "            exigir(pasta_f10.is_dir(),\n"
        "                   'Caminho teste_final nao e diretorio: ' + str(pasta_f10))\n"
        "            presentes = sorted(p.name for p in pasta_f10.iterdir())\n"
        "            exigir(not presentes,\n"
        "                   'Pasta teste_final NAO VAZIA: V4 bloqueada. '\n"
        "                   'Inspecione antes de continuar. Caminho=' + str(pasta_f10) +\n"
        "                   ' | entradas=' + repr(presentes))\n")


def exigir(condicao, msg):
    if not condicao:
        raise RuntimeError(msg)


def texto_fonte():
    exigir(FONTE.is_file(), f'Executor V4 nao encontrado: {FONTE}')
    data = FONTE.read_bytes()
    # Posicionar patch em bytes sem modificar o restante e preservando CRLF/LF.
    old_lf = ANTIGO.encode('utf-8')
    new_lf = NOVO.encode('utf-8')
    for forma, antigo, novo in (('LF', old_lf, new_lf),
                               ('CRLF', old_lf.replace(b'\n', b'\r\n'),
                                new_lf.replace(b'\n', b'\r\n'))):
        n_old, n_new = data.count(antigo), data.count(novo)
        if n_old == 1 and n_new == 0:
            return data, antigo, novo, forma, 'PENDENTE'
        if n_new == 1 and n_old == 0:
            return data, antigo, novo, forma, 'APLICADO'
    raise RuntimeError('Guarda V4 diferente da versao esperada; nada foi alterado. '
                       'Envie scripts/execucao/v4_oof_real_tsne2d_3pct.py para auditoria.')


def inspecionar_ensaio():
    exigir(YAML.is_file(), f'YAML congelado ausente: {YAML}')
    esperado = yaml.safe_load(YAML.read_text(encoding='utf-8-sig'))
    exigir(isinstance(esperado, dict) and esperado.get('reducao', {}).get('metodo') == 'TSNE' and
           esperado.get('tsne', {}).get('n_components') == [2] and
           float(esperado.get('dados', {}).get('fracao_base', -1)) == .03,
           'YAML TSNE 2D/3% nao corresponde ao esperado.')
    base = ROOT / 'outputs'
    pastas = []
    for p in sorted(base.glob('ensaio_tsne_gmm_*')):
        origem = p / 'configuracao.yaml'
        if not origem.is_file():
            continue
        try:
            atual = yaml.safe_load(origem.read_text(encoding='utf-8-sig'))
        except (yaml.YAMLError, OSError):
            continue
        if atual == esperado:
            pastas.append(p)
    exigir(len(pastas) == 1,
           f'Esperado um unico OOF TSNE 2D 3% com o YAML congelado; encontrados={len(pastas)}. '
           f'Pastas={pastas}')
    pasta = pastas[0]
    manifesto_path = pasta / 'manifesto.json'
    exigir(manifesto_path.is_file(), f'Manifesto ausente: {manifesto_path}')
    man = json.loads(manifesto_path.read_text(encoding='utf-8-sig'))
    exigir(man.get('status') == 'concluido_06_03' and
           man.get('visualizacoes') == 'concluidas',
           'OOF/galeria nao concluidos segundo o manifesto. Nao alterar nada.')
    html = pasta / 'vencedor' / 'visualizacao' / 'galeria' / 'index.html'
    exigir(html.is_file(), f'Galeria OOF ausente: {html}')
    f10 = pasta / 'vencedor' / 'teste_final'
    if f10.exists():
        exigir(f10.is_dir(), f'teste_final nao e pasta: {f10}')
        entradas = sorted(p.name for p in f10.iterdir())
        exigir(not entradas,
               'teste_final NAO VAZIA; correção bloqueada. Nao excluir dados. '
               f'Pasta={f10}; entradas={entradas}')
        estado = 'DIRETORIO_VAZIO_CRIADO_PELO_OOF'
    else:
        estado = 'DIRETORIO_AUSENTE_SEM_PROBLEMA'
    return pasta, f10, html, estado


def aplicar():
    data, antigo, novo, forma, estado = texto_fonte()
    if estado == 'APLICADO':
        print('HOTFIX_V4=JA_APLICADO', flush=True)
        return
    exigir(data.count(antigo) == 1, 'Guarda antiga nao e unica.')
    modificado = data.replace(antigo, novo, 1)
    # Falha antes de sobrescrever caso a fonte tenha alterado durante a inspecao.
    exigir(FONTE.read_bytes() == data, 'Executor V4 mudou durante a aplicacao.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    backup = ROOT / 'backups' / 'v4_guarda_teste_final_vazio' / (stamp + '_' + uuid.uuid4().hex[:6])
    backup.mkdir(parents=True, exist_ok=False)
    arquivo_backup = backup / FONTE.name
    shutil.copy2(FONTE, arquivo_backup)
    exigir(arquivo_backup.read_bytes() == data, 'Backup da V4 nao conferiu.')
    temporario = FONTE.with_name('.' + FONTE.name + '.hotfix_' + uuid.uuid4().hex)
    try:
        temporario.write_bytes(modificado)
        compilar = compile(modificado.decode('utf-8-sig'), str(FONTE), 'exec')
        del compilar
        exigir(FONTE.read_bytes() == data, 'Executor V4 foi alterado por outra sessao.')
        os.replace(temporario, FONTE)
    finally:
        temporario.unlink(missing_ok=True)
    exigir(FONTE.read_bytes() == modificado, 'Patch nao confirmado.')
    print('BACKUP_V4=' + str(arquivo_backup), flush=True)
    print('FONTE_CORRIGIDA=' + str(FONTE), flush=True)
    print('HOTFIX_V4=APLICADO', flush=True)


def validar():
    data, _antigo, _novo, _forma, estado = texto_fonte()
    exigir(estado == 'APLICADO', 'Hotfix ainda nao aplicado. Execute --aplicar.')
    # Esta chamada e exclusiva de leitura e nao possui caminho para o treinamento.
    cmd = [sys.executable, '-B', '-m',
           'scripts.execucao.v4_oof_real_tsne2d_3pct', '--validar']
    print('VALIDACAO_V4_SEM_TREINO=' + ' '.join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    exigir(proc.returncode == 0, f'V4 --validar falhou; EXIT_CODE={proc.returncode}. '
           'Nao execute OOF novamente. Envie o traceback para analise.')
    print('HOTFIX_STATUS=VALIDADO_OOF_EXISTENTE_SEM_TREINAMENTO', flush=True)
    print('F10_LIDO=NAO', flush=True)
    print('OOF_REEXECUTADO=NAO', flush=True)
    print('EXIT_CODE_HOTFIX=0', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument('--planejar', action='store_true')
    sel.add_argument('--aplicar', action='store_true')
    sel.add_argument('--validar', action='store_true')
    args = ap.parse_args(argv)
    try:
        data, antigo, novo, forma, estado = texto_fonte()
        pasta, f10, html, e = inspecionar_ensaio()
        print('OOF_EXISTENTE=' + str(pasta), flush=True)
        print('GALERIA_OOF=' + str(html), flush=True)
        print('TESTE_FINAL=' + str(f10), flush=True)
        print('TESTE_FINAL_ESTADO=' + e, flush=True)
        print('V4_GUARDA_ATUAL=' + estado, flush=True)
        if args.aplicar:
            aplicar()
        elif args.validar:
            validar()
        else:
            print('PLANEJAMENTO_SEM_MODIFICACOES=SIM', flush=True)
            print('OOF_REEXECUTADO=NAO', flush=True)
        return 0
    except Exception as exc:
        print('HOTFIX_STATUS=BLOQUEADO_SEM_SOBRESCRITA', file=sys.stderr, flush=True)
        print('MOTIVO=' + str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
