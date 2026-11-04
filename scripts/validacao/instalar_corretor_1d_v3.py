# -*- coding: utf-8 -*-
"""Instala somente a retificacao do script corretor do reteste 1D.

Nunca importa scripts do pipeline, nunca acessa dados e nunca altera resultados.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import os
import shutil
import sys
import traceback

RAIZ = Path(__file__).resolve().parents[2]
ALVO_REL = Path('scripts/validacao/corrigir_reteste_1d_oof_pre01.py')
ALVO = RAIZ / ALVO_REL
NOVO = RAIZ / 'hotfix/corrigir_reteste_1d_oof_pre01_v3.py'
SHA_ANTIGO = 'f87b0f152201eb0fb2b4c5e9a95c017af0570e11b2527f1dd5dd31247302a7cc'
SHA_NOVO = 'ba12aff10dc853067db8367f04ed2042feb5963c71463685e1fad11bfb9e8216'


def digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def conferir():
    if not ALVO.is_file() or not NOVO.is_file():
        raise RuntimeError('Corretor original ou arquivo de hotfix ausente. Confira a pasta do projeto.')
    antes = ALVO.read_bytes()
    depois = NOVO.read_bytes()
    if digest(depois) != SHA_NOVO:
        raise RuntimeError('SHA-256 do hotfix recebido e diferente do esperado.')
    ast.parse(depois.decode('utf-8-sig'), filename=str(NOVO))
    atual = digest(antes)
    if atual not in (SHA_ANTIGO, SHA_NOVO):
        raise RuntimeError('SHA-256 do corretor instalado e desconhecido: ' + atual +
                           '. Nao substituir: precisamos conferir modificacoes locais.')
    return antes, depois, atual


def aplicar(antes: bytes, depois: bytes, atual: str):
    if atual == SHA_NOVO:
        print('HOTFIX_CORRETOR_1D_V3=JA_APLICADO', flush=True)
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_UTC')
    backup = RAIZ / 'backups/hotfix_corretor_1d_v3' / stamp / ALVO_REL
    backup.parent.mkdir(parents=True, exist_ok=False)
    with backup.open('xb') as arq:
        arq.write(antes)
    if digest(backup.read_bytes()) != SHA_ANTIGO:
        raise RuntimeError('Backup nao confere com script original; sem alteracao.')
    temp = ALVO.with_name(ALVO.name + '.tmp_hotfix_v3')
    if temp.exists():
        raise RuntimeError('Temporario anterior encontrado; sem sobrescrever: '+str(temp))
    try:
        with temp.open('xb') as arq:
            arq.write(depois)
        if digest(temp.read_bytes()) != SHA_NOVO:
            raise RuntimeError('Arquivo temporario com SHA divergente.')
        os.replace(temp, ALVO)
        if digest(ALVO.read_bytes()) != SHA_NOVO:
            raise RuntimeError('Corretor instalado nao confere com SHA-256.')
    except BaseException:
        if temp.exists():
            temp.unlink()
        shutil.copy2(backup, ALVO)
        raise
    print('HOTFIX_CORRETOR_1D_V3=APLICADO', flush=True)
    print('BACKUP_CORRETOR='+str(backup), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modo', required=True, choices=['planejar','aplicar','validar'])
    args = parser.parse_args()
    try:
        antes, depois, atual = conferir()
        print('HOTFIX_CORRETOR_1D_V3_MODO='+args.modo, flush=True)
        print('SHA256_CORRETOR_ATUAL='+atual, flush=True)
        if args.modo == 'planejar':
            print('HOTFIX_NECESSARIO='+('SIM' if atual == SHA_ANTIGO else 'NAO'), flush=True)
            print('PLANEJAMENTO_SEM_ALTERACOES=SIM', flush=True)
        elif args.modo == 'aplicar':
            aplicar(antes, depois, atual)
        else:
            if atual != SHA_NOVO:
                raise RuntimeError('Hotfix ainda nao foi aplicado; SHA atual nao corresponde ao V3.')
            print('HOTFIX_CORRETOR_1D_V3=VALIDADO', flush=True)
        print('OOF_REEXECUTADO=NAO', flush=True)
        print('F10_RECALCULADO=NAO', flush=True)
        print('EXIT_CODE_HOTFIX_CORRETOR_1D_V3=0', flush=True)
        return 0
    except Exception as e:
        print('HOTFIX_CORRETOR_1D_V3=BLOQUEADO', file=sys.stderr, flush=True)
        print('MOTIVO='+str(e), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
