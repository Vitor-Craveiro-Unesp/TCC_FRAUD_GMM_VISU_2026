# -*- coding: utf-8 -*-
"""Corrige apenas a regra de detecção do rótulo TSNE13 do corretor 1D.

Não abre dados, não recalcula estatísticas nem altera renderizadores ou ensaios.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import shutil
import sys
import traceback

RAIZ = Path(__file__).resolve().parents[2]
REL = Path('scripts/validacao/corrigir_reteste_1d_oof_pre01.py')
ALVO = RAIZ / REL
MARCA = 'HOTFIX_PLANEJAMENTO_TSNE13_ROTULO_V1_20261103'
HASH_ANTIGO = '1ec5e6d8d006ce4e2c696c19b528539b416a44af340b8c1bc80a423ea7bd2c87'

ANTIGO = '''            src,n=re.subn(pattern,lambda _:rep,src)
            exigir(n==1,'Marcador TSNE13 ausente ou duplicado em '+rel)
            src,n=re.subn(r"(?m)^    ax\\.set_xlabel\\('t-SNE 1 \\(coordenada do embedding; sem variância explicada\\)'\\)$",
                          "    ax.set_xlabel('t-SNE 1')",src)
            exigir(n==1,'Rotulo do eixo TSNE13 inesperado em '+rel)
'''

NOVO = '''            src,n=re.subn(pattern,lambda _:rep,src)
            exigir(n==1,'Marcador TSNE13 ausente ou duplicado em '+rel)
            # HOTFIX_PLANEJAMENTO_TSNE13_ROTULO_V1_20261103:
            # O eixo pode estar na forma longa ou ja estar abreviado por um hotfix antigo.
            inicio=src.find('def _tsne_histograma_1d(')
            fim=src.find("    nome = '13_tsne_mapa_TSNE1'",inicio)
            exigir(inicio>=0 and fim>inicio,'Funcao TSNE13 nao localizada em '+rel)
            bloco=src[inicio:fim]
            expressoes=list(re.finditer(r"(?m)^    ax\\.set_xlabel\\((?P<arg>[^\\n]*)\\)[ \\t]*$",bloco))
            exigir(len(expressoes)==1,'Esperada uma definicao do eixo TSNE13; encontradas '+str(len(expressoes)))
            linha=expressoes[0]
            valor=linha.group('arg').strip()
            curtos=("'t-SNE 1'", '\"t-SNE 1\"')
            longos=("'t-SNE 1 (", '\"t-SNE 1 (')
            exigir(valor in curtos or (valor.startswith(longos) and
                   'coordenada do embedding' in valor and
                   'sem variância explicada' in valor),
                   'Rotulo do eixo TSNE13 nao reconhecido: '+valor[:120])
            if valor not in curtos:
                bloco=bloco[:linha.start()]+"    ax.set_xlabel('t-SNE 1')"+bloco[linha.end():]
                src=src[:inicio]+bloco+src[fim:]
'''


def digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def preparar() -> tuple[bytes, bytes, bool]:
    if not ALVO.is_file():
        raise RuntimeError('Corretor original ausente: ' + str(ALVO))
    before = ALVO.read_bytes()
    normal = before.decode('utf-8-sig').replace('\r\n', '\n')
    if MARCA in normal:
        if normal.count(MARCA) != 1 or ANTIGO in normal:
            raise RuntimeError('Hotfix existente com conteúdo divergente; nenhuma alteração permitida.')
        ast.parse(normal, filename=str(ALVO))
        return before, before, False
    if digest(before) != HASH_ANTIGO:
        raise RuntimeError('Hash do corretor difere do pacote original. SHA256 encontrado=' + digest(before)
                           + '. Nenhuma alteração realizada.')
    if normal.count(ANTIGO) != 1:
        raise RuntimeError('Regra antiga do rótulo TSNE13 não aparece exatamente uma vez.')
    after = normal.replace(ANTIGO, NOVO)
    ast.parse(after, filename=str(ALVO))
    bom = b'\xef\xbb\xbf' if before.startswith(b'\xef\xbb\xbf') else b''
    newline = '\r\n' if b'\r\n' in before else '\n'
    novo = bom + after.replace('\n', newline).encode('utf-8')
    return before, novo, True


def aplicar() -> None:
    antes, depois, precisa = preparar()
    if not precisa:
        print('HOTFIX_TSNE13=JA_APLICADO', flush=True)
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_UTC')
    backup = RAIZ / 'backups' / 'hotfix_planejamento_tsne13' / stamp / REL
    backup.parent.mkdir(parents=True, exist_ok=False)
    with backup.open('xb') as file:
        file.write(antes)
    if digest(backup.read_bytes()) != digest(antes):
        raise RuntimeError('Backup não corresponde à fonte; não aplicar hotfix.')
    temp = ALVO.with_name(ALVO.name + '.tmp_hotfix_tsne13')
    if temp.exists():
        raise RuntimeError('Arquivo temporário já existe: ' + str(temp))
    try:
        with temp.open('xb') as out:
            out.write(depois)
        temp.replace(ALVO)
        if ALVO.read_bytes() != depois:
            raise RuntimeError('Escrita do hotfix falhou na verificação pós-gravação.')
    except Exception:
        if temp.exists():
            temp.unlink()
        shutil.copy2(backup, ALVO)
        raise
    print('HOTFIX_TSNE13=APLICADO',flush=True)
    print('BACKUP_CORRETOR=' + str(backup),flush=True)
    print('SHA256_CORRETOR_ANTIGO=' + digest(antes),flush=True)
    print('SHA256_CORRETOR_NOVO=' + digest(depois),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modo', choices=['planejar','aplicar','validar'], required=True)
    args = parser.parse_args()
    try:
        antes, depois, precisa = preparar()
        print('HOTFIX_TSNE13_MODO=' + args.modo,flush=True)
        print('HOTFIX_TSNE13_NECESSARIO=' + ('SIM' if precisa else 'NAO'), flush=True)
        if args.modo == 'planejar':
            print('PLANEJAMENTO_SEM_ALTERACOES=SIM',flush=True)
        elif args.modo == 'aplicar':
            aplicar()
        else:
            if precisa:
                raise RuntimeError('Hotfix ainda não aplicado.')
            print('HOTFIX_TSNE13=VALIDADO',flush=True)
        print('OOF_REEXECUTADO=NAO',flush=True)
        print('F10_RECALCULADO=NAO',flush=True)
        print('EXIT_CODE_HOTFIX_TSNE13=0',flush=True)
        return 0
    except Exception as exc:
        print('HOTFIX_TSNE13=BLOQUEADO',file=sys.stderr,flush=True)
        print('HOTFIX_MOTIVO=' + str(exc),file=sys.stderr,flush=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
