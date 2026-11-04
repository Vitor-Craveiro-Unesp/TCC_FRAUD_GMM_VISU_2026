# -*- coding: utf-8 -*-
"""Hotfix V7: permitir avaliacao F10 real na legenda do GMM08 2D.

Conserva a integridade declarada pelo instalador V2; nao treina nem le F10.
Guarda os PNGs temporarios da V7 interrompida como backup, sem os apagar.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
GEO_REL = 'src/visualizacao_geometrica.py'
INST_REL = 'scripts/validacao/instalar_v2_2d.py'
GEO = ROOT / GEO_REL
INST = ROOT / INST_REL
OLD_HASH = 'e215974749eee6b75adcd55ac91011f67e167031bf3cad6d4a008c2e0e758d41'
MARCADOR = 'HOTFIX_V7_GMM08_ORIGEM_F10_20261103'
ENSAIO = 'ensaio_tsne_gmm_20261103_135207'
TESTE = ROOT / 'outputs' / ENSAIO / 'vencedor' / 'teste_final'
LOCK = ROOT / 'outputs' / '.ensaio_real_3pct_2d_tsne.lock'
BACKUPS = ROOT / 'backups' / 'hotfix_v7_gmm08_f10'
STAGE_RE = re.compile(r'^\.graficos_f10_3pct_incompletos_[0-9a-f]{12}$')


def exigir(ok, mensagem):
    if not ok:
        raise RuntimeError(mensagem)


def sha(b):
    return sha256(b).hexdigest()


def sha_path(p):
    return sha(p.read_bytes())


def _eol(texto):
    if '\r\r\n' in texto:
        return '\r\r\n'
    if '\r\n' in texto:
        return '\r\n'
    return '\n'


def patch_geo(bruto):
    """Duas mudanças locais e explícitas, sem relabel de F10 como OOF."""
    bom = bruto.startswith(b'\xef\xbb\xbf')
    s = bruto.decode('utf-8-sig')
    exigir(MARCADOR not in s, 'Geometria parece ja alterada; nao repatchear.')
    eol = _eol(s)
    old = "_exigir(origem_mcc in ('oof', 'simulado'),"
    new = "_exigir(origem_mcc in ('oof', 'simulado', 'f10_real_3pct'),"
    exigir(s.count(old) == 1, 'Ancora unica da origem GMM08 nao encontrada.')
    s = s.replace(old, new, 1)
    old2 = "    else:" + eol + "        nome_mcc = 'MCC (origem não identificada)'"
    new2 = ("    elif origem_mcc == 'f10_real_3pct':" + eol
            + "        # " + MARCADOR + eol
            + "        nome_mcc = 'MCC F10'" + eol
            + "        nota_mcc = 'MCC calculado no F10 com tau* congelado em OOF F1-F9'" + eol
            + old2)
    exigir(s.count(old2) == 1, 'Ancora unica de rotulagem GMM08 nao encontrada.')
    s = s.replace(old2, new2, 1)
    ast.parse(s, filename=GEO_REL)
    return (b'\xef\xbb\xbf' if bom else b'') + s.encode('utf-8')


def patch_inst(bruto, novo_hash):
    s = bruto.decode('utf-8-sig')
    exigir(s.count("'src/visualizacao_geometrica.py': '" + OLD_HASH + "'") == 1,
           'Instalador V2 nao contem o SHA anterior exato da geometria.')
    s = s.replace("'src/visualizacao_geometrica.py': '" + OLD_HASH + "'",
                  "'src/visualizacao_geometrica.py': '" + novo_hash + "'", 1)
    ast.parse(s, filename=INST_REL)
    return (b'\xef\xbb\xbf' if bruto.startswith(b'\xef\xbb\xbf') else b'') + s.encode('utf-8')


def conferir_sem_escrever():
    exigir(GEO.is_file() and INST.is_file(), 'Faltam os arquivos originais da V2.')
    exigir(TESTE.is_dir(), 'Pasta teste_final do OOF aprovado nao existe.')
    exigir(not LOCK.exists(), 'Lock TSNE2D presente. Nao alterar durante uma execucao.')
    atual = GEO.read_bytes()
    inst = INST.read_bytes()
    h = sha(atual)
    s_inst = inst.decode('utf-8-sig')
    s_geo = atual.decode('utf-8-sig')
    if h == OLD_HASH and MARCADOR not in s_geo:
        estado = 'PENDENTE'
        geo_novo = patch_geo(atual)
        inst_novo = patch_inst(inst, sha(geo_novo))
    elif MARCADOR in s_geo:
        # Nao aceitar simples presenca de marcador: conferir a fonte registrada.
        exigir('f10_real_3pct' in s_geo and "nome_mcc = 'MCC F10'" in s_geo,
               'Geometria F10 incompleta.')
        exigir("'src/visualizacao_geometrica.py': '" + h + "'" in s_inst,
               'SHA no instalador V2 nao corresponde ao arquivo corrigido.')
        ast.parse(s_geo, filename=GEO_REL)
        ast.parse(s_inst, filename=INST_REL)
        estado = 'APLICADO'
        geo_novo = atual
        inst_novo = inst
    else:
        raise RuntimeError('Fonte geometrica mudou desde V2; nao sobrescrever. SHA=' + h)
    presentes = {p.name for p in TESTE.iterdir()}
    permitidos = {'modelo_final_f1_f9', 'avaliacao_f10_3pct',
                  'graficos_f10_3pct', 'galeria_3pct'}
    temporarios = [p for p in TESTE.iterdir() if STAGE_RE.fullmatch(p.name)]
    inesperados = presentes - permitidos - {p.name for p in temporarios}
    exigir(not inesperados, 'Arquivos inesperados em teste_final: ' + repr(sorted(inesperados)))
    exigir((TESTE/'modelo_final_f1_f9'/'modelo_final_f1_f9.joblib').is_file() and
           (TESTE/'avaliacao_f10_3pct'/'metricas_f10.json').is_file(),
           'Modelo V5 ou avaliacao F10 V6 ausente; nao prosseguir.')
    for p in temporarios:
        exigir(p.is_dir() and not p.is_symlink(), 'Pasta temporaria nao e diretorio normal: ' + str(p))
        exigir(len(list(p.rglob('*'))) < 100,
               'Temporario possui quantidade inesperada de arquivos: ' + str(p))
    exigir(not (TESTE/'graficos_f10_3pct').exists() or
           (TESTE/'graficos_f10_3pct').is_dir(),
           'Destino final de graficos existe como arquivo.')
    exigir(not (TESTE/'galeria_3pct').exists() or
           (TESTE/'galeria_3pct').is_dir(),
           'Destino galeria existe como arquivo.')
    return estado, geo_novo, inst_novo, temporarios


def atomic_write(p, conteudo):
    tmp = p.with_name('.' + p.name + '.hotfix_' + uuid.uuid4().hex)
    with tmp.open('xb') as f:
        f.write(conteudo)
    try:
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def relatar(estado, temporarios):
    print('HOTFIX_V7_GMM08_ESTADO=' + estado)
    print('GMM08_ORIGEM_ANTIGA=oof|simulado')
    print('GMM08_ORIGEM_NOVA=oof|simulado|f10_real_3pct')
    print('PARCIAIS_ENCONTRADOS=' + str(len(temporarios)))
    for p in temporarios:
        print('PARCIAL=' + str(p))
        print('PARCIAL_ITENS=' + ', '.join(sorted(x.name for x in p.iterdir())))
    print('OOF_REEXECUTADO=NAO')
    print('MODELO_REAJUSTADO=NAO')
    print('F10_INFERENCIA_REPETIDA=NAO')


def planejar():
    estado, _, _, temporarios = conferir_sem_escrever()
    relatar(estado, temporarios)
    print('PLANEJAMENTO_SEM_MODIFICACOES=SIM')


def aplicar():
    estado, geo_novo, inst_novo, temporarios = conferir_sem_escrever()
    relatar(estado, temporarios)
    if estado == 'APLICADO' and not temporarios:
        print('HOTFIX_V7=JA_APLICADO_SEM_ALTERACOES')
        return
    bdir = BACKUPS / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC_') + uuid.uuid4().hex[:8])
    exigir(not bdir.exists(), 'Pasta backup ja existe.')
    bdir.mkdir(parents=True)
    moved = []
    orig_geo, orig_inst = GEO.read_bytes(), INST.read_bytes()
    try:
        for rel, data in ((GEO_REL, orig_geo), (INST_REL, orig_inst)):
            p = bdir/rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        # Guardar metadados do backup antes de alterar os arquivos oficiais.
        (bdir/'manifesto_backup.json').write_text(json.dumps({
            'experimento': ENSAIO,
            'sha256_geometria_original': sha(orig_geo),
            'sha256_instalador_original': sha(orig_inst),
            'temporarios': [p.name for p in temporarios],
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if estado == 'PENDENTE':
            atomic_write(GEO, geo_novo)
            atomic_write(INST, inst_novo)
        for p in temporarios:
            destino = bdir/'parciais_v7'/p.name
            destino.parent.mkdir(parents=True, exist_ok=True)
            exigir(not destino.exists(), 'Destino de backup do temporario ja existe.')
            p.rename(destino)
            moved.append((destino, p))
        from scripts.validacao import instalar_v2_2d as instalador
        instalador.testar_fonte_modificada(GEO_REL)
        from scripts.validacao.homologar_v3_2d import inspecionar_v2
        # A integridade de fontes e YAMLs da V3 permanecem preservadas.
        inspecionar_v2()
        print('BACKUP_V7=' + str(bdir))
        print('PARCIAIS_PRESERVADOS_NO_BACKUP=' + str(len(moved)))
        print('HOTFIX_V7=APLICADO')
    except BaseException:
        for dest, orig in reversed(moved):
            if dest.exists() and not orig.exists():
                dest.rename(orig)
        if estado == 'PENDENTE':
            atomic_write(GEO, orig_geo)
            atomic_write(INST, orig_inst)
        raise


def validar():
    estado, _, _, temporarios = conferir_sem_escrever()
    exigir(estado == 'APLICADO', 'Hotfix ainda nao aplicado.')
    exigir(not temporarios, 'Temporarios ainda nao guardados; execute --aplicar.')
    from scripts.validacao import instalar_v2_2d as instalador
    instalador.testar_fonte_modificada(GEO_REL)
    from scripts.validacao.homologar_v3_2d import inspecionar_v2
    inspecionar_v2()
    # Somente leitura. V7 --executar continuara gerando os PNGs do F10.
    print('HOTFIX_V7_STATUS=VALIDADO')
    print('F10_SCORES_INALTERADOS=SIM')
    print('OOF_REEXECUTADO=NAO')
    print('MODELO_REAJUSTADO=NAO')
    print('GERACAO_14_PNG_AINDA_PENDENTE=SIM')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    grupo = p.add_mutually_exclusive_group(required=True)
    grupo.add_argument('--planejar', action='store_true')
    grupo.add_argument('--aplicar', action='store_true')
    grupo.add_argument('--validar', action='store_true')
    args = p.parse_args()
    try:
        if args.planejar:
            planejar()
        elif args.aplicar:
            aplicar()
        else:
            validar()
        print('EXIT_CODE_HOTFIX_V7=0')
        return 0
    except Exception as exc:
        print('HOTFIX_V7_STATUS=BLOQUEADO_SEM_TREINAR', file=sys.stderr)
        print('HOTFIX_V7_MOTIVO=' + str(exc), file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
