"""Patch minimo e verificavel de fonte para futuras galerias 2D.

Modos: planejar, aplicar, validar. Nunca reexecuta OOF/F10.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

SINAL = "# TCC_LAYOUT_2D_LIMPO_PERMANENTE_V1"
ALVO = Path("src/visualizacao_oof.py")
AUX = Path("src/layout_limpo_2d.py")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def exigir(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def carregar(p):
    b = p.read_bytes()
    enc = "utf-8-sig" if b.startswith(b"\xef\xbb\xbf") else "utf-8"
    s = b.decode(enc)
    return b, s, enc


def preparar(texto):
    ast.parse(texto)
    arv = ast.parse(texto)
    funcs = [f for f in arv.body if isinstance(f, ast.FunctionDef) and f.name == "_salvar"]
    exigir(len(funcs) == 1, "Esperada uma unica funcao _salvar na fonte atual.")
    f = funcs[0]
    exigir(len(f.args.args) >= 2 and f.args.args[0].arg == "fig" and f.args.args[1].arg == "destino", "Assinatura _salvar divergente: inspecao manual necessaria.")
    exigir("fig.savefig" in ast.get_source_segment(texto, f), "O salvamento figura mudou; patch automatico bloqueado.")
    if SINAL in texto:
        exigir(texto.count(SINAL) == 1, "Marcador duplicado; nenhuma alteracao.")
        exigir("from src.layout_limpo_2d import limpar" in ast.get_source_segment(texto, f), "Marcador presente fora da funcao _salvar.")
        return texto, False

    linhas = texto.splitlines(keepends=True)
    index = f.body[0].lineno - 1
    eol = "\r\n" if "\r\n" in texto else "\n"
    indent = " " * (f.col_offset + 4)
    inj = (
        indent + SINAL + eol
        + indent + "from src.layout_limpo_2d import limpar as _limpar_layout_2d" + eol
        + indent + "_limpar_layout_2d(fig, destino)" + eol
    )
    novo = "".join(linhas[:index]) + inj + "".join(linhas[index:])
    ast.parse(novo)
    compile(novo, str(ALVO), "exec")
    exigir(novo.count(SINAL) == 1, "Falha ao inserir marcador unico.")
    return novo, True


def atomic_write(path, conteudo):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = tempfile.NamedTemporaryFile(delete=False, prefix=".tcc2d_", suffix=".tmp", dir=path.parent)
    tmp = Path(f.name)
    try:
        with f:
            f.write(conteudo)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo", required=True, choices=["planejar", "aplicar", "validar"])
    parser.add_argument("--raiz", type=Path, default=Path.cwd())
    args = parser.parse_args()
    raiz = args.raiz.resolve()
    p = raiz / ALVO
    dest = raiz / AUX
    exigir(p.is_file(), "Fonte src/visualizacao_oof.py nao encontrada.")
    antigo, texto, enc = carregar(p)
    novo_texto, alterar = preparar(texto)
    novo = novo_texto.encode(enc)
    base = Path(__file__).resolve().parents[2]
    aux_src = base / "src_layout_limpo_2d.py"
    exigir(aux_src.is_file(), "Auxiliar de instalacao nao encontrado no pacote.")
    aux_bytes = aux_src.read_bytes()
    ast.parse(aux_bytes.decode("utf-8"))
    compile(aux_bytes.decode("utf-8"), str(dest), "exec")
    if dest.exists():
        exigir(dest.read_bytes() == aux_bytes, "Modulo auxiliar existente difere do pacote; bloqueado sem sobrescrita.")
    print(f"MODO={args.modo}")
    print(f"FONTE_ATUAL_SHA256={sha(antigo)}")
    print(f"ALTERACAO_NECESSARIA={'SIM' if alterar or not dest.exists() else 'NAO'}")

    if args.modo == "planejar":
        print("PLANEJAMENTO=APROVADO_SEM_ALTERACOES")
        print("OOF_EXECUTADO=NAO")
        print("F10_EXECUTADO=NAO")
        return
    if args.modo == "validar":
        exigir(not alterar, "Hook ainda nao instalado; aplique a correcao primeiro.")
        exigir(dest.exists() and dest.read_bytes() == aux_bytes, "Modulo auxiliar ausente ou divergente.")
        print("LAYOUT_2D_PERMANENTE=VALIDADO")
        return
    if not alterar and dest.exists():
        print("LAYOUT_2D_PERMANENTE=JA_INSTALADO")
        return
    exigir(alterar, "Hook instalado, mas modulo auxiliar nao esta presente. Bloqueado para revisao.")
    backup = raiz / "backups" / "layout_2d_futuros" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    backup_origem = backup / ALVO
    backup_origem.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(p, backup_origem)
    try:
        atomic_write(dest, aux_bytes)
        atomic_write(p, novo)
        _, ver, _ = carregar(p)
        ver_p, pendente = preparar(ver)
        exigir(not pendente and ver_p == ver, "Validacao pos-escrita falhou.")
        exigir(dest.read_bytes() == aux_bytes, "Auxiliar divergiu apos escrita.")
    except BaseException:
        atomic_write(p, antigo)
        dest.unlink(missing_ok=True)
        raise
    print(f"BACKUP_FONTE={backup_origem}")
    print(f"FONTE_NOVA_SHA256={sha(novo)}")
    print("LAYOUT_2D_PERMANENTE=APLICADO")
    print("OOF_EXECUTADO=NAO")
    print("F10_EXECUTADO=NAO")


if __name__ == "__main__":
    try:
        main()
        print("EXIT_CODE_LAYOUT_2D=0")
    except Exception as exc:
        print("LAYOUT_2D_STATUS=BLOQUEADO", file=sys.stderr)
        print("MOTIVO=" + str(exc), file=sys.stderr)
        print("EXIT_CODE_LAYOUT_2D=1", file=sys.stderr)
        raise SystemExit(1)
