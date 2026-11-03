# -*- coding: utf-8 -*-
"""V2 hotfix: corrige bug Windows openTSNE+Annoy sem refazer OOF/F10.

--planejar: nao escreve
--aplicar: copia backup do codigo e injeta guardas em dump/load; arquiva
           pasta V2 parcial sem exclui-la; nao altera o modelo OOF/PCA.
--testar: testa pickle real do Annoy e seu funcionamento apos recarga,
          usando apenas dados sinteticos e sem t-SNE/GMM/Parquet/F10.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
from hashlib import sha256
import io
import json
import os
import re
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "src" / "treinamento_final_tsne.py"
COMPAT = ROOT / "src" / "compatibilidade_opentsne_windows.py"
OUT = ROOT / "outputs" / "ensaio_tsne_gmm_20261102_134606" / "vencedor" / "teste_final"
PARCIAL = OUT / "modelo_final_f1_f9.incompleto_v2"
DESTINO = OUT / "modelo_final_f1_f9"
LOCK = ROOT / "outputs" / ".ajuste_final_tsne1d_v2_em_andamento.lock"
ANTES_DUMP = "    joblib.dump(pacote, modelo, compress=3)"
DEPOIS_DUMP = ("    from src.compatibilidade_opentsne_windows import pickle_opentsne_annoy_windows\n"
               "    with pickle_opentsne_annoy_windows():\n"
               "        joblib.dump(pacote, modelo, compress=3)")
ANTES_LOAD = "    pacote = joblib.load(modelo)"
DEPOIS_LOAD = ("    from src.compatibilidade_opentsne_windows import pickle_opentsne_annoy_windows\n"
               "    with pickle_opentsne_annoy_windows():\n"
               "        pacote = joblib.load(modelo)")


def check(cond, message):
    if not cond:
        raise RuntimeError(message)


def sha(data):
    return sha256(data).hexdigest()


def status():
    check(COMPAT.is_file(), "Modulo src/compatibilidade_opentsne_windows.py nao encontrado.")
    check(TARGET.is_file(), "Arquivo src/treinamento_final_tsne.py nao encontrado.")
    check(OUT.is_dir(), "Pasta teste_final do experimento TSNE nao encontrada.")
    content = TARGET.read_text(encoding="utf-8")
    original = (len(re.findall(r"(?m)^    joblib\.dump\(pacote, modelo, compress=3\)$", content)) == 1
                and len(re.findall(r"(?m)^    pacote = joblib\.load\(modelo\)$", content)) == 1)
    installed = content.count(DEPOIS_DUMP) == 1 and content.count(DEPOIS_LOAD) == 1
    check(original ^ installed, "Pontos de injecao inesperados; nao modificar codigo. Verifique possivel edicao manual.")
    print("=== HOTFIX V2 TSNE ANNoy / WINDOWS ===", flush=True)
    print("SISTEMA=" + os.name, flush=True)
    print("CODIGO_V2_COMPAT=" + ("JA_INSTALADO" if installed else "AINDA_NAO_INSTALADO"), flush=True)
    print("PARCIAL_EXISTE=" + str(PARCIAL.exists()).upper(), flush=True)
    print("DESTINO_FINAL_EXISTE=" + str(DESTINO.exists()).upper(), flush=True)
    print("LOCK_V2_EXISTE=" + str(LOCK.exists()).upper(), flush=True)
    print("OOF_REEXECUTADO=NAO", flush=True)
    print("F10_ACESSADO=NAO", flush=True)
    return content, installed


def aplicar():
    content, installed = status()
    check(not LOCK.exists(), "Lock V2 existente; confirme se ha processo ativo antes de prosseguir.")
    if installed:
        print("HOTFIX_STATUS=JA_INSTALADO", flush=True)
    else:
        # Nunca alterar src original se nao for possivel criar backup.
        backup_root = ROOT / "outputs" / ("backup_hotfix_opentsne_v2_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        backup_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(TARGET, backup_root / TARGET.name)
        modified = content.replace(ANTES_DUMP, DEPOIS_DUMP).replace(ANTES_LOAD, DEPOIS_LOAD)
        compile(modified, str(TARGET), "exec")
        # Escrita atomica; se algo falhar, backup continua disponivel.
        temp = TARGET.with_name(TARGET.name + ".hotfix_win_tmp")
        try:
            temp.write_text(modified, encoding="utf-8")
            os.replace(temp, TARGET)
        finally:
            temp.unlink(missing_ok=True)
        print("BACKUP_CODIGO=" + str(backup_root / TARGET.name), flush=True)
        print("HOTFIX_STATUS=INSTALADO", flush=True)

    if PARCIAL.exists():
        check(not DESTINO.exists(), "Destino V2 final existe e parcial tambem; preservados sem alterar.")
        check(PARCIAL.is_dir(), "Parcial V2 nao e diretorio.")
        arquivado = OUT / ("modelo_final_f1_f9.incompleto_v2_ARQUIVO_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        PARCIAL.rename(arquivado)
        print("ARQUIVO_PARCIAL_PRESERVADO=" + str(arquivado), flush=True)
    print("PRONTO_TESTE_ANNoy=SIM", flush=True)
    return 0


def testar():
    _, installed = status()
    check(installed, "Instale o hotfix antes do teste.")
    from src.compatibilidade_opentsne_windows import pickle_opentsne_annoy_windows
    from openTSNE.nearest_neighbors import Annoy

    rng = np.random.default_rng(20261102)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    knn = Annoy(X, k=5, metric="euclidean", n_jobs=1, random_state=42)
    knn.build()
    idx_before, dis_before = knn.query(X[:7], 4)
    stream = io.BytesIO()
    with pickle_opentsne_annoy_windows():
        joblib.dump(knn, stream, compress=3)
    stream.seek(0)
    with pickle_opentsne_annoy_windows():
        loaded = joblib.load(stream)
    idx_after, dis_after = loaded.query(X[:7], 4)
    check(np.array_equal(idx_before, idx_after) and np.allclose(dis_before, dis_after, rtol=1e-6, atol=1e-6),
          "Annoy carregado mudou indices/distancias: bloquear refit grande.")
    del loaded, knn
    gc.collect()
    print("HOTFIX_TESTE_PICKLE_ANNoy=APROVADO", flush=True)
    print("HOTFIX_TESTE_RECARREGAMENTO_INDICE=APROVADO", flush=True)
    print("TREINO_F1_F9_EXECUTADO=NAO", flush=True)
    print("F10_ACESSADO=NAO", flush=True)
    return 0


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    ops = arg.add_mutually_exclusive_group(required=True)
    ops.add_argument("--planejar", action="store_true")
    ops.add_argument("--aplicar", action="store_true")
    ops.add_argument("--testar", action="store_true")
    opts = arg.parse_args()
    if opts.planejar:
        status()
        print("HOTFIX_STATUS=PLANEJADO_SEM_ESCRITA", flush=True)
        return 0
    if opts.aplicar:
        return aplicar()
    return testar()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as ex:
        print("HOTFIX_STATUS=BLOQUEADO", file=sys.stderr)
        print("HOTFIX_MOTIVO=" + str(ex), file=sys.stderr)
        raise
