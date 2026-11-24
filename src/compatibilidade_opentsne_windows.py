# -*- coding: utf-8 -*-
"""Compatibilidade pontual do pickle openTSNE/Annoy no Windows.

A versao 1.0.4 do openTSNE serializa Annoy com TemporaryDirectory, grava
`tmp.ann` e tenta remover o diretorio ainda com o arquivo mapeado no Windows.
Tambem faz isso ao desserializar, para preservar a busca OOS.

Este contexto NAO altera o algoritmo, o indice nem os parametros t-SNE.
Somente tolera a limpeza bloqueada do arquivo temporario, sem mudar a
representacao serializada. Em Windows, arquivos temporarios tcc_annoy_
podem permanecer ate limpeza posterior COM O PROCESSO ENCERRADO.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import os
import tempfile
from threading import RLock

_LOCK = RLock()


@contextmanager
def _temporario_windows_tolerante():
    original = tempfile.TemporaryDirectory

    def criar(*args, **kwargs):
        kwargs["ignore_cleanup_errors"] = True
        kwargs.setdefault("prefix", "tcc_annoy_")
        return original(*args, **kwargs)

    tempfile.TemporaryDirectory = criar
    try:
        yield
    finally:
        tempfile.TemporaryDirectory = original


@contextmanager
def pickle_opentsne_annoy_windows():
    """Corrige somente Annoy.__getstate__/__setstate__ durante dump/load.

    Nao altera o modulo site-packages nem o modelo/embedding treinado.
    Use esta protecao nos DOIS lados: joblib.dump e joblib.load.
    """
    if os.name != "nt":
        yield
        return

    # Evitar que dois salvamentos simultaneos substituam os metodos.
    with _LOCK:
        from openTSNE.nearest_neighbors import Annoy

        original_getstate = Annoy.__getstate__
        original_setstate = Annoy.__setstate__

        @wraps(original_getstate)
        def getstate_windows(self):
            with _temporario_windows_tolerante():
                return original_getstate(self)

        @wraps(original_setstate)
        def setstate_windows(self, state):
            with _temporario_windows_tolerante():
                return original_setstate(self, state)

        Annoy.__getstate__ = getstate_windows
        Annoy.__setstate__ = setstate_windows
        try:
            yield
        finally:
            Annoy.__getstate__ = original_getstate
            Annoy.__setstate__ = original_setstate
