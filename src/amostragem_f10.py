# -*- coding: utf-8 -*-
"""Regra unica de amostragem do F10, configurada exclusivamente no YAML.

Nao acessa arquivos, nao usa dados do desenvolvimento e nao executa modelos.
Serve PCA 1D/2D/3D e t-SNE em qualquer dimensao; a existencia de uma
regra de amostragem nao habilita um finalizador t-SNE 2D/3D.
"""
from __future__ import annotations

import math

import numpy as np


def fracao_amostra_f10(config: dict) -> float:
    """Exige divisao.fracao_amostra_f10 explicita (sem fallback silencioso)."""
    try:
        valor = config['divisao']['fracao_amostra_f10']
    except (TypeError, KeyError) as exc:
        raise ValueError(
            'Informe divisao.fracao_amostra_f10 no YAML (ex.: 0.10).'
        ) from exc
    if type(valor) not in (float, int) or not math.isfinite(valor) or not 0 < valor <= 1:
        raise ValueError('divisao.fracao_amostra_f10 precisa ser numerica em (0, 1].')
    return float(valor)


def selecionar_f10_estratificado(folds, y, config: dict) -> np.ndarray:
    """Seleciona proporcao de CADA classe do fold reservado, deterministica.

    Mantem o minimo historico de duas observacoes por classe, sem reposicao;
    valida explicitamente que as duas classes estao presentes. A fracao vem
    SOMENTE do YAML congelado da execucao. Nao reamostra F1-F9.
    """
    fracao = fracao_amostra_f10(config)
    try:
        seed = config['experimento']['semente']
        qual_fold = config['divisao']['qual_fold_teste']
    except (TypeError, KeyError) as exc:
        raise ValueError('Semente e fold reservado devem estar no YAML.') from exc
    if type(seed) is not int or type(qual_fold) is not int or qual_fold != 10:
        raise ValueError('A amostragem F10 requer semente inteira e fold reservado 10.')
    f = np.asarray(folds)
    rotulos = np.asarray(y)
    if (f.ndim != 1 or rotulos.shape != f.shape or
            not np.issubdtype(f.dtype, np.integer) or
            not np.isin(rotulos, [0, 1]).all()):
        raise ValueError('Folds e rotulos originais invalidos para F10.')
    f10 = np.flatnonzero(f == qual_fold)
    if len(f10) == 0:
        raise ValueError('Fold 10 ausente.')
    rng = np.random.default_rng(seed)
    partes = []
    for classe in (0, 1):
        pos = f10[rotulos[f10] == classe]
        if len(pos) < 2:
            raise ValueError('F10 exige ao menos duas observacoes por classe.')
        n = max(2, round(fracao * len(pos)))
        partes.append(rng.choice(pos, size=n, replace=False))
    indices = np.sort(np.concatenate(partes)).astype(np.int64)
    if len(indices) != len(np.unique(indices)) or np.any(f[indices] != 10):
        raise RuntimeError('Amostra F10 nao pertence exclusivamente ao fold 10.')
    return indices
