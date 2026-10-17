"""Passo 03.03: verifica NORMAL/RUS no treino ORIGINAL F1-F8 da base real.

Uso: python .\tests\verificar_amostragem_base_real.py
Nao altera YAML, Parquet, folds nem arquivos do projeto; nao ajusta GMM.
Utiliza somente os rotulos e as posicoes dos folds (pouca memoria).
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.amostragem import selecionar_indices_gmm
from src.folds import carregar_alvo, hash_arquivo


def carregar_divisao(config, total_linhas):
    arquivo_base = RAIZ / config["dados"]["arquivo"]
    hash_base = hash_arquivo(arquivo_base)
    compativeis = []
    for caminho in (RAIZ / "data" / "splits").glob("*/manifesto_folds.json"):
        manifesto = json.loads(caminho.read_text(encoding="utf-8"))
        p = manifesto["parametros"]
        if (
            p["caminho_relativo"] == config["dados"]["arquivo"]
            and p["coluna_alvo"] == config["dados"]["coluna_alvo"]
            and p["classe_positiva"] == config["dados"]["classe_positiva"]
            and p["total_linhas"] == total_linhas
            and p["total_folds"] == 10
            and p["fold_teste"] == 10
            and p["semente"] == config["experimento"]["semente"]
            and p["sha256_parquet"] == hash_base
        ):
            caminho_npy = caminho.parent / "folds.npy"
            if not caminho_npy.is_file() or (
                hash_arquivo(caminho_npy) != manifesto["sha256_folds"]
            ):
                raise ValueError("Divisao encontrada com folds.npy ausente ou alterado.")
            compativeis.append(caminho_npy)
    if len(compativeis) != 1:
        raise ValueError(f"Esperada uma divisao compativel; encontradas {len(compativeis)}.")
    folds = np.load(compativeis[0], allow_pickle=False)
    if len(folds) != total_linhas or not np.isin(folds, np.arange(1, 11)).all():
        raise ValueError("Atribuicoes de folds invalidas.")
    return folds


def main():
    with (RAIZ / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").open(
        encoding="utf-8-sig"
    ) as arq:
        config = yaml.safe_load(arq)

    y = carregar_alvo(
        RAIZ / config["dados"]["arquivo"], config["dados"]["coluna_alvo"]
    )
    folds = carregar_divisao(config, len(y))
    # Uma rodada representativa: treina somente em F1-F8, valida em F9.
    posicoes_treino = np.flatnonzero((folds >= 1) & (folds <= 8))
    y_treino = y[posicoes_treino]
    n_pos = int(np.count_nonzero(y_treino == 1))
    n_neg = int(np.count_nonzero(y_treino == 0))
    assert n_pos > 0 and n_neg > 0

    # O YAML original nao e alterado; verificamos os dois modos separadamente.
    cfg_normal = deepcopy(config)
    cfg_normal["amostragem"]["modo"] = "NORMAL"
    idx_normal = selecionar_indices_gmm(y_treino, cfg_normal, fold_validacao=9)
    np.testing.assert_array_equal(idx_normal, np.arange(len(y_treino)))

    cfg_rus = deepcopy(config)
    cfg_rus["amostragem"]["modo"] = "RUS"
    lista_razoes = cfg_rus["amostragem"]["razao_negativos_positivos"]
    if isinstance(lista_razoes, list):
        if not lista_razoes:
            raise ValueError("Grade de razoes RUS vazia.")
        razao = lista_razoes[0]  # Apenas verificacao; grade real sera varrida depois.
        selecionados = selecionar_indices_gmm(
            y_treino, cfg_rus, razao=razao, fold_validacao=9
        )
        repetidos = selecionar_indices_gmm(
            y_treino, cfg_rus, razao=razao, fold_validacao=9
        )
    else:
        razao = lista_razoes
        selecionados = selecionar_indices_gmm(y_treino, cfg_rus, fold_validacao=9)
        repetidos = selecionar_indices_gmm(y_treino, cfg_rus, fold_validacao=9)

    np.testing.assert_array_equal(selecionados, repetidos)
    assert np.all(np.diff(selecionados) > 0)
    # A verificacao do fold e feita em POSICOES ABSOLUTAS, nao IDs.
    assert np.isin(folds[posicoes_treino[selecionados]], np.arange(1, 9)).all()
    assert not np.isin(folds[posicoes_treino[selecionados]], [9, 10]).any()
    assert int(np.count_nonzero(y_treino[selecionados] == 1)) == n_pos
    assert int(np.count_nonzero(y_treino[selecionados] == 0)) == min(
        n_neg, razao * n_pos
    )

    print("[OK] Base real e indices dos folds conferidos com o manifesto.")
    print(f"[OK] Treino original F1-F8: {len(y_treino):,} linhas.")
    print(f"[OK] Classes originais: {n_neg:,} legitimas; {n_pos:,} fraudes.")
    print(f"[OK] NORMAL: {len(idx_normal):,} linhas (sem reducao).")
    print(f"[OK] RUS {razao}:1: {len(selecionados):,} linhas para ajuste do GMM.")
    print(f"[OK] Fraudes preservadas: {n_pos:,} de {n_pos:,}.")
    print("[OK] F9 e F10 nao incluidos no treino selecionado.")
    print("[OK] Resultado reprodutivel; arquivos de entrada preservados.")
    print("[NOTA] RUS verificado sem alterar o modo ativo no YAML.")
    print("[NOTA] RH_k sera estimado sobre TODO o treino original no passo do GMM.")
    print("PASSO 03.03 - VERIFICACAO COM BASE REAL APROVADA")


if __name__ == "__main__":
    main()
