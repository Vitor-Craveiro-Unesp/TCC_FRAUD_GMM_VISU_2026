"""Testes do Passo 03.03: indices do treino para GMM com NORMAL/RUS."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.amostragem import selecionar_indices_gmm


@pytest.fixture
def config():
    with (RAIZ / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").open(
        encoding="utf-8-sig"
    ) as arq:
        cfg = yaml.safe_load(arq)
    cfg["amostragem"]["modo"] = "RUS"
    cfg["amostragem"]["razao_negativos_positivos"] = 2
    return cfg


@pytest.fixture
def y():
    # 3 positivas e 12 negativas: RUS 2:1 => 6 negativas + 3 positivas.
    return np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0])


def test_normal_preserva_todas_as_linhas(config, y):
    config["amostragem"]["modo"] = "NORMAL"
    idx = selecionar_indices_gmm(y, config, fold_validacao=9)
    np.testing.assert_array_equal(idx, np.arange(len(y)))


def test_rus_preserva_todas_fraudes_e_respeita_razao(config, y):
    idx = selecionar_indices_gmm(y, config, fold_validacao=9)
    assert len(idx) == 9
    assert np.count_nonzero(y[idx] == 1) == 3
    assert np.count_nonzero(y[idx] == 0) == 6
    np.testing.assert_array_equal(np.flatnonzero(y == 1), idx[y[idx] == 1])
    assert np.all(np.diff(idx) > 0)


def test_rus_nao_faz_oversampling(config):
    y = np.array([1, 0, 1, 0, 0])
    idx = selecionar_indices_gmm(y, config)
    np.testing.assert_array_equal(idx, np.arange(len(y)))


def test_reprodutibilidade_por_fold(config, y):
    a = selecionar_indices_gmm(y, config, fold_validacao=9)
    b = selecionar_indices_gmm(y, config, fold_validacao=9)
    np.testing.assert_array_equal(a, b)


def test_outra_semente_produz_outra_amostra(config):
    y = np.array([0] * 100 + [1] * 5)
    a = selecionar_indices_gmm(y, config, fold_validacao=9)
    config["experimento"]["semente"] += 1
    b = selecionar_indices_gmm(y, config, fold_validacao=9)
    assert not np.array_equal(a, b)


def test_nao_modifica_y_ou_dataframe_original(config, y):
    y_original = y.copy()
    x = pd.DataFrame({"numero": np.arange(len(y))}, index=np.arange(len(y)) * 3)
    x_original = x.copy(deep=True)
    idx = selecionar_indices_gmm(y, config)
    assert len(x.iloc[idx]) == 9
    np.testing.assert_array_equal(y, y_original)
    pd.testing.assert_frame_equal(x, x_original)


def test_normal_nao_multiplica_grid_de_razao(config, y):
    config["amostragem"]["modo"] = ["NORMAL", "RUS"]
    config["amostragem"]["razao_negativos_positivos"] = [2, 3]
    idx = selecionar_indices_gmm(y, config, modo="NORMAL")
    np.testing.assert_array_equal(idx, np.arange(len(y)))


def test_grid_requer_escolha_explicita(config, y):
    config["amostragem"]["modo"] = ["NORMAL", "RUS"]
    with pytest.raises(ValueError, match="escolha um valor"):
        selecionar_indices_gmm(y, config)
    config["amostragem"]["modo"] = "RUS"
    config["amostragem"]["razao_negativos_positivos"] = [2, 3]
    with pytest.raises(ValueError, match="escolha um valor"):
        selecionar_indices_gmm(y, config)
    idx = selecionar_indices_gmm(y, config, razao=2)
    assert len(idx) == 9


def test_escolha_fora_do_yaml_falha(config, y):
    with pytest.raises(ValueError, match="nao consta"):
        selecionar_indices_gmm(y, config, modo="NORMAL")
    with pytest.raises(ValueError, match="nao consta"):
        selecionar_indices_gmm(y, config, razao=1)


@pytest.mark.parametrize("y_invalido", [
    [0, 0, 0], [1, 1], [0, 1, np.nan], [0, 1, 2], [[0], [1]], [],
])
def test_rotulos_invalidos_falham(config, y_invalido):
    with pytest.raises(ValueError):
        selecionar_indices_gmm(y_invalido, config)


@pytest.mark.parametrize("proporcao", [0, -1, 1.5, True])
def test_razao_invalida_falha(config, y, proporcao):
    config["amostragem"]["razao_negativos_positivos"] = proporcao
    with pytest.raises(ValueError, match="inteiro"):
        selecionar_indices_gmm(y, config)


def test_amostragem_somente_no_treino(config, y):
    config["amostragem"]["aplicar_somente_treino"] = False
    with pytest.raises(ValueError, match="somente ao treino"):
        selecionar_indices_gmm(y, config)


def test_normal_recusa_razao_desnecessaria(config, y):
    config["amostragem"]["modo"] = "NORMAL"
    with pytest.raises(ValueError, match="nao informe razao"):
        selecionar_indices_gmm(y, config, razao=2)


def test_marcacao_fold_nao_altera_y(config, y):
    antes = y.copy()
    selecionar_indices_gmm(y, config, fold_validacao=9)
    np.testing.assert_array_equal(y, antes)


def test_fold_invalido_falha(config, y):
    with pytest.raises(ValueError, match="fold_validacao"):
        selecionar_indices_gmm(y, config, fold_validacao=-1)
