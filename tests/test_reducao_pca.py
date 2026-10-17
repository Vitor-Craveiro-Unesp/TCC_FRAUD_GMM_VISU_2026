"""Passo 04.01: testes unitarios para PCA sem vazamento e grade YAML."""
from pathlib import Path
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reducao import RedutorPCA, grade_pca


@pytest.fixture
def config():
    with (ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").open(encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


@pytest.fixture
def dados():
    rng = np.random.default_rng(42)
    treino = pd.DataFrame(rng.normal(size=(64, 5)).astype(np.float32), columns=list("abcde"))
    validacao = pd.DataFrame(rng.normal(size=(12, 5)).astype(np.float32), columns=list("abcde"), index=range(100, 112))
    return treino, validacao


def parametros(config):
    return list(grade_pca(config))[0]


def test_grade_pca_respeita_yaml_sem_misturar_tsne(config):
    grade = list(grade_pca(config))
    assert [p["n_components"] for p in grade] == [2, 3]
    assert all(p["whiten"] is False for p in grade)
    assert all(p["svd_solver"] == "auto" for p in grade)


def test_grade_com_parametros_escalares_e_vetores(config):
    config["pca"]["n_components"] = 2
    config["pca"]["whiten"] = [False, True]
    assert len(list(grade_pca(config))) == 2


def test_grade_vazia_falha(config):
    config["pca"]["n_components"] = []
    with pytest.raises(ValueError, match="vazia"):
        list(grade_pca(config))


def test_instanciacao_sem_escolher_grade_falha(config):
    with pytest.raises(ValueError, match="varios hiperparametros"):
        RedutorPCA(config)


def test_pca_ajusta_e_transforma_sem_refit(config, dados):
    treino, validacao = dados
    modelo = RedutorPCA(config, parametros(config)).fit(treino)
    medias = modelo.modelo_.mean_.copy()
    comps = modelo.modelo_.components_.copy()
    val_1 = modelo.transform(validacao)
    alterada = validacao.copy()
    alterada.iloc[0, 0] += 1000
    modelo.transform(alterada)
    np.testing.assert_array_equal(modelo.modelo_.mean_, medias)
    np.testing.assert_array_equal(modelo.modelo_.components_, comps)
    assert val_1.shape == (12, 2)
    assert val_1.columns.tolist() == ["PC1", "PC2"]
    assert val_1.index.equals(validacao.index)
    assert np.isfinite(val_1.to_numpy()).all()


def test_pca_3_componentes(config, dados):
    treino, val = dados
    p = parametros(config)
    p["n_components"] = 3
    mod = RedutorPCA(config, p)
    assert mod.fit_transform(treino).shape == (64, 3)
    assert mod.transform(val).shape == (12, 3)
    razoes = mod.modelo_.explained_variance_ratio_
    assert len(razoes) == 3
    assert np.all((razoes >= 0) & (razoes <= 1))
    assert 0 < razoes.sum() <= 1 + 1e-6


def test_entrada_nao_modificada(config, dados):
    treino, validacao = dados
    original_treino = treino.copy(deep=True)
    original_validacao = validacao.copy(deep=True)
    mod = RedutorPCA(config, parametros(config))
    mod.fit(treino).transform(validacao)
    pd.testing.assert_frame_equal(treino, original_treino)
    pd.testing.assert_frame_equal(validacao, original_validacao)


def test_transform_antes_fit_falha(config, dados):
    with pytest.raises(RuntimeError, match="fit"):
        RedutorPCA(config, parametros(config)).transform(dados[1])


def test_ordem_ou_features_trocadas_falha(config, dados):
    treino, val = dados
    mod = RedutorPCA(config, parametros(config)).fit(treino)
    with pytest.raises(ValueError, match="Colunas ou ordem"):
        mod.transform(val[val.columns[::-1]])
    with pytest.raises(ValueError, match="Colunas ou ordem"):
        mod.transform(val.drop(columns=["e"]))


@pytest.mark.parametrize("invalidos", [np.nan, np.inf, -np.inf])
def test_nan_inf_falham(config, dados, invalidos):
    treino, val = dados
    treino.iloc[0, 0] = invalidos
    with pytest.raises(ValueError, match="NaN ou infinito"):
        RedutorPCA(config, parametros(config)).fit(treino)


def test_tipo_nao_numerico_falha(config, dados):
    treino, val = dados
    treino["a"] = "categoria"
    with pytest.raises(ValueError, match="numericas"):
        RedutorPCA(config, parametros(config)).fit(treino)


def test_n_components_invalidos_falham(config, dados):
    for k in [0, -1, 999, True, "2"]:
        p = parametros(config)
        p["n_components"] = k
        with pytest.raises(ValueError, match="n_components"):
            RedutorPCA(config, p).fit(dados[0])


def test_solver_arpack_limite(config, dados):
    p = parametros(config)
    p.update(n_components=5, svd_solver="arpack")
    with pytest.raises(ValueError, match="incompativel"):
        RedutorPCA(config, p).fit(dados[0])


def test_parametros_invalidos_falham(config):
    p = parametros(config)
    p["whiten"] = "yes"
    with pytest.raises(ValueError, match="whiten"):
        RedutorPCA(config, p)
    p = parametros(config)
    p["svd_solver"] = "teste"
    with pytest.raises(ValueError, match="svd_solver"):
        RedutorPCA(config, p)
    p = parametros(config)
    p["iterated_power"] = -1
    with pytest.raises(ValueError, match="iterated_power"):
        RedutorPCA(config, p)


def test_nao_aceita_metodo_tsne(config):
    config["reducao"]["metodo"] = "TSNE"
    with pytest.raises(ValueError, match="PCA"):
        list(grade_pca(config))
    with pytest.raises(ValueError, match="PCA"):
        RedutorPCA(config, {"n_components": 2, "whiten": False,
                           "svd_solver": "auto", "iterated_power": "auto"})


def test_transform_por_definicao_usa_media_treino(config, dados):
    treino, validacao = dados
    model = RedutorPCA(config, parametros(config)).fit(treino)
    rot_val = model.transform(validacao)
    esperado = model.modelo_.transform(validacao.to_numpy(dtype=np.float32))
    np.testing.assert_allclose(rot_val.to_numpy(), esperado, atol=1e-5)
