"""Testes sinteticos do Passo 05.01. Nao acessam Parquet real."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gmm import GMMNaoSupervisionado, grade_gmm
from src.amostragem import selecionar_indices_gmm


@pytest.fixture
def config():
    with (ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").open(
        encoding="utf-8-sig"
    ) as arq:
        return yaml.safe_load(arq)


@pytest.fixture
def dados():
    rng = np.random.default_rng(123)
    parte1 = rng.normal(loc=-2.0, scale=0.8, size=(90, 2))
    parte2 = rng.normal(loc=2.0, scale=0.7, size=(90, 2))
    return pd.DataFrame(
        np.vstack((parte1, parte2)).astype(np.float32),
        columns=["PC1", "PC2"],
        index=np.arange(1000, 1180),
    )


def p(config, k=2):
    return {**next(grade_gmm(config)), "n_components": k}


def test_grade_yaml_k2_k4_k6(config):
    grade = list(grade_gmm(config))
    assert [d["n_components"] for d in grade] == [2, 4, 6]
    assert len(grade) == 3


def test_grade_produto_cartesiano_sem_vetor_visualizacao(config):
    config["gmm"]["n_components"] = [1, 2]
    config["gmm"]["covariance_type"] = ["diag", "full"]
    config["pca"]["n_components"] = [2, 3, 4]
    assert len(list(grade_gmm(config))) == 4


def test_grade_vazia_falha(config):
    config["gmm"]["n_components"] = []
    with pytest.raises(ValueError, match="lista vazia"):
        list(grade_gmm(config))


def test_grade_precisa_de_escolha(config):
    with pytest.raises(ValueError, match="grade GMM"):
        GMMNaoSupervisionado(config)


def test_k1_responsabilidade_um(config, dados):
    gmm = GMMNaoSupervisionado(config, p(config, 1)).fit(dados)
    gamma = gmm.responsabilidades(dados.iloc[::5])
    assert gamma.shape == (36, 1)
    assert list(gamma.columns) == ["G1"]
    np.testing.assert_allclose(gamma.to_numpy(), 1.0)


def test_k2_soma_1_e_probabilidades(config, dados):
    gmm = GMMNaoSupervisionado(config, p(config, 2)).fit(dados)
    gamma = gmm.responsabilidades(dados)
    assert gamma.shape == (180, 2)
    assert np.isfinite(gamma.to_numpy()).all()
    assert gamma.to_numpy().min() >= 0
    np.testing.assert_allclose(gamma.sum(axis=1), 1, atol=1e-6)


def test_indices_preservados(config, dados):
    gmm = GMMNaoSupervisionado(config, p(config, 2)).fit(dados)
    externo = dados.iloc[[8, 2, 20]]
    assert gmm.responsabilidades(externo).index.equals(externo.index)


def test_semente_reprodutivel(config, dados):
    a = GMMNaoSupervisionado(config, p(config, 2)).fit(dados)
    b = GMMNaoSupervisionado(config, p(config, 2)).fit(dados)
    np.testing.assert_allclose(a.modelo_.means_, b.modelo_.means_)
    np.testing.assert_allclose(
        a.responsabilidades(dados).to_numpy(), b.responsabilidades(dados).to_numpy()
    )


def test_previsao_nao_muda_parametros(config, dados):
    gmm = GMMNaoSupervisionado(config, p(config, 2)).fit(dados)
    medias = gmm.modelo_.means_.copy()
    pesos = gmm.modelo_.weights_.copy()
    gmm.responsabilidades(dados.iloc[:4] + 100)
    np.testing.assert_array_equal(medias, gmm.modelo_.means_)
    np.testing.assert_array_equal(pesos, gmm.modelo_.weights_)


def test_nao_modifica_X(config, dados):
    copia = dados.copy(deep=True)
    gmm = GMMNaoSupervisionado(config, p(config, 2)).fit(dados)
    gmm.responsabilidades(dados)
    pd.testing.assert_frame_equal(dados, copia)


def test_y_nao_aceito_no_fit(config, dados):
    gmm = GMMNaoSupervisionado(config, p(config, 2))
    with pytest.raises(TypeError):
        gmm.fit(dados, np.zeros(len(dados)))


def test_alvo_e_id_proibidos(config, dados):
    gmm = GMMNaoSupervisionado(config, p(config, 2))
    for proibida in (config["dados"]["coluna_alvo"], config["dados"]["coluna_id"]):
        with pytest.raises(ValueError, match="Alvo, ID"):
            gmm.fit(dados.assign(**{proibida: 0}))


def test_nao_numericos_falham(config, dados):
    with pytest.raises(ValueError, match="numericas"):
        GMMNaoSupervisionado(config, p(config)).fit(dados.assign(TEXTO="oi"))


@pytest.mark.parametrize("valor", [np.nan, np.inf, -np.inf])
def test_nao_finitos_falham(config, dados, valor):
    invalidos = dados.copy()
    invalidos.iloc[0, 0] = valor
    with pytest.raises(ValueError, match="NaN ou infinito"):
        GMMNaoSupervisionado(config, p(config)).fit(invalidos)


def test_antes_do_fit_falha(config, dados):
    with pytest.raises(RuntimeError, match="fit"):
        GMMNaoSupervisionado(config, p(config)).responsabilidades(dados)


def test_colunas_ou_ordem_falham(config, dados):
    gmm = GMMNaoSupervisionado(config, p(config)).fit(dados)
    with pytest.raises(ValueError, match="Colunas ou ordem"):
        gmm.responsabilidades(dados[["PC2", "PC1"]])


def test_k_superior_ao_treino_falha(config, dados):
    with pytest.raises(ValueError, match="excede"):
        GMMNaoSupervisionado(config, p(config, 2)).fit(dados.iloc[:1])


@pytest.mark.parametrize("campo, valor", [
    ("n_components", 0), ("n_components", True),
    ("covariance_type", "indefinido"), ("reg_covar", -1),
    ("max_iter", 0), ("n_init", 0), ("tol", 0),
    ("init_params", "outro"),
])
def test_parametro_invalido_falha(config, campo, valor):
    parametros = p(config)
    parametros[campo] = valor
    with pytest.raises(ValueError):
        GMMNaoSupervisionado(config, parametros)


def test_rus_muda_apenas_selecao_para_ajuste(config, dados):
    y = np.array([1] * 20 + [0] * 160)
    config["amostragem"]["modo"] = ["NORMAL", "RUS"]
    config["amostragem"]["razao_negativos_positivos"] = [2, 32]
    indice = selecionar_indices_gmm(y, config, modo="RUS", razao=2, fold_validacao=9)
    assert len(indice) == 60
    assert np.count_nonzero(y[indice] == 1) == 20
    gmm = GMMNaoSupervisionado(config, p(config)).fit(dados.iloc[indice])
    # As responsabilidades sao calculadas sobre TODO o treino original.
    gamma = gmm.responsabilidades(dados)
    assert len(gamma) == 180
    assert gmm.n_ajuste_ == 60


def test_parametros_diagnosticos(config, dados):
    gmm = GMMNaoSupervisionado(config, p(config)).fit(dados)
    assert gmm.modelo_.converged_
    assert len(gmm.modelo_.weights_) == 2
    assert gmm.modelo_.means_.shape == (2, 2)
    assert gmm.modelo_.covariances_.shape == (2, 2, 2)
    assert gmm.modelo_.n_iter_ >= 1


def test_NaN_validacao_falha(config, dados):
    gmm = GMMNaoSupervisionado(config, p(config)).fit(dados)
    ruim = dados.iloc[:3].copy()
    ruim.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN ou infinito"):
        gmm.responsabilidades(ruim)


def test_grade_copias_independentes(config):
    grade = list(grade_gmm(config))
    grade[0]["n_components"] = 100
    assert grade[1]["n_components"] == 4
    assert config["gmm"]["n_components"] == [2, 4, 6]
