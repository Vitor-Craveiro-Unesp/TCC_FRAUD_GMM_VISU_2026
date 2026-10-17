"""Testes do Passo 04.02. Simulam o backend em vez de executar otimizacao pesada.

O script verificar_tsne_base_real.py usa o backend openTSNE verdadeiro.
"""
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.reducao_tsne import RedutorTSNE, grade_tsne


@pytest.fixture
def config():
    return {
        "experimento": {"semente": 42},
        "execucao": {"n_jobs": -1},
        "reducao": {"metodo": "TSNE"},
        "tsne": {
            "perplexity": [2], "n_components": 2,
            "initialization": ["random", "pca"],
            "metric": "euclidean", "learning_rate": "auto",
            "early_exaggeration": 12., "n_iter": 500, "theta": 0.5,
        },
    }


@pytest.fixture
def treino():
    return pd.DataFrame({"a": np.arange(10., dtype=float),
                         "b": np.linspace(-1, 1, 10)}, index=np.arange(101, 111))


class FakeEmbedding:
    def __init__(self, X, k):
        self.coords = np.zeros((len(X), k), dtype=np.float32)
        self.coords[:, 0] = np.arange(len(X))
        self.kl_divergence = 1.23
        self.k = k
        self.transforms = 0
        self.first_coords = self.coords.copy()

    def __array__(self, dtype=None, copy=None):
        return np.array(self.coords, dtype=dtype, copy=True)

    def transform(self, X):
        self.transforms += 1
        return np.full((len(X), self.k), 3.5, dtype=np.float32)


class FakeTSNE:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fits = 0
        FakeTSNE.instances.append(self)

    def fit(self, X):
        self.fits += 1
        self.embedding = FakeEmbedding(X, self.kwargs["n_components"])
        return self.embedding


@pytest.fixture
def backend(monkeypatch):
    FakeTSNE.instances.clear()
    monkeypatch.setitem(sys.modules, "openTSNE", SimpleNamespace(TSNE=FakeTSNE))
    return FakeTSNE


def caso(config):
    return list(grade_tsne(config))[0]


def test_grade_ativa_duas_configuracoes(config):
    grade = list(grade_tsne(config))
    assert len(grade) == 2
    assert [p["initialization"] for p in grade] == ["random", "pca"]


def test_grade_sem_ativar_tsne_falha(config):
    config["reducao"]["metodo"] = "PCA"
    with pytest.raises(ValueError, match="metodo"):
        list(grade_tsne(config))


def test_lista_vazia_falha(config):
    config["tsne"]["perplexity"] = []
    with pytest.raises(ValueError, match="vazia"):
        list(grade_tsne(config))


def test_grade_exige_escolha_explicita(config):
    with pytest.raises(ValueError, match="varias"):
        RedutorTSNE(config)


def test_parametro_invalido_falha(config):
    p = caso(config)
    p["theta"] = 1.5
    with pytest.raises(ValueError, match="theta"):
        RedutorTSNE(config, p)


def test_perplexidade_incompativel_falha(config, treino):
    p = caso(config)
    p["perplexity"] = 10
    with pytest.raises(ValueError, match="perplexity"):
        RedutorTSNE(config, p).fit(treino)


def test_coluna_categorica_falha(config, treino):
    x = treino.assign(nome="fraud_")
    with pytest.raises(ValueError, match="numericas"):
        RedutorTSNE(config, caso(config)).fit(x)


def test_nan_ou_infinito_falha(config, treino):
    x = treino.copy()
    x.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        RedutorTSNE(config, caso(config)).fit(x)
    x.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN"):
        RedutorTSNE(config, caso(config)).fit(x)


def test_antes_de_fit_falha(config, treino):
    r = RedutorTSNE(config, caso(config))
    with pytest.raises(RuntimeError, match="fit"):
        r.transform(treino)
    with pytest.raises(RuntimeError, match="fit"):
        r.transform_treino()


def test_fit_e_transform_sem_refit(config, treino, backend):
    r = RedutorTSNE(config, caso(config)).fit(treino)
    assert len(backend.instances) == 1
    instancia = backend.instances[0]
    assert instancia.fits == 1
    assert instancia.kwargs["n_jobs"] == -1
    assert instancia.kwargs["initialization"] == "random"
    x_novo = treino.iloc[:2].copy()
    out = r.transform(x_novo)
    assert list(out.columns) == ["TSNE1", "TSNE2"]
    assert out.index.equals(x_novo.index)
    assert instancia.fits == 1
    assert r.embedding_.transforms == 1
    np.testing.assert_array_equal(r.embedding_.coords, r.embedding_.first_coords)
    assert r.kl_divergence_ == 1.23


def test_treino_usa_coordenadas_aprendidas(config, treino, backend):
    r = RedutorTSNE(config, caso(config))
    coords = r.fit_transform(treino)
    assert coords.shape == (len(treino), 2)
    assert coords.index.equals(treino.index)
    assert r.embedding_.transforms == 0
    coords.iloc[0, 0] = -99
    assert r.transform_treino().iloc[0, 0] == 0  # nao altera cache


def test_ordem_das_colunas_precisa_ser_igual(config, treino, backend):
    r = RedutorTSNE(config, caso(config)).fit(treino)
    with pytest.raises(ValueError, match="ordem"):
        r.transform(treino[["b", "a"]])


def test_preserva_dados_de_entrada(config, treino, backend):
    antes = treino.copy(deep=True)
    r = RedutorTSNE(config, caso(config)).fit(treino)
    r.transform(treino.iloc[:2])
    pd.testing.assert_frame_equal(treino, antes)


def test_inicializacao_pca_interna(config, treino, backend):
    p = caso(config)
    p["initialization"] = "pca"
    RedutorTSNE(config, p).fit(treino)
    assert backend.instances[0].kwargs["initialization"] == "pca"


def test_n_jobs_zero_falha(config):
    config["execucao"]["n_jobs"] = 0
    with pytest.raises(ValueError, match="n_jobs"):
        RedutorTSNE(config, caso(config))


def test_parametros_individuais_sem_grade(config):
    config["tsne"]["initialization"] = "random"
    r = RedutorTSNE(config)
    assert r.parametros["initialization"] == "random"


def test_nome_feature_saida_so_apos_fit(config, treino, backend):
    r = RedutorTSNE(config, caso(config))
    with pytest.raises(RuntimeError, match="fit"):
        r.get_feature_names_out()
    r.fit(treino)
    assert r.get_feature_names_out().tolist() == ["TSNE1", "TSNE2"]
