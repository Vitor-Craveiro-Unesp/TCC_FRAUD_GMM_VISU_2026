"""Testes do Passo 05.03: responsabilidades x riscos => score de fraude."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.amostragem import selecionar_indices_gmm
from src.gmm import GMMNaoSupervisionado, grade_gmm
from src.risco import RiscoHistoricoGaussianas
from src.probabilidade import calcular_probabilidades, PreditorProbabilisticoGMM


@pytest.fixture
def cfg():
    with (ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").open(encoding="utf-8-sig") as arq:
        config = yaml.safe_load(arq)
    config["gmm"]["n_components"] = [1, 2] # componentes sinteticos nos testes, sem afetar YAML real
    return config


@pytest.fixture
def dados():
    rng = np.random.default_rng(42)
    n0, n1 = 60, 20
    x0 = rng.normal([0.0, 0.0], [0.9, 1.2], size=(n0, 2))
    x1 = rng.normal([3.0, 0.5], [0.9, 1.1], size=(n1, 2))
    X = pd.DataFrame(np.vstack([x0, x1]), columns=["PC1", "PC2"], index=range(100, 100 + n0 + n1))
    y = pd.Series(np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)], index=X.index)
    X_val = pd.DataFrame([[0.2, -0.1], [2.5, 0.4], [9.0, 10.0]], columns=X.columns,
                         index=[900, 1000, 1100])
    return X, y, X_val


@pytest.fixture
def exemplo():
    gamma = pd.DataFrame({"G1": [1.0, 0.75, 0.2], "G2": [0.0, 0.25, 0.8]},
                         index=[7, 11, 13])
    riscos = pd.Series({"G1": 0.1, "G2": 0.9}, name="risco_historico")
    return gamma, riscos


def montar(cfg, dados, k=2, modo="NORMAL"):
    X, y, X_val = dados
    p = next(p for p in grade_gmm(cfg) if p["n_components"] == k)
    c = deepcopy(cfg)
    c["amostragem"]["modo"] = modo
    # Razao 1 na copia temporaria para testar reducao efetiva; YAML oficial inalterado.
    c["amostragem"]["razao_negativos_positivos"] = 1
    pos = selecionar_indices_gmm(y.to_numpy(), c, fold_validacao=9)
    gmm = GMMNaoSupervisionado(cfg, p).fit(X.iloc[pos])
    rh = RiscoHistoricoGaussianas(cfg).fit(gmm.responsabilidades(X), y)
    pred = PreditorProbabilisticoGMM(gmm, rh)
    return pred, gmm, rh, pos


def test_formula_manual(exemplo):
    gamma, riscos = exemplo
    esperado = np.array([0.1, 0.75*0.1 + 0.25*0.9, 0.2*0.1 + 0.8*0.9])
    scores = calcular_probabilidades(gamma, riscos)
    np.testing.assert_allclose(scores.to_numpy(), esperado, atol=1e-15)
    assert scores.name == "prob_fraude"
    assert scores.index.tolist() == [7, 11, 13]


def test_score_soft_nao_e_argmax(exemplo):
    gamma, riscos = exemplo
    scores = calcular_probabilidades(gamma, riscos)
    assert scores.loc[11] == pytest.approx(0.3)
    assert scores.loc[11] != pytest.approx(riscos[gamma.loc[11].idxmax()])


def test_k1_constante_e_prevalencia(cfg, dados):
    X, y, X_val = dados
    preditor, _, rh, _ = montar(cfg, dados, k=1)
    scores = preditor.prever_probabilidade(X_val)
    assert rh.riscos_["G1"] == pytest.approx(y.mean())
    np.testing.assert_allclose(scores, y.mean(), atol=1e-7)


@pytest.mark.parametrize("modo", ["NORMAL", "RUS"])
@pytest.mark.parametrize("k", [1, 2])
def test_integracao_sem_labels_com_ambas_amostragens(cfg, dados, modo, k):
    X, y, X_val = dados
    preditor, gmm, rh, pos = montar(cfg, dados, k=k, modo=modo)
    assert len(pos) == (len(X) if modo == "NORMAL" else 2*int(y.sum()))
    assert rh.n_treino_ == len(X)
    scores = preditor.prever_probabilidade(X_val)
    assert scores.index.equals(X_val.index)
    assert scores.between(0, 1).all()
    assert np.isfinite(scores.to_numpy()).all()
    np.testing.assert_allclose(scores, gmm.responsabilidades(X_val).to_numpy() @ rh.riscos_.to_numpy(), atol=1e-7)
    assert preditor.prevalencia_treino_ == pytest.approx(y.mean())


def test_resultado_treino_conserva_prevalencia(cfg, dados):
    X, y, _ = dados
    pred, _, _, _ = montar(cfg, dados, k=2, modo="RUS")
    scores = pred.prever_probabilidade(X)
    assert float(scores.mean()) == pytest.approx(float(y.mean()), abs=2e-7)


def test_valores_limites_probabilidade(exemplo):
    gamma, _ = exemplo
    zeros = pd.Series({"G1": 0.0, "G2": 0.0})
    uns = pd.Series({"G1": 1.0, "G2": 1.0})
    np.testing.assert_array_equal(calcular_probabilidades(gamma, zeros), 0)
    np.testing.assert_array_equal(calcular_probabilidades(gamma, uns), 1)


def test_entrada_sem_mutacao(exemplo):
    gamma, riscos = exemplo
    gx = gamma.copy(deep=True)
    rx = riscos.copy(deep=True)
    calcular_probabilidades(gamma, riscos)
    pd.testing.assert_frame_equal(gamma, gx)
    pd.testing.assert_series_equal(riscos, rx)


def test_indices_desordenados_permanecem(exemplo):
    gamma, riscos = exemplo
    embaralhado = gamma.iloc[[2, 0, 1]]
    scores = calcular_probabilidades(embaralhado, riscos)
    assert scores.index.tolist() == [13, 7, 11]
    assert scores.loc[13] == pytest.approx(0.74)


@pytest.mark.parametrize("riscos", [
    pd.Series({"G2": 0.9, "G1": 0.1}),
    pd.Series({"G1": 0.1}),
    pd.Series({"G1": 0.1, "G3": 0.9}),
    [0.1, 0.9],
    pd.Series({"G1": 0.1, "G2": np.nan}),
    pd.Series({"G1": 0.1, "G2": np.inf}),
    pd.Series({"G1": 0.1, "G2": -0.1}),
    pd.Series({"G1": 0.1, "G2": 1.1}),
    pd.Series({"G1": "baixo", "G2": "alto"}),
    pd.Series([0.1, 0.9], index=["G1", "G1"]),
])
def test_riscos_invalidos_rejeitados(exemplo, riscos):
    gamma, _ = exemplo
    with pytest.raises(ValueError):
        calcular_probabilidades(gamma, riscos)


@pytest.mark.parametrize("gamma", [
    pd.DataFrame({"G1": [0.6, -0.1], "G2": [0.4, 1.1]}),
    pd.DataFrame({"G1": [0.6, np.nan], "G2": [0.4, np.nan]}),
    pd.DataFrame({"G1": [0.6, np.inf], "G2": [0.4, 0.0]}),
    pd.DataFrame({"G1": [0.3, 0.6], "G2": [0.4, 0.4]}),
    pd.DataFrame({"G2": [0.4, 0.4], "G1": [0.6, 0.6]}),
    pd.DataFrame({"G1": [], "G2": []}),
    np.array([[.2, .8]]),
    pd.DataFrame({"G1": [.5, .6], "G2": ["x", "y"]}),
    pd.DataFrame({"G1": [.3, .4], "G2": [.7, .6]}, index=[1, 1]),
])
def test_gamma_invalida_rejeitada(gamma):
    risco = pd.Series({"G1": .1, "G2": .9})
    with pytest.raises(ValueError):
        calcular_probabilidades(gamma, risco)


def test_preditor_rejeita_gmm_sem_fit(cfg, dados):
    _, _, rh, _ = montar(cfg, dados)
    parametros = next(p for p in grade_gmm(cfg) if p["n_components"] == 2)
    gmm = GMMNaoSupervisionado(cfg, parametros)
    with pytest.raises(RuntimeError, match="fit"):
        PreditorProbabilisticoGMM(gmm, rh)


def test_preditor_rejeita_rh_sem_fit(cfg, dados):
    _, gmm, _, _ = montar(cfg, dados)
    with pytest.raises(RuntimeError, match="fit"):
        PreditorProbabilisticoGMM(gmm, RiscoHistoricoGaussianas(cfg))


def test_preditor_rejeita_componentes_diferentes(cfg, dados):
    _, gmm, rh, _ = montar(cfg, dados, k=2)
    rh.riscos_ = rh.riscos_.rename(index={"G2": "G9"})
    with pytest.raises(ValueError, match="gaussianas"):
        PreditorProbabilisticoGMM(gmm, rh)


def test_rh_nao_pode_ser_menor_que_ajuste_gmm(cfg, dados):
    X, y, _ = dados
    _, gmm, _, _ = montar(cfg, dados, k=2)
    rh = RiscoHistoricoGaussianas(cfg).fit(gmm.responsabilidades(X.iloc[:5]), y.iloc[:5])
    with pytest.raises(ValueError, match="treino original"):
        PreditorProbabilisticoGMM(gmm, rh)


def test_preditor_rejeita_tipos_invalidos(cfg, dados):
    pred, gmm, rh, _ = montar(cfg, dados)
    with pytest.raises(TypeError, match="gmm"):
        PreditorProbabilisticoGMM(object(), rh)
    with pytest.raises(TypeError, match="risco"):
        PreditorProbabilisticoGMM(gmm, object())


def test_riscos_congelados_no_preditor(cfg, dados):
    X, _, X_val = dados
    pred, _, rh, _ = montar(cfg, dados)
    antes = pred.prever_probabilidade(X_val)
    rh.riscos_.iloc[:] = 1
    depois = pred.prever_probabilidade(X_val)
    pd.testing.assert_series_equal(antes, depois)


def test_nao_permite_reajustar_gmm_apos_rh(cfg, dados):
    X, _, X_val = dados
    pred, gmm, _, _ = montar(cfg, dados)
    gmm.fit(X)
    with pytest.raises(RuntimeError, match="reajustado"):
        pred.prever_probabilidade(X_val)


def test_predicao_repetida_deterministica(cfg, dados):
    _, _, X_val = dados
    pred, _, _, _ = montar(cfg, dados)
    pd.testing.assert_series_equal(pred.prever_probabilidade(X_val), pred.prever_probabilidade(X_val))


def test_colunas_invertidas_rejeitadas(cfg, dados):
    _, _, X_val = dados
    pred, _, _, _ = montar(cfg, dados)
    with pytest.raises(ValueError, match="Colunas ou ordem"):
        pred.prever_probabilidade(X_val[["PC2", "PC1"]])


def test_rejeita_nao_finito_ou_id_no_gmm(cfg, dados):
    _, _, X_val = dados
    pred, _, _, _ = montar(cfg, dados)
    with pytest.raises(ValueError, match="NaN ou infinito"):
        pred.prever_probabilidade(X_val.assign(PC1=np.nan))
    with pytest.raises(ValueError, match="Alvo, ID"):
        pred.prever_probabilidade(X_val.assign(NID_ALPHA=range(len(X_val))))
