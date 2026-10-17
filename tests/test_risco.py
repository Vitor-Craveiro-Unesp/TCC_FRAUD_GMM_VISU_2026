"""Passo 05.02: testes da estimativa de RH com soft counts."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.risco import RiscoHistoricoGaussianas, grade_suavizacao
from src.gmm import GMMNaoSupervisionado, grade_gmm
from src.amostragem import selecionar_indices_gmm


@pytest.fixture
def cfg():
    with (ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").open(encoding="utf-8-sig") as arq:
        return yaml.safe_load(arq)


@pytest.fixture
def exemplo():
    gamma = pd.DataFrame({"G1": [1.0, .6, .2, 0.0],
                          "G2": [0.0, .4, .8, 1.0]}, index=[10, 20, 30, 40])
    y = pd.Series([1, 0, 1, 0], index=gamma.index)
    return gamma, y


def test_formulas_exatas(cfg, exemplo):
    gamma, y = exemplo
    rh = RiscoHistoricoGaussianas(cfg).fit(gamma, y)
    np.testing.assert_allclose(rh.massas_, [1.8, 2.2])
    np.testing.assert_allclose(rh.fraudes_ponderadas_, [1.2, .8])
    np.testing.assert_allclose(rh.riscos_, [1.2/1.8, .8/2.2])
    assert rh.n_treino_ == 4
    assert rh.n_fraudes_treino_ == 2
    assert rh.prevalencia_treino_ == 0.5


def test_k1_risco_igual_prevalencia(cfg):
    gamma = pd.DataFrame({"G1": np.ones(10)}, index=range(10))
    y = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
    rh = RiscoHistoricoGaussianas(cfg).fit(gamma, y)
    assert rh.riscos_["G1"] == pytest.approx(0.2)


def test_media_ponderada_rh_recupera_prevalencia_sem_suavizacao(cfg, exemplo):
    gamma, y = exemplo
    rh = RiscoHistoricoGaussianas(cfg).fit(gamma, y)
    estimativa = float(np.dot(rh.riscos_, rh.massas_) / rh.n_treino_)
    assert estimativa == pytest.approx(y.mean())


def test_tabela_ordem_e_copia(cfg, exemplo):
    gamma, y = exemplo
    rh = RiscoHistoricoGaussianas(cfg).fit(gamma, y)
    tab = rh.tabela()
    assert tab.index.tolist() == ["G1", "G2"]
    assert tab.columns.tolist() == ["massa_efetiva", "fraudes_ponderadas", "risco_historico"]
    assert tab.loc["G2", "risco_historico"] == pytest.approx(.8/2.2)
    tab.iloc[0, 0] = -1
    assert rh.massas_["G1"] > 0


def test_antes_do_fit_falha(cfg):
    with pytest.raises(RuntimeError, match="fit"):
        RiscoHistoricoGaussianas(cfg).tabela()


def test_nao_modifica_entradas(cfg, exemplo):
    gamma, y = exemplo
    gc, yc = gamma.copy(deep=True), y.copy(deep=True)
    RiscoHistoricoGaussianas(cfg).fit(gamma, y)
    pd.testing.assert_frame_equal(gamma, gc)
    pd.testing.assert_series_equal(y, yc)


def test_y_pandas_indice_desalinhado_falha(cfg, exemplo):
    gamma, y = exemplo
    with pytest.raises(ValueError, match="Indice/ordem"):
        RiscoHistoricoGaussianas(cfg).fit(gamma, y.iloc[::-1])


def test_y_array_posicional_valido(cfg, exemplo):
    gamma, y = exemplo
    rh = RiscoHistoricoGaussianas(cfg).fit(gamma, y.to_numpy())
    assert rh.riscos_["G1"] == pytest.approx(1.2 / 1.8)


@pytest.mark.parametrize("y", [[0, 1], [0, 1, 2, 0], [0, -1, 0, 1], [0, np.nan, 0, 1],
                                  [0, np.inf, 0, 1], [0, 'fraude', 0, 1], [[0], [1], [0], [1]]])
def test_y_invalido(cfg, exemplo, y):
    gamma, _ = exemplo
    with pytest.raises(ValueError, match="y_treino"):
        RiscoHistoricoGaussianas(cfg).fit(gamma, y)


@pytest.mark.parametrize("mudanca", [
    lambda g: g.assign(G1=[.5, np.nan, .2, 0.]),
    lambda g: g.assign(G1=[.5, np.inf, .2, 0.]),
    lambda g: g.assign(G1=[-.1, .6, .2, 0.]),
    lambda g: g.assign(G1=[1., .6, 2., 0.]),
    lambda g: g.rename(columns={"G2": "G3"}),
    lambda g: g.rename(columns={"G1": "texto"}),
    lambda g: g.assign(G1=["x", "y", "z", "w"]),
    lambda g: g.iloc[:0],
    lambda g: pd.concat([g.iloc[:2], g.iloc[:2]]),
])
def test_gamma_invalido(cfg, exemplo, mudanca):
    gamma, y = exemplo
    mal = mudanca(gamma)
    with pytest.raises(ValueError):
        RiscoHistoricoGaussianas(cfg).fit(mal, y.to_numpy() if len(mal) == 4 else y)


def test_gamma_soma_diferente_de_um(cfg, exemplo):
    gamma, y = exemplo
    mal = gamma.copy()
    mal.iloc[0] = [.4, .4]
    with pytest.raises(ValueError, match="somar 1"):
        RiscoHistoricoGaussianas(cfg).fit(mal, y)


def test_gamma_massas_zero_falha_sem_suavizacao(cfg):
    g = pd.DataFrame({"G1": [1., 1.], "G2": [0., 0.]})
    with pytest.raises(ValueError, match="massa zero"):
        RiscoHistoricoGaussianas(cfg).fit(g, [1, 0])


def test_suavizacao_ancorada_prevalencia(cfg, exemplo):
    cfg["risco_historico"]["suavizacao"] = [0.0, 5.0]
    gamma, y = exemplo
    rh = RiscoHistoricoGaussianas(cfg, suavizacao=5).fit(gamma, y)
    np.testing.assert_allclose(rh.riscos_, [(1.2 + 5*.5)/(1.8+5), (.8+5*.5)/(2.2+5)])


def test_suavizacao_massa_zero_retorna_prevalencia(cfg):
    cfg["risco_historico"]["suavizacao"] = 5.
    g = pd.DataFrame({"G1": [1., 1.], "G2": [0., 0.]})
    rh = RiscoHistoricoGaussianas(cfg).fit(g, [1, 0])
    assert rh.riscos_["G2"] == pytest.approx(.5)


def test_grade_suavizacao_e_erro_grade_nao_escolhida(cfg):
    cfg["risco_historico"]["suavizacao"] = [0.0, 5, 10]
    assert list(grade_suavizacao(cfg)) == [0., 5., 10.]
    with pytest.raises(ValueError, match="Grade de suavizacao"):
        RiscoHistoricoGaussianas(cfg)
    assert RiscoHistoricoGaussianas(cfg, suavizacao=5).suavizacao_ == 5
    with pytest.raises(ValueError, match="nao consta"):
        RiscoHistoricoGaussianas(cfg, suavizacao=8)


@pytest.mark.parametrize("valor", [-1, np.inf, np.nan, True, "0", []])
def test_suavizacao_invalida(cfg, valor):
    cfg["risco_historico"]["suavizacao"] = valor
    with pytest.raises(ValueError, match="suavizacao"):
        RiscoHistoricoGaussianas(cfg)


@pytest.mark.parametrize("chave,valor,trecho", [
    ("metodo", "hard_counts", "soft_counts"),
    ("estimar_com", "amostra_rus", "treino_original"),
    ("previsao", "hard", "soma_ponderada"),
])
def test_config_incompativel(cfg, chave, valor, trecho):
    cfg["risco_historico"][chave] = valor
    with pytest.raises(ValueError, match=trecho):
        RiscoHistoricoGaussianas(cfg)


def test_classe_positiva_deve_ser_um(cfg):
    cfg["dados"]["classe_positiva"] = 0
    with pytest.raises(ValueError, match="positiva"):
        RiscoHistoricoGaussianas(cfg)


def test_rus_risco_treino_original_nao_amostra(cfg):
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["PC1", "PC2"])
    y = np.array([0] * 180 + [1] * 20)
    cfg_rus = deepcopy(cfg)
    cfg_rus["amostragem"]["modo"] = "RUS"
    cfg_rus["amostragem"]["razao_negativos_positivos"] = 2
    pos = selecionar_indices_gmm(y, cfg_rus, fold_validacao=9)
    assert len(pos) == 60
    gmm = GMMNaoSupervisionado(cfg, {**next(grade_gmm(cfg)), "n_components": 1}).fit(X.iloc[pos])
    gamma = gmm.responsabilidades(X)  # TODAS as 200 observacoes
    rh = RiscoHistoricoGaussianas(cfg).fit(gamma, y)
    assert rh.n_treino_ == 200
    assert rh.riscos_["G1"] == pytest.approx(0.10)
    assert y[pos].mean() == pytest.approx(1/3)


def test_gamma_float32_produz_resultado_estavel(cfg):
    g = pd.DataFrame({"G1": np.asarray([.3, .8, .2, .4], dtype=np.float32),
                      "G2": np.asarray([.7, .2, .8, .6], dtype=np.float32)})
    rh = RiscoHistoricoGaussianas(cfg).fit(g, [1, 0, 0, 1])
    assert rh.riscos_.between(0., 1.).all()


def test_treino_original_tamanho_integral(cfg):
    g = pd.DataFrame({"G1": np.ones(50)}, index=range(50))
    y = np.array([0] * 45 + [1] * 5)
    rh = RiscoHistoricoGaussianas(cfg).fit(g, y)
    assert rh.n_treino_ == 50 and rh.prevalencia_treino_ == pytest.approx(.1)
