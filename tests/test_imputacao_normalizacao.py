"""Testes do passo 03.02: mediana/StandardScaler aprendidos apenas no treino."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import (
    EncodersPorSufixo, ImputadorPadronizador, selecionar_features,
)


@pytest.fixture
def config():
    with (ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").open(
        encoding="utf-8-sig"
    ) as f:
        return yaml.safe_load(f)


@pytest.fixture
def dados():
    return pd.DataFrame({
        "NID_ALPHA": [101, 102, 103, 104],
        "TARGET_OMEGA": [0, 0, 0, 1],
        "SEND_JOB_FEWF": ["A", "A", "A", "B"],
        "SEND_GENDER_BE": ["F", "M", "F", "M"],
        "TRANS_WEEK_OHEWI": ["Monday", "Tuesday", "Monday", "Tuesday"],
        "TRANS_MONTH_SIN": [0.5, np.nan, -0.5, 0.5],
        "TRANS_MONTH_COS": [0.0, 1.0, 0.0, -1.0],
        "TRANS_VALUE": [10.0, 20.0, np.nan, 40.0],
        "TRANS_CONSTANTE": [7.0, 7.0, 7.0, 7.0],
    })


@pytest.fixture
def codificado(config, dados):
    x = selecionar_features(dados, config)
    encoder = EncodersPorSufixo(config).fit(x)
    return encoder, encoder.transform(x)


def test_imputa_com_mediana_do_treino(config, codificado):
    encoder, x = codificado
    proc = ImputadorPadronizador(config, encoder).fit(x)
    out = proc.transform(x)
    assert proc.mediana_["TRANS_VALUE"] == pytest.approx(20)
    assert proc.mediana_["TRANS_MONTH_SIN"] == pytest.approx(0.5)
    assert np.isfinite(out.to_numpy()).all()
    assert out["TRANS_VALUE"].mean() == pytest.approx(0, abs=1e-6)
    assert out["TRANS_VALUE"].std(ddof=0) == pytest.approx(1, abs=1e-6)


def test_nao_aprende_nada_da_validacao(config, codificado):
    encoder, x = codificado
    proc = ImputadorPadronizador(config, encoder).fit(x)
    medias_antes = proc.scaler_.mean_.copy()
    mediana_antes = proc.mediana_.copy()
    val = x.iloc[[0]].copy()
    val["TRANS_VALUE"] = 10000.0
    out = proc.transform(val)
    assert proc.mediana_ == mediana_antes
    np.testing.assert_allclose(proc.scaler_.mean_, medias_antes)
    j = proc.padronizadas_.index("TRANS_VALUE")
    esperado = (10000.0 - proc.scaler_.mean_[j]) / proc.scaler_.scale_[j]
    assert out["TRANS_VALUE"].iloc[0] == pytest.approx(esperado, rel=1e-6)


def test_ciclicas_sem_padronizacao_e_binarias_intactas(config, codificado):
    encoder, x = codificado
    proc = ImputadorPadronizador(config, encoder).fit(x)
    out = proc.transform(x)
    np.testing.assert_allclose(out["TRANS_MONTH_SIN"], [0.5, 0.5, -0.5, 0.5])
    np.testing.assert_allclose(out["TRANS_MONTH_COS"], x["TRANS_MONTH_COS"])
    np.testing.assert_allclose(out["SEND_GENDER_BE"], x["SEND_GENDER_BE"])
    onehot = [c for c in x.columns if c.startswith("TRANS_WEEK_OHEWI__")]
    np.testing.assert_allclose(out[onehot], x[onehot])


def test_frequencias_padronizadas_e_coluna_constante(config, codificado):
    encoder, x = codificado
    proc = ImputadorPadronizador(config, encoder).fit(x)
    out = proc.transform(x)
    assert "SEND_JOB_FEWF" in proc.padronizadas_
    assert out["SEND_JOB_FEWF"].mean() == pytest.approx(0, abs=1e-6)
    np.testing.assert_allclose(out["TRANS_CONSTANTE"], 0, atol=1e-6)


def test_coluna_sem_valores_observados_falha(config, codificado):
    encoder, x = codificado
    x["TRANS_VALUE"] = np.nan
    with pytest.raises(ValueError, match="sem valores observados"):
        ImputadorPadronizador(config, encoder).fit(x)


def test_inf_falha_explicitamente(config, codificado):
    encoder, x = codificado
    x.loc[x.index[0], "TRANS_VALUE"] = np.inf
    with pytest.raises(ValueError, match="infinitos"):
        ImputadorPadronizador(config, encoder).fit(x)


def test_colunas_ou_ordem_alteradas_falha(config, codificado):
    encoder, x = codificado
    proc = ImputadorPadronizador(config, encoder).fit(x)
    with pytest.raises(ValueError, match="Colunas codificadas"):
        proc.transform(x[x.columns[::-1]])


def test_transform_antes_de_fit_falha(config, codificado):
    encoder, x = codificado
    with pytest.raises(RuntimeError, match="fit"):
        ImputadorPadronizador(config, encoder).transform(x)


def test_config_incompativel_falha(config, codificado):
    encoder, x = codificado
    config["preprocessamento"]["numericas"]["imputacao"] = "media"
    with pytest.raises(ValueError, match="mediana"):
        ImputadorPadronizador(config, encoder).fit(x)


def test_nao_modifica_dados_de_entrada(config, codificado):
    encoder, x = codificado
    original = x.copy(deep=True)
    proc = ImputadorPadronizador(config, encoder)
    proc.fit(x).transform(x)
    pd.testing.assert_frame_equal(x, original)
