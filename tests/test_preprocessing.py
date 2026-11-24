"""Testes rapidos do passo 03.01, inclusive categorias jamais vistas no treino."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import EncodersPorSufixo, selecionar_features


@pytest.fixture
def config():
    with (ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").open(
        encoding="utf-8-sig"
    ) as f:
        return yaml.safe_load(f)


@pytest.fixture
def treino():
    return pd.DataFrame({
        "NID_ALPHA": [1, 2, 3, 4],
        "TARGET_OMEGA": [0, 0, 0, 1],
        "SEND_JOB_FEWF": ["A", "A", "A", "B"],
        "SEND_GENDER_BE": ["F", "M", "F", "M"],
        "TRANS_WEEK_OHEWI": ["Monday", "Tuesday", "Monday", "Tuesday"],
        "TRANS_MONTH_SIN": [0.5, 0.0, -0.5, 0.5],
        "TRANS_VALUE": [10., 20., 30., 40.],
    })


def test_aprende_somente_treino(config, treino):
    X = selecionar_features(treino, config)
    enc = EncodersPorSufixo(config).fit(X)
    resultado = enc.transform(X)
    assert "TARGET_OMEGA" not in resultado.columns
    assert "NID_ALPHA" not in resultado.columns
    np.testing.assert_allclose(resultado["SEND_JOB_FEWF"], [0.75, 0.75, 0.75, 0.25])
    np.testing.assert_allclose(resultado["SEND_GENDER_BE"], [0, 1, 0, 1])
    np.testing.assert_allclose(resultado["TRANS_MONTH_SIN"], treino["TRANS_MONTH_SIN"])
    assert list(resultado.columns) == list(enc.get_feature_names_out())


def test_categorias_nao_vistas_na_validacao(config, treino):
    X = selecionar_features(treino, config)
    enc = EncodersPorSufixo(config).fit(X)
    validacao = X.iloc[[0]].copy()
    validacao["SEND_JOB_FEWF"] = "NOVA"  # FE: fallback 1/4
    validacao["TRANS_WEEK_OHEWI"] = "Friday"  # OHE: tudo zero
    novo = enc.transform(validacao)
    assert novo["SEND_JOB_FEWF"].iloc[0] == pytest.approx(0.25)
    assert novo.filter(like="TRANS_WEEK_OHEWI__").sum(axis=1).iloc[0] == 0
    assert "NOVA" not in enc.freq_["SEND_JOB_FEWF"]  # Sem aprendizado na validacao


def test_be_desconhecido_gera_erro(config, treino):
    X = selecionar_features(treino, config)
    enc = EncodersPorSufixo(config).fit(X)
    X_bad = X.iloc[[0]].copy()
    X_bad["SEND_GENDER_BE"] = "NAO_MAPEADO"
    with pytest.raises(ValueError, match="sem mapeamento"):
        enc.transform(X_bad)


def test_impede_vazamento_de_alvo_e_id(config, treino):
    with pytest.raises(ValueError, match="proibidas"):
        EncodersPorSufixo(config).fit(treino)


def test_transform_antes_do_fit_falha(config, treino):
    with pytest.raises(RuntimeError, match="fit"):
        EncodersPorSufixo(config).transform(selecionar_features(treino, config))


def test_nao_modifica_dados_originais(config, treino):
    original = treino.copy(deep=True)
    X = selecionar_features(treino, config)
    enc = EncodersPorSufixo(config).fit(X)
    enc.transform(X)
    pd.testing.assert_frame_equal(treino, original)


def test_categorias_parquet_dictionary(config, treino):
    """Parquet dictionary pode ser lido como pandas.Categorical."""
    df = treino.copy()
    for coluna in ("SEND_JOB_FEWF", "SEND_GENDER_BE", "TRANS_WEEK_OHEWI"):
        df[coluna] = df[coluna].astype("category")
    X = selecionar_features(df, config)
    encoder = EncodersPorSufixo(config).fit(X)
    convertido = encoder.transform(X)
    assert len(convertido) == len(df)
    assert np.isfinite(convertido.to_numpy()).all()
    assert convertido["SEND_JOB_FEWF"].iloc[0] == pytest.approx(0.75)
