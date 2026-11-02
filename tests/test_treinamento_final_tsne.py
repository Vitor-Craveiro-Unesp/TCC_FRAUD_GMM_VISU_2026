"""Regressoes isoladas: somente base sintetica F1-F9, sem Parquet ou F10 real.

Executar: .venv\\Scripts\\python.exe -m pytest tests/test_treinamento_final_tsne.py -q
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.treinamento_final_tsne import (
    PacoteFinalTSNE, ajustar_final_f1_f9, salvar_pacote,
    carregar_pacote_confiavel,
)
from src.validacao_oof import grade_candidatos_oof


def dados_e_config(dim):
    raiz = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((raiz / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").read_text(encoding="utf-8-sig"))
    cfg = deepcopy(cfg)  # NUNCA altera o YAML oficial
    cfg["reducao"]["metodo"] = "TSNE"
    cfg["dados"]["fracao_base"] = 0.1
    cfg["tsne"]["n_components"] = dim
    cfg["tsne"]["initialization"] = ["random"]
    cfg["tsne"]["perplexity"] = [10]
    cfg["gmm"]["n_components"] = [4]
    cfg["gmm"]["covariance_type"] = "full"
    cfg["gmm"]["n_init"] = 5
    cfg["amostragem"]["modo"] = "RUS"
    cfg["amostragem"]["razao_negativos_positivos"] = 16
    cfg["risco_historico"]["suavizacao"] = 0.0
    cfg["execucao"]["n_jobs"] = 1

    rng = np.random.default_rng(42)
    folds = np.repeat(np.arange(1, 10, dtype=np.uint8), 50)
    n = len(folds)
    y = np.zeros(n, dtype=np.uint8)
    for k in range(9):
        y[50 * k:50 * k + 2] = 1
    ruido = rng.normal(size=n)
    df = pd.DataFrame({
        "TARGET_OMEGA": y,
        "NID_ALPHA": np.arange(n, dtype=np.int64),
        "TRANS_VALUE": ruido + 0.45 * y,
        "TRANS_NUM_CARD_FEWF": rng.choice(["R", "S", "T"], n),
        "SEND_GENDER_BE": rng.choice(["F", "M"], n),
        "TRANS_MONTH_SIN": np.sin(np.arange(n) * 2 * np.pi / 12),
        "TRANS_MONTH_COS": np.cos(np.arange(n) * 2 * np.pi / 12),
        "VALOR_CONTINUO": rng.normal(size=n) + 0.5 * y,
    })
    grade = list(grade_candidatos_oof(cfg))
    assert len(grade) == 1
    escolha = {
        "id_candidato": f"sintetico_tsne_{dim}d",
        "configuracao": grade[0],
        "limiar_mcc_oof": 0.3,
        "status": "selecionado_em_ensaio",
        "teste_final_usado": False,
    }
    origem = {
        "sha256_parquet": "SINTETICO_SEM_PARQUET",
        "sha256_folds": "SINTETICO_F1_F9",
        "sha256_yaml": "CONFIG_EM_MEMORIA",
        "experimento_oof": "TESTE_SINTETICO",
    }
    return df, folds, cfg, escolha, origem


@pytest.fixture(scope="module", params=(1, 2, 3), ids=("1d", "2d", "3d"))
def ajustado(request):
    dim = request.param
    df, folds, cfg, escolha, origem = dados_e_config(dim)
    copia_df = df.copy(deep=True)
    copia_folds = folds.copy()
    pacote = ajustar_final_f1_f9(
        df, folds, cfg, escolha, origem, registrar_historico_em=False
    )
    pd.testing.assert_frame_equal(df, copia_df)
    np.testing.assert_array_equal(folds, copia_folds)
    return dim, df, folds, cfg, escolha, origem, pacote


def test_treino_f1_f9_rus_e_risco_historico(ajustado):
    dim, df, folds, cfg, escolha, origem, pacote = ajustado
    assert isinstance(pacote, PacoteFinalTSNE)
    assert set(np.unique(folds).tolist()) == set(range(1, 10))
    assert pacote.n_treino_original == len(df)
    assert pacote.n_ajuste_gmm == 18 + 16 * 18
    assert pacote.n_ajuste_gmm < len(df)
    assert pacote.risco.n_treino_ == len(df)
    assert pacote.tsne.n_treino_ == len(df)
    assert pacote.limiar_oof == 0.3
    assert pacote.tsne.transform_treino().shape == (len(df), dim)
    if dim == 3:
        assert pacote.tsne.modelo_.negative_gradient_method == "bh"


def test_inferencia_oos_e_bloqueios(ajustado):
    dim, df, folds, cfg, escolha, origem, pacote = ajustado
    entrada = df.drop(columns=["TARGET_OMEGA"]).iloc[:25].copy()
    entrada.index = pd.Index(range(8000, 8025))
    scores = pacote.prever_sem_rotulos(entrada)
    assert scores.index.equals(entrada.index)
    assert np.isfinite(scores.to_numpy()).all()
    assert scores.between(0, 1).all()
    with pytest.raises(ValueError, match="Remova o alvo"):
        pacote.prever_sem_rotulos(df.iloc[:10])
    with pytest.raises(ValueError, match="Features ou ordem"):
        pacote.prever_sem_rotulos(entrada.drop(columns=["VALOR_CONTINUO"]))


def test_bloqueia_f10_sintetico_e_candidato_pca(ajustado):
    dim, df, folds, cfg, escolha, origem, pacote = ajustado
    falso_folds = folds.copy()
    falso_folds[-1] = 10
    with pytest.raises(ValueError, match="F10 proibido"):
        ajustar_final_f1_f9(df, falso_folds, cfg, escolha, origem, registrar_historico_em=False)
    falso_escolha = deepcopy(escolha)
    falso_escolha["configuracao"]["metodo"] = "PCA"
    with pytest.raises(ValueError, match="metodo TSNE"):
        ajustar_final_f1_f9(df, folds, cfg, falso_escolha, origem, registrar_historico_em=False)


def test_persistencia_hash_e_scores(ajustado, tmp_path):
    dim, df, folds, cfg, escolha, origem, pacote = ajustado
    entrada = df.drop(columns=["TARGET_OMEGA"]).iloc[:25].copy()
    esperado = pacote.prever_sem_rotulos(entrada)
    pasta = tmp_path / f"pacote_tsne_{dim}d"
    modelo, manifesto = salvar_pacote(pacote, pasta)
    assert modelo.is_file() and manifesto.is_file()
    info = json.loads(manifesto.read_text(encoding="utf-8"))
    assert info["teste_f10_executado"] is False
    assert info["metodo"] == "TSNE"
    assert info["dimensao_tsne"] == dim
    assert info["limiar_mcc_oof_congelado"] == 0.3
    outro = carregar_pacote_confiavel(pasta)
    encontrado = outro.prever_sem_rotulos(entrada)
    np.testing.assert_allclose(encontrado.to_numpy(), esperado.to_numpy(), atol=1e-7, rtol=0)
    with pytest.raises(FileExistsError):
        salvar_pacote(pacote, pasta)
    manifesto.write_text(manifesto.read_text(encoding="utf-8").replace('"TSNE"', '"PCA"'), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Manifesto"):
        carregar_pacote_confiavel(pasta)
