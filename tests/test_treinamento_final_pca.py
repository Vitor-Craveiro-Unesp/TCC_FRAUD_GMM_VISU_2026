"""Testes sinteticos: F10 real jamais e lido nem incluido nos dados dos testes.

Execute na raiz: python -m pytest tests/test_treinamento_final_pca.py -q
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.treinamento_final_pca import (
    ajustar_final_f1_f9,
    salvar_pacote,
    carregar_pacote_confiavel,
)
from src.validacao_oof import grade_candidatos_oof


def _dados_e_config():
    raiz = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((raiz / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").read_text(encoding="utf-8-sig"))
    cfg = deepcopy(cfg)
    cfg["reducao"]["metodo"] = "PCA"
    cfg["dados"]["fracao_base"] = 0.1
    cfg["pca"]["n_components"] = [2]
    cfg["gmm"]["n_components"] = [4]
    cfg["gmm"]["n_init"] = 5
    cfg["amostragem"]["modo"] = "RUS"
    cfg["amostragem"]["razao_negativos_positivos"] = 16
    cfg["risco_historico"]["suavizacao"] = 0.0

    rng = np.random.default_rng(42)
    n = 450
    folds = np.repeat(np.arange(1, 10, dtype=np.uint8), 50)
    y = np.zeros(n, dtype=np.uint8)
    for k in range(9):
        y[50 * k:50 * k + 2] = 1
    x1 = rng.normal(0, 1, n) + 0.45 * y
    x2 = rng.normal(0, 1, n) + 0.55 * y
    df = pd.DataFrame({
        "TARGET_OMEGA": y,
        "NID_ALPHA": np.arange(n, dtype=np.int64),
        "TRANS_VALUE": x1,
        "TRANS_NUM_CARD_FEWF": rng.choice(["R", "S", "T"], n),
        "SEND_GENDER_BE": rng.choice(["F", "M"], n),
        "TRANS_MONTH_SIN": np.sin(np.arange(n) * 2 * np.pi / 12),
        "TRANS_MONTH_COS": np.cos(np.arange(n) * 2 * np.pi / 12),
        "VALOR_CONTINUO": x2,
    })
    candidato = next(grade_candidatos_oof(cfg))
    escolha = {
        "id_candidato": "sintetico_teste",
        "configuracao": candidato,
        "limiar_mcc_oof": 0.3,
        "status": "selecionado_em_ensaio",
        "teste_final_usado": False,
    }
    origem = {
        "sha256_parquet": "DADOS_SINTETICOS_SEM_F10",
        "sha256_folds": "FOLDS_SINTETICOS_F1_F9",
        "sha256_yaml": "CONFIG_SINTETICA",
        "experimento_oof": "TESTE_SINTETICO",
    }
    return df, folds, cfg, escolha, origem


@pytest.fixture(scope="module")
def ajustado():
    df, folds, cfg, escolha, origem = _dados_e_config()
    pacote = ajustar_final_f1_f9(df, folds, cfg, escolha, origem, registrar_historico_em=False)
    return df, folds, pacote


def test_apenas_f1_f9_e_rus_nao_altera_risco_original(ajustado):
    df, folds, pacote = ajustado
    assert set(np.unique(folds)) == set(range(1, 10))
    assert pacote.n_treino_original == len(df)
    assert pacote.n_ajuste_gmm < len(df)  # RUS deve reduzir nesta base sintetica
    assert pacote.risco.n_treino_ == len(df)  # RH usa a totalidade, sem RUS
    assert pacote.limiar_oof == 0.3


def test_inferencia_sem_alvo_e_alinhamento(ajustado):
    df, _, pacote = ajustado
    entrada = df.drop(columns=["TARGET_OMEGA"]).iloc[:20].copy()
    p = pacote.prever_sem_rotulos(entrada)
    assert p.index.equals(entrada.index)
    assert len(p) == len(entrada)
    assert np.isfinite(p.to_numpy()).all()
    assert p.between(0, 1).all()
    with pytest.raises(ValueError, match="Remova o alvo"):
        pacote.prever_sem_rotulos(df.iloc[:20])
    with pytest.raises(ValueError, match="Features ou ordem"):
        pacote.prever_sem_rotulos(entrada.drop(columns=["VALOR_CONTINUO"]))


def test_impede_inclusao_de_f10(ajustado):
    df, folds, pacote = ajustado
    config, escolha, origem = pacote.configuracao, pacote.escolha_oof, pacote.proveniencia
    falso_folds = folds.copy()
    falso_folds[-1] = 10
    with pytest.raises(ValueError, match="F10 proibido"):
        ajustar_final_f1_f9(df, falso_folds, config, escolha, origem, registrar_historico_em=False)


def test_salvar_carregar_sem_mudar_scores(ajustado, tmp_path):
    df, _, pacote = ajustado
    entrada = df.drop(columns=["TARGET_OMEGA"]).iloc[:25].copy()
    esperado = pacote.prever_sem_rotulos(entrada)
    pasta = tmp_path / "modelo_congelado"
    modelo, manifesto = salvar_pacote(pacote, pasta)
    assert modelo.exists() and manifesto.exists()
    info = json.loads(manifesto.read_text(encoding="utf-8"))
    assert info["teste_f10_executado"] is False
    outro = carregar_pacote_confiavel(pasta)
    encontrado = outro.prever_sem_rotulos(entrada)
    np.testing.assert_array_equal(encontrado.to_numpy(), esperado.to_numpy())
    with pytest.raises(FileExistsError):
        salvar_pacote(pacote, pasta)
