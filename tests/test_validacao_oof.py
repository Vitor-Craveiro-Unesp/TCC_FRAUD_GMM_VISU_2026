"""Passo 06.01: testes da rotacao OOF, integracao estatistica e isolamento do F10."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validacao_oof import (
    grade_candidatos_oof, validar_divisao, indices_do_fold,
    prever_fold_oof, executar_oof,
)


@pytest.fixture
def cfg():
    return yaml.safe_load((ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").read_text(encoding="utf-8-sig"))


@pytest.fixture
def dados():
    rng = np.random.default_rng(99)
    folds = np.repeat(np.arange(1, 11), 12).astype(np.uint8)
    n = len(folds)
    y = np.tile(np.r_[np.zeros(10, dtype=int), np.ones(2, dtype=int)], 10)
    v = rng.normal(size=n)
    df = pd.DataFrame({
        "NID_ALPHA": np.arange(n) + 1000,
        "SEND_JOB_FEWF": np.where(np.arange(n) % 3 == 0, "a", "b"),
        "SEND_GENDER_BE": np.where(np.arange(n) % 2 == 0, "F", "M"),
        "TRANS_WEEK_OHEWI": np.where(np.arange(n) % 3 == 0, "Monday", "Tuesday"),
        "TRANS_HOUR_SIN": np.sin(v),
        "TRANS_HOUR_COS": np.cos(v),
        "TRANS_VALUE": v + y*2,
        "TARGET_OMEGA": y,
    })
    return df, folds


def candidato(cfg, *, k=1, dim=2, modo="NORMAL"):
    return next(c for c in grade_candidatos_oof(cfg)
                if c["gmm"]["n_components"] == k
                and c["reducao"]["n_components"] == dim
                and c["modo"] == modo)


def test_grade_original_tem_quatro_configuracoes(cfg):
    cs = list(grade_candidatos_oof(cfg))
    assert len(cs) == 4
    assert {(c["reducao"]["n_components"], c["gmm"]["n_components"]) for c in cs} == {
        (2, 1), (2, 2), (3, 1), (3, 2)
    }
    assert all(c["modo"] == "NORMAL" and c["razao"] is None for c in cs)


def test_normal_nao_duplicado_por_grade_de_razao(cfg):
    cfg["amostragem"]["modo"] = ["NORMAL", "RUS"]
    cfg["amostragem"]["razao_negativos_positivos"] = [1, 2, 3]
    cs = list(grade_candidatos_oof(cfg))
    assert sum(c["modo"] == "NORMAL" for c in cs) == 4
    assert sum(c["modo"] == "RUS" for c in cs) == 12


def test_grade_rejeita_metodo_invalido(cfg):
    cfg["reducao"]["metodo"] = "PCA_TSNE"
    with pytest.raises(ValueError, match="PCA ou TSNE"):
        list(grade_candidatos_oof(cfg))


def test_grade_rejeita_razao_rus_zero(cfg):
    cfg["amostragem"]["modo"] = "RUS"
    cfg["amostragem"]["razao_negativos_positivos"] = [0]
    with pytest.raises(ValueError, match="inteiro positivo"):
        list(grade_candidatos_oof(cfg))


def test_rotacao_isola_f10_e_val(cfg, dados):
    _, folds = dados
    for k in range(1, 10):
        t, v = indices_do_fold(folds, k)
        assert len(t) == 96 and len(v) == 12
        assert not np.any(folds[t] == 10)
        assert (folds[v] == k).all()
        assert not np.intersect1d(t, v).size


def test_nao_permita_f10_validacao(dados):
    _, folds = dados
    with pytest.raises(ValueError, match="nunca F10"):
        indices_do_fold(folds, 10)


def test_nao_permita_fold_invalido(dados):
    _, folds = dados
    with pytest.raises(ValueError):
        indices_do_fold(folds, 0)
    with pytest.raises(ValueError):
        indices_do_fold(folds, 1.5)


def test_divisao_deve_corresponder_base(cfg, dados):
    df, f = dados
    with pytest.raises(ValueError, match="tamanho"):
        validar_divisao(f[:-1], cfg, len(df))


def test_divisao_exige_todos_folds(cfg, dados):
    df, f = dados
    sem_f10 = f.copy()
    sem_f10[sem_f10 == 10] = 9
    with pytest.raises(ValueError, match="F1..F10 presentes"):
        validar_divisao(sem_f10, cfg, len(df))


def test_divisao_respeita_flags(cfg, dados):
    df, folds = dados
    cfg["validacao_cruzada"]["evitar_vazamento"] = False
    with pytest.raises(ValueError, match="evitar_vazamento"):
        validar_divisao(folds, cfg, len(df))


def test_uma_rodada_gera_indices_corretos(cfg, dados):
    df, folds = dados
    resp = prever_fold_oof(df, folds, cfg, candidato(cfg), 9)
    assert len(resp) == 12
    np.testing.assert_array_equal(resp["linha_original"], np.flatnonzero(folds == 9))
    assert (resp["fold"] == 9).all()
    assert resp["prob_fraude"].between(0, 1).all()
    assert resp["y_true"].tolist() == df.loc[folds == 9, "TARGET_OMEGA"].tolist()


def test_k1_equivale_prevalencia_treino_original(cfg, dados):
    df, folds = dados
    resp = prever_fold_oof(df, folds, cfg, candidato(cfg, k=1), 9)
    treino, _ = indices_do_fold(folds, 9)
    np.testing.assert_allclose(resp["prob_fraude"], df.iloc[treino]["TARGET_OMEGA"].mean(), atol=1e-7)


def test_f10_alvo_nao_afeta_oof(cfg, dados):
    df, f = dados
    original = prever_fold_oof(df, f, cfg, candidato(cfg), 9)
    alterado = df.copy()
    alterado.loc[f == 10, "TARGET_OMEGA"] = 1
    alterado.loc[f == 10, "TRANS_VALUE"] = 1e9
    novo = prever_fold_oof(alterado, f, cfg, candidato(cfg), 9)
    pd.testing.assert_frame_equal(original, novo)


def test_y_validacao_nao_afeta_predicao(cfg, dados):
    df, f = dados
    p = candidato(cfg, k=2)
    original = prever_fold_oof(df, f, cfg, p, 9)
    alterado = df.copy()
    alterado.loc[f == 9, "TARGET_OMEGA"] ^= 1
    novo = prever_fold_oof(alterado, f, cfg, p, 9)
    np.testing.assert_allclose(original["prob_fraude"], novo["prob_fraude"], atol=1e-7)
    assert not np.array_equal(original["y_true"], novo["y_true"])


def test_oof_nove_rodadas_sem_duplicar_nem_f10(cfg, dados):
    df, f = dados
    resp = executar_oof(df, f, cfg, candidato(cfg))
    assert len(resp) == 108
    assert resp["linha_original"].is_unique
    np.testing.assert_array_equal(resp["linha_original"], np.flatnonzero(f != 10))
    np.testing.assert_array_equal(resp["fold"], f[f != 10])
    assert resp["prob_fraude"].between(0, 1).all()


def test_oof_nao_modifica_entrada(cfg, dados):
    df, folds = dados
    antigo = df.copy(deep=True)
    divisao = folds.copy()
    executar_oof(df, folds, cfg, candidato(cfg))
    pd.testing.assert_frame_equal(df, antigo)
    np.testing.assert_array_equal(folds, divisao)


def test_candidato_fora_grade_falha(cfg, dados):
    df, folds = dados
    c = deepcopy(candidato(cfg))
    c["gmm"]["n_components"] = 999
    with pytest.raises(ValueError, match="nao pertence"):
        prever_fold_oof(df, folds, cfg, c, 9)


def test_treino_sem_duas_classes_falha(cfg, dados):
    df, folds = dados
    df.loc[folds != 9, "TARGET_OMEGA"] = 0
    with pytest.raises(ValueError, match="ambas as classes"):
        prever_fold_oof(df, folds, cfg, candidato(cfg), 9)


def test_indice_exige_posicional(cfg, dados):
    df, f = dados
    df.index = pd.Index(np.arange(len(df))+1)
    with pytest.raises(ValueError, match="posicional"):
        prever_fold_oof(df, f, cfg, candidato(cfg), 9)


def test_rus_preserva_prevalencia_original_k1(cfg, dados):
    df, folds = dados
    c = deepcopy(cfg)
    c["amostragem"]["modo"] = ["NORMAL", "RUS"]
    c["amostragem"]["razao_negativos_positivos"] = [1]
    treino, _ = indices_do_fold(folds, 9)
    valor = df.iloc[treino]["TARGET_OMEGA"].mean()
    oof = prever_fold_oof(df, folds, c, candidato(c, k=1, modo="RUS"), 9)
    np.testing.assert_allclose(oof["prob_fraude"], valor, atol=1e-7)
