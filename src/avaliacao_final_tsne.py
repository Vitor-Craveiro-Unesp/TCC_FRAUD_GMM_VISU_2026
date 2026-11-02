"""Avaliacao isolada TSNE + GMM com limiar MCC congelado no OOF.

Somente funcoes em memoria; sem CLI, sem leitura de Parquet/folds,
sem treinamento, sem acesso automatico a F10 e sem alteracao de PCA.
Somente um executor futuro, autorizado, podera fornecer dados reais de F10.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from src.treinamento_final_tsne import PacoteFinalTSNE

VERSAO_AVALIACAO = "avaliacao-tsne-f10-tau-oof-congelado-v1"


def _validar_limiar_oof(tau: Any) -> float:
    if isinstance(tau, (bool, np.bool_)) or not isinstance(
        tau, (int, float, np.integer, np.floating)
    ):
        raise ValueError("Limiar OOF precisa ser numero finito.")
    tau = float(tau)
    if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise ValueError("Limiar OOF fora de [0, 1].")
    return tau


def _validar_vetores(y_true: Any, prob_fraude: Any):
    if isinstance(y_true, pd.Series) != isinstance(prob_fraude, pd.Series):
        raise ValueError("Forneca ambos Series ou ambos vetores.")
    if isinstance(y_true, pd.Series) and not y_true.index.equals(prob_fraude.index):
        raise ValueError("Indices de rotulos e scores nao coincidem.")
    y = np.asarray(y_true)
    p = np.asarray(prob_fraude)
    if y.ndim != 1 or p.ndim != 1 or not len(y) or len(y) != len(p):
        raise ValueError("Rotulos e scores devem ser vetores de mesmo tamanho.")
    if not (np.issubdtype(y.dtype, np.number) or np.issubdtype(y.dtype, np.bool_)):
        raise ValueError("Rotulos devem ser numericos 0/1.")
    if not np.isin(y, [0, 1]).all() or y.sum() in (0, len(y)):
        raise ValueError("Rotulos precisam ser 0/1 com ambas as classes.")
    if not np.issubdtype(p.dtype, np.number):
        raise ValueError("Scores devem ser numericos.")
    p = p.astype(np.float64)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Scores devem ser finitos em [0, 1].")
    return y.astype(np.uint8), p


def _avaliar_corte(y: np.ndarray, scores: np.ndarray, tau: float) -> dict:
    predito = (scores >= tau).astype(np.uint8)
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(y, predito, labels=[0, 1]).ravel()]
    n0, n1 = tn + fp, fn + tp
    n = len(y)
    return {
        "limiar": float(tau),
        "mcc": float(matthews_corrcoef(y, predito)),
        "matriz_confusao": {
            "ordem_linhas_reais": [0, 1],
            "ordem_colunas_previstas": [0, 1],
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "contagens": [[tn, fp], [fn, tp]],
            "percentuais_por_classe_real": [
                [100.0 * tn / n0, 100.0 * fp / n0],
                [100.0 * fn / n1, 100.0 * tp / n1],
            ],
            "percentuais_do_total": [
                [100.0 * tn / n, 100.0 * fp / n],
                [100.0 * fn / n, 100.0 * tp / n],
            ],
        },
        "precision": float(precision_score(y, predito, zero_division=0)),
        "recall_sensibilidade": float(recall_score(y, predito, zero_division=0)),
        "especificidade": float(tn / n0),
        "f1": float(f1_score(y, predito, zero_division=0)),
        "fpr": float(fp / n0),
        "fnr": float(fn / n1),
        "accuracy": float((tn + tp) / n),
        "positivos_preditos": int(fp + tp),
    }


def avaliar_scores_com_limiar_oof_congelado(
    y_true: Any, prob_fraude: Any, *, limiar_mcc_oof: float
) -> dict:
    """Calcula metricas sem nunca escolher/otimizar limiar com estes rotulos."""
    y, scores = _validar_vetores(y_true, prob_fraude)
    tau = _validar_limiar_oof(limiar_mcc_oof)
    ap = float(average_precision_score(y, scores))
    ll = float(log_loss(y, scores, labels=[0, 1]))
    cortes = {
        "tau_0_50": _avaliar_corte(y, scores, 0.5),
        "tau_mcc_oof": _avaliar_corte(y, scores, tau),
    }
    mcc = cortes["tau_mcc_oof"]["mcc"]
    return {
        "versao": VERSAO_AVALIACAO,
        "tipo_avaliacao": "TESTE_FINAL_F10_LIMIAR_OOF_CONGELADO",
        "limiar_otimizado_no_f10": False,
        "n": int(len(y)),
        "n_legitimas": int(len(y) - y.sum()),
        "n_fraudes": int(y.sum()),
        "prevalencia": float(y.mean()),
        "ap": ap,
        "logloss": ll,
        "exp_menos_logloss": float(math.exp(-ll)),
        "mcc_limiar_oof": mcc,
        "score_composto_limiar_oof": float((ap + (mcc + 1) / 2 + math.exp(-ll)) / 3),
        "limiar_mcc_oof_congelado": tau,
        "comparacao_limiares": cortes,
    }


def avaliar_f10_em_memoria(
    pacote: PacoteFinalTSNE,
    df_f10: pd.DataFrame,
    folds_f10: Any,
    linhas_originais: Any,
) -> tuple[pd.DataFrame, dict]:
    """Apenas avalia dados entregues explicitamente; nunca acessa arquivos.

    Atenção: testar com F10 REAL depende de autorização posterior e
    verificação externa de proveniência, hashes e condição exploratória.
    """
    if not isinstance(pacote, PacoteFinalTSNE):
        raise TypeError("A avaliacao exige PacoteFinalTSNE.")
    if not isinstance(df_f10, pd.DataFrame) or df_f10.empty or not df_f10.index.equals(pd.RangeIndex(len(df_f10))):
        raise ValueError("Lote de avaliacao requer DataFrame com RangeIndex.")
    if not df_f10.columns.is_unique:
        raise ValueError("Colunas duplicadas no lote de avaliacao.")
    cfg = pacote.configuracao
    escolha = pacote.escolha_oof
    if cfg.get("reducao", {}).get("metodo", "").upper() != "TSNE":
        raise ValueError("Modelo nao corresponde a TSNE.")
    if cfg.get("divisao", {}).get("qual_fold_teste") != 10:
        raise ValueError("Configuracao nao reserva F10.")
    if (escolha.get("teste_final_usado") is not False
            or escolha.get("status") not in {"selecionado_em_ensaio", "selecionado_em_oof_integral"}
            or escolha.get("configuracao", {}).get("metodo") != "TSNE"):
        raise ValueError("Escolha OOF t-SNE nao esta congelada.")
    tau = _validar_limiar_oof(escolha.get("limiar_mcc_oof"))
    alvo = cfg["dados"]["coluna_alvo"]
    if alvo not in df_f10.columns:
        raise ValueError("Alvo ausente no lote de avaliacao.")
    n = len(df_f10)
    folds = np.asarray(folds_f10)
    ids = np.asarray(linhas_originais)
    if folds.shape != (n,) or not np.issubdtype(folds.dtype, np.integer) or not (folds == 10).all():
        raise ValueError("Somente folds de avaliacao identificados como 10 sao aceitos.")
    if (ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer)
            or (ids < 0).any() or len(np.unique(ids)) != n
            or (n > 1 and not (np.diff(ids) > 0).all())):
        raise ValueError("Indices originais devem ser inteiros crescentes e unicos.")
    # VALIDAR os rotulos antes da inferencia, mas NUNCA repassa-los ao preditor.
    y = df_f10[alvo]
    _validar_vetores(y, pd.Series(np.full(n, 0.5), index=y.index))
    entrada = df_f10.drop(columns=[alvo]).copy()
    scores = pacote.prever_sem_rotulos(entrada)
    if not isinstance(scores, pd.Series) or not scores.index.equals(df_f10.index):
        raise RuntimeError("Scores nao preservaram a ordem de avaliacao.")
    resultado = avaliar_scores_com_limiar_oof_congelado(y, scores, limiar_mcc_oof=tau)
    resultado["metodo"] = "TSNE"
    resultado["id_candidato_oof"] = escolha["id_candidato"]
    resultado["dimensao_tsne"] = int(escolha["configuracao"]["reducao"]["n_components"])
    s = scores.to_numpy(dtype=np.float64)
    tabela = pd.DataFrame({
        "linha_original": ids.astype(np.int64),
        "fold": np.full(n, 10, dtype=np.uint8),
        "y_true": y.to_numpy(dtype=np.uint8),
        "prob_fraude": s,
        "predicao_tau_0_50": (s >= 0.5).astype(np.uint8),
        "predicao_tau_mcc_oof": (s >= tau).astype(np.uint8),
    })
    return tabela, resultado
