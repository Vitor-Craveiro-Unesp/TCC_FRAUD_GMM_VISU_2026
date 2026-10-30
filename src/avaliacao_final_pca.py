"""Avaliacao final de GMM + PCA com limiar congelado em OOF.

ETAPA 5/9: somente funcoes PURAS / em memoria. Este modulo NAO possui CLI,
nao le Parquet nem os folds do disco, nao salva arquivos, nao treina ou ajusta,
nao seleciona configuracoes e NAO otimiza o limiar com rotulos do F10.

A funcao de integracao para F10 so sera usada pelo futuro executor, depois da
validacao de proveniencia e de autorizacao explicita da avaliacao independente.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
)


VERSAO_AVALIACAO = "avaliacao-pca-f10-tau-oof-congelado-v1"


def _validar_limiar_oof(tau: Any) -> float:
    if isinstance(tau, (bool, np.bool_)) or not isinstance(tau, (int, float, np.integer, np.floating)):
        raise ValueError("Limiar OOF precisa ser um numero finito.")
    tau = float(tau)
    if not math.isfinite(tau) or not 0.0 <= tau <= float(np.nextafter(1.0, np.inf)):
        raise ValueError("Limiar OOF fora dos limites validos.")
    return tau


def _validar_vetores(y_true: Any, prob_fraude: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(y_true, pd.Series) != isinstance(prob_fraude, pd.Series):
        raise ValueError("Para garantir alinhamento, passe ambos como Series ou ambos como vetores.")
    if isinstance(y_true, pd.Series) and not y_true.index.equals(prob_fraude.index):
        raise ValueError("Indices dos rotulos e dos scores nao coincidem.")
    y = np.asarray(y_true)
    p = np.asarray(prob_fraude)
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p):
        raise ValueError("Rotulos e probabilidades precisam ser vetores 1D nao vazios de mesmo tamanho.")
    if (not (np.issubdtype(y.dtype, np.number) or np.issubdtype(y.dtype, np.bool_))
            or not np.isin(y, [0, 1]).all()):
        raise ValueError("Rotulos precisam ser exclusivamente 0/1, sem valores ausentes.")
    if not np.issubdtype(p.dtype, np.number):
        raise ValueError("Scores precisam ser numericos.")
    p = p.astype(np.float64)
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Scores precisam ser probabilidades finitas entre 0 e 1.")
    y = y.astype(np.uint8)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("Metricas finais exigem pelo menos um exemplo de cada classe.")
    return y, p


def _avaliar_corte(y: np.ndarray, p: np.ndarray, corte: float) -> dict:
    pred = (p >= corte).astype(np.uint8)  # comparacao inclusiva >=, igual ao OOF
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    n_leg = tn + fp
    n_fra = fn + tp
    n = n_leg + n_fra
    mcc = float(matthews_corrcoef(y, pred))
    precision = float(precision_score(y, pred, zero_division=0))
    recall = float(recall_score(y, pred, zero_division=0))
    f1 = float(f1_score(y, pred, zero_division=0))
    especificidade = tn / n_leg
    return {
        "limiar": corte,
        "mcc": mcc,
        "matriz_confusao": {
            "ordem_linhas_reais": [0, 1],
            "ordem_colunas_previstas": [0, 1],
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "contagens": [[tn, fp], [fn, tp]],
            "percentuais_por_classe_real": [
                [100.0 * tn / n_leg, 100.0 * fp / n_leg],
                [100.0 * fn / n_fra, 100.0 * tp / n_fra],
            ],
            "percentuais_do_total": [
                [100.0 * tn / n, 100.0 * fp / n],
                [100.0 * fn / n, 100.0 * tp / n],
            ],
        },
        "precision": precision,
        "recall_sensibilidade": recall,
        "especificidade": float(especificidade),
        "f1": f1,
        "fpr": float(fp / n_leg),
        "fnr": float(fn / n_fra),
        "accuracy": float((tp + tn) / n),
        "positivos_preditos": int(tp + fp),
    }


def avaliar_scores_com_limiar_oof_congelado(
    y_true: Any, prob_fraude: Any, *, limiar_mcc_oof: float,
) -> dict:
    """Calcula AP, LogLoss, score e duas CMs sem otimizar nenhum limiar no teste.

    NUNCA chamar otimizar_limiar_mcc() ou avaliar_probabilidades() aqui:
    essas funcoes do OOF usam os labels da entrada para selecionar o corte.
    O score composto do teste usa o MCC calculado em tau* CONGELADO.
    """
    y, p = _validar_vetores(y_true, prob_fraude)
    tau = _validar_limiar_oof(limiar_mcc_oof)
    ap = float(average_precision_score(y, p))
    ll = float(log_loss(y, p, labels=[0, 1]))
    cortes = {"tau_0_50": _avaliar_corte(y, p, 0.5),
              "tau_mcc_oof": _avaliar_corte(y, p, tau)}
    mcc_tau = cortes["tau_mcc_oof"]["mcc"]
    score = float((ap + (mcc_tau + 1.0) / 2.0 + math.exp(-ll)) / 3.0)
    return {
        "versao": VERSAO_AVALIACAO,
        "tipo_avaliacao": "TESTE_FINAL_F10_LIMIAR_OOF_CONGELADO",
        "limiar_otimizado_no_f10": False,
        "n": int(len(y)),
        "n_legitimas": int(len(y) - int(y.sum())),
        "n_fraudes": int(y.sum()),
        "prevalencia": float(y.mean()),
        "ap": ap,
        "logloss": ll,
        "exp_menos_logloss": float(math.exp(-ll)),
        "mcc_limiar_oof": mcc_tau,
        "score_composto_limiar_oof": score,
        "limiar_mcc_oof_congelado": tau,
        "comparacao_limiares": cortes,
    }


def avaliar_f10_em_memoria(
    pacote: Any,
    df_f10: pd.DataFrame,
    folds_f10: Any,
    linhas_originais: Any,
) -> tuple[pd.DataFrame, dict]:
    """Integra modelo previamente ajustado e dados recebidos em memoria.

    Nao carrega o F10 sozinho. O executor externo DEVE conferir o hash do
    parquet, hash dos folds e a amostragem antes de fornecer as entradas.
    Esta funcao verifica que TODAS as linhas indicadas estao em F10,
    remove TARGET antes da previsao e somente depois calcula as metricas.
    """
    if not isinstance(df_f10, pd.DataFrame) or df_f10.empty or not df_f10.index.equals(pd.RangeIndex(len(df_f10))):
        raise ValueError("df_f10 deve ser DataFrame nao vazio com RangeIndex posicional.")
    if not df_f10.columns.is_unique:
        raise ValueError("Colunas duplicadas no F10.")
    cfg = getattr(pacote, "configuracao", None)
    escolha = getattr(pacote, "escolha_oof", None)
    if not isinstance(cfg, dict) or not isinstance(escolha, dict):
        raise ValueError("Pacote final precisa conter configuracao e escolha OOF congeladas.")
    if cfg.get("divisao", {}).get("qual_fold_teste") != 10:
        raise ValueError("Modelo nao reserva F10 como teste final.")
    if escolha.get("teste_final_usado") is not False or escolha.get("status") not in {"selecionado_em_ensaio", "selecionado_em_oof_integral"}:
        raise ValueError("Pacote nao possui escolha OOF valida e congelada.")
    _validar_limiar_oof(escolha.get("limiar_mcc_oof"))
    alvo = cfg["dados"]["coluna_alvo"]
    if alvo not in df_f10.columns:
        raise ValueError("Coluna de rotulos ausente no conjunto destinado a avaliacao.")
    folds = np.asarray(folds_f10)
    ids = np.asarray(linhas_originais)
    n = len(df_f10)
    if folds.shape != (n,) or not np.issubdtype(folds.dtype, np.integer) or not np.all(folds == 10):
        raise ValueError("Somente linhas autenticadas como F10 podem entrar na avaliacao.")
    if (ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer)
            or (ids < 0).any() or len(np.unique(ids)) != n or not np.all(np.diff(ids) > 0)):
        raise ValueError("linhas_originais devem ser inteiros nao negativos, unicos e crescentes.")
    # A coluna alvo nao e entregue ao pipeline preditivo.
    df_sem_alvo = df_f10.drop(columns=[alvo]).copy()
    scores = pacote.prever_sem_rotulos(df_sem_alvo)
    if not isinstance(scores, pd.Series) or not scores.index.equals(df_f10.index):
        raise RuntimeError("Scores nao estao alinhados aos registros de teste.")
    resultado = avaliar_scores_com_limiar_oof_congelado(
        df_f10[alvo], scores, limiar_mcc_oof=float(escolha["limiar_mcc_oof"]),
    )
    resultado["id_candidato_oof"] = escolha.get("id_candidato")
    resultado["dimensao_pca"] = int(escolha["configuracao"]["reducao"]["n_components"])
    tau = float(escolha["limiar_mcc_oof"])
    tabela = pd.DataFrame({
        "linha_original": ids.astype(np.int64),
        "fold": np.full(n, 10, dtype=np.uint8),
        "y_true": df_f10[alvo].to_numpy(dtype=np.uint8),
        "prob_fraude": scores.to_numpy(dtype=np.float64),
        "predicao_tau_0_50": (scores.to_numpy() >= 0.5).astype(np.uint8),
        "predicao_tau_mcc_oof": (scores.to_numpy() >= tau).astype(np.uint8),
    })
    return tabela, resultado
