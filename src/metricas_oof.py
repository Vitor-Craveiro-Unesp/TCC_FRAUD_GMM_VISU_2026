"""Passo 06.02: metricas probabilisticas e limiar MCC para predicoes OOF.

Nao treina, nao escolhe hiperparametros, nao grava resultados e NAO toca o F10.
O ajuste do limiar deve usar a consolidacao das nove rodadas F1..F9.

Uma avaliacao isolada, por exemplo F9 de uma AMOSTRA, e apenas um smoke test;
nao representa a avaliacao OOF completa nem seu limiar definitivo.
"""
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, matthews_corrcoef

from src.validacao_oof import validar_divisao


_FORMULA = "(AP+(MCC+1)/2+exp(-LogLoss))/3"


def _validar_config(config):
    if not isinstance(config, dict) or not isinstance(config.get("avaliacao"), dict):
        raise ValueError("O YAML precisa conter a secao avaliacao.")
    a = config["avaliacao"]
    if a.get("metrica_pr") != "average_precision":
        raise ValueError("avaliacao.metrica_pr deve ser average_precision (nao AUC trapezoidal).")
    for item in ("calcular_mcc", "calcular_logloss", "calcular_composite"):
        if a.get(item) is not True:
            raise ValueError(f"avaliacao.{item} deve ser true para este passo.")
    if a.get("selecao_limiar") != "mcc_oof":
        raise ValueError("avaliacao.selecao_limiar deve ser mcc_oof.")
    formula = re.sub(r"\s+", "", str(a.get("formula_composite", "")))
    if formula != _FORMULA:
        raise ValueError("avaliacao.formula_composite diferente da formula implementada.")
    if a.get("congelar_modelo_antes_teste") is not True:
        raise ValueError("avaliacao.congelar_modelo_antes_teste deve ser true.")


def _vetores(y_true, prob_fraude):
    y = np.asarray(y_true)
    p = np.asarray(prob_fraude)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("y_true e prob_fraude devem ser vetores 1D, nao vazios e de mesmo tamanho.")
    if not np.issubdtype(y.dtype, np.number) and y.dtype != np.dtype("bool"):
        raise ValueError("y_true deve conter apenas rotulos numericos 0/1.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true deve conter apenas 0/1, sem ausentes.")
    if not np.issubdtype(p.dtype, np.number):
        raise ValueError("prob_fraude precisa conter valores numericos.")
    p = p.astype(np.float64)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("prob_fraude deve conter probabilidades finitas em [0,1].")
    y = y.astype(np.uint8)
    if int(y.sum()) == 0 or int(y.sum()) == len(y):
        raise ValueError("A avaliacao requer exemplos de ambas as classes 0 e 1.")
    return y, p


def otimizar_limiar_mcc(y_true, prob_fraude):
    """Busca EXATA dos limiares com regra pred=1 quando prob>=limiar.

    Ordena uma unica vez (O(n log n)) e agrega empates: todos os registros
    de mesmo score recebem a mesma classe. Inclui a referencia 0.5 e a
    opcao prever nenhum positivo (limiar imediatamente acima do maior score).
    Para empate exato de MCC, prefere o limiar mais proximo de 0.5;
    em novo empate, prefere o maior limiar. Deterministico.
    """
    y, p = _vetores(y_true, prob_fraude)
    ordem = np.argsort(p)[::-1]
    ordenados = p[ordem]
    labels = y[ordem]
    # Fim de cada bloco de scores identicos. Nenhum empate e dividido.
    fins = np.flatnonzero(np.r_[ordenados[:-1] != ordenados[1:], True])
    n_pos_preditos = fins.astype(np.int64) + 1
    verdadeiros_positivos = np.cumsum(labels, dtype=np.int64)[fins]
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    # MCC = (N*TP - n_pos*n_pred_pos) / sqrt(n_pred_pos*n_pos*n_neg*(N-n_pred_pos))
    numerador = n * verdadeiros_positivos - n_pos * n_pos_preditos
    denominador = np.sqrt(
        n_pos_preditos.astype(np.float64) * n_pos * n_neg * (n - n_pos_preditos)
    )
    mcc = np.divide(
        numerador, denominador, out=np.zeros(len(fins), dtype=np.float64),
        where=denominador > 0,
    )

    mascara05 = p >= 0.5
    n05 = int(np.count_nonzero(mascara05))
    tp05 = int(np.count_nonzero(y[mascara05] == 1))
    denominador05 = np.sqrt(float(n05) * n_pos * n_neg * (n - n05))
    mcc05 = (n * tp05 - n_pos * n05) / denominador05 if denominador05 else 0.0

    # Admite limiar >1 apenas se o maior score for 1, para tornar possivel
    # a decisao "nenhum positivo" com comparacao inclusiva >=.
    acima_maximo = float(np.nextafter(ordenados[0], np.inf))
    limiares = np.r_[acima_maximo, ordenados[fins], 0.5]
    valores_mcc = np.r_[0.0, mcc, mcc05]
    melhor = float(np.max(valores_mcc))
    empatados = np.flatnonzero((valores_mcc == melhor))
    # Primeiro: menor distancia a 0.5; segundo: maior limiar.
    ordenacao = np.lexsort((-limiares[empatados], np.abs(limiares[empatados] - 0.5)))
    limiar = float(limiares[empatados[ordenacao[0]]])
    mcc_final = float(matthews_corrcoef(y, (p >= limiar).astype(np.uint8)))
    return {"limiar_mcc": limiar, "mcc": mcc_final,
            "total_limiares_avaliados": int(len(limiares))}


def avaliar_probabilidades(y_true, prob_fraude, config):
    """AP sklearn, MCC maximizado, LogLoss e score composto; sem I/O."""
    _validar_config(config)
    y, p = _vetores(y_true, prob_fraude)
    limiar = otimizar_limiar_mcc(y, p)
    ap = float(average_precision_score(y, p))
    ll = float(log_loss(y, p, labels=[0, 1]))
    composite = float((ap + (limiar["mcc"] + 1.0) / 2.0 + np.exp(-ll)) / 3.0)
    return {
        "n": int(len(y)), "n_legitimas": int(len(y) - y.sum()),
        "n_fraudes": int(y.sum()), "prevalencia": float(np.mean(y)),
        "ap": ap, "mcc": limiar["mcc"], "logloss": ll,
        "score": composite, "limiar_mcc": limiar["limiar_mcc"],
        "total_limiares_avaliados": limiar["total_limiares_avaliados"],
        "mcc_limiar_05": float(matthews_corrcoef(y, (p >= 0.5).astype(np.uint8))),
    }


def avaliar_oof_consolidado(predicoes, folds, config, *, y_referencia=None):
    """So aceita OOF COMPLETO de F1..F9, uma predicao por linha de desenvolvimento.

    Opcional y_referencia e vetor de rotulos da base original, apenas para
    verificar alinhamento nos indices de desenvolvimento (nunca avalia F10).
    """
    a = validar_divisao(folds, config, len(folds), exigir_todos=True)
    if not isinstance(predicoes, pd.DataFrame):
        raise ValueError("predicoes deve ser um DataFrame OOF.")
    necessarias = {"linha_original", "fold", "y_true", "prob_fraude"}
    if not necessarias.issubset(predicoes.columns):
        raise ValueError(f"Predicoes OOF precisam das colunas: {sorted(necessarias)}.")
    ids = predicoes["linha_original"].to_numpy()
    f_pred = predicoes["fold"].to_numpy()
    if (not np.issubdtype(ids.dtype, np.integer)
            or not np.issubdtype(f_pred.dtype, np.integer)):
        raise ValueError("linha_original e fold devem ser inteiros.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Predicoes OOF possuem indices duplicados.")
    esperado = np.flatnonzero(a != 10)
    # Ordenar indices posicionais da base. Impede omissao, F10 e extra.
    ordem = np.argsort(ids)
    if not np.array_equal(ids[ordem], esperado):
        raise ValueError("OOF incompleto ou com F10/indices indevidos; esperado F1..F9 completo.")
    if not np.array_equal(f_pred[ordem], a[esperado]):
        raise ValueError("Identificacao dos folds nao corresponde aos indices da base.")
    y_ordenado = predicoes["y_true"].to_numpy()[ordem]
    p_ordenado = predicoes["prob_fraude"].to_numpy()[ordem]
    if y_referencia is not None:
        y_ref = np.asarray(y_referencia)
        if y_ref.ndim != 1 or len(y_ref) != len(a):
            raise ValueError("y_referencia precisa conter uma linha por registro da base.")
        if not np.array_equal(y_ordenado, y_ref[esperado]):
            raise ValueError("y_true OOF nao corresponde aos rotulos originais da base.")
    resultado = avaliar_probabilidades(y_ordenado, p_ordenado, config)
    resultado["tipo_avaliacao"] = "OOF_9_FOLDS"
    resultado["quantidade_folds_cv"] = 9
    return resultado
