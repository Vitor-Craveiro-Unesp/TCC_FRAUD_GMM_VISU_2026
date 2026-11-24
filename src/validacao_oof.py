"""Passo 06.01: motor de previsoes Out-of-Fold do TCC_EXPERIMENTAL_V3.

Esta e a unidade estatistica/testavel da validacao cruzada; NAO e ainda o
orquestrador de execucoes longas. Nao grava resultados, nao paraleliza e nao
ignora silenciosamente flags operacionais: persistencia, cache, retomada,
paralelismo e selecao por metricas serao implementados no Passo 06.02/06.03.

Em cada F1..F9, fit de encoding, imputacao, reducao, GMM e RH somente nos
outros 8 folds. GMM nunca recebe labels. RH recebe TODOS os labels originais
do treino, mesmo com RUS. O F10 e sempre excluido de ajuste e validacao OOF.
"""
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd

from src.preprocessing import selecionar_features, EncodersPorSufixo, ImputadorPadronizador
from src.amostragem import selecionar_indices_gmm
from src.reducao import RedutorPCA, grade_pca
from src.reducao_tsne import RedutorTSNE, grade_tsne
from src.gmm import GMMNaoSupervisionado, grade_gmm
from src.risco import RiscoHistoricoGaussianas, grade_suavizacao
from src.probabilidade import PreditorProbabilisticoGMM


def _lista(valor):
    return valor if isinstance(valor, list) else [valor]


def grade_candidatos_oof(config):
    """Grade metodologica (nao inclui opcoes visuais nem operacionais).

    Retorna combinacoes unicas de reducao, amostragem, GMM e RH. A razao RUS
    nao cria variantes duplicadas para NORMAL.
    """
    metodo = str(config["reducao"]["metodo"]).upper()
    if metodo == "PCA":
        reducoes = grade_pca(config)
    elif metodo == "TSNE":
        reducoes = grade_tsne(config)
    else:
        raise ValueError("reducao.metodo deve ser PCA ou TSNE.")
    modos = _lista(config["amostragem"]["modo"])
    razoes = _lista(config["amostragem"]["razao_negativos_positivos"])
    if not modos or not razoes:
        raise ValueError("A grade de amostragem nao pode ser vazia.")
    if config["amostragem"].get("aplicar_somente_treino") is not True:
        raise ValueError("A amostragem deve ocorrer somente no treino.")
    modos_razoes = []
    for modo in modos:
        if modo == "NORMAL":
            modos_razoes.append(("NORMAL", None))
        elif modo == "RUS":
            for razao in razoes:
                if isinstance(razao, (bool, np.bool_)) or not isinstance(razao, (int, np.integer)) or razao < 1:
                    raise ValueError("Razao RUS deve ser inteiro positivo.")
                modos_razoes.append(("RUS", int(razao)))
        else:
            raise ValueError(f"Modo de amostragem nao suportado: {modo!r}.")
    vistos = set()
    for red, (modo, razao), gmm, suav in product(
        reducoes, modos_razoes, grade_gmm(config), grade_suavizacao(config)
    ):
        # product materializa os iteraveis finitos, sem altera-los.
        c = {"metodo": metodo, "reducao": dict(red), "modo": modo,
             "razao": razao, "gmm": dict(gmm), "suavizacao": float(suav)}
        chave = repr(c)
        if chave not in vistos:
            vistos.add(chave)
            yield c


def validar_divisao(folds, config, n_linhas, *, exigir_todos=True):
    """Retorna vetor numerico 1..10; bloqueia qualquer uso de F10 na CV."""
    d = config["divisao"]
    cv = config["validacao_cruzada"]
    if (d["total_folds"] != 10 or d["quantidade_folds_teste"] != 1
            or d["qual_fold_teste"] != 10):
        raise ValueError("Este projeto exige dez folds e F10 como teste reservado.")
    if cv.get("estrategia") != "rotacao_folds_restantes" or cv.get("evitar_vazamento") is not True:
        raise ValueError("A CV deve ser rotacao_folds_restantes com evitar_vazamento=true.")
    a = np.asarray(folds)
    if a.ndim != 1 or len(a) != n_linhas or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("Vetor de folds inteiro com tamanho igual a base e obrigatorio.")
    if not np.isin(a, np.arange(1, 11)).all():
        raise ValueError("Folds devem pertencer a F1..F10.")
    if exigir_todos and set(np.unique(a)) != set(range(1, 11)):
        raise ValueError("A execucao OOF completa exige F1..F10 presentes.")
    return a


def indices_do_fold(folds, fold_validacao):
    """Treino=F1..F9 exceto a validacao; F10 nao pertence a nenhum deles."""
    if isinstance(fold_validacao, (bool, np.bool_)) or not isinstance(
        fold_validacao, (int, np.integer)
    ) or not 1 <= fold_validacao <= 9:
        raise ValueError("O fold de validacao deve ser inteiro entre 1 e 9; nunca F10.")
    treino = np.flatnonzero((folds >= 1) & (folds <= 9) & (folds != fold_validacao))
    validacao = np.flatnonzero(folds == fold_validacao)
    if not len(treino) or not len(validacao):
        raise ValueError("Treino/validacao vazios para o fold solicitado.")
    if np.intersect1d(treino, validacao).size:
        raise AssertionError("Sobreposicao entre treino e validacao.")
    return treino, validacao


def _validar_candidato(candidato, config):
    if not isinstance(candidato, dict):
        raise ValueError("Forneca um candidato retornado por grade_candidatos_oof.")
    if candidato not in list(grade_candidatos_oof(config)):
        raise ValueError("O candidato nao pertence a grade do YAML atual.")


def prever_fold_oof(df, folds, config, candidato, fold_validacao):
    """Treina uma rodada e retorna scores em ordem POSICIONAL original.

    df precisa estar alinhado aos folds por posicao (index 0..n-1). A amostra
    reduzida do verificador pode conter so F1..F9; o motor nao a confunde
    com uma execucao OOF completa. Nao recebe/reusa componentes prefit.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError("df deve ser DataFrame nao vazio com indice posicional 0..n-1.")
    a = validar_divisao(folds, config, len(df), exigir_todos=False)
    _validar_candidato(candidato, config)
    alvo = config["dados"]["coluna_alvo"]
    if alvo not in df.columns:
        raise ValueError(f"Coluna alvo ausente: {alvo}.")
    y = df[alvo].to_numpy()
    if not np.issubdtype(y.dtype, np.number) or not np.isin(y, [0, 1]).all():
        raise ValueError("Alvo precisa conter apenas 0/1 sem ausentes.")

    indice_t, indice_v = indices_do_fold(a, fold_validacao)
    y_t = df.iloc[indice_t][alvo].copy()
    if (y_t == 1).sum() == 0 or (y_t == 0).sum() == 0:
        raise ValueError("Treino do fold requer exemplos de ambas as classes.")
    # Encarna apenas as particoes da rodada; F10 nao entra em nenhum objeto fit.
    x_t = selecionar_features(df.iloc[indice_t], config)
    x_v = selecionar_features(df.iloc[indice_v], config)

    encoder = EncodersPorSufixo(config).fit(x_t)
    t_enc = encoder.transform(x_t)
    v_enc = encoder.transform(x_v)
    imputador = ImputadorPadronizador(config, encoder).fit(t_enc)
    t_pre = imputador.transform(t_enc)
    v_pre = imputador.transform(v_enc)
    del x_t, x_v, t_enc, v_enc

    if candidato["metodo"] == "PCA":
        redutor = RedutorPCA(config, candidato["reducao"]).fit(t_pre)
        t_red = redutor.transform(t_pre)
        v_red = redutor.transform(v_pre)
    elif candidato["metodo"] == "TSNE":
        redutor = RedutorTSNE(config, candidato["reducao"]).fit(t_pre)
        t_red = redutor.transform_treino()
        v_red = redutor.transform(v_pre)
    else:
        raise ValueError("Redutor nao suportado.")
    del t_pre, v_pre

    amostra = selecionar_indices_gmm(
        y_t.to_numpy(), config, modo=candidato["modo"],
        razao=candidato["razao"], fold_validacao=int(fold_validacao),
    )
    modelo = GMMNaoSupervisionado(config, candidato["gmm"]).fit(t_red.iloc[amostra])
    # RH SEMPRE usa o conjunto de treino inteiro, nao o subconjunto RUS.
    gamma_t = modelo.responsabilidades(t_red)
    riscos = RiscoHistoricoGaussianas(config, candidato["suavizacao"]).fit(gamma_t, y_t)
    preditor = PreditorProbabilisticoGMM(modelo, riscos)
    scores = preditor.prever_probabilidade(v_red)
    if not scores.index.equals(df.index[indice_v]):
        raise RuntimeError("A ordem dos scores da validacao foi alterada.")
    return pd.DataFrame({
        "linha_original": indice_v.astype(np.int64),
        "fold": np.full(len(indice_v), fold_validacao, dtype=np.uint8),
        "y_true": df.iloc[indice_v][alvo].to_numpy(dtype=np.uint8),
        "prob_fraude": scores.to_numpy(dtype=np.float64),
    })


def executar_oof(df, folds, config, candidato):
    """Nove rodadas CV, um score OOF por observacao de F1..F9.

    Funcao IN-MEMORY para testar rigor/alinhamento. NAO deve ser chamada com
    o dataset completo antes do orquestrador de disco, RAM e retomada.
    """
    a = validar_divisao(folds, config, len(df), exigir_todos=True)
    saidas = [prever_fold_oof(df, a, config, candidato, k) for k in range(1, 10)]
    resultado = pd.concat(saidas, ignore_index=True).sort_values("linha_original")
    resultado = resultado.reset_index(drop=True)
    esperado = np.flatnonzero(a != 10)
    if not np.array_equal(resultado["linha_original"].to_numpy(), esperado):
        raise RuntimeError("OOF incompleto, duplicado ou com F10 incluido.")
    if not np.isfinite(resultado["prob_fraude"]).all() or not resultado["prob_fraude"].between(0, 1).all():
        raise RuntimeError("Scores OOF invalidos.")
    return resultado
