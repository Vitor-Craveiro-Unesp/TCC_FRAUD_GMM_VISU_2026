"""Passo 05.02: risco historico por gaussiana usando soft counts.

Entrada: responsabilidades gamma do GMM para TODO o treino ORIGINAL e seus
rotulos 0/1 na MESMA ordem. Mesmo com RUS, gamma deve incluir o treino todo.

A classe nao ajusta GMM, nao calcula previsoes de validacao, nao usa prior
externo nem otimiza limiares. A composicao do score fica para o Passo 05.03.
"""

import numpy as np
import pandas as pd


def grade_suavizacao(config):
    """Valores de risco_historico.suavizacao (escalar ou grade)."""
    secao = config["risco_historico"]
    valores = secao["suavizacao"]
    valores = valores if isinstance(valores, list) else [valores]
    if not valores:
        raise ValueError("risco_historico.suavizacao: grade vazia.")
    for valor in valores:
        _validar_suavizacao(valor)
        yield float(valor)


def _validar_suavizacao(valor):
    if (isinstance(valor, (bool, np.bool_)) or
            not isinstance(valor, (int, float, np.integer, np.floating)) or
            not np.isfinite(valor) or valor < 0):
        raise ValueError("risco_historico.suavizacao deve ser numero finito >= 0.")


def _matriz_responsabilidades(gamma):
    if not isinstance(gamma, pd.DataFrame) or gamma.empty:
        raise ValueError("Responsabilidades devem ser DataFrame nao vazio.")
    if not gamma.index.is_unique:
        raise ValueError("Indice de responsabilidades deve ser unico.")
    nomes = [f"G{k}" for k in range(1, gamma.shape[1] + 1)]
    if list(gamma.columns) != nomes:
        raise ValueError("Colunas das responsabilidades devem ser G1, G2, ... na ordem.")
    if any(not pd.api.types.is_numeric_dtype(t) for t in gamma.dtypes):
        raise ValueError("Responsabilidades devem conter somente numeros.")
    matriz = gamma.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(matriz).all():
        raise ValueError("Responsabilidades contem NaN ou infinito.")
    if (matriz < 0).any() or (matriz > 1 + 1e-6).any():
        raise ValueError("Responsabilidades devem pertencer a [0, 1].")
    if not np.allclose(matriz.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("Responsabilidades devem somar 1 por observacao.")
    return matriz


def _rotulos(y, indice, n):
    # Series recebe validacao estrita de indices; arrays recebem alinhamento posicional.
    if isinstance(y, pd.Series):
        if not y.index.equals(indice):
            raise ValueError("Indice/ordem de y_treino nao coincide com responsabilidades.")
        y = y.to_numpy(copy=False)
    else:
        y = np.asarray(y)
    if y.ndim != 1 or len(y) != n:
        raise ValueError("y_treino deve ter uma observacao para cada linha do treino original.")
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("y_treino deve conter apenas 0/1 numericos.")
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("y_treino deve conter apenas 0/1, sem ausentes.")
    return y.astype(np.float64, copy=False)


class RiscoHistoricoGaussianas:
    """Estima RH_k de gamma do treino original e y original (sem RUS).

    Sem suavizacao: RH_k = sum_i gamma_ik * y_i / sum_i gamma_ik.

    Com suavizacao s>0, encolhe cada RH_k para a prevalencia EMPIRICA
    original p: (fraudes_soft_k + s * p) / (massa_soft_k + s).
    O valor s nao e uma priori externa; p e calculado de y_treino.

    A classe NAO distingue automaticamente um treino completo de uma amostra
    RUS: a responsabilidade de passar o conjunto completo e do pipeline.
    """

    def __init__(self, config, suavizacao=None):
        secao = config["risco_historico"]
        if secao.get("metodo") != "soft_counts":
            raise ValueError("risco_historico.metodo deve ser soft_counts.")
        if secao.get("estimar_com") != "treino_original":
            raise ValueError("risco_historico.estimar_com deve ser treino_original.")
        if secao.get("previsao") != "soma_ponderada":
            raise ValueError("risco_historico.previsao deve ser soma_ponderada.")
        if config["dados"].get("classe_positiva") != 1:
            raise ValueError("Classe positiva deve ser 1 (fraude).")
        opcoes = list(grade_suavizacao(config))
        if suavizacao is None:
            if len(opcoes) != 1:
                raise ValueError("Grade de suavizacao: escolha uma opcao explicitamente.")
            suavizacao = opcoes[0]
        _validar_suavizacao(suavizacao)
        if float(suavizacao) not in opcoes:
            raise ValueError("Suavizacao escolhida nao consta no YAML.")
        self.suavizacao_ = float(suavizacao)

    def fit(self, gamma_treino_original, y_treino_original):
        """Estima riscos sem modificar gamma/y e sem usar validacao/teste."""
        gamma = _matriz_responsabilidades(gamma_treino_original)
        y = _rotulos(y_treino_original, gamma_treino_original.index, len(gamma))
        massa = gamma.sum(axis=0)
        fraudes = gamma.T @ y
        p = float(y.mean())
        s = self.suavizacao_
        if s == 0 and np.any(massa <= 0):
            raise ValueError("Gaussiana com massa zero: utilize suavizacao > 0.")
        risco = (fraudes + s * p) / (massa + s)
        if not np.isfinite(risco).all() or ((risco < -1e-10) | (risco > 1 + 1e-10)).any():
            raise RuntimeError("Riscos por gaussiana invalidos.")
        colunas = gamma_treino_original.columns
        self.riscos_ = pd.Series(np.clip(risco, 0.0, 1.0), index=colunas, name="risco_historico")
        self.massas_ = pd.Series(massa, index=colunas, name="massa_efetiva")
        self.fraudes_ponderadas_ = pd.Series(fraudes, index=colunas, name="fraudes_ponderadas")
        self.prevalencia_treino_ = p
        self.n_treino_ = len(gamma)
        self.n_fraudes_treino_ = int(y.sum())
        return self

    def tabela(self):
        """Tabela descritiva, com identificadores G1...Gk preservados."""
        if not hasattr(self, "riscos_"):
            raise RuntimeError("Chame fit() antes de tabela().")
        return pd.concat(
            (self.massas_, self.fraudes_ponderadas_, self.riscos_), axis=1
        ).rename_axis("gaussiana")
