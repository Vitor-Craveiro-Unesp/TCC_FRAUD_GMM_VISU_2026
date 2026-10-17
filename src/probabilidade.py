"""Passo 05.03: score probabilistico do GMM com riscos soft por componente.

Formula: p(y=1 | x) = soma_k P(Z=k | x) * RH_k.

As responsabilidades para treino/validacao/teste sao produzidas pelo MESMO GMM
ja ajustado em X_treino (NORMAL/RUS). RH_k deve ser previamente estimado
com responsabilidades e rotulos de TODO o treino original do fold.

Este modulo nao ajusta modelos, nao utiliza y_validacao, nao escolhe limiar,
nao calcula MCC e nao aplica correcoes de prevalencia externas.
"""

import numpy as np
import pandas as pd

from src.gmm import GMMNaoSupervisionado
from src.risco import RiscoHistoricoGaussianas, _matriz_responsabilidades


def calcular_probabilidades(gamma, riscos):
    """Compoe probabilidades de fraude a partir de responsabilidades e RH.

    Parameters
    ----------
    gamma : pandas.DataFrame
        Linhas = observacoes; colunas ordenadas G1, G2, ..., GK.
        Cada linha soma 1 e contem valores em [0, 1].
    riscos : pandas.Series
        RH_k indexados por G1, ..., GK, na MESMA ordem de gamma.

    Returns
    -------
    pandas.Series
        Probabilidade de fraude por observacao; preserva indice e ordem.
    """
    matriz = _matriz_responsabilidades(gamma)
    if not isinstance(riscos, pd.Series):
        raise ValueError("Riscos historicos devem ser pandas.Series indexada por gaussiana.")
    if not riscos.index.is_unique or not riscos.index.equals(gamma.columns):
        raise ValueError("Indice/ordem dos riscos nao coincide com as gaussianas G1..GK.")
    if not pd.api.types.is_numeric_dtype(riscos.dtype):
        raise ValueError("Riscos historicos devem ser numericos.")
    valores = riscos.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(valores).all() or (valores < 0).any() or (valores > 1).any():
        raise ValueError("Riscos historicos devem ser finitos e pertencer a [0, 1].")

    score = matriz @ valores
    if not np.isfinite(score).all() or (score < -1e-6).any() or (score > 1 + 1e-6).any():
        raise RuntimeError("Scores de fraude invalidos.")
    return pd.Series(np.clip(score, 0, 1), index=gamma.index, name="prob_fraude")


class PreditorProbabilisticoGMM:
    """Associa o GMM ajustado ao RH do treino original para previsao.

    Contrato: o `risco` DEVE ter sido ajustado com responsabilidades do
    `gmm` fornecido para TODO o treino original. Nao e possivel verificar
    essa proveniencia automaticamente; a etapa de orquestracao a garantira.

    O preditor congela uma copia dos riscos, e bloqueia predicoes caso
    o objeto GMM seja reajustado posteriormente.
    """

    def __init__(self, gmm, risco):
        if not isinstance(gmm, GMMNaoSupervisionado):
            raise TypeError("gmm deve ser GMMNaoSupervisionado do Passo 05.01.")
        if not isinstance(risco, RiscoHistoricoGaussianas):
            raise TypeError("risco deve ser RiscoHistoricoGaussianas do Passo 05.02.")
        if not hasattr(gmm, "modelo_"):
            raise RuntimeError("GMM sem fit; ajuste primeiro com X_treino_gmm.")
        if not hasattr(risco, "riscos_"):
            raise RuntimeError("RH sem fit; estime primeiro com o treino original.")
        if not risco.riscos_.index.equals(pd.Index(gmm.saidas_)):
            raise ValueError("Riscos e GMM possuem gaussianas diferentes ou fora de ordem.")
        if risco.n_treino_ < gmm.n_ajuste_:
            raise ValueError("RH deve utilizar o treino original completo, nao menor que o ajuste GMM.")
        self.gmm_ = gmm
        self.riscos_ = risco.riscos_.copy(deep=True)
        self._modelo_id = id(gmm.modelo_)
        self.n_treino_rh_ = risco.n_treino_
        self.prevalencia_treino_ = risco.prevalencia_treino_

    def prever_probabilidade(self, X):
        """Previsao probabilistica sem labels e sem fit (inclusive para F9/F10)."""
        if id(self.gmm_.modelo_) != self._modelo_id:
            raise RuntimeError("GMM reajustado apos definir RH; recalcule riscos e preditor.")
        gamma = self.gmm_.responsabilidades(X)
        return calcular_probabilidades(gamma, self.riscos_)
