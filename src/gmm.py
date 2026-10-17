"""Passo 05.01: GMM unico nao supervisionado no espaco PCA ou t-SNE.

Entrada: DataFrame numerico reduzido (saida do Passo 04).
Ajuste: SOMENTE X escolhido para o GMM (NORMAL ou RUS), SEM rotulos.
Saida: responsabilidades soft por gaussiana, preservando indices das linhas.

NAO estima risco por componente, nao usa y, nao seleciona limiar e nao
reajusta o GMM ao prever validacao/teste. Esses trabalhos serao feitos depois.
"""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


_PARAMETROS = (
    "n_components", "covariance_type", "reg_covar", "max_iter",
    "n_init", "tol", "init_params",
)
_COVARIANCIAS = {"full", "diag", "tied", "spherical"}
_INICIALIZACOES = {"kmeans", "k-means++", "random", "random_from_data"}


def grade_gmm(config):
    """Produto cartesiano APENAS dos hiperparametros ativos de gmm."""
    opcoes = []
    secao = config["gmm"]
    for chave in _PARAMETROS:
        valor = secao[chave]
        valores = valor if isinstance(valor, list) else [valor]
        if not valores:
            raise ValueError(f"gmm.{chave}: lista vazia.")
        opcoes.append(valores)
    for valores in product(*opcoes):
        yield dict(zip(_PARAMETROS, valores))


def _inteiro_positivo(valor, nome):
    if isinstance(valor, (bool, np.bool_)) or not isinstance(
        valor, (int, np.integer)
    ) or valor <= 0:
        raise ValueError(f"gmm.{nome} deve ser inteiro positivo.")


def _real_positivo(valor, nome):
    if isinstance(valor, (bool, np.bool_)) or not isinstance(
        valor, (float, int, np.floating, np.integer)
    ) or not np.isfinite(valor) or valor <= 0:
        raise ValueError(f"gmm.{nome} deve ser numero positivo finito.")


class GMMNaoSupervisionado:
    """Wrapper por fold em torno de um unico sklearn GaussianMixture.

    A escolha NORMAL/RUS acontece FORA desta classe, via
    `selecionar_indices_gmm(y_treino, ...)` do Passo 03.03.
    Nunca passar y para `fit`. `responsabilidades(X)` serve para calcular,
    posteriormente, riscos soft usando TODO o treino original do fold.
    """

    def __init__(self, config, parametros=None):
        self.config = config
        if parametros is None:
            opcoes = list(grade_gmm(config))
            if len(opcoes) != 1:
                raise ValueError(
                    "O YAML possui uma grade GMM. Escolha uma combinacao "
                    "de grade_gmm(config)."
                )
            parametros = opcoes[0]
        if not isinstance(parametros, dict) or set(parametros) != set(_PARAMETROS):
            raise ValueError("Parametros GMM devem conter as sete chaves do YAML.")
        self.parametros = parametros.copy()
        self._validar_parametros()

    def _validar_parametros(self):
        p = self.parametros
        for chave in ("n_components", "max_iter", "n_init"):
            _inteiro_positivo(p[chave], chave)
        for chave in ("reg_covar", "tol"):
            _real_positivo(p[chave], chave)
        if p["covariance_type"] not in _COVARIANCIAS:
            raise ValueError("gmm.covariance_type invalido.")
        if p["init_params"] not in _INICIALIZACOES:
            raise ValueError("gmm.init_params invalido.")
        semente = self.config["experimento"]["semente"]
        if isinstance(semente, bool) or not isinstance(semente, (int, np.integer)) or semente < 0:
            raise ValueError("experimento.semente deve ser inteiro nao negativo.")

    def _validar_X(self, X):
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("GMM requer DataFrame numerico nao vazio.")
        if not X.columns.is_unique or not all(isinstance(c, str) for c in X.columns):
            raise ValueError("GMM requer nomes de colunas unicos e textuais.")
        proibidas = [
            self.config["dados"]["coluna_alvo"],
            self.config["dados"].get("coluna_id"),
            *self.config["dados"].get("colunas_excluir", []),
        ]
        if set(X.columns).intersection(c for c in proibidas if c):
            raise ValueError("Alvo, ID ou colunas excluidas foram enviados ao GMM.")
        if any(not pd.api.types.is_numeric_dtype(tipo) for tipo in X.dtypes):
            raise ValueError("GMM requer somente colunas numericas pos-reducao.")
        matriz = X.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(matriz).all():
            raise ValueError("GMM recebeu NaN ou infinito.")
        return matriz

    def fit(self, X):
        """Ajusta um GMM sem labels. X deve ser somente treino NORMAL/RUS."""
        matriz = self._validar_X(X)
        n = len(X)
        k = int(self.parametros["n_components"])
        if k > n:
            raise ValueError(f"GMM K={k} excede n_treino_gmm={n}.")
        modelo = GaussianMixture(
            n_components=k,
            covariance_type=self.parametros["covariance_type"],
            reg_covar=float(self.parametros["reg_covar"]),
            max_iter=int(self.parametros["max_iter"]),
            n_init=int(self.parametros["n_init"]),
            tol=float(self.parametros["tol"]),
            init_params=self.parametros["init_params"],
            random_state=int(self.config["experimento"]["semente"]),
        )
        modelo.fit(matriz)
        if not modelo.converged_:
            raise RuntimeError(
                "GMM nao convergiu dentro de gmm.max_iter; ajuste a configuracao "
                "antes de usar o resultado."
            )
        self.modelo_ = modelo
        self.colunas_ = list(X.columns)
        self.saidas_ = [f"G{i + 1}" for i in range(k)]
        self.n_ajuste_ = n
        return self

    def responsabilidades(self, X):
        """Probabilidades P(Z=k|x) para TODO treino, validacao ou teste.

        Nao reajusta parametros e nao usa rotulos nem hard assignment.
        """
        if not hasattr(self, "modelo_"):
            raise RuntimeError("Chame fit(X_treino_gmm) antes de responsabilidades().")
        matriz = self._validar_X(X)
        if list(X.columns) != self.colunas_:
            raise ValueError("Colunas ou ordem diferentes das do ajuste GMM.")
        gamma = self.modelo_.predict_proba(matriz)
        if gamma.shape != (len(X), len(self.saidas_)) or not np.isfinite(gamma).all():
            raise RuntimeError("Responsabilidades GMM com formato ou valores invalidos.")
        if np.any(gamma < -1e-12) or not np.allclose(gamma.sum(axis=1), 1.0, atol=1e-7):
            raise RuntimeError("Responsabilidades devem ser nao negativas e somar 1.")
        return pd.DataFrame(
            gamma.astype(np.float32, copy=False), columns=self.saidas_, index=X.index
        )
