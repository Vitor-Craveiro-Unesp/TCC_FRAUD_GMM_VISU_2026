"""Passo 04.01: reducao PCA ajustada somente no treino de cada fold.

Entrada: DataFrame numerico de saida do passo 03.02 (sem alvo ou ID).
Saida: DataFrame PC1, PC2, ..., mantendo indice e ordem das linhas.
Este modulo NAO ajusta t-SNE, GMM, nem usa rotulos.
"""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


_PARAMETROS_PCA = ("n_components", "whiten", "svd_solver", "iterated_power")
_SOLVERS = {"auto", "full", "randomized", "covariance_eigh", "arpack"}


def grade_pca(config):
    """Gera o produto cartesiano dos quatro parametros PCA ativos no YAML.

    Nao inclui listas de outras secoes (p.ex. t-SNE ou visualizacao).
    """
    if config.get("reducao", {}).get("metodo", "").upper() != "PCA":
        raise ValueError("grade_pca requer reducao.metodo: PCA no YAML.")
    configuracao = config["pca"]
    opcoes = []
    for nome in _PARAMETROS_PCA:
        valor = configuracao[nome]
        valores = valor if isinstance(valor, list) else [valor]
        if not valores:
            raise ValueError(f"pca.{nome} possui lista vazia.")
        opcoes.append(valores)
    for combinacao in product(*opcoes):
        yield dict(zip(_PARAMETROS_PCA, combinacao))


class RedutorPCA:
    """PCA por fold: fit somente no treino e transform sem reaprender.

    Para o YAML em grade, use `for params in grade_pca(config)` e instancie
    `RedutorPCA(config, params)` para cada combinacao.
    """

    def __init__(self, config, parametros=None):
        self.config = config
        if config.get("reducao", {}).get("metodo", "").upper() != "PCA":
            raise ValueError("RedutorPCA requer reducao.metodo: PCA.")
        if parametros is None:
            grade = list(grade_pca(config))
            if len(grade) != 1:
                raise ValueError(
                    "O YAML possui varios hiperparametros PCA. "
                    "Forneca uma combinacao obtida de grade_pca(config)."
                )
            parametros = grade[0]
        if not isinstance(parametros, dict) or set(parametros) != set(_PARAMETROS_PCA):
            raise ValueError("Parametros PCA devem conter exatamente as quatro chaves previstas.")
        self.parametros = parametros.copy()
        self._validar_parametros()

    def _validar_parametros(self):
        p = self.parametros
        if isinstance(p["n_components"], (bool, np.bool_)) or not isinstance(
            p["n_components"], (int, np.integer)
        ) or p["n_components"] < 1:
            raise ValueError("pca.n_components deve ser inteiro positivo.")
        if not isinstance(p["whiten"], (bool, np.bool_)):
            raise ValueError("pca.whiten deve ser booleano.")
        if p["svd_solver"] not in _SOLVERS:
            raise ValueError(f"pca.svd_solver invalido: {p['svd_solver']}")
        ip = p["iterated_power"]
        if not (ip == "auto" or (isinstance(ip, (int, np.integer)) and
                                 not isinstance(ip, (bool, np.bool_)) and ip >= 0)):
            raise ValueError("pca.iterated_power deve ser 'auto' ou inteiro >= 0.")

    @staticmethod
    def _validar_X(X):
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("PCA requer DataFrame numerico nao vazio.")
        if not X.columns.is_unique:
            raise ValueError("PCA nao permite nomes de features duplicados.")
        if not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
            raise ValueError("PCA requer somente features numericas pos-processadas.")
        valores = X.to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(valores).all():
            raise ValueError("PCA recebeu NaN ou infinito; execute o passo 03.02.")
        return valores

    def fit(self, X):
        X_np = self._validar_X(X)
        n_linhas, n_features = X_np.shape
        minimo = min(n_linhas, n_features)
        k = int(self.parametros["n_components"])
        if k > minimo or (self.parametros["svd_solver"] == "arpack" and k >= minimo):
            raise ValueError(
                f"PCA n_components={k} incompativel com dimensoes "
                f"{X_np.shape} e solver={self.parametros['svd_solver']}."
            )
        # Nunca faz amostragem aqui: entrada deve ser TODO o treino original do fold.
        modelo = PCA(
            n_components=k,
            whiten=bool(self.parametros["whiten"]),
            svd_solver=self.parametros["svd_solver"],
            iterated_power=self.parametros["iterated_power"],
            random_state=int(self.config["experimento"]["semente"]),
        )
        modelo.fit(X_np)
        self.modelo_ = modelo
        self.colunas_ = list(X.columns)
        self.n_treino_ = len(X)
        self.saidas_ = [f"PC{i + 1}" for i in range(k)]
        return self

    def transform(self, X):
        if not hasattr(self, "modelo_"):
            raise RuntimeError("Chame fit(X_treino) antes de transform().")
        X_np = self._validar_X(X)
        if list(X.columns) != self.colunas_:
            raise ValueError("Colunas ou ordem diferentes do treinamento no PCA.")
        coords = self.modelo_.transform(X_np)
        if not np.isfinite(coords).all():
            raise ValueError("PCA produziu coordenadas nao finitas.")
        return pd.DataFrame(
            np.asarray(coords, dtype=np.float32),
            columns=self.saidas_, index=X.index,
        )

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_feature_names_out(self):
        if not hasattr(self, "saidas_"):
            raise RuntimeError("Chame fit antes de obter nomes das componentes.")
        return np.array(self.saidas_, dtype=object)
