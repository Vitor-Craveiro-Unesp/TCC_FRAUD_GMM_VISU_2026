"""Passo 04.02: t-SNE openTSNE em treino, com insercao de novos dados.

Entrada: DataFrame numerico ja codificado/imputado/escalado pelo Passo 03.
Nao aplica PCA previamente: 'pca' no YAML e apenas a inicializacao interna
opcional do t-SNE. Nunca recebe alvo ou ajusta parametros na validacao.

Diferenca de interfaces:
    .fit(X_treino) -> instancia treinada
    .transform_treino() -> coordenadas aprendidas durante o fit
    .transform(X_validacao) -> posiciona novos dados, sem reotimizar o treino
    .fit_transform(X_treino) -> fit + coordenadas de treino
"""

from itertools import product
import numpy as np
import pandas as pd

_PARAMETROS = (
    "perplexity", "n_components", "initialization", "metric",
    "learning_rate", "early_exaggeration", "n_iter", "theta",
)
_INICIOS = {"random", "pca"}
_METRICAS = {"euclidean", "cosine", "manhattan", "correlation"}


def grade_tsne(config):
    """Expande apenas os 8 hiperparametros t-SNE ativos do YAML."""
    if str(config.get("reducao", {}).get("metodo", "")).upper() != "TSNE":
        raise ValueError("grade_tsne requer reducao.metodo: TSNE.")
    opcoes = []
    for chave in _PARAMETROS:
        valor = config["tsne"][chave]
        valores = valor if isinstance(valor, list) else [valor]
        if not valores:
            raise ValueError(f"tsne.{chave}: lista vazia.")
        opcoes.append(valores)
    for combinacao in product(*opcoes):
        yield dict(zip(_PARAMETROS, combinacao))


def _inteiro_positivo(valor):
    return (isinstance(valor, (int, np.integer))
            and not isinstance(valor, (bool, np.bool_)) and valor >= 1)


def _numero_positivo(valor):
    return (isinstance(valor, (int, float, np.integer, np.floating))
            and not isinstance(valor, (bool, np.bool_))
            and np.isfinite(valor) and valor > 0)


class RedutorTSNE:
    """t-SNE sobre os dados de treinamento e extensao OOS na validacao.

    Cada instancia e um fold/configuracao. A funcao transform() nao chama
    fit(), nao modifica o embedding de treinamento e nao usa rotulos.
    """

    def __init__(self, config, parametros=None):
        if str(config.get("reducao", {}).get("metodo", "")).upper() != "TSNE":
            raise ValueError("RedutorTSNE requer reducao.metodo: TSNE.")
        self.config = config
        if parametros is None:
            opcoes = list(grade_tsne(config))
            if len(opcoes) != 1:
                raise ValueError(
                    "Grade t-SNE possui varias configuracoes. "
                    "Forneca parametros de grade_tsne(config)."
                )
            parametros = opcoes[0]
        if not isinstance(parametros, dict) or set(parametros) != set(_PARAMETROS):
            raise ValueError("Forneca os oito parametros t-SNE previstos no YAML.")
        self.parametros = parametros.copy()
        self._validar_parametros()

    def _validar_parametros(self):
        p = self.parametros
        if not _numero_positivo(p["perplexity"]):
            raise ValueError("tsne.perplexity deve ser um numero positivo.")
        if not _inteiro_positivo(p["n_components"]) or p["n_components"] > 3:
            raise ValueError("tsne.n_components deve ser inteiro de 1 a 3.")
        if p["initialization"] not in _INICIOS:
            raise ValueError("tsne.initialization deve ser random ou pca.")
        if p["metric"] not in _METRICAS:
            raise ValueError("tsne.metric nao suportada por este modulo.")
        if p["learning_rate"] != "auto" and not _numero_positivo(p["learning_rate"]):
            raise ValueError("tsne.learning_rate deve ser auto ou numero positivo.")
        if not _numero_positivo(p["early_exaggeration"]):
            raise ValueError("tsne.early_exaggeration deve ser positivo.")
        if not _inteiro_positivo(p["n_iter"]):
            raise ValueError("tsne.n_iter deve ser inteiro positivo.")
        if (not isinstance(p["theta"], (int, float, np.integer, np.floating))
                or isinstance(p["theta"], (bool, np.bool_))
                or not np.isfinite(p["theta"]) or not 0 <= p["theta"] <= 1):
            raise ValueError("tsne.theta deve estar entre 0 e 1.")
        semente = self.config["experimento"]["semente"]
        if not isinstance(semente, int) or isinstance(semente, bool):
            raise ValueError("experimento.semente deve ser inteiro.")
        jobs = self.config.get("execucao", {}).get("n_jobs", 1)
        if not isinstance(jobs, int) or isinstance(jobs, bool) or jobs == 0 or jobs < -1:
            raise ValueError("execucao.n_jobs deve ser -1 ou inteiro positivo.")

    @staticmethod
    def _validar_X(X):
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("t-SNE requer DataFrame numerico nao vazio.")
        if not X.columns.is_unique:
            raise ValueError("t-SNE nao permite nomes de features duplicados.")
        if not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
            raise ValueError("t-SNE requer somente features numericas pos-processadas.")
        valores = X.to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(valores).all():
            raise ValueError("t-SNE recebeu NaN ou infinito; execute o Passo 03.02.")
        return valores

    def fit(self, X):
        """Calcula o embedding apenas com X_treino; nao usa PCA externo."""
        X_np = self._validar_X(X)
        p = self.parametros
        if len(X_np) < 4 or p["perplexity"] >= len(X_np):
            raise ValueError("t-SNE: perplexity deve ser menor que n_treino (>=4).")
        if p["initialization"] == "pca" and p["n_components"] > X_np.shape[1]:
            raise ValueError("Inicializacao pca exige n_components <= n_features.")
        try:
            from openTSNE import TSNE
        except ImportError as exc:
            raise ImportError(
                "Dependencia ausente: execute 'python -m pip install openTSNE' "
                "no .venv do projeto."
            ) from exc
        modelo = TSNE(
            perplexity=float(p["perplexity"]),
            n_components=int(p["n_components"]),
            initialization=p["initialization"],
            metric=p["metric"],
            learning_rate=p["learning_rate"],
            early_exaggeration=float(p["early_exaggeration"]),
            n_iter=int(p["n_iter"]),
            theta=float(p["theta"]),
            random_state=int(self.config["experimento"]["semente"]),
            n_jobs=int(self.config.get("execucao", {}).get("n_jobs", 1)),
        )
        embedding = modelo.fit(X_np)
        coords = np.asarray(embedding, dtype=np.float32)
        formato = (len(X), int(p["n_components"]))
        if coords.shape != formato or not np.isfinite(coords).all():
            raise ValueError("t-SNE produziu embedding de treino invalido.")
        # Somente apos sucesso publicar o estado da instancia.
        self.modelo_ = modelo
        self.embedding_ = embedding
        self.coordenadas_treino_ = coords.copy()
        self.indice_treino_ = X.index.copy()
        self.colunas_ = list(X.columns)
        self.n_treino_ = len(X)
        self.saidas_ = [f"TSNE{i+1}" for i in range(int(p["n_components"]))]
        self.kl_divergence_ = getattr(embedding, "kl_divergence", None)
        return self

    def transform_treino(self):
        """Retorna coordenadas JA ajustadas; nunca recalcula t-SNE."""
        if not hasattr(self, "embedding_"):
            raise RuntimeError("Chame fit(X_treino) antes de transform_treino().")
        return pd.DataFrame(
            self.coordenadas_treino_.copy(), index=self.indice_treino_.copy(),
            columns=self.saidas_,
        )

    def transform(self, X):
        """Insere observacoes novas no embedding de treino, sem refit."""
        if not hasattr(self, "embedding_"):
            raise RuntimeError("Chame fit(X_treino) antes de transform().")
        valores = self._validar_X(X)
        if list(X.columns) != self.colunas_:
            raise ValueError("Colunas ou ordem diferentes do treino no t-SNE.")
        # Default openTSNE para OOS: inicializacao por mediana, LR=0.1,
        # n_iter=250. Sao parametros fixos para os novos pontos;
        # NAO usa n_iter de treino, nem refaz o embedding original.
        novos = np.asarray(self.embedding_.transform(valores), dtype=np.float32)
        esperado = (len(X), len(self.saidas_))
        if novos.shape != esperado or not np.isfinite(novos).all():
            raise ValueError("t-SNE transform produziu embedding invalido.")
        return pd.DataFrame(novos, index=X.index, columns=self.saidas_)

    def fit_transform(self, X):
        return self.fit(X).transform_treino()

    def get_feature_names_out(self):
        if not hasattr(self, "saidas_"):
            raise RuntimeError("Chame fit antes de obter nomes das componentes.")
        return np.asarray(self.saidas_, dtype=object)
