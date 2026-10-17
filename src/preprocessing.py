"""Passos 03.01 e 03.02: encoders, imputacao e padronizacao sem vazamento.

Fluxo: selecionar_features -> EncodersPorSufixo.fit/transform ->
ImputadorPadronizador.fit/transform, sempre ajustando somente no treino.
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


_AUSENTE = "__TCC_AUSENTE__"


def _categorias_com_ausentes(serie):
    """Compatibiliza Series Categorical/Arrow com marcador de ausentes."""
    serie = serie.astype(object)
    return serie.where(serie.notna(), _AUSENTE)


def selecionar_features(df, config):
    """Exclui alvo, ID e colunas explicitamente descartadas no YAML."""
    dados = config["dados"]
    excluir = [dados["coluna_alvo"]]
    if dados.get("coluna_id") is not None:
        excluir.append(dados["coluna_id"])
    excluir += dados.get("colunas_excluir", [])
    faltando = [c for c in excluir[:1] if c not in df.columns]
    if faltando:
        raise ValueError(f"Coluna alvo nao encontrada: {faltando}")
    return df.drop(columns=excluir, errors="ignore").copy()


class EncodersPorSufixo:
    """Aprende frequencias e categorias de One-Hot apenas durante fit(X_treino)."""

    def __init__(self, config):
        self.config = config

    def _validar_X(self, X):
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("X deve ser um DataFrame nao vazio.")
        dados = self.config["dados"]
        proibidas = [dados["coluna_alvo"], dados.get("coluna_id")]
        proibidas += dados.get("colunas_excluir", [])
        vazamento = set(X.columns).intersection(c for c in proibidas if c)
        if vazamento:
            raise ValueError(f"Colunas proibidas nas features: {sorted(vazamento)}")
        if not X.columns.is_unique:
            raise ValueError("Nomes de features duplicados.")

    def fit(self, X):
        self._validar_X(X)
        cfg = self.config["preprocessamento"]
        if cfg.get("identificar_por_sufixo") is not True:
            raise ValueError("Esta etapa exige identificar_por_sufixo: true")
        enc = cfg["encodings"]
        self.colunas_ = list(X.columns)
        self.regras_ = {}
        self.n_treino_ = len(X)
        self.saidas_ = []
        self.freq_ = {}
        self.categorias_ = {}
        self.binarios_ = {}
        fe = enc["frequency"]["sufixo"]
        be = enc["binary"]["sufixo"]
        oh = enc["onehot"]["sufixo"]
        cic = tuple(cfg["ciclicas"]["sufixos"])
        if enc["frequency"].get("tipo") != "frequencia_relativa":
            raise ValueError("Somente frequencia_relativa implementada.")
        if enc["frequency"].get("fallback") != "1/n":
            raise ValueError("Somente fallback 1/n implementado.")
        if enc["onehot"].get("handle_unknown") != "ignore":
            raise ValueError("Somente One-Hot handle_unknown: ignore implementado.")
        if not cfg["ciclicas"].get("ja_transformadas", False):
            raise ValueError("As variaveis ciclicas devem estar transformadas.")
        if cfg["ciclicas"].get("padronizar", False):
            raise ValueError("O YAML pede preservar as ciclicas sem padronizar.")

        for coluna in self.colunas_:
            s = X[coluna]
            if coluna.endswith(fe):
                self.regras_[coluna] = "frequency"
                self.freq_[coluna] = _categorias_com_ausentes(s).value_counts(
                    normalize=True, dropna=False
                ).to_dict()
                self.saidas_.append(coluna)
            elif coluna.endswith(be):
                self.regras_[coluna] = "binary"
                mapping = enc["binary"].get("mapeamentos", {}).get(coluna)
                if not isinstance(mapping, dict) or not mapping:
                    raise ValueError(
                        f"Mapeamento BE ausente para {coluna}. "
                        "Nao assumir codigos de categorias automaticamente."
                    )
                self.binarios_[coluna] = mapping
                self._codificar_binaria(s, coluna)  # conferir o treino
                self.saidas_.append(coluna)
            elif coluna.endswith(oh):
                self.regras_[coluna] = "onehot"
                valores = _categorias_com_ausentes(s).unique().tolist()
                categorias = sorted(valores, key=lambda v: str(v))
                self.categorias_[coluna] = categorias
                self.saidas_ += [f"{coluna}__{c}" for c in categorias]
            elif coluna.endswith(cic):
                if not is_numeric_dtype(s):
                    raise ValueError(f"Ciclica {coluna} deve ser numerica.")
                self.regras_[coluna] = "ciclica"
                self.saidas_.append(coluna)
            elif is_numeric_dtype(s):
                self.regras_[coluna] = "numerica"
                self.saidas_.append(coluna)
            else:
                raise ValueError(
                    f"Coluna categórica sem sufixo reconhecido: {coluna}. "
                    "Defina encoding no nome/YAML antes de prosseguir."
                )
        if len(self.saidas_) != len(set(self.saidas_)):
            raise ValueError("Nomes de colunas apos encoding repetidos.")
        return self

    def _codificar_binaria(self, s, coluna):
        valores = s.map(self.binarios_[coluna])
        invalidas = valores.isna()
        if invalidas.any():
            exemplos = s.loc[invalidas].drop_duplicates().head(5).tolist()
            raise ValueError(
                f"BE {coluna}: categorias sem mapeamento (ou ausentes): {exemplos}."
            )
        return valores.to_numpy(dtype=np.float32)

    def transform(self, X):
        if not hasattr(self, "colunas_"):
            raise RuntimeError("Chame fit(X_treino) antes de transform().")
        self._validar_X(X)
        if list(X.columns) != self.colunas_:
            raise ValueError("Colunas ou ordem diferentes do treino.")
        n = len(X)
        matriz = np.empty((n, len(self.saidas_)), dtype=np.float32)
        j = 0
        for coluna in self.colunas_:
            s = X[coluna]
            regra = self.regras_[coluna]
            if regra == "frequency":
                p = _categorias_com_ausentes(s).map(self.freq_[coluna])
                matriz[:, j] = p.fillna(1.0 / self.n_treino_).to_numpy(dtype=np.float32)
                j += 1
            elif regra == "binary":
                matriz[:, j] = self._codificar_binaria(s, coluna)
                j += 1
            elif regra == "onehot":
                valores = _categorias_com_ausentes(s).to_numpy()
                for categoria in self.categorias_[coluna]:
                    matriz[:, j] = (valores == categoria).astype(np.float32)
                    j += 1
            else:  # numericas e ciclicas: apenas passagem; passo 03.02 trata NaN/scaler
                matriz[:, j] = pd.to_numeric(s, errors="raise").to_numpy(dtype=np.float32)
                j += 1
        return pd.DataFrame(matriz, columns=self.saidas_, index=X.index)

    def get_feature_names_out(self):
        if not hasattr(self, "saidas_"):
            raise RuntimeError("Chame fit antes de obter os nomes.")
        return np.array(self.saidas_, dtype=object)


class ImputadorPadronizador:
    """Passo 03.02: mediana e StandardScaler aprendidos apenas no treino.

    Use depois de EncodersPorSufixo: fit(X_treino_codificado) e
    transform(X_validacao_codificada). Nao aprende nada no transform.

    Padroniza numericas originais e frequencias FEWF. Preserva os valores
    binarios, one-hot e ciclicos; ausentes ciclicos recebem mediana do treino.
    """

    def __init__(self, config, encoder):
        self.config = config
        self.encoder = encoder

    def _validar_configuracao(self):
        numericas = self.config["preprocessamento"]["numericas"]
        ciclicas = self.config["preprocessamento"]["ciclicas"]
        if numericas.get("imputacao") != "mediana":
            raise ValueError("03.02 implementa apenas imputacao: mediana.")
        if numericas.get("padronizacao") is not True:
            raise ValueError("03.02 exige padronizacao: true (StandardScaler).")
        if ciclicas.get("padronizar") is not False:
            raise ValueError("03.02 exige ciclicas.padronizar: false.")

    def _validar_entrada(self, X):
        if not hasattr(self.encoder, "saidas_"):
            raise RuntimeError("Ajuste EncodersPorSufixo no treino antes do passo 03.02.")
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("X codificado deve ser um DataFrame nao vazio.")
        if list(X.columns) != self.encoder.saidas_:
            raise ValueError("Colunas codificadas diferentes do treino ou fora de ordem.")
        matriz = X.to_numpy(dtype=np.float32, copy=True)
        if np.isinf(matriz).any():
            raise ValueError("Valores infinitos encontrados nas features codificadas.")
        return matriz

    def _selecionar_colunas(self):
        padronizar, imputar, protegidas = [], [], []
        for original in self.encoder.colunas_:
            regra = self.encoder.regras_[original]
            if regra == "onehot":
                protegidas.extend(
                    f"{original}__{categoria}"
                    for categoria in self.encoder.categorias_[original]
                )
            elif regra in ("numerica", "frequency"):
                imputar.append(original)
                padronizar.append(original)
            elif regra == "ciclica":
                imputar.append(original)
            elif regra == "binary":
                protegidas.append(original)
            else:
                raise ValueError(f"Regra de encoding desconhecida: {regra}")
        return padronizar, imputar, protegidas

    def fit(self, X):
        from sklearn.preprocessing import StandardScaler

        self._validar_configuracao()
        matriz = self._validar_entrada(X)
        self.colunas_ = list(X.columns)
        self.padronizadas_, self.imputadas_, protegidas = self._selecionar_colunas()
        pos = {nome: i for i, nome in enumerate(self.colunas_)}
        self.indices_padronizar_ = np.array([pos[c] for c in self.padronizadas_], dtype=int)
        self.indices_protegidos_ = np.array([pos[c] for c in protegidas], dtype=int)
        self.mediana_ = {}

        if np.isnan(matriz[:, self.indices_protegidos_]).any():
            raise ValueError("Valor ausente em feature binaria ou One-Hot.")

        for coluna in self.imputadas_:
            j = pos[coluna]
            valores = matriz[:, j]
            finitos = valores[np.isfinite(valores)]
            if finitos.size == 0:
                raise ValueError(f"Coluna sem valores observados no treino: {coluna}")
            mediana = float(np.median(finitos))
            self.mediana_[coluna] = mediana
            valores[np.isnan(valores)] = mediana

        self.scaler_ = StandardScaler(copy=True)
        if self.indices_padronizar_.size:
            self.scaler_.fit(matriz[:, self.indices_padronizar_])
        else:
            self.scaler_ = None
        return self

    def transform(self, X):
        if not hasattr(self, "colunas_"):
            raise RuntimeError("Chame fit(X_treino_codificado) antes de transform().")
        matriz = self._validar_entrada(X)
        if list(X.columns) != self.colunas_:
            raise ValueError("As features devem manter os nomes e a ordem do treino.")
        if np.isnan(matriz[:, self.indices_protegidos_]).any():
            raise ValueError("Valor ausente em feature binaria ou One-Hot.")

        pos = {nome: i for i, nome in enumerate(self.colunas_)}
        for coluna, mediana in self.mediana_.items():
            valores = matriz[:, pos[coluna]]
            valores[np.isnan(valores)] = mediana

        if self.scaler_ is not None:
            indices = self.indices_padronizar_
            matriz[:, indices] = (
                (matriz[:, indices].astype(np.float64) - self.scaler_.mean_)
                / self.scaler_.scale_
            ).astype(np.float32)

        if not np.isfinite(matriz).all():
            raise ValueError("Valores nao finitos encontrados apos imputacao/padronizacao.")
        return pd.DataFrame(matriz, columns=self.colunas_, index=X.index)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_feature_names_out(self):
        if not hasattr(self, "colunas_"):
            raise RuntimeError("Chame fit antes de obter os nomes.")
        return np.array(self.colunas_, dtype=object)
