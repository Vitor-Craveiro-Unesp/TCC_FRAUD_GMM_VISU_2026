# -*- coding: utf-8 -*-
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import json
import time

import numpy as np
import pandas as pd
import yaml

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from src.reducao_tsne import RedutorTSNE
from src.gmm import GMMNaoSupervisionado
from src.risco import RiscoHistoricoGaussianas
from src import visualizacao_geometrica as geo

RAIZ = Path.cwd()

SAIDA = (
    RAIZ / "outputs/validacao_tsne_v2_1d"
    / "TSNE13_mapa_1d_original_v1"
)

PNG = SAIDA / "13_tsne_mapa_TSNE1.png"
CSV = SAIDA / "13_tsne_mapa_TSNE1.csv"

protegidos = [
    RAIZ / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml",
    RAIZ / "src/reducao_tsne.py",
    RAIZ / "src/gmm.py",
    RAIZ / "src/risco.py",
    RAIZ / "src/visualizacao_geometrica.py",
]

def hash_arquivo(p):
    return sha256(p.read_bytes()).hexdigest()

antes = {str(p): hash_arquivo(p) for p in protegidos}

cfg = deepcopy(
    yaml.safe_load(
        protegidos[0].read_text(encoding="utf-8-sig")
    )
)

cfg["reducao"]["metodo"] = "TSNE"
cfg["experimento"]["semente"] = 42
cfg["execucao"]["n_jobs"] = 2

# Parametros apenas desta simulacao.
parametros_tsne = {
    "perplexity": 10,
    "n_components": 1,
    "initialization": "pca",
    "metric": "euclidean",
    "learning_rate": "auto",
    "early_exaggeration": 12.0,
    "n_iter": 300,
    "theta": 0.5,
}

cfg["tsne"].update(parametros_tsne)

op = cfg["visualizacao"]["graficos"]["tsne_mapa"]

op["colorir_por"] = "classe_real"
op["bins_histograma_1d"] = 36
op["curvas_suavizadas_1d"] = True

# 01 - BASE ARTIFICIAL
X, y = make_classification(
    n_samples=1200,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    n_repeated=0,
    n_clusters_per_class=2,
    weights=[0.92, 0.08],
    class_sep=1.2,
    random_state=42,
)

# Padronizacao apenas para a amostra artificial.
X = StandardScaler().fit_transform(X)

features = pd.DataFrame(
    X,
    columns=[f"FEATURE_{i:02d}" for i in range(1, 13)]
)

y = np.asarray(y, dtype=np.uint8)

assert len(features) == len(y) == 1200
assert set(np.unique(y)) == {0, 1}

# 02 - TSNE 1D REAL SOBRE DADOS SINTETICOS
inicio = time.perf_counter()

redutor = RedutorTSNE(
    cfg, parametros_tsne
).fit(features)

coord = redutor.transform_treino()

tempo_tsne = time.perf_counter() - inicio

assert coord.shape == (1200, 1)
assert list(coord.columns) == ["TSNE1"]
assert np.isfinite(coord.to_numpy()).all()

# 03 - GMM FULL SINTETICO
parametros_gmm = {
    "n_components": 4,
    "covariance_type": "full",
    "reg_covar": 1e-6,
    "max_iter": 300,
    "n_init": 5,
    "tol": 0.001,
    "init_params": "kmeans",
}

gmm = GMMNaoSupervisionado(
    cfg, parametros_gmm
).fit(coord)

gamma = gmm.responsabilidades(coord)

assert gamma.shape == (1200, 4)

# 04 - RISCOS HISTORICOS ILUSTRATIVOS
# Os rotulos nao participam do ajuste t-SNE/GMM.
rh = RiscoHistoricoGaussianas(
    cfg, suavizacao=0.0
).fit(
    gamma,
    pd.Series(y, index=coord.index)
)

scores = (
    gamma.to_numpy(dtype=float)
    @ rh.riscos_.to_numpy(dtype=float)
)

assert np.isfinite(scores).all()
assert np.all((scores >= 0) & (scores <= 1))

# Este limiar nao foi otimizado em OOF.
tau_ficticio = 0.35

dados = {
    "coord": coord,
    "redutor": redutor,
    "gmm": gmm,
    "gamma": gamma,
    "rh": rh,
    "y": y,
    "scores": scores,
    "limiar_oof": tau_ficticio,
}

# 05 - RENDERIZADOR OFICIAL DO TSNE13
if SAIDA.exists():
    raise FileExistsError(
        "Pasta da previa ja existe: " + str(SAIDA)
    )

SAIDA.mkdir(parents=True, exist_ok=False)

# A rotina original menciona F1-F9 em seu rodape.
# Na previa, substituimos esse texto apenas em memoria
# para nao atribuir os dados artificiais ao desenvolvimento real.
guardar_original = geo._guardar

def guardar_previa(fig, destino, nome, ext, dpi):
    for item in fig.texts:
        texto = item.get_text()
        texto = texto.replace(
            "Desenvolvimento F1–F9; F10 não utilizado.",
            "Dados exclusivamente artificiais; F10 real não utilizado."
        )
        item.set_text(texto)

    return guardar_original(
        fig, destino, nome, ext, dpi
    )

geo._guardar = guardar_previa

try:
    geo._tsne_histograma_1d(
        cfg,
        dados,
        SAIDA,
        "png",
        180,
        True,
        op,
        origem="simulado",
    )
finally:
    geo._guardar = guardar_original

# 06 - VALIDACOES
assert PNG.is_file(), "PNG nao gerado."
assert CSV.is_file(), "CSV nao gerado."

tabela = pd.read_csv(CSV)

assert len(tabela) == 1200

colunas = {
    "TSNE1",
    "classe_real",
    "classe_predita_limiar_oof",
    "risco_estimado_desenvolvimento",
    "gaussiana_dominante",
}

assert colunas.issubset(tabela.columns)

np.testing.assert_allclose(
    tabela["TSNE1"].to_numpy(),
    coord["TSNE1"].to_numpy(),
    rtol=1e-7,
    atol=1e-7,
)

np.testing.assert_array_equal(
    tabela["classe_real"].to_numpy(),
    y,
)

np.testing.assert_allclose(
    tabela["risco_estimado_desenvolvimento"].to_numpy(),
    scores,
    rtol=1e-7,
    atol=1e-8,
)

np.testing.assert_array_equal(
    tabela["gaussiana_dominante"].to_numpy(),
    np.argmax(gamma.to_numpy(), axis=1) + 1,
)

np.testing.assert_array_equal(
    tabela["classe_predita_limiar_oof"].to_numpy(),
    (scores >= tau_ficticio).astype(int),
)

# 07 - MANIFESTO
meta = {
    "grafico": "TSNE13",
    "tipo": "PREVIA_SINTETICA_NAO_OFICIAL",
    "metodo": "TSNE",
    "dimensao": 1,
    "eixo": "TSNE1",
    "n_sintetico": int(len(y)),
    "n_fraudes_sinteticas": int(y.sum()),
    "perplexity": 10,
    "initialization": "pca",
    "gmm_componentes": 4,
    "covariance_type": "full",
    "gmm_n_init": 5,
    "tempo_tsne_segundos": tempo_tsne,
    "limiar_ficticio_nao_oof": tau_ficticio,
    "treinamento_somente_sintetico": True,
    "oof_real_executado": False,
    "f10_real_acessado": False,
}

(SAIDA / "manifesto_previa.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

depois = {str(p): hash_arquivo(p) for p in protegidos}

assert antes == depois, "Arquivo oficial alterado."

print()
print("=== V2 / TSNE 1D / TSNE13 ===")
print("RENDERIZADOR_OFICIAL_TSNE13=SIM")
print("DIMENSAO=1D")
print("EIXO=TSNE1")
print("TIPO=HISTOGRAMA_DO_EMBEDDING")
print("COLORACAO=CLASSE_REAL_SINTETICA")
print(f"N_SINTETICO={len(y)}")
print(f"FRAUDES_SINTETICAS={int(y.sum())}")
print(f"PERPLEXITY={parametros_tsne['perplexity']}")
print(f"TEMPO_TSNE_SINTETICO={tempo_tsne:.3f}s")
print("GMM_FULL_K4_N_INIT5=SIM")
print("DADOS_REAIS_UTILIZADOS=NAO")
print("OOF_REAL_EXECUTADO=NAO")
print("F10_REAL_ACESSADO=NAO")
print("ARQUIVOS_OFICIAIS_ALTERADOS=NAO")
print(f"PNG_PARA_AVALIAR={PNG}")
print(f"CSV_PARA_AVALIAR={CSV}")
print("V2_TSNE13_VALIDACAO=AGUARDANDO_APROVACAO_VISUAL")