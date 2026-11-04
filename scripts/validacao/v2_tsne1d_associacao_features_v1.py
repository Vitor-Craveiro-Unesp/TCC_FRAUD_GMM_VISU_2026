# -*- coding: utf-8 -*-
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from src.reducao_tsne import RedutorTSNE

RAIZ = Path.cwd()
SAIDA = (
    RAIZ / "outputs/validacao_tsne_v2_1d"
    / "TSNE_EXTRA_associacao_features_v1"
)

PNG = SAIDA / "TSNE_EXTRA_associacao_features_TSNE1.png"
CSV = SAIDA / "TSNE_EXTRA_associacao_features_TSNE1.csv"

protegidos = [
    RAIZ / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml",
    RAIZ / "src/reducao_tsne.py",
    RAIZ / "src/visualizacao_pca_biplot.py",
]

def hash_arquivo(p):
    return sha256(p.read_bytes()).hexdigest()

antes = {str(p): hash_arquivo(p) for p in protegidos}

config = deepcopy(
    yaml.safe_load(
        protegidos[0].read_text(encoding="utf-8-sig")
    )
)

config["reducao"]["metodo"] = "TSNE"
config["experimento"]["semente"] = 42
config["execucao"]["n_jobs"] = 2

parametros = {
    "perplexity": 10,
    "n_components": 1,
    "initialization": "pca",
    "metric": "euclidean",
    "learning_rate": "auto",
    "early_exaggeration": 12.0,
    "n_iter": 300,
    "theta": 0.5,
}

# 01 - Dados sinteticos
# Mesma especificacao da previa TSNE13.
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

X = StandardScaler().fit_transform(X)

features = pd.DataFrame(
    X,
    columns=[
        f"FEATURE_{i:02d}"
        for i in range(1, 13)
    ]
)

# 02 - Ajustar TSNE1 sem utilizar rotulos
redutor = RedutorTSNE(
    config,
    parametros,
).fit(features)

coordenadas = redutor.transform_treino()

assert coordenadas.shape == (1200, 1)
assert list(coordenadas.columns) == ["TSNE1"]
assert features.index.equals(coordenadas.index)

z = coordenadas["TSNE1"]

# 03 - Associacoes descritivas
# Spearman mede associacao monotônica entre a
# variavel preprocessada e a coordenada TSNE1.
# NAO e loading, peso ou coeficiente do t-SNE.

registros = []

for nome in features.columns:
    rho = float(
        features[nome].corr(z, method="spearman")
    )

    assert np.isfinite(rho)
    assert -1 <= rho <= 1

    registros.append({
        "variavel_preprocessada": nome,
        "correlacao_spearman_TSNE1": rho,
        "associacao_absoluta": abs(rho),
    })

tabela = pd.DataFrame(registros)

tabela = tabela.sort_values(
    ["associacao_absoluta", "variavel_preprocessada"],
    ascending=[False, True],
    kind="stable",
).reset_index(drop=True)

assert len(tabela) == features.shape[1]
assert tabela["variavel_preprocessada"].is_unique

if SAIDA.exists():
    raise FileExistsError(
        "Pasta de saida ja existente."
    )

SAIDA.mkdir(parents=True, exist_ok=False)

tabela.to_csv(
    CSV, index=False, encoding="utf-8-sig",
    float_format="%.17g"
)

# 04 - Grafico com mesmo conceito visual do PCA
n = len(tabela)

altura = max(8.0, 2.0 + 0.42 * n)

fig, ax = plt.subplots(
    figsize=(14.0, altura),
    facecolor="white",
)

yy = np.arange(n)

labels = tabela["variavel_preprocessada"].tolist()
valores = tabela["correlacao_spearman_TSNE1"].to_numpy()

ax.axvline(
    0,
    color="#778391",
    linewidth=1.1,
    zorder=2,
)

cores = plt.get_cmap("tab20")

for i, (pos, valor) in enumerate(zip(yy, valores)):

    cor = cores(i % 20)

    ax.annotate(
        "",
        xy=(valor, pos),
        xytext=(0, pos),
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.9,
            "color": cor,
            "mutation_scale": 13,
        },
        zorder=4,
    )

    exibido = (
        0.0 if abs(valor) < 0.00000005 else valor
    )

    ax.text(
        valor + (0.040 if valor >= 0 else -0.040),
        pos,
        f"{exibido:+.7f}",
        ha="left" if valor >= 0 else "right",
        va="center",
        fontsize=9,
        color=cor,
    )

ax.set_yticks(yy, labels=labels)

ax.set_ylim(n - 0.85, -0.85)
ax.set_xlim(-1.35, 1.35)

ax.set_xticks([-1, -0.5, 0, 0.5, 1])

ax.xaxis.set_major_formatter(
    FuncFormatter(lambda v, _: f"{v:.7f}")
)

ax.set_xlabel(
    "Correlacao de Spearman com TSNE1",
    fontsize=11,
    labelpad=12,
)

ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=9, length=0, pad=8)

ax.grid(
    axis="x",
    linestyle=":",
    alpha=0.22,
)

ax.set_axisbelow(True)

fig.suptitle(
    "ASSOCIACAO DAS VARIAVEIS COM TSNE1",
    fontsize=17,
    fontweight="bold",
    y=0.988,
)

fig.text(
    0.5,
    0.948,
    "t-SNE 1D | Correlacoes de Spearman | Dados sinteticos",
    fontsize=10,
    ha="center",
    color="#52647a",
)

fig.text(
    0.5,
    0.025,
    "Associacao descritiva; nao representa coeficientes "
    "ou importancia causal do t-SNE. O sinal pode inverter "
    "com a orientacao do embedding.",
    ha="center",
    fontsize=8.5,
    color="#52647a",
)

fig.subplots_adjust(
    left=0.24,
    right=0.96,
    top=0.89,
    bottom=0.11,
)

fig.savefig(
    PNG,
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)

# 05 - Validacao
assert PNG.is_file()
assert CSV.is_file()

conferencia = pd.read_csv(CSV)

assert len(conferencia) == n

np.testing.assert_allclose(
    conferencia["correlacao_spearman_TSNE1"],
    valores,
    atol=1e-12,
    rtol=1e-12,
)

depois = {
    str(p): hash_arquivo(p)
    for p in protegidos
}

assert antes == depois, "Arquivo oficial alterado."

# 06 - Manifesto
manifesto = {
    "grafico": "TSNE_EXTRA_ASSOCIACAO_FEATURES",
    "tipo": "PREVIA_SINTETICA_NAO_OFICIAL",
    "metodo": "TSNE",
    "dimensao": 1,
    "n_sintetico": 1200,
    "n_variaveis": n,
    "medida": "correlacao_spearman",
    "medida_e_coeficiente_tsne": False,
    "sinal_dependente_orientacao": True,
    "rotulos_usados_no_ajuste": False,
    "treinamento_real_executado": False,
    "oof_real_executado": False,
    "f10_real_acessado": False,
}

(SAIDA / "manifesto_previa.json").write_text(
    json.dumps(manifesto, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print()
print("=== V2 / TSNE 1D / ASSOCIACAO DAS VARIAVEIS ===")
print(f"N_SINTETICO=1200")
print(f"N_VARIAVEIS={n}")
print("MEDIDA=SPEARMAN")
print("COEFICIENTES_LINEARES_TSNE=NAO_EXISTEM")
print("ROTULOS_USADOS_NO_AJUSTE=NAO")
print("TREINAMENTO_REAL_EXECUTADO=NAO")
print("OOF_REAL_EXECUTADO=NAO")
print("F10_REAL_ACESSADO=NAO")
print("ARQUIVOS_OFICIAIS_ALTERADOS=NAO")
print(f"PNG_PARA_AVALIAR={PNG}")
print(f"CSV_PARA_AVALIAR={CSV}")
print("V2_TSNE_ASSOCIACAO_VALIDACAO=AGUARDANDO_APROVACAO_VISUAL")