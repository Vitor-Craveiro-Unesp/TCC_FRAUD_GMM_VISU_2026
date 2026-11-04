# -*- coding: utf-8 -*-
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness

from openTSNE import TSNE

SEED = 42
OUTDIR = Path("outputs/validacao_tsne_v2_1d/TSNE02_trustworthiness_1d")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Dados sinteticos apenas para validacao visual do grafico original.
# Nao acessa F10, nao usa YAML, nao executa OOF real.
# ------------------------------------------------------------------
X, y = make_classification(
    n_samples=1200,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    n_repeated=0,
    n_clusters_per_class=2,
    weights=[0.92, 0.08],
    class_sep=1.2,
    random_state=SEED
)

X = StandardScaler().fit_transform(X)

perplexidades = [5, 10, 15, 20, 30, 40, 50]
linhas = []

for p in perplexidades:
    t0 = time.perf_counter()

    tsne = TSNE(
        n_components=1,
        perplexity=p,
        initialization="pca",
        random_state=SEED,
        metric="euclidean",
        n_jobs=8,
        verbose=False
    )

    emb = tsne.fit(X)
    emb_np = np.asarray(emb).reshape(-1, 1)

    tempo = time.perf_counter() - t0

    n_vizinhos = int(max(5, min(p, 50)))
    tw = float(trustworthiness(X, emb_np, n_neighbors=n_vizinhos))

    kl = getattr(emb, "kl_divergence", np.nan)
    try:
        kl = float(kl)
    except Exception:
        kl = np.nan

    linhas.append(
        {
            "perplexity": int(p),
            "trustworthiness": tw,
            "kl_final": kl,
            "tempo_segundos": tempo,
            "n_vizinhos_trustworthiness": n_vizinhos,
        }
    )

df = pd.DataFrame(linhas).sort_values("perplexity").reset_index(drop=True)

idx_best = int(df["trustworthiness"].idxmax())
best = df.loc[idx_best]

csv_path = OUTDIR / "TSNE02_trustworthiness_vs_perplexity_1d.csv"
png_path = OUTDIR / "TSNE02_trustworthiness_vs_perplexity_1d.png"
json_path = OUTDIR / "TSNE02_trustworthiness_vs_perplexity_1d_manifesto.json"

df.to_csv(csv_path, index=False, encoding="utf-8-sig")

manifesto = {
    "metodo_simulado": "TSNE",
    "dimensao_simulada": "1D",
    "grafico": "TSNE02",
    "titulo": "Trustworthiness vs Perplexity",
    "dados": "sinteticos",
    "f10_real_acessado": False,
    "treinamento_real_executado": False,
    "yaml_oficial_modificado": False,
    "oof_completo_executado": False,
    "semente": SEED,
    "melhor_perplexity": int(best["perplexity"]),
    "melhor_trustworthiness": float(best["trustworthiness"]),
}
json_path.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")

# ------------------------------------------------------------------
# Grafico
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(
    df["perplexity"],
    df["trustworthiness"],
    linewidth=2.5,
    marker="o",
    label="Trustworthiness"
)

ax.scatter(
    [best["perplexity"]],
    [best["trustworthiness"]],
    s=90,
    zorder=5,
    label=f"Melhor perplexity = {int(best['perplexity'])}"
)

ax.axvline(
    best["perplexity"],
    linestyle="--",
    linewidth=1.5,
    alpha=0.9
)

for _, row in df.iterrows():
    ax.annotate(
        f"{row['trustworthiness']:.4f}",
        (row["perplexity"], row["trustworthiness"]),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=9
    )

ax.set_title("TRUSTWORTHINESS DO t-SNE POR PERPLEXITY", fontsize=22, weight="bold", pad=18)
fig.text(0.5, 0.92, "TSNE | 1D | VALIDACAO SINTETICA", ha="center", fontsize=14)

ax.set_xlabel("Perplexity", fontsize=14)
ax.set_ylabel("Trustworthiness", fontsize=14)
ax.grid(True, alpha=0.25)
ax.legend(loc="best")

rodape_1 = (
    f"Melhor perplexity = {int(best['perplexity'])}"
    f"  |  Trustworthiness = {best['trustworthiness']:.7f}"
)
rodape_2 = (
    "Dados sinteticos | Sem F10 | Sem OOF completo | "
    "Sem alteracao de YAML | Grafico original do t-SNE"
)

fig.text(0.10, 0.05, rodape_1, fontsize=12, weight="bold")
fig.text(0.10, 0.02, rodape_2, fontsize=10)

plt.tight_layout(rect=[0.03, 0.09, 0.98, 0.90])
fig.savefig(png_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print("=== V2 / TSNE 1D / TSNE02 - VALIDACAO ORIGINAL ===")
print("METODO_SIMULADO=TSNE")
print("DIMENSAO_SIMULADA=1D")
print("GRAFICO=TSNE02")
print("TIPO=TRUSTWORTHINESS_VS_PERPLEXITY")
print(f"N_SINTETICO={len(X)}")
print(f"MELHOR_PERPLEXITY={int(best['perplexity'])}")
print(f"MELHOR_TRUSTWORTHINESS={best['trustworthiness']:.7f}")
print("F10_REAL_ACESSADO=NAO")
print("TREINAMENTO_REAL_EXECUTADO=NAO")
print("OOF_COMPLETO_EXECUTADO=NAO")
print("ARQUIVOS_OFICIAIS_ALTERADOS=NAO")
print(f"PNG_PARA_AVALIAR={png_path.resolve()}")
print(f"CSV_PARA_AVALIAR={csv_path.resolve()}")
print(f"MANIFESTO={json_path.resolve()}")
print("V2_TSNE02_VALIDACAO=APROVADA_TECNICAMENTE_AGUARDANDO_AVALIACAO_VISUAL")
