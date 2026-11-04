# -*- coding: utf-8 -*-
from pathlib import Path
from copy import deepcopy
import json
import time

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter1d
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from src.reducao_tsne import RedutorTSNE

RAIZ = Path.cwd()
SAIDA = RAIZ / "outputs" / "validacao_tsne_v2_1d" / "TSNE13_mapa_1d_estilo_pca_v2"
SAIDA.mkdir(parents=True, exist_ok=True)

with open(RAIZ / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml", "r", encoding="utf-8-sig") as f:
    cfg = yaml.safe_load(f)

cfg = deepcopy(cfg)
cfg["reducao"]["metodo"] = "TSNE"
cfg["experimento"]["semente"] = 42
cfg["execucao"]["n_jobs"] = 2

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

# Base sintética
X, y = make_classification(
    n_samples=1200,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    n_clusters_per_class=2,
    weights=[0.92, 0.08],
    class_sep=1.2,
    random_state=42,
)

X = StandardScaler().fit_transform(X)

df = pd.DataFrame(
    X,
    columns=[f"FEATURE_{i:02d}" for i in range(1, 13)]
)

y = np.asarray(y, dtype=np.uint8)

# t-SNE 1D real
t0 = time.perf_counter()
redutor = RedutorTSNE(cfg, parametros=parametros_tsne).fit(df)
coord = redutor.transform_treino()
tempo = time.perf_counter() - t0

x = coord["TSNE1"].to_numpy(dtype=float)
leg = x[y == 0]
frd = x[y == 1]

# Histograma estilo PCA
bins = 36
lo = min(x.min(), leg.min(), frd.min())
hi = max(x.max(), leg.max(), frd.max())
edges = np.linspace(lo, hi, bins + 1)
centros = 0.5 * (edges[:-1] + edges[1:])

dens_leg, _ = np.histogram(leg, bins=edges, density=True)
dens_frd, _ = np.histogram(frd, bins=edges, density=True)

curva_leg = gaussian_filter1d(dens_leg, sigma=1.2, mode="nearest")
curva_frd = gaussian_filter1d(dens_frd, sigma=1.2, mode="nearest")

media_global = float(np.mean(x))

fig, ax = plt.subplots(figsize=(13.8, 8.2), dpi=140)

# Linha média global primeiro para aparecer primeiro na legenda
ax.axvline(
    media_global,
    color="#25364d",
    linestyle="--",
    linewidth=1.9,
    label=f"Média global da TSNE1 = {media_global:.7f}",
    zorder=5
)

ax.hist(
    leg,
    bins=edges,
    density=True,
    alpha=0.35,
    color="#6fa8d6",
    edgecolor="#6fa8d6",
    linewidth=0.6,
    label=f"Legítima (n={len(leg)})",
    zorder=1
)

ax.hist(
    frd,
    bins=edges,
    density=True,
    alpha=0.35,
    color="#e57373",
    edgecolor="#e57373",
    linewidth=0.6,
    label=f"Fraude (n={len(frd)})",
    zorder=2
)

ax.plot(centros, curva_leg, color="#1f77b4", linewidth=2.4, zorder=6)
ax.plot(centros, curva_frd, color="#d62728", linewidth=2.4, zorder=7)

ax.set_title("TSNE 1D | Histograma da projeção em TSNE1", fontsize=24, pad=16)
ax.set_ylabel("Densidade por classe (área = 1)", fontsize=17)
ax.set_xlabel("TSNE1", fontsize=15)

ax.grid(axis="y", alpha=0.25, linestyle=":")
ax.set_axisbelow(True)

fmt = FuncFormatter(lambda v, pos: f"{v:.7f}")
ax.xaxis.set_major_formatter(fmt)
ax.yaxis.set_major_formatter(fmt)
ax.tick_params(axis="both", labelsize=12)

legenda = ax.legend(
    loc="upper right",
    frameon=True,
    fontsize=12
)
legenda.get_frame().set_alpha(0.95)

rodape = (
    "Prévia sintética para validação visual do TSNE13 | "
    "t-SNE ajustado apenas em dados artificiais | "
    "F10 real não acessado"
)
fig.text(0.5, 0.02, rodape, ha="center", va="bottom", fontsize=10, color="#4b5563")

fig.tight_layout(rect=[0, 0.04, 1, 1])

png = SAIDA / "13_tsne_mapa_TSNE1_ESTILO_PCA_V2.png"
csv = SAIDA / "13_tsne_mapa_TSNE1_ESTILO_PCA_V2.csv"
meta = SAIDA / "manifesto_previa.json"

fig.savefig(png, bbox_inches="tight")
plt.close(fig)

pd.DataFrame({
    "TSNE1": x,
    "classe_real": y
}).to_csv(csv, index=False, encoding="utf-8-sig")

meta.write_text(json.dumps({
    "grafico": "TSNE13",
    "versao": "ESTILO_PCA_V2",
    "metodo": "TSNE",
    "dimensao": "1D",
    "n_total": int(len(x)),
    "n_legitimas": int((y == 0).sum()),
    "n_fraudes": int((y == 1).sum()),
    "media_global_tsne1": media_global,
    "tempo_tsne_segundos": tempo,
    "f10_real_acessado": False,
    "dados_reais_utilizados": False
}, indent=2, ensure_ascii=False), encoding="utf-8")

print("=== TSNE13 ESTILO PCA V2 ===")
print(f"PNG={png}")
print(f"CSV={csv}")
print(f"N_TOTAL={len(x)}")
print(f"N_LEGITIMAS={(y == 0).sum()}")
print(f"N_FRAUDES={(y == 1).sum()}")
print(f"MEDIA_GLOBAL_TSNE1={media_global:.7f}")
print(f"TEMPO_TSNE={tempo:.3f}s")
print("F10_REAL_ACESSADO=NAO")
print("DADOS_REAIS_UTILIZADOS=NAO")