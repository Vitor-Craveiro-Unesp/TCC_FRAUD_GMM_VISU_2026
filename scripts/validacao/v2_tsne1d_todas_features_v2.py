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

from src import orquestracao_oof as orq
from src import preprocessing as prep
from src.reducao_tsne import RedutorTSNE

RAIZ = Path.cwd()

EXP = RAIZ / (
    "outputs/ensaio_pca_gmm_20261030_114404"
)

SAIDA = RAIZ / (
    "outputs/validacao_tsne_v2_1d/"
    "TSNE_EXTRA_TODAS_FEATURES_V2"
)

PNG = SAIDA / "TSNE_ASSOCIACAO_TODAS_FEATURES.png"
CSV = SAIDA / "TSNE_ASSOCIACAO_TODAS_FEATURES.csv"

config_path = EXP / "configuracao.yaml"
manifesto_path = EXP / "manifesto.json"
npz_path = EXP / "vencedor/oof/predicoes.npz"

protegidos = [
    RAIZ / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml",
    RAIZ / "src/preprocessing.py",
    RAIZ / "src/reducao_tsne.py",
    config_path,
    manifesto_path,
    npz_path,
]

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

for p in protegidos:
    assert p.is_file(), f"Arquivo ausente: {p}"

antes = {str(p): sha(p) for p in protegidos}

assert not SAIDA.exists()

# 01 - Recuperar somente registros de desenvolvimento
config_pca = yaml.safe_load(
    config_path.read_text(encoding="utf-8-sig")
)
manifesto = json.loads(
    manifesto_path.read_text(encoding="utf-8-sig")
)

assert config_pca["reducao"]["metodo"].upper() == "PCA"
assert int(config_pca["divisao"]["qual_fold_teste"]) == 10

with np.load(npz_path, allow_pickle=False) as z:
    ids = np.asarray(z["linha_original"], dtype=np.int64)
    folds_oof = np.asarray(z["fold"], dtype=np.uint8)
    fold_esperado = int(z["fold_esperado"].item())

assert fold_esperado == 0
assert len(ids) > 1200
assert len(ids) == len(folds_oof)
assert np.all(np.diff(ids) > 0)
assert set(np.unique(folds_oof)) == set(range(1, 10))

parquet = RAIZ / config_pca["dados"]["arquivo"]
assert parquet.is_file()

folds_originais, arquivo_folds, hash_dados = (
    orq._identificar_folds(RAIZ, config_pca, parquet)
)

assert hash_dados == manifesto["sha256_parquet"]
assert orq.hash_arquivo(arquivo_folds) == manifesto["sha256_folds"]
assert np.array_equal(folds_originais[ids], folds_oof)
assert np.all(folds_originais[ids] != 10)

print(f"REGISTROS_DESENVOLVIMENTO={len(ids)}", flush=True)
print("F10_REGISTROS_INCLUIDOS=0", flush=True)

# Leitura seletiva conforme o pipeline oficial.
df = orq._ler_desenvolvimento(parquet, ids)

assert len(df) == len(ids)

# 02 - Todas as features originais e derivadas
X_raw = prep.selecionar_features(df, config_pca)

assert config_pca["dados"]["coluna_alvo"] not in X_raw.columns
assert config_pca["dados"]["coluna_id"] not in X_raw.columns

n_originais = len(X_raw.columns)

# Encoder e scaler ajustados somente na referencia F1-F9.
# Referencia descritiva: nao substitui os ajustes 8F do OOF.
encoder = prep.EncodersPorSufixo(config_pca).fit(X_raw)

X_enc_full = encoder.transform(X_raw)

imputador = prep.ImputadorPadronizador(
    config_pca, encoder
).fit(X_enc_full)

del X_enc_full, df

n_codificadas = len(encoder.saidas_)

assert n_codificadas > 12

# 03 - Amostra auxiliar para ajuste t-SNE
# Todos os atributos sao mantidos; somente as LINHAS
# sao amostradas para limitar o custo computacional.
rng = np.random.default_rng(42)

n_amostra = 1200

posicoes = np.sort(
    rng.choice(len(X_raw), n_amostra, replace=False)
)

X_amostra_raw = X_raw.iloc[posicoes].copy()

X_amostra_cod = encoder.transform(X_amostra_raw)

X_pre = imputador.transform(X_amostra_cod)

assert X_pre.shape == (n_amostra, n_codificadas)
assert list(X_pre.columns) == encoder.saidas_
assert X_pre.columns.is_unique
assert np.isfinite(X_pre.to_numpy()).all()

print(f"FEATURES_ORIGINAIS={n_originais}", flush=True)
print(f"FEATURES_APOS_ENCODING={n_codificadas}", flush=True)
print(f"FEATURES_USADAS_NO_TSNE={X_pre.shape[1]}", flush=True)

del X_raw, X_amostra_raw, X_amostra_cod

# 04 - t-SNE auxiliar 1D
config = deepcopy(config_pca)

config["reducao"]["metodo"] = "TSNE"
config["experimento"]["semente"] = 42
config["execucao"]["n_jobs"] = 2

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

redutor = RedutorTSNE(
    config, parametros_tsne
).fit(X_pre)

coordenadas = redutor.transform_treino()

assert coordenadas.shape == (n_amostra, 1)
assert coordenadas.index.equals(X_pre.index)

z = coordenadas["TSNE1"]

# 05 - Spearman para TODAS as features
registros = []

for nome in X_pre.columns:
    s = X_pre[nome]

    constante = s.nunique(dropna=False) < 2

    if constante:
        rho = np.nan
    else:
        rho = float(s.corr(z, method="spearman"))
        assert np.isfinite(rho)
        assert -1 <= rho <= 1

    registros.append({
        "variavel_preprocessada": nome,
        "correlacao_spearman_TSNE1": rho,
        "associacao_absoluta": abs(rho) if np.isfinite(rho) else np.nan,
        "constante_na_amostra": bool(constante),
    })

tabela = pd.DataFrame(registros)

assert len(tabela) == n_codificadas
assert tabela["variavel_preprocessada"].is_unique
assert set(tabela["variavel_preprocessada"]) == set(X_pre.columns)

# Sem Top N: todas as variaveis entram.
tabela = tabela.sort_values(
    ["associacao_absoluta", "variavel_preprocessada"],
    ascending=[False, True],
    na_position="last",
    kind="stable",
).reset_index(drop=True)

SAIDA.mkdir(parents=True, exist_ok=False)

tabela.to_csv(
    CSV,
    index=False,
    encoding="utf-8-sig",
    float_format="%.17g",
)

# 06 - Visual no mesmo padrao do PCA
n = len(tabela)
altura = max(9, 2.2 + 0.39 * n)

fig, ax = plt.subplots(
    figsize=(19, altura),
    facecolor="white",
)

ax.axvline(
    0,
    color="#79838f",
    linewidth=1.1,
)

cores = plt.get_cmap("tab20")
posicoes_y = np.arange(n)

for i, row in tabela.iterrows():
    valor = row["correlacao_spearman_TSNE1"]

    if pd.isna(valor):
        ax.plot(
            0, i, marker="x",
            color="#777777",
            markersize=5,
        )
        ax.text(
            0.035, i, "n/d (constante)",
            ha="left",
            va="center",
            fontsize=8,
            color="#777777",
        )
        continue

    cor = cores(i % 20)

    ax.annotate(
        "",
        xy=(valor, i),
        xytext=(0, i),
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.8,
            "color": cor,
            "mutation_scale": 12,
        },
    )

    valor_exibido = (
        0.0 if abs(valor) < 0.00000005 else valor
    )

    ax.text(
        valor + (0.045 if valor >= 0 else -0.045),
        i,
        f"{valor_exibido:+.7f}",
        ha="left" if valor >= 0 else "right",
        va="center",
        fontsize=8,
        color=cor,
    )

ax.set_yticks(
    posicoes_y,
    labels=tabela["variavel_preprocessada"].tolist(),
)

ax.set_ylim(n - 0.85, -0.85)
ax.set_xlim(-1.35, 1.35)

ax.set_xticks([-1, -0.5, 0, 0.5, 1])

ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f"{x:.7f}")
)

ax.set_xlabel(
    "Correlação de Spearman com TSNE1",
    fontsize=12,
)

ax.set_title(
    "ASSOCIAÇÃO DE TODAS AS FEATURES COM TSNE1",
    fontsize=19,
    fontweight="bold",
    pad=20,
)

ax.tick_params(
    axis="y",
    labelsize=8,
    length=0,
    pad=7,
)
ax.tick_params(axis="x", labelsize=10)

ax.grid(
    axis="x",
    alpha=0.22,
    linestyle=":",
)
ax.set_axisbelow(True)

fig.text(
    0.5,
    0.012,
    "t-SNE 1D auxiliar | Todas as features preprocessadas | "
    "Amostra F1-F9 | Correlações descritivas, não coeficientes do t-SNE",
    ha="center",
    fontsize=9,
    color="#52647a",
)

fig.subplots_adjust(
    left=0.35,
    right=0.96,
    top=0.94,
    bottom=0.075,
)

fig.savefig(
    PNG,
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)

# 07 - Conferencia completa
salvo = pd.read_csv(CSV)

assert len(salvo) == n_codificadas
assert salvo["variavel_preprocessada"].is_unique

assert set(salvo["variavel_preprocessada"]) == set(X_pre.columns)

np.testing.assert_allclose(
    salvo["correlacao_spearman_TSNE1"].to_numpy(),
    tabela["correlacao_spearman_TSNE1"].to_numpy(),
    equal_nan=True,
    rtol=1e-10,
    atol=1e-10,
)

depois = {str(p): sha(p) for p in protegidos}

assert antes == depois, "Arquivo protegido alterado."
assert PNG.is_file() and PNG.stat().st_size > 1000

# 08 - Manifesto
meta = {
    "tipo": "PREVIA_DIAGNOSTICA_NAO_OFICIAL",
    "grafico": "TSNE_ASSOCIACAO_TODAS_FEATURES",
    "metodo": "TSNE",
    "dimensao": 1,
    "origem": str(EXP),
    "n_registros_desenvolvimento": int(len(ids)),
    "n_amostra_tsne": n_amostra,
    "n_features_originais": n_originais,
    "n_features_preprocessadas": n_codificadas,
    "n_features_exibidas": len(tabela),
    "n_features_constantes_na_amostra": int(
        tabela["constante_na_amostra"].sum()
    ),
    "medida": "correlacao_spearman",
    "coeficientes_lineares_tsne": False,
    "preprocessamento_referencia": "F1-F9 global; apenas diagnostico",
    "oof_completo_executado": False,
    "f10_incluido": False,
}

(SAIDA / "manifesto.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print()
print("=== V2 / TSNE1 / TODAS AS FEATURES ===")
print(f"FEATURES_ORIGINAIS={n_originais}")
print(f"FEATURES_PREPROCESSADAS={n_codificadas}")
print(f"FEATURES_NO_GRAFICO={len(tabela)}")
print(f"FEATURES_CONSTANTES={int(tabela['constante_na_amostra'].sum())}")
print("TOP_N_APLICADO=NAO")
print("TODAS_FEATURES_INCLUIDAS=SIM")
print("REGISTROS_F10_INCLUIDOS=0")
print("OOF_COMPLETO_EXECUTADO=NAO")
print("ARQUIVOS_OFICIAIS_ALTERADOS=NAO")
print(f"PNG={PNG}")
print(f"CSV={CSV}")
print("V2_TSNE_FEATURES=AGUARDANDO_APROVACAO_VISUAL")