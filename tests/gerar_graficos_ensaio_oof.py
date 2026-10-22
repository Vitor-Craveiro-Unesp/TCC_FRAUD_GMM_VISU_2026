r"""Gera visualizacoes exploratorias do ensaio OOF concluido, sem retreinar.

Uso (na raiz do projeto):
    python .\tests\gerar_graficos_ensaio_oof.py outputs\ensaio_pca_gmm_20261017_195252

Saidas: <experimento>/vencedor/visualizacao/oof/ensaio_*.png
Nunca carrega dados do F10 nem sobrescreve arquivos existentes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
)


def salvar(fig, destino: Path):
    """Nao sobrescrever figuras existentes."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        print(f"[EXISTENTE] {destino.name} (preservado)")
    else:
        fig.savefig(destino, dpi=180, bbox_inches="tight")
        print(f"[GERADO] {destino.name}")
    plt.close(fig)


def mcc_vetorizado(y: np.ndarray, p: np.ndarray):
    """MCC para todos os limiares observados, O(n log n)."""
    ordem = np.argsort(-p, kind="mergesort")
    s = p[ordem]
    y_ = y[ordem].astype(np.int64)
    n = len(y)
    positivos = int(y_.sum())
    negativos = n - positivos
    limites = np.flatnonzero(np.r_[s[:-1] != s[1:], True])
    tp = np.cumsum(y_)[limites].astype(float)
    fp = (limites + 1).astype(float) - tp
    fn = positivos - tp
    tn = negativos - fp
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denom, out=np.zeros_like(tp), where=denom > 0)
    return s[limites], mcc


def comparar_grade(df: pd.DataFrame, destino: Path):
    xs = np.arange(len(df))
    rotulos = []
    for r in df.itertuples():
        cfg = json.loads(r.configuracao)
        rotulos.append(f"PCA {cfg['reducao']['n_components']} | K {cfg['gmm']['n_components']}")

    for coluna, titulo, arquivo in (
        ("ap", "Average Precision (AP)", "comparacao_ap"),
        ("mcc", "MCC (limiar otimizado no OOF)", "comparacao_mcc"),
        ("logloss", "LogLoss (menor e melhor)", "comparacao_logloss"),
        ("score", "Score composto (maior e melhor)", "comparacao_score"),
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        vals = df[coluna].to_numpy(float)
        barras = ax.bar(xs, vals)
        barras[0].set_hatch("///")  # Primeiro candidato apos ordenar pelo score
        ax.set_xticks(xs, rotulos, rotation=20, ha="right")
        ax.set_ylabel(coluna.upper())
        ax.set_title(f"Ensaio OOF (10% desenvolvimento) — {titulo}")
        ax.grid(axis="y", alpha=.25)
        for barra, val in zip(barras, vals):
            ax.annotate(f"{val:.5f}", (barra.get_x() + barra.get_width()/2, barra.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        amplitude = max(np.ptp(vals), .002)
        ax.set_ylim(bottom=min(0, np.min(vals) - .35*amplitude), top=max(vals) + .55*amplitude)
        salvar(fig, destino / f"ensaio_{arquivo}.png")


def matriz(y, p, limiar, titulo, destino):
    cm = confusion_matrix(y, p >= limiar, labels=[0, 1]); ideal = np.diag(cm.sum(axis=1))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            total_classe = cm[i, :].sum()
            pct = 100 * cm[i,j] / total_classe if total_classe else float('nan')
            ax.text(j, i, f"Observado: {cm[i,j]:,}\nIdeal: {ideal[i,j]:,}\n{pct:.2f}% da classe",
                    ha="center", va="center", fontsize=10,
                    color="white" if cm[i,j] > cm.max()*.5 else "black")
    ax.set_xticks([0, 1], ["Legítima", "Fraude"])
    ax.set_yticks([0, 1], ["Legítima", "Fraude"])
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real")
    ax.set_title(titulo)
    salvar(fig, destino)


def vencedor_oof(df, pasta, saidas):
    win = df.iloc[0]
    cid = str(win["id_candidato"])
    arq = pasta / "vencedor" / "oof" / "predicoes.npz"
    if not arq.is_file():
        raise FileNotFoundError(f"Predicoes OOF do candidato vencedor ausentes: {arq}")
    with np.load(arq, allow_pickle=False) as z:
        y = z["y_true"].astype(int)
        p = z["prob_fraude"].astype(float)
        folds = z["fold"].astype(int)
        esperado = str(z["id_candidato"].item())
        fold_esperado = int(z["fold_esperado"].item())
    if esperado != cid or fold_esperado != 0:
        raise ValueError("Arquivo de predicoes nao corresponde ao candidato OOF consolidado")
    if not (np.isin(folds, range(1,10)).all() and len(y) == int(win["n"])):
        raise ValueError("Predicoes incluem folds indevidos ou quantidade inconsistente")
    if not (np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()):
        raise ValueError("Probabilidades invalidas")
    ap = average_precision_score(y, p)
    if not np.isclose(ap, float(win["ap"]), rtol=1e-6, atol=1e-8):
        raise ValueError("AP nao confere com CSV; resultados inconsistentes")
    limiar = float(win["limiar_mcc"])
    legenda = f"PCA {json.loads(win['configuracao'])['reducao']['n_components']} | GMM {json.loads(win['configuracao'])['gmm']['n_components']}"

    precisao, recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(recall, precisao, lw=1.5, label=f"{legenda} | AP={ap:.5f}")
    ax.axhline(y.mean(), ls="--", color="gray", label=f"Prevalência={y.mean():.4%}")
    ax.set(xlabel="Recall", ylabel="Precisão", xlim=(0, 1), ylim=(0, 1),
           title="Curva Precision–Recall — vencedor OOF (ensaio 10%)")
    ax.legend(); ax.grid(alpha=.25)
    salvar(fig, saidas / "ensaio_vencedor_curva_pr.png")

    thresholds, mcc = mcc_vetorizado(y, p)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Subamostragem SOMENTE para desenhar, sem otimizar de novo
    inds = np.unique(np.linspace(0, len(thresholds)-1, min(2000, len(thresholds)), dtype=int))
    ax.plot(thresholds[inds], mcc[inds], lw=1.1, label="MCC por limiar (OOF)")
    ax.axvline(limiar, ls="--", color="black", label=f"Limiar OOF={limiar:.5f}")
    ax.scatter([limiar], [float(win['mcc'])], zorder=5, marker="o", label=f"MCC={win['mcc']:.5f}")
    ax.set(xlabel="Limiar de probabilidade", ylabel="MCC", title="MCC por limiar — vencedor OOF (ensaio 10%)")
    ax.grid(alpha=.25); ax.legend()
    salvar(fig, saidas / "ensaio_vencedor_mcc_por_limiar.png")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(p[y==0], bins=50, histtype="step", density=True, label="Legítimas")
    ax.hist(p[y==1], bins=50, histtype="step", density=True, label="Fraudes")
    ax.axvline(limiar, ls="--", color="black", label=f"Limiar OOF={limiar:.5f}")
    ax.set(xlabel="Score de fraude", ylabel="Densidade por classe",
           title="Distribuição dos scores — vencedor OOF (ensaio 10%)")
    ax.grid(alpha=.2); ax.legend()
    salvar(fig, saidas / "ensaio_vencedor_distribuicao_scores.png")

    matriz(y,p,.5,"Matriz de confusão — limiar 0,5 (OOF amostral)",saidas / "ensaio_vencedor_matriz_confusao_05.png")
    matriz(y,p,limiar, "Matriz de confusão — limiar MCC OOF amostral", saidas / "ensaio_vencedor_matriz_confusao_mcc.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experimento", type=Path, help="Pasta do ensaio OOF ja concluido")
    args = parser.parse_args()
    pasta = args.experimento.resolve()
    arquivo = pasta / "vencedor" / "metricas" / "resultados_grade.csv"
    if not arquivo.is_file():
        raise FileNotFoundError(f"Nao encontrei {arquivo}")
    tabela = pd.read_csv(arquivo)
    obrigatorias = {'id_candidato','ap','mcc','logloss','score','limiar_mcc','n','configuracao'}
    if not obrigatorias.issubset(tabela.columns):
        raise ValueError(f"Colunas ausentes no CSV: {obrigatorias - set(tabela.columns)}")
    tabela = tabela.sort_values(['score','ap'], ascending=False, kind='stable').reset_index(drop=True)
    destino = pasta / "vencedor" / "visualizacao" / "oof"
    comparar_grade(tabela, destino)
    vencedor_oof(tabela, pasta, destino)
    print(f"\n9 graficos de diagnostico OOF em: {destino}")
    print("F10 nao acessado. Nenhum modelo foi treinado. Arquivos existentes preservados.")

if __name__ == '__main__':
    main()
