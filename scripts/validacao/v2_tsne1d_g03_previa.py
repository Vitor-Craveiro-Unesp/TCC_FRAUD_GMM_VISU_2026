"""V2 | t-SNE 1D OOF | grafico 03 | PREVIA SINTETICA NAO OFICIAL.

Reutiliza _matriz de src.visualizacao_oof sem modificar o renderizador.
Nenhum treinamento; nao carrega parquet, folds.npy ou F10. Os 360 valores
sao sinteticos e reproduziveis com seed fixa, como os graficos 00/01/02.
Executar na raiz TCC_EXPERIMENTAL_V3:
    python scripts/validacao/v2_tsne1d_g03_previa.py
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
)

from src import visualizacao_oof as vis


RAIZ = Path(__file__).resolve().parents[2]
YAML_OFICIAL = RAIZ / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml"
RENDERIZADOR = RAIZ / "src" / "visualizacao_oof.py"
SAIDA = (RAIZ / "outputs" / "validacao_tsne_v2_1d" /
         "03_matriz_confusao_050_previa_sintetica")
NOME = "03_matriz_confusao_05"


def _hash(arquivo: Path) -> str:
    return sha256(arquivo.read_bytes()).hexdigest()


def _base_sintetica():
    # Reproduz exatamente a semente e a geracao de scores de G00/G01/G02.
    rng = np.random.default_rng(20261101)
    folds = np.repeat(np.arange(1, 10, dtype=np.uint8), 40)
    y = np.zeros(360, dtype=np.uint8)
    for k in range(9):
        locais = rng.choice(40, size=3, replace=False)
        y[k * 40 + locais] = 1

    scores = rng.beta(1.8, 10.0, size=360)
    scores[y == 1] = rng.beta(2.2, 2.8, size=int(y.sum()))
    scores = np.clip(scores, 1e-8, 1.0 - 1e-8)
    return folds, y, scores


def main() -> None:
    hashes_antes = {str(p): _hash(p) for p in (YAML_OFICIAL, RENDERIZADOR)}
    config = deepcopy(yaml.safe_load(YAML_OFICIAL.read_text(encoding="utf-8-sig")))
    # Toda alteracao de configuracao acontece APENAS em memoria.
    config["reducao"]["metodo"] = "TSNE"
    config["tsne"]["n_components"] = 1
    config["visualizacao"]["graficos"]["matriz_confusao_05"]["habilitar"] = True

    folds, y, scores = _base_sintetica()
    tau_fixo = 0.5
    cm = confusion_matrix(y, scores >= tau_fixo, labels=[0, 1])
    tn, fp, fn, tp = [int(n) for n in cm.ravel()]
    assert len(folds) == len(y) == len(scores) == 360
    assert int(y.sum()) == 27
    assert set(np.unique(folds)) == set(range(1, 10))
    assert np.array_equal(cm, np.array([[331, 2], [18, 9]])), cm
    assert np.isclose(average_precision_score(y, scores), 0.7096816, atol=5e-8)
    assert np.isclose(matthews_corrcoef(y, scores >= 0.35), 0.6292825, atol=5e-8)

    SAIDA.mkdir(parents=True, exist_ok=True)
    vis._matriz(config, y, scores, tau_fixo, "matriz_confusao_05", SAIDA, "png", 180, True)

    png = SAIDA / f"{NOME}.png"
    csv = SAIDA / f"{NOME}.csv"
    assert png.is_file() and png.stat().st_size > 0, "PNG nao gerado."
    assert csv.is_file(), "CSV nao gerado."
    tabela = pd.read_csv(csv)
    esperado = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    assert len(tabela) == 4
    assert set(tabela["quadrante"]) == set(esperado)
    for quadrante, quantidade in esperado.items():
        registro = tabela.loc[tabela["quadrante"] == quadrante].iloc[0]
        assert int(registro["observado"]) == quantidade, quadrante
        denominador = tn + fp if quadrante in ("TN", "FP") else fn + tp
        assert np.isclose(float(registro["pct_da_classe_real"]),
                          100.0 * quantidade / denominador)
    assert tn + fp + fn + tp == len(y)
    assert {str(p): _hash(p) for p in (YAML_OFICIAL, RENDERIZADOR)} == hashes_antes

    meta = {
        "tipo": "PREVIA_SINTETICA_NAO_OFICIAL",
        "metodo_simulado": "TSNE",
        "dimensao_simulada": 1,
        "folds_sinteticos": list(range(1, 10)),
        "n_sintetico": len(y),
        "fraudes_sinteticas": int(y.sum()),
        "limiar": tau_fixo,
        "contagens": esperado,
        "percentuais_condicionais_a_classe_real": True,
        "f10_real_acessado": False,
        "treinamento_real_executado": False,
        "arquivos_oficiais_alterados": False,
        "observacao": "A simulacao nao representa o desempenho real do t-SNE + GMM.",
    }
    manifesto = SAIDA / "LEIA_ME_PREVIA_SINTETICA.json"
    if manifesto.exists():
        assert json.loads(manifesto.read_text(encoding="utf-8")) == meta, (
            "Manifesto anterior diverge; nao sobrescrever.")
    else:
        manifesto.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")

    ap = average_precision_score(y, scores)
    logloss = log_loss(y, scores, labels=[0, 1])
    print("\n=== V2 / TSNE 1D / GRAFICO 03 - MATRIZ DE CONFUSAO 0.50 ===")
    print("METODO_SIMULADO=TSNE | DIMENSAO_SIMULADA=1D")
    print("FOLDS_DESENVOLVIMENTO=1,2,3,4,5,6,7,8,9")
    print(f"N_SINTETICO={len(y)} | FRAUDES_SINTETICAS={int(y.sum())}")
    print(f"AP_SINTETICA={ap:.7f} | LOGLOSS_SINTETICO={logloss:.7f}")
    print(f"LIMIAR_FIXO={tau_fixo:.7f}")
    print(f"TN={tn} | FP={fp} | FN={fn} | TP={tp}")
    print(f"LEGITIMAS={tn+fp} | FRAUDES={fn+tp}")
    print("PERCENTUAIS=CONDICIONAIS_A_CLASSE_REAL")
    print("F10_REAL_ACESSADO=NAO")
    print("TREINAMENTO_REAL_EXECUTADO=NAO")
    print("ARQUIVOS_OFICIAIS_ALTERADOS=NAO")
    print(f"PNG_PARA_AVALIAR={png}")
    print(f"CSV_VERIFICACAO={csv}")
    print("V2_G03_VALIDACAO=APROVADA_TECNICAMENTE_AGUARDANDO_AVALIACAO_VISUAL")


if __name__ == "__main__":
    main()
