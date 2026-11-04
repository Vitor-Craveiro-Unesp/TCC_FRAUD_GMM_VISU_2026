"""V2 | t-SNE 1D OOF | Grafico 04 | PREVIA SINTETICA NAO OFICIAL.

Reutiliza a funcao _matriz de src.visualizacao_oof sem alterar codigo PCA.
Trabalha SOMENTE com 360 observacoes artificiais de F1..F9 e tau*=0.35
ficticio, congelado para consistencia com as previas 00..03.
Nao abre parquet, folds.npy ou dados F10; nao realiza fit de modelo.

Execucao a partir da raiz TCC_EXPERIMENTAL_V3:
    python -m scripts.validacao.v2_tsne1d_g04_previa
ou
    python scripts/validacao/v2_tsne1d_g04_previa.py
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
)

# Permite rodar como script isolado ou via `python -m`.
RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
from src import visualizacao_oof as vis

YAML_OFICIAL = RAIZ / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml"
RENDERIZADOR = RAIZ / "src" / "visualizacao_oof.py"
SAIDA = (
    RAIZ / "outputs" / "validacao_tsne_v2_1d" /
    "04_matriz_confusao_mcc_previa_sintetica"
)
NOME = "04_matriz_confusao_mcc"
TAU_FICTICIO = 0.35


def _hash(arquivo: Path) -> str:
    return sha256(arquivo.read_bytes()).hexdigest()


def _base_sintetica():
    # Mesma geracao e ordem de consumo do RNG dos Graficos 00, 01, 02 e 03.
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
    arquivos_protegidos = (YAML_OFICIAL, RENDERIZADOR)
    hashes_antes = {str(p): _hash(p) for p in arquivos_protegidos}
    config = deepcopy(yaml.safe_load(YAML_OFICIAL.read_text(encoding="utf-8-sig")))

    # Configuracao somente em memoria: nenhum YAML e editado.
    config["reducao"]["metodo"] = "TSNE"
    config["tsne"]["n_components"] = 1
    config["visualizacao"]["graficos"]["matriz_confusao_mcc"]["habilitar"] = True

    folds, y, scores = _base_sintetica()
    tau = TAU_FICTICIO
    cm = confusion_matrix(y, scores >= tau, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())

    # Valores de referencia deterministas da simulacao, nao resultados OOF reais.
    assert len(folds) == len(y) == len(scores) == 360
    assert int(y.sum()) == 27
    assert set(np.unique(folds)) == set(range(1, 10))
    assert np.array_equal(cm, [[319, 14], [7, 20]]), cm
    assert np.isclose(average_precision_score(y, scores), 0.7096816, atol=5e-8)
    mcc_ficticio = float(matthews_corrcoef(y, scores >= tau))
    assert np.isclose(mcc_ficticio, 0.6292825, atol=5e-8)
    assert tn + fp + fn + tp == len(y)

    SAIDA.mkdir(parents=True, exist_ok=True)
    vis._matriz(config, y, scores, tau, "matriz_confusao_mcc", SAIDA, "png", 180, True)

    png = SAIDA / (NOME + ".png")
    csv = SAIDA / (NOME + ".csv")
    assert png.is_file() and png.stat().st_size > 0, "PNG nao gerado."
    assert csv.is_file() and csv.stat().st_size > 0, "CSV nao gerado."
    tabela = pd.read_csv(csv)
    esperado = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    assert len(tabela) == 4 and set(tabela["quadrante"]) == set(esperado)
    for quadrante, quantidade in esperado.items():
        registro = tabela.loc[tabela["quadrante"] == quadrante].iloc[0]
        assert int(registro["observado"]) == quantidade, quadrante
        denominador = tn + fp if quadrante in ("TN", "FP") else fn + tp
        assert np.isclose(
            float(registro["pct_da_classe_real"]),
            100.0 * quantidade / denominador,
            atol=1e-7,
        ), quadrante

    assert {str(p): _hash(p) for p in arquivos_protegidos} == hashes_antes, (
        "Arquivo YAML ou renderizador oficial foi alterado."
    )

    meta = {
        "tipo": "PREVIA_SINTETICA_NAO_OFICIAL",
        "metodo_simulado": "TSNE",
        "dimensao_simulada": 1,
        "folds_sinteticos": list(range(1, 10)),
        "n_sintetico": len(y),
        "fraudes_sinteticas": int(y.sum()),
        "limiar_mcc_oof_ficticio_congelado": tau,
        "limiar_nao_otimizado_sobre_dados_reais": True,
        "contagens": esperado,
        "percentuais_condicionais_a_classe_real": True,
        "f10_real_acessado": False,
        "treinamento_real_executado": False,
        "arquivos_oficiais_alterados": False,
        "observacao": (
            "O rotulo de limiar otimo no PNG e ilustrativo. "
            "Tau=0.35 e ficticio e foi congelado para manter os graficos "
            "sinteticos consistentes; nao representa selecao OOF real."
        ),
    }
    manifesto = SAIDA / "LEIA_ME_PREVIA_SINTETICA.json"
    if manifesto.exists():
        assert json.loads(manifesto.read_text(encoding="utf-8")) == meta, (
            "Manifesto existente divergente; nao sobrescrever."
        )
    else:
        manifesto.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    ap = average_precision_score(y, scores)
    ll = log_loss(y, scores, labels=[0, 1])
    print("\n=== V2 / TSNE 1D / GRAFICO 04 - MATRIZ MCC OOF (SINTETICA) ===")
    print("METODO_SIMULADO=TSNE | DIMENSAO_SIMULADA=1D")
    print("FOLDS_SINTETICOS=1,2,3,4,5,6,7,8,9")
    print(f"N_SINTETICO={len(y)} | FRAUDES_SINTETICAS={int(y.sum())}")
    print(f"AP_SINTETICA={ap:.7f} | LOGLOSS_SINTETICO={ll:.7f}")
    print(f"TAU_FICTICIO_CONGELADO={tau:.7f} | MCC_SINTETICO={mcc_ficticio:.7f}")
    print(f"TN={tn} | FP={fp} | FN={fn} | TP={tp}")
    print("LIMIAR_RECALCULADO_POR_F10=NAO")
    print("F10_REAL_ACESSADO=NAO")
    print("TREINAMENTO_REAL_EXECUTADO=NAO")
    print("ARQUIVOS_OFICIAIS_ALTERADOS=NAO")
    print(f"PNG_PARA_AVALIAR={png}")
    print(f"CSV_VERIFICACAO={csv}")
    print("V2_G04_VALIDACAO=APROVADA_TECNICAMENTE_AGUARDANDO_AVALIACAO_VISUAL")


if __name__ == "__main__":
    main()
