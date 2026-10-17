"""Passo 06.02: smoke test das metricas no F9 do Parquet real.

Executar da raiz do projeto:
    python .\tests\verificar_metricas_base_real.py

IMPORTANTE: avalia AP/MCC/LogLoss em uma AMOSTRA de F9, apenas para testar
integracao. O limiar MCC aqui NAO e o limiar OOF definitivo; ele so sera
estimado depois de consolidar TODAS as nove rodadas F1..F9.
"""
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import matthews_corrcoef

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from verificar_pca_base_real import localizar_folds
from verificar_oof_base_real import ler_amostra_com_fold
from src.validacao_oof import prever_fold_oof, grade_candidatos_oof
from src.metricas_oof import avaliar_probabilidades


def exigir(condicao, mensagem):
    if not condicao:
        raise RuntimeError(mensagem)


def main():
    arquivo_yaml = RAIZ / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml"
    anterior = arquivo_yaml.read_bytes()
    config = yaml.safe_load(anterior.decode("utf-8-sig"))
    exigir(str(config["reducao"]["metodo"]).upper() == "PCA",
           "Este smoke test curto usa PCA; mantenha reducao.metodo=PCA no YAML.")
    exigir(config["divisao"]["qual_fold_teste"] == 10,
           "A configuracao precisa reservar F10.")

    arquivo_parquet = RAIZ / config["dados"]["arquivo"]
    parquet = pq.ParquetFile(arquivo_parquet)
    folds = localizar_folds(config, arquivo_parquet, parquet.metadata.num_rows)
    df, amostra_folds = ler_amostra_com_fold(parquet, folds, n_treino=10_000, n_validacao=2_000)
    exigir(not (amostra_folds == 10).any(), "F10 apareceu na amostra!")
    exigir(set(np.unique(amostra_folds)) == set(range(1, 10)),
           "A amostra precisa conter F1..F9.")
    prevalencia_treino = float(df.loc[amostra_folds != 9, config["dados"]["coluna_alvo"]].mean())

    candidatos = [c for c in grade_candidatos_oof(config)
                  if c["metodo"] == "PCA" and c["reducao"]["n_components"] == 2
                  and c["modo"] == "NORMAL"
                  and c["gmm"]["n_components"] in (1,2)]
    exigir(len(candidatos) == 2, "Esperados PCA2D, NORMAL e K=1/K=2 no YAML.")

    for candidato in candidatos:
        pred = prever_fold_oof(df, amostra_folds, config, candidato, 9)
        exigir(len(pred) == 2_000 and (pred["fold"] == 9).all(), "Predicoes de F9 invalidas.")
        res = avaliar_probabilidades(pred["y_true"], pred["prob_fraude"], config)
        exigir(res["n_fraudes"] > 0 and res["n_legitimas"] > 0,
               "F9 amostral exige ambas as classes.")
        exigir(res["n"] == 2000 and 0 <= res["ap"] <= 1, "AP ou contagem invalida.")
        exigir(-1 <= res["mcc"] <= 1 and res["logloss"] >= 0, "MCC ou LogLoss invalidos.")
        exigir(0 <= res["score"] <= 1, "Score composto invalido.")
        y = pred["y_true"].to_numpy()
        p = pred["prob_fraude"].to_numpy()
        mcc = matthews_corrcoef(y, (p >= res["limiar_mcc"]).astype(int))
        exigir(np.isclose(res["mcc"], mcc), "MCC nao corresponde ao limiar retornado.")
        if candidato["gmm"]["n_components"] == 1:
            np.testing.assert_allclose(p, prevalencia_treino, atol=1e-7)
            exigir(np.isclose(res["mcc"], 0.0), "Para K=1 scores constantes exigem MCC=0.")
        print(f"[OK] F9 amostral | PCA2D NORMAL K={candidato['gmm']['n_components']}: "
              f"AP={res['ap']:.6f} MCC={res['mcc']:.6f} "
              f"LogLoss={res['logloss']:.6f} S={res['score']:.6f} "
              f"limiar={res['limiar_mcc']:.8f}")

    exigir(arquivo_yaml.read_bytes() == anterior, "YAML oficial foi alterado.")
    print("[OK] Treino: 10.000 linhas F1..F8; validacao: 2.000 linhas F9.")
    print("[OK] F10 isolado. YAML e Parquet mantidos intactos.")
    print("[ATENCAO] Limiares/metricas desta amostra F9 NAO sao os resultados OOF finais.")
    print("PASSO 06.02 - VERIFICACAO METRICAS COM BASE REAL APROVADA")


if __name__ == "__main__":
    main()
