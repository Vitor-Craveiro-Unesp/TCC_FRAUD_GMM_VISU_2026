"""Passo 06.01: smoke test de uma rodada F9 OOF no Parquet real.

Executar da raiz:
    python .\tests\verificar_oof_base_real.py

Leitura em lotes; F1-F8 = treinamento amostral, F9 = validacao amostral.
F10 NAO e lido para os dados do experimento, nem recebe previsoes.
Nao grava outputs; o motor completo com persistencia/retomada virá depois.
"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verificar_pca_base_real import localizar_folds
from src.validacao_oof import grade_candidatos_oof, prever_fold_oof


def exigir(condicao, msg):
    if not condicao:
        raise RuntimeError(msg)


def ler_amostra_com_fold(parquet, folds, n_treino=10_000, n_validacao=2_000):
    limites = {"treino": n_treino, "validacao": n_validacao}
    partes = {k: [] for k in limites}
    identificadores = {k: [] for k in limites}
    totais = {k: 0 for k in limites}
    inicio = 0
    for lote in parquet.iter_batches(batch_size=5000):
        df = lote.to_pandas()
        fs = folds[inicio:inicio+len(df)]
        inicio += len(df)
        for nome, mascara in (
            ("treino", (fs >= 1) & (fs <= 8)),
            ("validacao", fs == 9),
        ):
            faltam = limites[nome]-totais[nome]
            if faltam <= 0:
                continue
            posicoes = np.flatnonzero(mascara)[:faltam]
            if len(posicoes):
                partes[nome].append(df.iloc[posicoes].copy())
                identificadores[nome].append(fs[posicoes].copy())
                totais[nome] += len(posicoes)
        if all(totais[k] == limites[k] for k in limites):
            break
    if any(totais[k] != limites[k] for k in limites):
        raise RuntimeError(f"Amostras incompletas: {totais}")
    treino = pd.concat(partes["treino"], ignore_index=True)
    validacao = pd.concat(partes["validacao"], ignore_index=True)
    f_t = np.concatenate(identificadores["treino"])
    f_v = np.concatenate(identificadores["validacao"])
    return pd.concat([treino, validacao], ignore_index=True), np.r_[f_t, f_v]


def main():
    arquivo_yaml = ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml"
    bytes_yaml = arquivo_yaml.read_bytes()
    cfg = yaml.safe_load(bytes_yaml.decode("utf-8-sig"))
    exigir((cfg["divisao"]["total_folds"], cfg["divisao"]["qual_fold_teste"]) == (10, 10),
           "Este teste requer F10 isolado.")
    exigir(cfg["reducao"]["metodo"].upper() == "PCA",
           "Verificacao curta 06.01 usa PCA para evitar custo t-SNE.")

    arquivo_parquet = ROOT / cfg["dados"]["arquivo"]
    parquet = pq.ParquetFile(arquivo_parquet)
    folds = localizar_folds(cfg, arquivo_parquet, parquet.metadata.num_rows)
    df, f_amostra = ler_amostra_com_fold(parquet, folds)
    exigir(not np.any(f_amostra == 10), "F10 apareceu na amostra da CV.")
    exigir(set(np.unique(f_amostra)) == set(range(1, 10)),
           "Os folds F1-F9 devem estar presentes na amostra.")

    # Copia temporaria habilita RUS em memoria sem mexer no YAML oficial.
    teste_cfg = deepcopy(cfg)
    teste_cfg["amostragem"]["modo"] = ["NORMAL", "RUS"]
    casos = [c for c in grade_candidatos_oof(teste_cfg)
             if c["reducao"]["n_components"] == 2
             and c["gmm"]["n_components"] in (1, 2)]
    exigir(len(casos) == 4, "Esperadas quatro combinacoes PCA2/K1,K2/NORMAL,RUS.")
    alvo = cfg["dados"]["coluna_alvo"]
    p = float(df.loc[f_amostra != 9, alvo].mean())
    exigir(0 < p < 1, "Treino amostral precisa de ambas as classes.")

    for c in casos:
        pred = prever_fold_oof(df, f_amostra, teste_cfg, c, 9)
        exigir(len(pred) == 2000, "Quantidade de previsoes F9 errada.")
        exigir((pred["fold"] == 9).all(), "Scores de outro fold incluídos.")
        exigir(np.array_equal(pred["linha_original"], np.flatnonzero(f_amostra == 9)),
               "Indices posicionais F9 nao preservados.")
        exigir(pred["prob_fraude"].between(0, 1).all(), "Score fora de [0,1].")
        exigir(np.isfinite(pred["prob_fraude"]).all(), "Score nao finito.")
        if c["gmm"]["n_components"] == 1:
            np.testing.assert_allclose(pred["prob_fraude"], p, atol=1e-7)
        print(f"[OK] {c['modo']} | PCA 2D | GMM K={c['gmm']['n_components']} "
              f"| F9: {len(pred):,} scores | media: {pred['prob_fraude'].mean():.6%}")

    exigir(arquivo_yaml.read_bytes() == bytes_yaml, "YAML oficial alterado.")
    print("[OK] 10.000 exemplos F1-F8 para fit; 2.000 F9 apenas para score.")
    print("[OK] RH utiliza todos os exemplos originais do treino amostral.")
    print("[OK] F10 isolado. Nenhum modelo foi ajustado com a validacao.")
    print("[OK] Nenhum resultado experimental gravado; apenas smoke test.")
    print("PASSO 06.01 - VERIFICACAO OOF COM BASE REAL APROVADA")


if __name__ == "__main__":
    main()
