"""Passo 03.02: verifica imputacao e padronizacao com amostra da base real.

Executar na raiz: python .\tests\verificar_imputacao_normalizacao_base_real.py
Ajusta encoder/imputador/scaler exclusivamente em F1-F8; transforma F9.
F10 e o Parquet original permanecem intocados.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.preprocessing import (
    EncodersPorSufixo,
    ImputadorPadronizador,
    selecionar_features,
)


def main():
    with (RAIZ / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").open(
        encoding="utf-8-sig"
    ) as arq:
        config = yaml.safe_load(arq)

    dados = config["dados"]
    parquet = pq.ParquetFile(RAIZ / dados["arquivo"])
    manifestos = list((RAIZ / "data" / "splits").glob("*/manifesto_folds.json"))
    compativeis = []
    for arquivo in manifestos:
        meta = json.loads(arquivo.read_text(encoding="utf-8"))["parametros"]
        if (
            meta["caminho_relativo"] == dados["arquivo"]
            and meta["coluna_alvo"] == dados["coluna_alvo"]
            and meta["total_linhas"] == parquet.metadata.num_rows
            and meta["fold_teste"] == 10
        ):
            compativeis.append(arquivo.parent / "folds.npy")
    if len(compativeis) != 1:
        raise RuntimeError(
            "E necessaria exatamente uma divisao compativel em data/splits; "
            f"encontradas: {len(compativeis)}."
        )
    folds = np.load(compativeis[0], allow_pickle=False)
    if len(folds) != parquet.metadata.num_rows:
        raise ValueError("Folds e Parquet possuem quantidades de linhas diferentes.")

    limites = {"treino": 10_000, "validacao": 2_000}
    fatias = {"treino": [], "validacao": []}
    quantidades = {"treino": 0, "validacao": 0}
    inicio = 0
    for batch in parquet.iter_batches(batch_size=5_000):
        df = batch.to_pandas()
        ids_fold = folds[inicio:inicio + len(df)]
        inicio += len(df)
        mascaras = {
            "treino": (ids_fold >= 1) & (ids_fold <= 8),
            "validacao": ids_fold == 9,
        }
        for grupo, mascara in mascaras.items():
            faltam = limites[grupo] - quantidades[grupo]
            if faltam > 0:
                amostra = df.loc[mascara].head(faltam).copy()
                if not amostra.empty:
                    fatias[grupo].append(amostra)
                    quantidades[grupo] += len(amostra)
        if all(quantidades[k] == limites[k] for k in limites):
            break
    if any(quantidades[k] < limites[k] for k in limites):
        raise ValueError(f"Amostras insuficientes: {quantidades}")

    df_treino = pd.concat(fatias["treino"], ignore_index=True)
    df_validacao = pd.concat(fatias["validacao"], ignore_index=True)
    x_treino = selecionar_features(df_treino, config)
    x_validacao = selecionar_features(df_validacao, config)

    encoder = EncodersPorSufixo(config).fit(x_treino)
    enc_treino = encoder.transform(x_treino)
    enc_validacao = encoder.transform(x_validacao)

    processador = ImputadorPadronizador(config, encoder).fit(enc_treino)
    proc_treino = processador.transform(enc_treino)
    proc_validacao = processador.transform(enc_validacao)

    assert list(proc_treino.columns) == list(proc_validacao.columns)
    assert proc_treino.shape == enc_treino.shape
    assert proc_validacao.shape == enc_validacao.shape
    assert np.isfinite(proc_treino.to_numpy()).all()
    assert np.isfinite(proc_validacao.to_numpy()).all()
    for coluna in encoder.colunas_:
        if encoder.regras_[coluna] in ("ciclica", "binary"):
            # Preservar todos os valores que ja eram finitos antes do 03.02.
            esperado = enc_validacao[coluna].to_numpy()
            produzido = proc_validacao[coluna].to_numpy()
            mascara = np.isfinite(esperado)
            np.testing.assert_array_equal(produzido[mascara], esperado[mascara])
    colunas_onehot = [
        c for c in enc_treino.columns
        if c not in encoder.colunas_ and any(
            c == f"{orig}__{categoria}"
            for orig, regra in encoder.regras_.items() if regra == "onehot"
            for categoria in encoder.categorias_[orig]
        )
    ]
    if colunas_onehot:
        np.testing.assert_array_equal(
            proc_validacao[colunas_onehot].to_numpy(),
            enc_validacao[colunas_onehot].to_numpy(),
        )

    print("[OK] Parquet lido por lotes, sem alterar o arquivo original.")
    print(f"[OK] Treino: {len(proc_treino):,} linhas de F1-F8.")
    print(f"[OK] Validacao: {len(proc_validacao):,} linhas de F9.")
    print("[OK] F10 fora do ajuste e da verificacao de validacao.")
    print(f"[OK] Features apos preprocessing: {proc_treino.shape[1]}.")
    print(f"[OK] Continuas padronizadas: {len(processador.padronizadas_)}.")
    print(f"[OK] Medianas aprendidas no treino: {len(processador.mediana_)}.")
    print("[OK] Sem NaN ou infinito em treino/validacao.")
    print("[OK] Ciclicas, binarias e One-Hot preservadas quando observadas.")
    print("PASSO 03.02 - VERIFICACAO COM BASE REAL APROVADA")


if __name__ == "__main__":
    main()
