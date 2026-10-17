"""Teste de integracao do passo 03.01 em amostras reais de desenvolvimento.

Executar na raiz do projeto: python tests/verificar_encoders_base_real.py
Nao ajusta transformacoes com os dados de F9 nem le registros de F10 para fit.
Nao altera o Parquet e nao produz artefatos de treino.
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
from src.preprocessing import EncodersPorSufixo, selecionar_features



def main():
    with (RAIZ / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").open(
        encoding="utf-8-sig"
    ) as arquivo:
        config = yaml.safe_load(arquivo)

    dados = config["dados"]
    parquet = pq.ParquetFile(RAIZ / dados["arquivo"])
    manifests = list((RAIZ / "data" / "splits").glob("*/manifesto_folds.json"))
    validos = []
    for manifesto in manifests:
        meta = json.loads(manifesto.read_text(encoding="utf-8"))["parametros"]
        if (
            meta["caminho_relativo"] == dados["arquivo"]
            and meta["coluna_alvo"] == dados["coluna_alvo"]
            and meta["total_linhas"] == parquet.metadata.num_rows
            and meta["fold_teste"] == 10
        ):
            validos.append(manifesto.parent / "folds.npy")
    if len(validos) != 1:
        raise RuntimeError(
            f"Esperava exatamente uma divisao compatível em data/splits, encontrei {len(validos)}. "
            "Verifique a divisao salva antes de continuar."
        )

    folds = np.load(validos[0], allow_pickle=False)
    if len(folds) != parquet.metadata.num_rows:
        raise ValueError("Numero de folds diferente da quantidade de linhas da base.")

    # Pequena amostra: treino dos F1..F8; validacao exclusivamente no F9.
    limites = {"treino": 10_000, "validacao": 2_000}
    selecionados = {"treino": [], "validacao": []}
    totais = {"treino": 0, "validacao": 0}
    inicio = 0
    for batch in parquet.iter_batches(batch_size=5_000):
        bloco = batch.to_pandas()
        numeros = folds[inicio:inicio + len(bloco)]
        inicio += len(bloco)
        selecoes = {
            "treino": (numeros >= 1) & (numeros <= 8),
            "validacao": numeros == 9,
        }
        for nome, mascara in selecoes.items():
            faltam = limites[nome] - totais[nome]
            if faltam <= 0:
                continue
            fatia = bloco.loc[mascara].head(faltam).copy()
            if not fatia.empty:
                selecionados[nome].append(fatia)
                totais[nome] += len(fatia)
        if all(totais[k] >= limites[k] for k in limites):
            break

    if any(totais[k] < limites[k] for k in limites):
        raise ValueError(f"Amostras insuficientes para o teste: {totais}")

    df_treino = pd.concat(selecionados["treino"], ignore_index=True)
    df_validacao = pd.concat(selecionados["validacao"], ignore_index=True)
    X_treino = selecionar_features(df_treino, config)
    X_validacao = selecionar_features(df_validacao, config)
    codificador = EncodersPorSufixo(config).fit(X_treino)
    X_treino_codificado = codificador.transform(X_treino)
    X_validacao_codificado = codificador.transform(X_validacao)

    assert X_treino_codificado.shape[1] == X_validacao_codificado.shape[1]
    assert list(X_treino_codificado.columns) == list(X_validacao_codificado.columns)
    assert dados["coluna_alvo"] not in X_treino_codificado.columns
    if dados.get("coluna_id"):
        assert dados["coluna_id"] not in X_treino_codificado.columns

    print("[OK] Base real lida em lotes, sem modificar o Parquet.")
    print(f"[OK] Treinamento: {len(X_treino):,} linhas de F1-F8.")
    print(f"[OK] Validacao: {len(X_validacao):,} linhas de F9.")
    print("[OK] F10 reservado e nao utilizado no ajuste.")
    print(f"[OK] Features originais: {X_treino.shape[1]}")
    print(f"[OK] Features apos encoding: {X_treino_codificado.shape[1]}")
    print("[OK] Mesma estrutura de colunas em treino e validacao.")
    print("PASSO 03.01 — VERIFICACAO COM BASE REAL APROVADA")


if __name__ == "__main__":
    main()
