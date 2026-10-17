"""Passo 04.01: testar PCA apos 03.01/03.02 em amostra real.

Execute na raiz: python .\tests\verificar_pca_base_real.py
F1-F8: treino; F9: validacao; F10: reservado. Nao grava resultados na base.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.folds import hash_arquivo
from src.preprocessing import EncodersPorSufixo, ImputadorPadronizador, selecionar_features
from src.reducao import RedutorPCA, grade_pca


def localizar_folds(config, caminho_parquet, n_linhas):
    """Aceita somente o manifesto exato da base e das regras atuais."""
    p = config["dados"]
    d = config["divisao"]
    sha_parquet = hash_arquivo(caminho_parquet)
    compativeis = []
    for arq in (ROOT / "data" / "splits").glob("*/manifesto_folds.json"):
        manifesto = json.loads(arq.read_text(encoding="utf-8"))
        parametros = manifesto["parametros"]
        if (
            parametros.get("caminho_relativo") == p["arquivo"]
            and parametros.get("coluna_alvo") == p["coluna_alvo"]
            and parametros.get("classe_positiva") == p["classe_positiva"]
            and parametros.get("total_folds") == d["total_folds"]
            and parametros.get("fold_teste") == d["qual_fold_teste"]
            and parametros.get("estratificada") == d["estratificada"]
            and parametros.get("embaralhar") == d["embaralhar"]
            and parametros.get("semente") == config["experimento"]["semente"]
            and parametros.get("sha256_parquet") == sha_parquet
            and parametros.get("total_linhas") == n_linhas
        ):
            caminho_folds = arq.parent / "folds.npy"
            if hash_arquivo(caminho_folds) != manifesto["sha256_folds"]:
                raise RuntimeError(f"Divisao corrompida: {caminho_folds}")
            compativeis.append(caminho_folds)
    if len(compativeis) != 1:
        raise RuntimeError(
            "Esperado exatamente um manifesto compatível da divisão. "
            f"Encontrados: {len(compativeis)}. Execute novamente src/folds.py."
        )
    folds = np.load(compativeis[0], allow_pickle=False)
    if len(folds) != n_linhas:
        raise RuntimeError("Quantidade de folds nao corresponde ao Parquet.")
    return folds


def ler_amostras(parquet, folds):
    limites = {"treino": 10_000, "validacao": 2_000}
    fatias = {nome: [] for nome in limites}
    totais = {nome: 0 for nome in limites}
    inicio = 0
    for lote in parquet.iter_batches(batch_size=5000):
        df = lote.to_pandas()
        b_folds = folds[inicio:inicio + len(df)]
        inicio += len(df)
        # Divisao da CV: 8 folds de treino, 1 de validacao; F10 fora.
        mascaras = {
            "treino": (b_folds >= 1) & (b_folds <= 8),
            "validacao": b_folds == 9,
        }
        for nome, mascara in mascaras.items():
            faltam = limites[nome] - totais[nome]
            if faltam > 0:
                parte = df.loc[mascara].head(faltam).copy()
                if len(parte):
                    fatias[nome].append(parte)
                    totais[nome] += len(parte)
        if all(totais[nome] == limites[nome] for nome in limites):
            break
    if any(totais[k] != limites[k] for k in limites):
        raise ValueError(f"Amostras insuficientes: {totais}")
    return (pd.concat(fatias["treino"], ignore_index=True),
            pd.concat(fatias["validacao"], ignore_index=True))


def main():
    with (ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").open(
        encoding="utf-8-sig"
    ) as arq:
        config = yaml.safe_load(arq)
    d = config["divisao"]
    if (d["total_folds"], d["qual_fold_teste"]) != (10, 10):
        raise ValueError("O teste 04.01 usa 10 folds e reserva F10, conforme o projeto.")
    if config["reducao"]["metodo"].upper() != "PCA":
        raise ValueError("A verificacao 04.01 exige reducao.metodo: PCA.")

    caminho = ROOT / config["dados"]["arquivo"]
    parquet = pq.ParquetFile(caminho)
    folds = localizar_folds(config, caminho, parquet.metadata.num_rows)
    treino, validacao = ler_amostras(parquet, folds)

    x_treino = selecionar_features(treino, config)
    x_validacao = selecionar_features(validacao, config)
    enc = EncodersPorSufixo(config).fit(x_treino)
    t_enc = enc.transform(x_treino)
    v_enc = enc.transform(x_validacao)
    imp = ImputadorPadronizador(config, enc).fit(t_enc)
    t_proc = imp.transform(t_enc)
    v_proc = imp.transform(v_enc)

    grade = list(grade_pca(config))
    if not grade:
        raise RuntimeError("Grade PCA vazia.")
    for parametros in grade:
        pca = RedutorPCA(config, parametros).fit(t_proc)
        medias = pca.modelo_.mean_.copy()
        t_pca = pca.transform(t_proc)
        v_pca = pca.transform(v_proc)
        np.testing.assert_array_equal(pca.modelo_.mean_, medias)
        assert len(v_pca) == 2000
        assert t_pca.shape[1] == v_pca.shape[1] == parametros["n_components"]
        assert t_pca.columns.equals(v_pca.columns)
        assert np.isfinite(t_pca.to_numpy()).all()
        assert np.isfinite(v_pca.to_numpy()).all()
        explicado = float(pca.modelo_.explained_variance_ratio_.sum())
        assert -1e-8 <= explicado <= 1 + 1e-5
        print(f"[OK] PCA {parametros}: "
              f"treino {t_pca.shape}, validacao {v_pca.shape}, "
              f"variancia explicada = {explicado:.2%}.")

    print("[OK] Base Parquet original preservada (somente leitura).")
    print(f"[OK] F1-F8: 10.000 linhas; F9: 2.000 linhas; F10: nao utilizado.")
    print(f"[OK] Features apos pre-processamento: {t_proc.shape[1]}.")
    print(f"[OK] {len(grade)} configuracao(oes) PCA ativa(s) verificadas.")
    print("[OK] PCA ajustado somente no treino; transform da validacao sem refit.")
    print("PASSO 04.01 - VERIFICACAO PCA COM BASE REAL APROVADA")


if __name__ == "__main__":
    main()
