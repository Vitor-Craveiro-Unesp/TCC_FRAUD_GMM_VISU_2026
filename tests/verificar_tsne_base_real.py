"""Passo 04.02: verificar openTSNE real, sem alterar YAML nem Parquet.

Execute na raiz: python .\tests\verificar_tsne_base_real.py

Carrega amostras exclusivamente dos folds F1-F8 (treino) e F9 (validacao).
Usa 1000 exemplos para ajustar o t-SNE e 200 para transformar; este e um
smoke test, nao um experimento de qualidade nem uma validacao sobre a base toda.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import EncodersPorSufixo, ImputadorPadronizador, selecionar_features
from src.reducao_tsne import RedutorTSNE, grade_tsne
from verificar_pca_base_real import localizar_folds, ler_amostras


N_TREINO_TSNE = 1_000
N_VALIDACAO_TSNE = 200


def main():
    yaml_path = ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml"
    with yaml_path.open(encoding="utf-8-sig") as arquivo:
        config = yaml.safe_load(arquivo)

    if (config["divisao"]["total_folds"], config["divisao"]["qual_fold_teste"]) != (10, 10):
        raise ValueError("Esperado 10 folds com F10 reservado.")

    # Testa a alternativa t-SNE SEM modificar o YAML oficial,
    # que pode continuar com reducao.metodo: PCA.
    config_tsne = copy.deepcopy(config)
    config_tsne["reducao"]["metodo"] = "TSNE"

    caminho = ROOT / config["dados"]["arquivo"]
    parquet = pq.ParquetFile(caminho)
    folds = localizar_folds(config, caminho, parquet.metadata.num_rows)
    treino, validacao = ler_amostras(parquet, folds)
    # Primeiro apenas uma amostra: nao executar t-SNE com 1,6 milhao de
    # observacoes neste passo de verificacao funcional.
    treino = treino.iloc[:N_TREINO_TSNE].copy()
    validacao = validacao.iloc[:N_VALIDACAO_TSNE].copy()

    x_treino = selecionar_features(treino, config)
    x_validacao = selecionar_features(validacao, config)
    encoder = EncodersPorSufixo(config).fit(x_treino)
    t_enc = encoder.transform(x_treino)
    v_enc = encoder.transform(x_validacao)
    normalizador = ImputadorPadronizador(config, encoder).fit(t_enc)
    t_proc = normalizador.transform(t_enc)
    v_proc = normalizador.transform(v_enc)

    grade = list(grade_tsne(config_tsne))
    if not grade:
        raise ValueError("Grade t-SNE vazia.")

    for i, parametros in enumerate(grade, start=1):
        redutor = RedutorTSNE(config_tsne, parametros).fit(t_proc)
        antes = redutor.coordenadas_treino_.copy()
        t_tsne = redutor.transform_treino()
        v_tsne = redutor.transform(v_proc)

        assert t_tsne.shape == (N_TREINO_TSNE, parametros["n_components"])
        assert v_tsne.shape == (N_VALIDACAO_TSNE, parametros["n_components"])
        assert t_tsne.columns.equals(v_tsne.columns)
        assert t_tsne.index.equals(t_proc.index)
        assert v_tsne.index.equals(v_proc.index)
        assert np.isfinite(t_tsne.to_numpy()).all()
        assert np.isfinite(v_tsne.to_numpy()).all()
        np.testing.assert_array_equal(redutor.coordenadas_treino_, antes)

        kl = redutor.kl_divergence_
        if kl is not None and not np.isfinite(kl):
            raise ValueError("KL final do t-SNE nao e finito.")
        print(
            f"[OK] t-SNE {i}/{len(grade)}: init={parametros['initialization']}, "
            f"perplexity={parametros['perplexity']}, "
            f"treino={t_tsne.shape}, validacao={v_tsne.shape}, "
            f"KL final={kl}."
        )

    print(f"[OK] Preprocessamento produziu {t_proc.shape[1]} features numericas.")
    print(f"[OK] Ajuste: {N_TREINO_TSNE} registros dos F1-F8.")
    print(f"[OK] Validacao: {N_VALIDACAO_TSNE} registros do F9 sem refit.")
    print("[OK] F10 nao utilizado; Parquet e YAML oficial nao modificados.")
    print("[OK] PCA externo nao utilizado: opcao 'pca' e somente inicializacao do t-SNE.")
    print("PASSO 04.02 - VERIFICACAO TSNE COM BASE REAL APROVADA")


if __name__ == "__main__":
    main()
