"""Passo 05.01: GMM em amostra real, treino F1-F8 e validacao F9.

Rodar: python .\tests\verificar_gmm_base_real.py
Reutiliza os folds e o leitor do verificador PCA, sem tocar no F10.
Executa a grade PCA ativa, K do GMM e amostragem NORMAL/RUS.
NAO usa rotulos no GMM: y_treino serve APENAS para selecionar RUS.
"""
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verificar_pca_base_real import ler_amostras, localizar_folds
from src.preprocessing import EncodersPorSufixo, ImputadorPadronizador, selecionar_features
from src.reducao import RedutorPCA, grade_pca
from src.amostragem import selecionar_indices_gmm
from src.gmm import GMMNaoSupervisionado, grade_gmm


def exigir(condicao, mensagem):
    if not condicao:
        raise RuntimeError(mensagem)


def main():
    with (ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").open(
        encoding="utf-8-sig"
    ) as arq:
        cfg = yaml.safe_load(arq)
    if cfg["reducao"]["metodo"].upper() != "PCA":
        raise ValueError(
            "Este verificador 05.01 usa PCA para os exemplos. "
            "Mantenha reducao.metodo: PCA no YAML; t-SNE sera integrado depois."
        )
    exigir(cfg["divisao"]["qual_fold_teste"] == 10, "F10 precisa ser reservado")
    caminho = ROOT / cfg["dados"]["arquivo"]
    parquet = pq.ParquetFile(caminho)
    folds = localizar_folds(cfg, caminho, parquet.metadata.num_rows)
    treino, validacao = ler_amostras(parquet, folds)
    exigir(len(treino) == 10_000 and len(validacao) == 2_000, "Amostras incompletas")

    # Preprocessing ajustado SOMENTE no treino de F1-F8.
    x_treino = selecionar_features(treino, cfg)
    x_validacao = selecionar_features(validacao, cfg)
    enc = EncodersPorSufixo(cfg).fit(x_treino)
    enc_t = enc.transform(x_treino)
    enc_v = enc.transform(x_validacao)
    imp = ImputadorPadronizador(cfg, enc).fit(enc_t)
    pre_t = imp.transform(enc_t)
    pre_v = imp.transform(enc_v)

    y_treino = treino[cfg["dados"]["coluna_alvo"]].to_numpy()
    graus = list(grade_gmm(cfg))
    exigir(len(graus) >= 1, "Grade GMM vazia")
    feitos = 0

    config_amostragem_teste = deepcopy(cfg)
    # Habilita os dois modos apenas na copia em memoria, sem editar o YAML oficial.
    config_amostragem_teste["amostragem"]["modo"] = ["NORMAL", "RUS"]

    for pcfg in grade_pca(cfg):
        redutor = RedutorPCA(cfg, pcfg).fit(pre_t)
        t_reduzido = redutor.transform(pre_t)
        v_reduzido = redutor.transform(pre_v)

        # Ambos os modos sao verificados sem alterar o YAML oficial.
        for modo in ("NORMAL", "RUS"):
            kwargs = {"modo": modo, "fold_validacao": 9}
            if modo == "RUS":
                razoes = cfg["amostragem"]["razao_negativos_positivos"]
                kwargs["razao"] = razoes[0] if isinstance(razoes, list) else razoes
            posicoes = selecionar_indices_gmm(y_treino, config_amostragem_teste, **kwargs)
            exigir(np.isin(posicoes, np.arange(len(treino))).all(), "RUS fora do treino")
            exigir(np.count_nonzero(y_treino[posicoes] == 1) ==
                   np.count_nonzero(y_treino == 1), "RUS eliminou fraude")

            for gc in graus:
                gmm = GMMNaoSupervisionado(cfg, gc).fit(t_reduzido.iloc[posicoes])
                # Responsabilidades sobre o treino ORIGINAL e sobre F9, sem refit.
                gamma_t = gmm.responsabilidades(t_reduzido)
                gamma_v = gmm.responsabilidades(v_reduzido)
                exigir(gamma_t.shape == (len(treino), gc["n_components"]),
                       "Responsabilidades do treino invalidas")
                exigir(gamma_v.shape == (len(validacao), gc["n_components"]),
                       "Responsabilidades da validacao invalidas")
                exigir(np.allclose(gamma_t.sum(axis=1), 1, atol=1e-6),
                       "Soft assignments nao somam 1 no treino")
                exigir(np.allclose(gamma_v.sum(axis=1), 1, atol=1e-6),
                       "Soft assignments nao somam 1 na validacao")
                exigir(np.isfinite(gamma_t.to_numpy()).all() and
                       np.isfinite(gamma_v.to_numpy()).all(), "Probabilidades nao finitas")
                feitos += 1
                print(
                    f"[OK] PCA={pcfg['n_components']}d; modo={modo}; "
                    f"K={gc['n_components']}; GMM ajuste={len(posicoes):,}; "
                    f"responsabilidades treino={len(gamma_t):,}, F9={len(gamma_v):,}; "
                    f"iteracoes={gmm.modelo_.n_iter_}; convergiu={gmm.modelo_.converged_}"
                )
    print("[OK] GMM recebeu apenas features reduzidas, nunca y ou NID_ALPHA.")
    print("[OK] RUS selecionou apenas linhas de treinamento e preservou fraudes.")
    print("[OK] Responsabilidades do treino original e da validacao calculadas sem refit.")
    print("[OK] F10 nao utilizado. Dataset Parquet preservado.")
    print(f"[OK] Configuracoes verificadas: {feitos}.")
    print("PASSO 05.01 - VERIFICACAO GMM COM BASE REAL APROVADA")


if __name__ == "__main__":
    main()
