"""Passo 05.02: valida RH_k com Parquet real, sem tocar em F10.

Rodar: python .\tests\verificar_risco_base_real.py
Reusa a mesma leitura de amostras e verificacao de manifesto da etapa 04.01.
O YAML oficial NAO e modificado. Ajustes de GMM usam PCA, NORMAL/RUS,
K=1 ou 2. Os riscos usam sempre gamma de TODO treino F1-F8 original.
"""
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verificar_pca_base_real import localizar_folds, ler_amostras
from src.preprocessing import EncodersPorSufixo, ImputadorPadronizador, selecionar_features
from src.reducao import RedutorPCA, grade_pca
from src.amostragem import selecionar_indices_gmm
from src.gmm import GMMNaoSupervisionado, grade_gmm
from src.risco import RiscoHistoricoGaussianas, grade_suavizacao


def exigir(condicao, mensagem):
    if not condicao:
        raise RuntimeError(mensagem)


def main():
    with (ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").open(encoding="utf-8-sig") as arquivo:
        cfg = yaml.safe_load(arquivo)

    exigir(cfg["divisao"]["qual_fold_teste"] == 10, "F10 deve permanecer reservado.")
    if cfg["reducao"]["metodo"].upper() != "PCA":
        raise ValueError("Verificador 05.02 usa PCA. Mantenha reducao.metodo: PCA no YAML.")
    arquivo_base = ROOT / cfg["dados"]["arquivo"]
    parquet = pq.ParquetFile(arquivo_base)
    folds = localizar_folds(cfg, arquivo_base, parquet.metadata.num_rows)
    treino, validacao = ler_amostras(parquet, folds)
    exigir(len(treino) == 10000 and len(validacao) == 2000, "Amostras incompletas.")

    x_t = selecionar_features(treino, cfg)
    x_v = selecionar_features(validacao, cfg)
    enc = EncodersPorSufixo(cfg).fit(x_t)
    imp = ImputadorPadronizador(cfg, enc).fit(enc.transform(x_t))
    proc_t = imp.transform(enc.transform(x_t))
    proc_v = imp.transform(enc.transform(x_v))
    y_t = treino[cfg["dados"]["coluna_alvo"]].to_numpy()
    prevalencia = float(y_t.mean())
    exigir(np.isin(y_t, (0, 1)).all(), "Rotulos invalidos.")
    exigir(0 < prevalencia < 1, "Amostra de treino precisa conter ambas as classes.")

    cfg_amostragem = deepcopy(cfg)
    cfg_amostragem["amostragem"]["modo"] = ["NORMAL", "RUS"]
    total = 0
    for parametros_pca in grade_pca(cfg):
        pca = RedutorPCA(cfg, parametros_pca).fit(proc_t)
        t_pca = pca.transform(proc_t)
        v_pca = pca.transform(proc_v)
        for modo in ("NORMAL", "RUS"):
            args = {"modo": modo, "fold_validacao": 9}
            if modo == "RUS":
                razao = cfg["amostragem"]["razao_negativos_positivos"]
                args["razao"] = razao[0] if isinstance(razao, list) else razao
            posicoes = selecionar_indices_gmm(y_t, cfg_amostragem, **args)
            exigir(np.count_nonzero(y_t[posicoes] == 1) == np.count_nonzero(y_t == 1),
                   "RUS removeu fraudes.")
            for parametros_gmm in grade_gmm(cfg):
                gmm = GMMNaoSupervisionado(cfg, parametros_gmm).fit(t_pca.iloc[posicoes])
                gamma_t = gmm.responsabilidades(t_pca)  # TODO o treino, sem RUS
                gamma_v = gmm.responsabilidades(v_pca)  # F9 somente para conferir dimensoes
                exigir(len(gamma_t) == len(y_t), "RH deve usar treino original inteiro.")
                exigir(len(gamma_v) == len(validacao), "Responsabilidades F9 invalidas.")

                for suavizacao in grade_suavizacao(cfg):
                    rh = RiscoHistoricoGaussianas(cfg, suavizacao).fit(gamma_t, y_t)
                    tab = rh.tabela()
                    exigir(rh.n_treino_ == len(treino), "RH nao recebeu treino completo.")
                    exigir(np.isclose(tab["massa_efetiva"].sum(), len(treino), rtol=1e-6),
                           "Massas RH nao representam treino original.")
                    exigir(np.isclose(tab["fraudes_ponderadas"].sum(), y_t.sum(), rtol=1e-5, atol=1e-4),
                           "Fraudes ponderadas nao somam positivos originais.")
                    exigir(tab["risco_historico"].between(0, 1).all(), "RH fora de [0,1].")
                    if parametros_gmm["n_components"] == 1:
                        exigir(np.isclose(rh.riscos_.iloc[0], prevalencia, atol=1e-7),
                               "Com K=1 o risco deve ser prevalencia ORIGINAL, mesmo com RUS.")
                    if suavizacao == 0:
                        recomposicao = float(np.dot(rh.massas_, rh.riscos_) / len(treino))
                        exigir(np.isclose(recomposicao, prevalencia, atol=1e-7),
                               "Prevalencia nao recuperada pelas massas e RH.")
                    total += 1
                    print(f"[OK] PCA={parametros_pca['n_components']}D | {modo} | "
                          f"K={parametros_gmm['n_components']} | suavizacao={suavizacao:g} | "
                          f"ajuste_GMM={len(posicoes):,} | RH_treino_original={rh.n_treino_:,}")
                    for gaussiana, linha in tab.iterrows():
                        print(f"     {gaussiana}: massa={linha['massa_efetiva']:.2f}, "
                              f"fraudes_soft={linha['fraudes_ponderadas']:.2f}, "
                              f"RH={linha['risco_historico']:.6%}")
    print(f"[OK] Prevalencia ORIGINAL do treino: {prevalencia:.6%}.")
    print(f"[OK] Configuracoes verificadas: {total}; F9 sem rotulos no RH; F10 preservado.")
    print("[OK] Arquivos de dados e YAML original nao modificados.")
    print("PASSO 05.02 - VERIFICACAO RISCO COM BASE REAL APROVADA")


if __name__ == "__main__":
    main()
