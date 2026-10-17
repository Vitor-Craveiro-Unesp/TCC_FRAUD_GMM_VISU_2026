r"""Passo 05.03: scores de fraude no Parquet real, com F9 para prever e F10 intocado.

Rodar na raiz do projeto: python .\tests\verificar_probabilidade_base_real.py

Este verificador usa PCA para uma execucao curta, independentemente de o
projeto futuramente permitir PCA OU t-SNE como alternativas. Testa NORMAL/RUS
em memoria (sem modificar o YAML) e K=1/K=2. NAO calcula limiares nem MCC.
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
from src.probabilidade import PreditorProbabilisticoGMM, calcular_probabilidades


def exigir(condicao, mensagem):
    if not condicao:
        raise RuntimeError(mensagem)


def main():
    arquivo_yaml = ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml"
    yaml_original = arquivo_yaml.read_bytes()
    cfg = yaml.safe_load(yaml_original.decode("utf-8-sig"))

    exigir(cfg["divisao"]["total_folds"] == 10 and
           cfg["divisao"]["qual_fold_teste"] == 10,
           "Verificador exige 10 folds e F10 como teste reservado.")
    if cfg["reducao"]["metodo"].upper() != "PCA":
        raise ValueError("Verificacao curta 05.03 requer reducao.metodo: PCA no YAML.")

    caminho = ROOT / cfg["dados"]["arquivo"]
    parquet = pq.ParquetFile(caminho)
    folds = localizar_folds(cfg, caminho, parquet.metadata.num_rows)
    treino, validacao = ler_amostras(parquet, folds)
    exigir(len(treino) == 10000 and len(validacao) == 2000, "Amostras incompletas.")

    x_t = selecionar_features(treino, cfg)
    x_v = selecionar_features(validacao, cfg)
    encoder = EncodersPorSufixo(cfg).fit(x_t)
    t_enc, v_enc = encoder.transform(x_t), encoder.transform(x_v)
    imputer = ImputadorPadronizador(cfg, encoder).fit(t_enc)
    t_pre = imputer.transform(t_enc)
    v_pre = imputer.transform(v_enc)
    y_t = treino[cfg["dados"]["coluna_alvo"]]
    prevalencia_original = float(y_t.mean())
    exigir(np.isin(y_t.to_numpy(), (0, 1)).all(), "Rotulos de treino invalidos.")
    exigir(0 < prevalencia_original < 1, "Amostra deve ter ambas as classes.")

    cfg_rus = deepcopy(cfg)
    cfg_rus["amostragem"]["modo"] = ["NORMAL", "RUS"]
    total = 0
    for parametros_pca in grade_pca(cfg):
        pca = RedutorPCA(cfg, parametros_pca).fit(t_pre)
        t_pca, v_pca = pca.transform(t_pre), pca.transform(v_pre)
        for modo in ("NORMAL", "RUS"):
            args = {"modo": modo, "fold_validacao": 9}
            if modo == "RUS":
                razao = cfg["amostragem"]["razao_negativos_positivos"]
                args["razao"] = razao[0] if isinstance(razao, list) else razao
            indices = selecionar_indices_gmm(y_t.to_numpy(), cfg_rus, **args)
            exigir(int((y_t.to_numpy()[indices] == 1).sum()) == int(y_t.sum()),
                   "RUS removeu fraudes originais.")
            for parametros_gmm in grade_gmm(cfg):
                modelo_gmm = GMMNaoSupervisionado(cfg, parametros_gmm).fit(t_pca.iloc[indices])
                gamma_t = modelo_gmm.responsabilidades(t_pca)  # TODO o treino original
                exigir(len(gamma_t) == len(y_t), "RH deve receber treino completo.")

                for suavizacao in grade_suavizacao(cfg):
                    rh = RiscoHistoricoGaussianas(cfg, suavizacao).fit(gamma_t, y_t)
                    preditor = PreditorProbabilisticoGMM(modelo_gmm, rh)
                    scores_t = preditor.prever_probabilidade(t_pca)
                    scores_v = preditor.prever_probabilidade(v_pca)  # SEM y_validacao
                    gamma_v = modelo_gmm.responsabilidades(v_pca)
                    scores_manuais = calcular_probabilidades(gamma_v, rh.riscos_)
                    np.testing.assert_allclose(scores_v, scores_manuais, rtol=0, atol=1e-7)

                    exigir(scores_v.index.equals(v_pca.index), "Indice F9 foi alterado.")
                    exigir(scores_t.index.equals(t_pca.index), "Indice treino foi alterado.")
                    exigir(scores_v.between(0, 1).all() and scores_t.between(0, 1).all(),
                           "Scores fora de [0,1].")
                    exigir(np.isfinite(scores_v).all() and np.isfinite(scores_t).all(),
                           "Scores contem NaN/infinito.")
                    exigir(len(scores_v) == 2000 and len(scores_t) == 10000,
                           "Numero de scores incorreto.")
                    exigir(rh.n_treino_ == len(treino), "RH nao utilizou treino completo.")
                    if parametros_gmm["n_components"] == 1:
                        np.testing.assert_allclose(scores_v, prevalencia_original, rtol=0, atol=1e-7)
                    if suavizacao == 0:
                        exigir(np.isclose(float(scores_t.mean()), prevalencia_original, atol=2e-7),
                               "Media dos scores do treino deve recuperar prevalencia original.")
                    total += 1
                    print(f"[OK] PCA={parametros_pca['n_components']}D | {modo} | "
                          f"K={parametros_gmm['n_components']} | suavizacao={suavizacao:g} | "
                          f"GMM={len(indices):,} | RH_original={rh.n_treino_:,} | "
                          f"scores_F9={len(scores_v):,} | "
                          f"media_scores_F9={scores_v.mean():.6%}")

    exigir(arquivo_yaml.read_bytes() == yaml_original, "YAML oficial foi alterado.")
    print(f"[OK] Configuracoes verificadas: {total}.")
    print(f"[OK] Prevalencia no treino original: {prevalencia_original:.6%}.")
    print("[OK] F1-F8 usados no treino; F9 somente para scores; F10 isolado.")
    print("[OK] Nenhum limiar, MCC ou rotulo de validacao usado na predicao.")
    print("[OK] YAML e Parquet original preservados (somente leitura).")
    print("PASSO 05.03 - VERIFICACAO PROBABILIDADES COM BASE REAL APROVADA")


if __name__ == "__main__":
    main()
