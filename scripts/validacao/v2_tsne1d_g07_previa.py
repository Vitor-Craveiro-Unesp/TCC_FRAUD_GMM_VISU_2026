"""V2 | t-SNE 1D OOF | Grafico 07 — previsualizacao sintética isolada.

Executar da RAIZ do projeto:
    python -m scripts.validacao.v2_tsne1d_g07_previa

Não carrega dados reais, não ajusta modelos, não chama OOF completo/F10.
Reutiliza o renderizador compartilhado homologado no PCA+GMM.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    matthews_corrcoef,
)

from src import estrutura_saida as layout
from src import visualizacao_oof as vis


def exigir(condicao: bool, descricao: str) -> None:
    if not condicao:
        raise AssertionError(descricao)


def hash_bytes(caminho: Path) -> str:
    return sha256(caminho.read_bytes()).hexdigest()


def amostra_ficticia() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mesma amostra artificial deterministica dos graficos 00 a 05."""
    rng = np.random.default_rng(20261101)
    folds = np.repeat(np.arange(1, 10, dtype=np.uint8), 40)
    y = np.zeros(360, dtype=np.uint8)
    for k in range(9):
        locais = rng.choice(40, size=3, replace=False)
        y[k * 40 + locais] = 1
    p = rng.beta(1.8, 10.0, size=360)
    p[y == 1] = rng.beta(2.2, 2.8, size=int(y.sum()))
    p = np.clip(p, 1e-8, 1 - 1e-8)
    exigir(len(y) == len(p) == len(folds) == 360, "Tamanho artificial invalido")
    exigir(int(y.sum()) == 27, "Contagem artificial de fraudes invalida")
    exigir(set(np.unique(folds)) == set(range(1, 10)), "Folds artificiais invalidos")
    exigir(np.isclose(average_precision_score(y, p), 0.7096816, atol=5e-8),
           "Amostra artificial diverge daquela dos graficos anteriores")
    return y, p, folds


def main() -> None:
    raiz = Path.cwd()
    yaml_oficial = raiz / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml"
    renderizador = raiz / "src/visualizacao_oof.py"
    layout_codigo = raiz / "src/estrutura_saida.py"
    arquivos_protegidos = [yaml_oficial, renderizador, layout_codigo]
    for arquivo in arquivos_protegidos:
        exigir(arquivo.is_file(), f"Arquivo oficial ausente: {arquivo}")
    antes = {str(p): hash_bytes(p) for p in arquivos_protegidos}

    config = deepcopy(yaml.safe_load(yaml_oficial.read_text(encoding="utf-8-sig")))
    config["reducao"]["metodo"] = "TSNE"
    config["tsne"]["n_components"] = 1
    config["visualizacao"]["graficos"]["metricas_por_fold"]["habilitar"] = True
    config["visualizacao"]["graficos"]["metricas_por_fold"]["gerar_em"] = ["oof"]

    y, p, folds = amostra_ficticia()
    tau_ficticio = 0.35
    id_ficticio = "TSNE_1D_PREVIA_SINTETICA"
    escolha = {"id_candidato": id_ficticio,
               "limiar_mcc_oof": tau_ficticio,
               "teste_final_usado": False}
    exigir(config["reducao"]["metodo"] == "TSNE", "Metodo inesperado")

    destino = raiz / (
        "outputs/validacao_tsne_v2_1d/"
        "07_metricas_por_fold_previa_sintetica"
    )
    destino.mkdir(parents=True, exist_ok=True)

    # O renderizador so precisa de um NPZ OOF sintético na estrutura padrão.
    # O diretorio e temporario e e excluido automaticamente apos o desenho.
    with TemporaryDirectory(prefix="v2_tsne_g07_sintetico_") as diretorio:
        experimento = Path(diretorio)
        caminho_npz = layout.predicoes_vencedor(experimento)
        caminho_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            caminho_npz,
            linha_original=np.arange(360, dtype=np.int64) + 10000,
            fold=folds,
            y_true=y,
            prob_fraude=p,
            id_candidato=np.asarray(id_ficticio),
            fold_esperado=np.asarray(0, dtype=np.uint8),
        )
        vis._plot_metricas_por_fold(
            experimento, config, y, p, escolha,
            destino, "png", 180, True,
        )

    png = destino / "07_metricas_por_fold.png"
    csv = destino / "07_metricas_por_fold.csv"
    exigir(png.is_file() and png.stat().st_size > 10000,
           "Imagem nao foi gerada ou parece incompleta")
    exigir(csv.is_file(), "CSV das metricas por fold nao foi gerado")
    tabela = pd.read_csv(csv)
    exigir(len(tabela) == 9, "CSV deve possuir exatamente nove folds")
    exigir(tabela["fold"].tolist() == list(range(1, 10)), "Folds 1 a 9 incorretos")
    exigir((tabela["n_observacoes"] == 40).all(), "Tamanho por fold incorreto")
    exigir((tabela["n_fraudes"] == 3).all(), "Fraudes por fold incorretas")
    exigir((tabela["limiar_otimo_oof_global"] == tau_ficticio).all(),
           "O limiar global ficticio foi alterado")
    exigir(int(tabela["n_observacoes"].sum()) == len(y), "Soma de observacoes incorreta")
    exigir(int(tabela["n_fraudes"].sum()) == int(y.sum()), "Soma de fraudes incorreta")

    for fold in range(1, 10):
        mascara = folds == fold
        yi, pi = y[mascara], p[mascara]
        linha = tabela.iloc[fold - 1]
        esperado = {
            "prevalencia_fraude": float(yi.mean()),
            "ap": float(average_precision_score(yi, pi)),
            "mcc_corte_050": float(matthews_corrcoef(yi, pi >= 0.5)),
            "mcc_corte_otimo_oof": float(matthews_corrcoef(yi, pi >= tau_ficticio)),
            "logloss": float(log_loss(yi, pi, labels=[0, 1])),
        }
        esperado["exp_menos_logloss"] = float(np.exp(-esperado["logloss"]))
        for coluna, valor in esperado.items():
            exigir(np.isclose(float(linha[coluna]), valor, rtol=1e-9, atol=1e-9),
                   f"Divergencia em {coluna} para fold F{fold}")

    depois = {str(p): hash_bytes(p) for p in arquivos_protegidos}
    exigir(antes == depois, "Um arquivo oficial foi modificado")
    print("\n=== V2 / TSNE 1D / GRAFICO 07 - METRICAS POR FOLD ===")
    print("METODO_SIMULADO=TSNE | DIMENSAO_SIMULADA=1D")
    print("FOLDS_DESENVOLVIMENTO=1,2,3,4,5,6,7,8,9")
    print("N_SINTETICO=360 | FRAUDES_SINTETICAS=27")
    print(f"LIMIAR_GLOBAL_FICTICIO={tau_ficticio:.7f}")
    print("PAINEIS=AP_E_PREVALENCIA,MCC_050_E_TAU,METRICA_EXP_MENOS_LOGLOSS")
    print("DADOS_CSV_VALIDOS=SIM")
    print("F10_REAL_ACESSADO=NAO")
    print("TREINAMENTO_REAL_EXECUTADO=NAO")
    print("ARQUIVOS_OFICIAIS_ALTERADOS=NAO")
    print(f"PNG_PARA_AVALIAR={png}")
    print(f"CSV_VERIFICACAO={csv}")
    print("V2_G07_VALIDACAO=APROVADA_TECNICAMENTE_AGUARDANDO_AVALIACAO_VISUAL")


if __name__ == "__main__":
    main()
