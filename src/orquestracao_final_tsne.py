"""Parte 6: orquestrador SEPARADO e fechado por padrao da finalizacao t-SNE.

CLI --planejar: consulta somente artefatos OOF F1..F9 ja concluidos.
Ajuste final so e possivel por chamada Python EXPLICITA, com dados F1..F9
fornecidos em memoria e autorizar_ajuste=True; NUNCA ha CLI para treinar.
Nao le dataset bruto, folds.npy, F10, nem chama modulos PCA; nao escreve
arquivos nem vincula o fluxo ao --executar do orquestrador OOF existente.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ponte_oof_final_tsne import (
    PreflightFinalizacaoTSNE,
    verificar_oof_concluido_tsne,
)
from src.finalizacao_controlada_tsne import (
    ResultadoFinalizacaoControladaTSNE,
    executar_ajuste_final_tsne_em_memoria,
)


def planejar_orquestracao_final_tsne(experimento_oof: str | Path) -> PreflightFinalizacaoTSNE:
    """Somente leitura dos artefatos OOF do vencedor; nenhuma autorizacao."""
    pre = verificar_oof_concluido_tsne(experimento_oof)
    if pre.plano.executar_treinamento or pre.plano.acessar_f10:
        raise RuntimeError("Planejamento TSNE deve permanecer sem treinamento e sem F10.")
    return pre


def validar_entrada_f1_f9_contra_oof(
    experimento_oof: str | Path,
    df_dev: pd.DataFrame,
    folds_dev: Any,
    linhas_originais: Any,
) -> PreflightFinalizacaoTSNE:
    """Confere correspondencia exata com o OOF; nao efetua fit, grava ou le F10.

    Importante: indices originais sao metadados fornecidos pelo chamador.
    A autenticidade dos hashes do Parquet/folds nao e conferida aqui; esta
    verificacao devera ser implementada no futuro carregador autorizado.
    """
    pre = planejar_orquestracao_final_tsne(experimento_oof)
    if not isinstance(df_dev, pd.DataFrame) or df_dev.empty:
        raise ValueError("Desenvolvimento F1-F9 deve ser um DataFrame nao vazio.")
    if not df_dev.index.equals(pd.RangeIndex(len(df_dev))) or not df_dev.columns.is_unique:
        raise ValueError("Desenvolvimento exige RangeIndex e colunas unicas.")
    if len(df_dev) != pre.n_predicoes_oof:
        raise ValueError("Quantidade de linhas difere das predicoes OOF.")
    y_coluna = pre.configuracao["dados"]["coluna_alvo"]
    if y_coluna not in df_dev.columns:
        raise ValueError("Coluna alvo do desenvolvimento ausente.")

    folds = np.asarray(folds_dev)
    ids = np.asarray(linhas_originais)
    y = df_dev[y_coluna].to_numpy()
    n = len(df_dev)
    if (folds.shape != (n,) or not np.issubdtype(folds.dtype, np.integer)
            or set(np.unique(folds).tolist()) != set(range(1, 10))):
        raise ValueError("Folds fornecidos devem conter SOMENTE F1-F9; F10 proibido.")
    if (ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or not np.all(np.diff(ids) > 0)):
        raise ValueError("Linhas originais devem ser unicas, inteiras e crescentes.")
    if not np.issubdtype(y.dtype, np.number) or not np.isin(y, [0, 1]).all():
        raise ValueError("Alvos de desenvolvimento devem ser 0/1.")

    # A ponte ja conferiu metadados, ids de candidato, scores e hashes
    # declarados. Aqui conferimos que os DADOS EM MEMORIA representam
    # precisamente aquelas MESMAS linhas, folds e rotulos do OOF.
    caminho = Path(experimento_oof) / "vencedor" / "oof" / "predicoes.npz"
    with np.load(caminho, allow_pickle=False) as arr:
        ids_oof = arr["linha_original"]
        folds_oof = arr["fold"]
        y_oof = arr["y_true"]
    if not np.array_equal(ids, ids_oof):
        raise ValueError("Linhas originais divergentes do OOF congelado.")
    if not np.array_equal(folds, folds_oof):
        raise ValueError("Atribuicao de folds divergente do OOF congelado.")
    if not np.array_equal(y, y_oof):
        raise ValueError("Rotulos divergentes do OOF congelado.")
    return pre


def executar_ajuste_final_tsne_pos_oof_em_memoria(
    experimento_oof: str | Path,
    df_dev: pd.DataFrame,
    folds_dev: Any,
    linhas_originais: Any,
    *,
    autorizar_ajuste: bool = False,
    registrar_historico_em: bool = False,
) -> ResultadoFinalizacaoControladaTSNE:
    """Ajusta SOMENTE depois de autorizacao explicita e alinhamento OOF.

    Nao existe carregador de Parquet neste modulo. Uso real de F1..F9 requer
    um executor externo com verificacao de hashes/particao, ainda pendente.
    """
    if autorizar_ajuste is not True:
        raise PermissionError("Ajuste final TSNE bloqueado: autorizar_ajuste=True exigido.")
    pre = validar_entrada_f1_f9_contra_oof(
        experimento_oof, df_dev, folds_dev, linhas_originais
    )
    return executar_ajuste_final_tsne_em_memoria(
        df_dev, folds_dev, pre.configuracao, pre.escolha_oof,
        pre.proveniencia, registrar_historico_em=registrar_historico_em,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parte 6: planejamento seguro e SOMENTE LEITURA de OOF TSNE."
    )
    parser.add_argument("--planejar", action="store_true", required=True,
                        help="Conferir ponte OOF e exibir o plano, sem treinar.")
    parser.add_argument("--experimento", type=Path, required=True,
                        help="Diretorio de um OOF TSNE concluido.")
    args = parser.parse_args(argv)
    pre = planejar_orquestracao_final_tsne(args.experimento)
    plano = pre.plano
    print("[OK] ORQUESTRACAO FINAL TSNE: SOMENTE PLANEJAMENTO")
    print(f"metodo={plano.metodo}; dimensao={plano.dimensao_tsne}D")
    print(f"candidato_oof={plano.id_candidato_oof}")
    print(f"tau_mcc_oof_congelado={plano.limiar_mcc_oof_congelado:.7f}")
    print(f"predicoes_oof={pre.n_predicoes_oof}")
    print("treino=F1-F9; F10=RESERVADO")
    print("treinamento_executado=NAO; arquivo_criado=NAO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
