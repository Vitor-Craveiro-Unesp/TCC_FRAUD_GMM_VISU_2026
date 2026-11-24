"""Integração CONTROLADA t-SNE + GMM: somente F1..F9 fornecidos em memória.

Não tem CLI, não lê Parquet/YAML/folds de disco, não acessa F10, não
executa OOF e não chama nenhum finalizador PCA. A persistência é uma ação
separada, com destino novo explicitamente fornecido pelo chamador.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.roteamento_final_tsne import PlanoFinalizacaoTSNE, planejar_finalizacao_tsne
from src.treinamento_final_tsne import (
    PacoteFinalTSNE,
    ajustar_final_f1_f9,
    salvar_pacote,
)


@dataclass(frozen=True)
class ResultadoFinalizacaoControladaTSNE:
    plano: PlanoFinalizacaoTSNE
    pacote: PacoteFinalTSNE


def executar_ajuste_final_tsne_em_memoria(
    df_desenvolvimento: pd.DataFrame,
    folds_desenvolvimento: Any,
    config: dict,
    escolha_oof: dict,
    proveniencia: dict,
    *,
    registrar_historico_em: bool = False,
) -> ResultadoFinalizacaoControladaTSNE:
    """Executa uma única vez o refit F1..F9 do candidato OOF congelado.

    Exige invocação explícita; não há fallback PCA nem escolha de limiar.
    A validação de F1..F9/F10 é reforçada em ajustar_final_f1_f9.
    """
    plano = planejar_finalizacao_tsne(config, escolha_oof)
    pacote = ajustar_final_f1_f9(
        df_desenvolvimento,
        folds_desenvolvimento,
        config,
        escolha_oof,
        proveniencia,
        registrar_historico_em=registrar_historico_em,
    )
    if not isinstance(pacote, PacoteFinalTSNE):
        raise RuntimeError("Executor recebeu pacote que nao e PacoteFinalTSNE.")
    if pacote.configuracao["reducao"]["metodo"].upper() != "TSNE":
        raise RuntimeError("Pacote nao pertence ao metodo TSNE.")
    if pacote.escolha_oof["id_candidato"] != plano.id_candidato_oof:
        raise RuntimeError("ID do candidato foi alterado no ajuste final.")
    if pacote.limiar_oof != plano.limiar_mcc_oof_congelado:
        raise RuntimeError("Limiar OOF foi alterado no ajuste final.")
    if pacote.escolha_oof.get("teste_final_usado") is not False:
        raise RuntimeError("Uso do F10 nao autorizado no pacote final.")
    if int(pacote.tsne.parametros["n_components"]) != plano.dimensao_tsne:
        raise RuntimeError("Dimensionalidade t-SNE alterada no ajuste final.")
    if pacote.risco.n_treino_ != pacote.n_treino_original:
        raise RuntimeError("Risco historico nao utilizou todo o treino original.")
    if pacote.n_ajuste_gmm > pacote.n_treino_original:
        raise RuntimeError("Amostra GMM maior que o treino original.")
    return ResultadoFinalizacaoControladaTSNE(plano=plano, pacote=pacote)


def prever_sem_rotulos_controlado(
    resultado: ResultadoFinalizacaoControladaTSNE,
    df_sem_alvo: pd.DataFrame,
) -> pd.Series:
    """Previsão OOS do pacote congelado; não recebe rótulos ou folds."""
    if not isinstance(resultado, ResultadoFinalizacaoControladaTSNE):
        raise TypeError("Resultado de finalizacao TSNE invalido.")
    scores = resultado.pacote.prever_sem_rotulos(df_sem_alvo)
    if not scores.index.equals(df_sem_alvo.index):
        raise RuntimeError("Scores fora de ordem.")
    if not np.isfinite(scores.to_numpy()).all() or not scores.between(0, 1).all():
        raise RuntimeError("Scores nao finitos ou fora de [0,1].")
    return scores


def persistir_finalizacao_controlada(
    resultado: ResultadoFinalizacaoControladaTSNE,
    diretorio_novo: str | Path,
) -> tuple[Path, Path]:
    """Ação explícita: salva somente pacote TSNE em pasta que não existe."""
    if not isinstance(resultado, ResultadoFinalizacaoControladaTSNE):
        raise TypeError("Resultado de finalizacao TSNE invalido.")
    return salvar_pacote(resultado.pacote, diretorio_novo)
