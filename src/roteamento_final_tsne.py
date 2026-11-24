"""Planejador seguro da finalizacao TSNE + GMM (somente memoria).

Nao realiza ajuste, nao le dataset/folds/F10, nao cria arquivos nem
ativa a finalizacao PCA. A automacao real sera integrada em etapa posterior.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from src.validacao_oof import grade_candidatos_oof


@dataclass(frozen=True)
class PlanoFinalizacaoTSNE:
    metodo: str
    dimensao_tsne: int
    id_candidato_oof: str
    limiar_mcc_oof_congelado: float
    folds_treino: tuple[int, ...]
    fold_teste_reservado: int
    modulo_treinamento: str
    modulo_avaliacao: str
    executar_treinamento: bool = False
    acessar_f10: bool = False


def planejar_finalizacao_tsne(config: dict, escolha_oof: dict) -> PlanoFinalizacaoTSNE:
    """Valida o contrato de roteamento sem side effects ou permissao de execucao."""
    if not isinstance(config, dict) or not isinstance(escolha_oof, dict):
        raise TypeError("Config e escolha OOF precisam ser dict.")
    if str(config.get("reducao", {}).get("metodo", "")).upper() != "TSNE":
        raise ValueError("Roteador exclusivo TSNE; nao aceitar PCA.")
    divisao = config.get("divisao", {})
    if (divisao.get("total_folds") != 10 or
            divisao.get("quantidade_folds_teste") != 1 or
            divisao.get("qual_fold_teste") != 10):
        raise ValueError("Divisao precisa reservar exclusivamente F10.")

    if escolha_oof.get("teste_final_usado") is not False:
        raise ValueError("F10 deve permanecer sem uso na selecao OOF.")
    if escolha_oof.get("status") not in ("selecionado_em_ensaio", "selecionado_em_oof_integral"):
        raise ValueError("Escolha OOF nao esta congelada em status valido.")
    id_candidato = escolha_oof.get("id_candidato")
    if not isinstance(id_candidato, str) or not id_candidato.strip():
        raise ValueError("ID do candidato OOF ausente.")
    tau = escolha_oof.get("limiar_mcc_oof")
    if (isinstance(tau, bool) or not isinstance(tau, (int, float)) or
            not isfinite(tau) or not 0.0 <= tau <= 1.0):
        raise ValueError("Limiar MCC OOF congelado invalido.")

    candidato = escolha_oof.get("configuracao")
    if not isinstance(candidato, dict) or candidato.get("metodo") != "TSNE":
        raise ValueError("Candidato deve ser TSNE.")
    if candidato not in grade_candidatos_oof(config):
        raise ValueError("Candidato nao pertence a grade TSNE em memoria.")
    reducao = candidato.get("reducao", {})
    dimensao = reducao.get("n_components")
    if type(dimensao) is not int or dimensao not in (1, 2, 3):
        raise ValueError("Dimensao TSNE deve ser 1, 2 ou 3.")
    from src.contrato_finalizacao_dinamica import validar_configuracao_finalizacao
    # A rota so planeja e nunca abre F10 ou persiste resultados. Ela aceita
    # identificadores sinteticos em testes isolados; a ponte oficial OOF
    # verifica o hash canonico e a identidade do vencedor antes do refit.
    validar_configuracao_finalizacao(config, metodo='TSNE', dimensao=dimensao)

    return PlanoFinalizacaoTSNE(
        metodo="TSNE",
        dimensao_tsne=dimensao,
        id_candidato_oof=id_candidato,
        limiar_mcc_oof_congelado=float(tau),
        folds_treino=tuple(range(1, 10)),
        fold_teste_reservado=10,
        modulo_treinamento="src.treinamento_final_tsne",
        modulo_avaliacao="src.avaliacao_final_tsne",
    )
