"""Ponte de leitura SOMENTE entre OOF concluido e plano de finalizacao t-SNE.

Le apenas artefatos de um experimento OOF ja concluido: configuracao.yaml,
manifesto.json, escolha do vencedor, metricas e predicoes OOF do vencedor.
NAO le Parquet, folds.npy, dados F10, nem realiza fit/inferencia/escrita.
NAO altera src/orquestracao_oof.py ou qualquer componente PCA.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import yaml

from src.roteamento_final_tsne import PlanoFinalizacaoTSNE, planejar_finalizacao_tsne


def _json_canonico(valor: dict) -> str:
    return json.dumps(valor, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _ler_json(caminho: Path) -> dict:
    dado = json.loads(caminho.read_text(encoding="utf-8-sig"))
    if not isinstance(dado, dict):
        raise ValueError(f"JSON deve ser objeto: {caminho.name}")
    return dado


def _texto_nao_vazio(obj: dict, chave: str) -> str:
    valor = obj.get(chave)
    if not isinstance(valor, str) or not valor.strip():
        raise ValueError(f"Metadado obrigatorio ausente: {chave}")
    return valor


@dataclass(frozen=True)
class PreflightFinalizacaoTSNE:
    plano: PlanoFinalizacaoTSNE
    configuracao: dict
    escolha_oof: dict
    proveniencia: dict
    n_predicoes_oof: int


def verificar_oof_concluido_tsne(pasta_experimento: str | Path) -> PreflightFinalizacaoTSNE:
    """Retorna plano e parametros congelados; nao ajusta e nao persiste modelos.

    Mantem o nome de funcao distinto do executor da Parte 4 para impossibilitar
    acionamento implicito de treinamento ou teste final.
    """
    raiz = Path(pasta_experimento)
    if not raiz.is_dir():
        raise FileNotFoundError(f"Pasta OOF inexistente: {raiz}")
    manifesto = _ler_json(raiz / "manifesto.json")
    if manifesto.get("status") != "concluido_06_03" or manifesto.get("versao_layout") != 2:
        raise ValueError("Experimento OOF nao concluido no layout 2.")
    if manifesto.get("tipo") not in ("ensaio", "oficial"):
        raise ValueError("Tipo de OOF invalido.")

    cfg = yaml.safe_load((raiz / "configuracao.yaml").read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict) or str(cfg.get("reducao", {}).get("metodo", "")).upper() != "TSNE":
        raise ValueError("Ponte aceita somente configuracao TSNE; PCA proibido.")
    fracao = float(cfg["dados"]["fracao_base"])
    if fracao < 1.0 and manifesto["tipo"] != "ensaio":
        raise ValueError("OOF amostral nao pode ser identificado como oficial.")

    pasta_metricas = raiz / "vencedor" / "metricas"
    escolha = _ler_json(pasta_metricas / "melhor_configuracao.json")
    metricas = _ler_json(pasta_metricas / "metricas.json")
    plano = planejar_finalizacao_tsne(cfg, escolha)
    cid = plano.id_candidato_oof
    esperado = sha256(_json_canonico(escolha["configuracao"]).encode("utf-8")).hexdigest()[:14]
    if cid != esperado:
        raise ValueError("ID do vencedor diverge do hash canonico de sua configuracao.")
    if cid not in manifesto.get("candidatos_concluidos", []):
        raise ValueError("Vencedor nao consta dos candidatos OOF concluidos.")
    if metricas.get("id_candidato") != cid or metricas.get("configuracao") != escolha["configuracao"]:
        raise ValueError("Metadados de metricas nao correspondem ao vencedor OOF.")
    tau_metrica = metricas.get("limiar_mcc")
    if isinstance(tau_metrica, bool) or not isinstance(tau_metrica, (int, float)) or not np.isfinite(tau_metrica) or tau_metrica != plano.limiar_mcc_oof_congelado:
        raise ValueError("Limiar MCC do vencedor nao coincide com as metricas OOF.")
    if escolha["status"] != ("selecionado_em_ensaio" if manifesto["tipo"] == "ensaio" else "selecionado_em_oof_integral"):
        raise ValueError("Status da escolha diverge do tipo de experimento.")

    caminho_predicoes = raiz / "vencedor" / "oof" / "predicoes.npz"
    with np.load(caminho_predicoes, allow_pickle=False) as arr:
        necessarios = {"linha_original", "fold", "y_true", "prob_fraude", "chave_execucao", "id_candidato", "fold_esperado"}
        if not necessarios.issubset(set(arr.files)):
            raise ValueError("Predicoes OOF sem campos obrigatorios.")
        ids = arr["linha_original"]
        folds = arr["fold"]
        rotulos = arr["y_true"]
        scores = arr["prob_fraude"]
        chave = str(arr["chave_execucao"].item())
        cid_cache = str(arr["id_candidato"].item())
        fold_cache = int(arr["fold_esperado"].item())
    n = len(ids)
    if (n == 0 or ids.ndim != 1 or folds.shape != (n,) or rotulos.shape != (n,) or scores.shape != (n,)
            or not np.issubdtype(ids.dtype, np.integer) or not np.all(ids >= 0)
            or not np.all(np.diff(ids) > 0)):
        raise ValueError("Linhas e vetores OOF invalidos ou fora de ordem.")
    if (not np.issubdtype(folds.dtype, np.integer)
            or set(np.unique(folds).tolist()) != set(range(1, 10))
            or not np.issubdtype(rotulos.dtype, np.integer)
            or not np.isin(rotulos, [0, 1]).all()
            or not np.issubdtype(scores.dtype, np.number)
            or not np.isfinite(scores).all()
            or np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("Folds, rotulos ou probabilidades OOF invalidos; F10 proibido.")
    if cid_cache != cid or chave != _texto_nao_vazio(manifesto, "chave_execucao") or fold_cache != 0:
        raise ValueError("Cache do vencedor possui chave ou identificacao incompativel.")
    if n != manifesto.get("n_desenvolvimento_trabalhado") or n != metricas.get("n"):
        raise ValueError("Quantidade de previsoes nao corresponde ao manifesto/metricas.")

    proveniencia = {
        "sha256_parquet": _texto_nao_vazio(manifesto, "sha256_parquet"),
        "sha256_folds": _texto_nao_vazio(manifesto, "sha256_folds"),
        "sha256_yaml": _texto_nao_vazio(manifesto, "sha256_yaml"),
        "experimento_oof": str(raiz.resolve()),
    }
    return PreflightFinalizacaoTSNE(
        plano=plano, configuracao=cfg, escolha_oof=escolha,
        proveniencia=proveniencia, n_predicoes_oof=n,
    )
