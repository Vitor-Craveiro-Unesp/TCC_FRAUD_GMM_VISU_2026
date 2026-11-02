"""Pacote final t-SNE + GMM: ajuste estritamente em F1..F9.

Nao e CLI, nao le F10/Parquet, nao executa OOF, nao altera YAML/PCA.
A selecao OOF e seu limiar devem ter sido congelados anteriormente.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import selecionar_features, EncodersPorSufixo, ImputadorPadronizador
from src.amostragem import selecionar_indices_gmm
from src.reducao_tsne import RedutorTSNE
from src.gmm import GMMNaoSupervisionado
from src.risco import RiscoHistoricoGaussianas
from src.probabilidade import PreditorProbabilisticoGMM
from src.validacao_oof import grade_candidatos_oof

VERSAO_PACOTE = "tsne-gmm-final-f1f9-v1"


def _sha256_arquivo(caminho: Path) -> str:
    resumo = sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def _validar_entrada(df_dev: pd.DataFrame, folds_dev: Any,
                     config: dict, escolha: dict, proveniencia: dict):
    if not isinstance(df_dev, pd.DataFrame) or df_dev.empty or not df_dev.index.equals(pd.RangeIndex(len(df_dev))):
        raise ValueError("Desenvolvimento precisa ser DataFrame com RangeIndex 0..n-1.")
    if not df_dev.columns.is_unique:
        raise ValueError("Colunas duplicadas no desenvolvimento.")
    if not all(isinstance(x, dict) for x in (config, escolha, proveniencia)):
        raise TypeError("Config, escolha OOF e proveniencia devem ser dict.")
    if str(config.get("reducao", {}).get("metodo", "")).upper() != "TSNE":
        raise ValueError("Este modulo atende exclusivamente reducao TSNE.")
    divisao = config["divisao"]
    if divisao.get("total_folds") != 10 or divisao.get("qual_fold_teste") != 10 or divisao.get("quantidade_folds_teste") != 1:
        raise ValueError("A configuracao deve reservar F10 como unico teste final.")
    fracao = float(config["dados"]["fracao_base"])
    if not 0 < fracao <= 1:
        raise ValueError("Fracao da base invalida.")
    folds = np.asarray(folds_dev)
    if (folds.ndim != 1 or len(folds) != len(df_dev)
            or not np.issubdtype(folds.dtype, np.integer)
            or set(np.unique(folds).tolist()) != set(range(1, 10))):
        raise ValueError("Ajuste final exige exclusivamente F1-F9, com todos presentes; F10 proibido.")
    alvo = config["dados"]["coluna_alvo"]
    if alvo not in df_dev.columns:
        raise ValueError("Alvo ausente no desenvolvimento F1-F9.")
    y = df_dev[alvo].to_numpy()
    if not np.issubdtype(y.dtype, np.number) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Alvo de desenvolvimento deve ser 0/1 com ambas as classes.")
    if escolha.get("teste_final_usado") is not False:
        raise ValueError("Escolha OOF precisa declarar teste_final_usado=false.")
    if escolha.get("status") not in {"selecionado_em_ensaio", "selecionado_em_oof_integral"}:
        raise ValueError("Status de selecao OOF ausente ou desconhecido.")
    if not isinstance(escolha.get("id_candidato"), str) or not escolha["id_candidato"]:
        raise ValueError("ID do candidato OOF ausente.")
    candidato = escolha.get("configuracao")
    if not isinstance(candidato, dict) or candidato.get("metodo") != "TSNE":
        raise ValueError("Candidato OOF precisa ter metodo TSNE.")
    if candidato not in grade_candidatos_oof(config):
        raise ValueError("Candidato OOF nao pertence a grade da configuracao fornecida.")
    tau = escolha.get("limiar_mcc_oof")
    if (isinstance(tau, (bool, np.bool_))
            or not isinstance(tau, (float, int, np.floating, np.integer))
            or not np.isfinite(tau) or not 0 <= tau <= 1):
        raise ValueError("Limiar MCC congelado OOF invalido.")
    for campo in ("sha256_parquet", "sha256_folds", "sha256_yaml", "experimento_oof"):
        if not isinstance(proveniencia.get(campo), str) or not proveniencia[campo]:
            raise ValueError(f"Proveniencia obrigatoria ausente: {campo}")
    if fracao < 1 and escolha["status"] != "selecionado_em_ensaio":
        raise ValueError("Fracao amostral requer escolha OOF classificada como ensaio.")
    return y, candidato


@dataclass
class PacoteFinalTSNE:
    """Objeto t-SNE/GMM congelado para inferencia sem rotulos.

    PreditorProbabilisticoGMM e criado na inferencia: seu teste interno
    por id() nao deve ser persistido via joblib.
    """
    encoder: EncodersPorSufixo
    imputador: ImputadorPadronizador
    tsne: RedutorTSNE
    gmm: GMMNaoSupervisionado
    risco: RiscoHistoricoGaussianas
    configuracao: dict
    escolha_oof: dict
    proveniencia: dict
    n_treino_original: int
    n_ajuste_gmm: int
    versao: str = VERSAO_PACOTE

    @property
    def limiar_oof(self) -> float:
        return float(self.escolha_oof["limiar_mcc_oof"])

    def prever_sem_rotulos(self, df_sem_alvo: pd.DataFrame) -> pd.Series:
        """Inferencia OOS: sem fit, sem y e sem otimizacao de limiar."""
        if not isinstance(df_sem_alvo, pd.DataFrame) or df_sem_alvo.empty:
            raise ValueError("Entrada de inferencia deve ser DataFrame nao vazio.")
        alvo = self.configuracao["dados"]["coluna_alvo"]
        if alvo in df_sem_alvo.columns:
            raise ValueError("Remova o alvo da entrada de inferencia; nao fornecer rotulos ao preditor.")
        dados = self.configuracao["dados"]
        excluidas = [dados.get("coluna_id"), *dados.get("colunas_excluir", [])]
        x = df_sem_alvo.drop(columns=[c for c in excluidas if c], errors="ignore").copy()
        if list(x.columns) != self.encoder.colunas_:
            raise ValueError("Features ou ordem divergentes do treino final F1-F9.")
        x_enc = self.encoder.transform(x)
        x_pre = self.imputador.transform(x_enc)
        x_red = self.tsne.transform(x_pre)  # out-of-sample, nunca refit
        scores = PreditorProbabilisticoGMM(self.gmm, self.risco).prever_probabilidade(x_red)
        if not scores.index.equals(df_sem_alvo.index) or not np.isfinite(scores.to_numpy()).all() or not scores.between(0, 1).all():
            raise RuntimeError("Scores invalidos ou fora de ordem na inferencia t-SNE.")
        return scores


def ajustar_final_f1_f9(df_desenvolvimento: pd.DataFrame,
                        folds_desenvolvimento: Any, config: dict,
                        escolha_oof: dict, proveniencia: dict, *,
                        registrar_historico_em: bool = True) -> PacoteFinalTSNE:
    """Ajusta UMA VEZ em F1-F9 fornecidos em memoria; nunca le F10."""
    y, candidato = _validar_entrada(
        df_desenvolvimento, folds_desenvolvimento, config, escolha_oof, proveniencia
    )
    cfg = deepcopy(config)
    escolha = deepcopy(escolha_oof)
    x = selecionar_features(df_desenvolvimento, cfg)
    encoder = EncodersPorSufixo(cfg).fit(x)
    x_enc = encoder.transform(x)
    imputador = ImputadorPadronizador(cfg, encoder).fit(x_enc)
    x_pre = imputador.transform(x_enc)
    tsne = RedutorTSNE(cfg, candidato["reducao"]).fit(x_pre)
    x_red = tsne.transform_treino()  # nao recalcular o embedding de treino
    if int(candidato["reducao"]["n_components"]) == 3 and tsne.modelo_.negative_gradient_method != "bh":
        raise RuntimeError("TSNE 3D deve usar Barnes-Hut explicitamente.")

    indices_rus = selecionar_indices_gmm(
        y, cfg, modo=candidato["modo"], razao=candidato["razao"], fold_validacao=0
    )
    gmm = GMMNaoSupervisionado(cfg, candidato["gmm"]).fit(
        x_red.iloc[indices_rus], registrar_historico_em=registrar_historico_em
    )
    gamma_original = gmm.responsabilidades(x_red)  # TODO F1-F9 original; NAO so RUS
    risco = RiscoHistoricoGaussianas(cfg, candidato["suavizacao"]).fit(
        gamma_original, pd.Series(y, index=x_red.index)
    )
    PreditorProbabilisticoGMM(gmm, risco)
    return PacoteFinalTSNE(
        encoder=encoder, imputador=imputador, tsne=tsne, gmm=gmm, risco=risco,
        configuracao=cfg, escolha_oof=escolha, proveniencia=deepcopy(proveniencia),
        n_treino_original=len(df_desenvolvimento), n_ajuste_gmm=len(indices_rus),
    )


def salvar_pacote(pacote: PacoteFinalTSNE, pasta_nova: str | Path) -> tuple[Path, Path]:
    """Persistencia exclusiva em pasta nova; nao substitui resultados existentes."""
    if not isinstance(pacote, PacoteFinalTSNE) or pacote.versao != VERSAO_PACOTE:
        raise ValueError("Pacote final TSNE incompativel.")
    pasta = Path(pasta_nova)
    pasta.mkdir(parents=True, exist_ok=False)
    modelo = pasta / "modelo_final_f1_f9.joblib"
    manifesto = pasta / "manifesto_modelo_final.json"
    joblib.dump(pacote, modelo, compress=3)
    resumo = {
        "versao": pacote.versao,
        "tipo": "TREINAMENTO_FINAL_TSNE_F1_F9",
        "metodo": "TSNE",
        "teste_f10_executado": False,
        "fracao_base": float(pacote.configuracao["dados"]["fracao_base"]),
        "dimensao_tsne": int(pacote.escolha_oof["configuracao"]["reducao"]["n_components"]),
        "id_candidato_oof": pacote.escolha_oof["id_candidato"],
        "limiar_mcc_oof_congelado": pacote.limiar_oof,
        "n_treino_original": pacote.n_treino_original,
        "n_ajuste_gmm": pacote.n_ajuste_gmm,
        "proveniencia": pacote.proveniencia,
        "sha256_modelo_final": _sha256_arquivo(modelo),
    }
    manifesto.write_text(json.dumps(resumo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return modelo, manifesto


def carregar_pacote_confiavel(pasta: str | Path) -> PacoteFinalTSNE:
    """Carregar somente joblib local CONFIAVEL; pickle nao e formato seguro."""
    pasta = Path(pasta)
    modelo = pasta / "modelo_final_f1_f9.joblib"
    manifesto = json.loads((pasta / "manifesto_modelo_final.json").read_text(encoding="utf-8"))
    if (manifesto.get("versao") != VERSAO_PACOTE or manifesto.get("metodo") != "TSNE"
            or manifesto.get("teste_f10_executado") is not False
            or _sha256_arquivo(modelo) != manifesto.get("sha256_modelo_final")):
        raise RuntimeError("Manifesto ou hash do modelo final TSNE incompativel.")
    pacote = joblib.load(modelo)
    if not isinstance(pacote, PacoteFinalTSNE) or pacote.versao != VERSAO_PACOTE:
        raise RuntimeError("Objeto final TSNE desserializado incompativel.")
    return pacote
