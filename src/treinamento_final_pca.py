"""Ajuste final GMM + PCA exclusivamente em F1–F9 (sem leitura de F10).

Etapa preparatoria do TCC_EXPERIMENTAL_V3. Nao possui CLI: so executa
quando suas funcoes forem chamadas explicitamente pelo futuro orquestrador.
Nao altera YAML, nao le Parquet/folds no disco e nao gera graficos.
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
from src.reducao import RedutorPCA
from src.gmm import GMMNaoSupervisionado
from src.risco import RiscoHistoricoGaussianas
from src.probabilidade import PreditorProbabilisticoGMM
from src.validacao_oof import grade_candidatos_oof

VERSAO_PACOTE = "pca-gmm-final-f1f9-v1"


def _sha256_arquivo(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validar_entrada(df_dev: pd.DataFrame, folds_dev: Any, config: dict, escolha: dict, proveniencia: dict):
    if not isinstance(df_dev, pd.DataFrame) or df_dev.empty or not df_dev.index.equals(pd.RangeIndex(len(df_dev))):
        raise ValueError("Desenvolvimento precisa ser DataFrame com RangeIndex 0..n-1.")
    if not df_dev.columns.is_unique:
        raise ValueError("Colunas duplicadas no desenvolvimento.")
    if not isinstance(config, dict) or not isinstance(escolha, dict) or not isinstance(proveniencia, dict):
        raise TypeError("Config, escolha OOF e proveniencia devem ser dict.")
    if config.get("reducao", {}).get("metodo", "").upper() != "PCA":
        raise ValueError("Este modulo atende exclusivamente reducao PCA.")
    div = config["divisao"]
    if div.get("total_folds") != 10 or div.get("qual_fold_teste") != 10:
        raise ValueError("A configuracao deve reservar F10 como teste final.")
    fracao = float(config["dados"]["fracao_base"])
    if not 0 < fracao <= 1:
        raise ValueError("Fracao da base invalida.")
    f = np.asarray(folds_dev)
    if (f.ndim != 1 or len(f) != len(df_dev) or
            not np.issubdtype(f.dtype, np.integer) or
            set(np.unique(f).tolist()) != set(range(1, 10))):
        raise ValueError("Ajuste final exige exclusivamente F1–F9, com todos presentes; F10 proibido.")
    alvo = config["dados"]["coluna_alvo"]
    if alvo not in df_dev:
        raise ValueError("Alvo ausente no desenvolvimento F1–F9.")
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
    if not isinstance(candidato, dict) or candidato.get("metodo") != "PCA":
        raise ValueError("Configuracao do candidato OOF invalida.")
    if candidato not in grade_candidatos_oof(config):
        raise ValueError("Candidato OOF nao pertence a grade da configuracao fornecida.")
    tau = escolha.get("limiar_mcc_oof")
    if not isinstance(tau, (float, int)) or isinstance(tau, bool) or not np.isfinite(tau) or not 0 <= tau <= np.nextafter(1.0, np.inf):
        raise ValueError("Limiar MCC congelado OOF invalido.")
    for campo in ("sha256_parquet", "sha256_folds", "sha256_yaml", "experimento_oof"):
        if not isinstance(proveniencia.get(campo), str) or not proveniencia[campo]:
            raise ValueError(f"Proveniencia obrigatoria ausente: {campo}")
    if fracao < 1 and escolha["status"] != "selecionado_em_ensaio":
        raise ValueError("Fracao amostral requer escolha OOF classificada como ensaio.")
    return f, y, candidato, float(tau)


@dataclass
class PacoteFinalPCA:
    """Modelo congelado. O preditor e recriado ao inferir, inclusive apos joblib.load.

    Nao guardar PreditorProbabilisticoGMM no arquivo: seu bloqueio por id() do
    modelo em memoria nao sobrevive a desserializacao.
    """
    encoder: EncodersPorSufixo
    imputador: ImputadorPadronizador
    pca: RedutorPCA
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
        """Transforma dados sem rótulos e devolve scores; nunca chama fit."""
        if not isinstance(df_sem_alvo, pd.DataFrame) or df_sem_alvo.empty:
            raise ValueError("Entrada de inferencia deve ser DataFrame nao vazio.")
        alvo = self.configuracao["dados"]["coluna_alvo"]
        if alvo in df_sem_alvo.columns:
            raise ValueError("Remova o alvo da entrada de inferencia; F10 nao pode informar labels ao preditor.")
        dados = self.configuracao["dados"]
        excluidas = [dados.get("coluna_id"), *dados.get("colunas_excluir", [])]
        x = df_sem_alvo.drop(columns=[c for c in excluidas if c], errors="ignore").copy()
        if list(x.columns) != self.encoder.colunas_:
            raise ValueError("Features ou ordem divergentes do treino final F1–F9.")
        codificado = self.encoder.transform(x)
        normalizado = self.imputador.transform(codificado)
        reduzido = self.pca.transform(normalizado)
        preditor = PreditorProbabilisticoGMM(self.gmm, self.risco)
        scores = preditor.prever_probabilidade(reduzido)
        if len(scores) != len(df_sem_alvo) or not scores.index.equals(df_sem_alvo.index):
            raise RuntimeError("Tamanho ou ordem de scores alterados.")
        return scores


def ajustar_final_f1_f9(
    df_desenvolvimento: pd.DataFrame,
    folds_desenvolvimento: Any,
    config: dict,
    escolha_oof: dict,
    proveniencia: dict,
    *,
    registrar_historico_em: bool = True,
) -> PacoteFinalPCA:
    """Refit uma vez em F1–F9, seguindo exatamente a sequencia validada no OOF.

    df_desenvolvimento DEVE conter somente a amostra congelada de F1–F9.
    A chamada nao abre arquivos nem acessa o F10. Nao selecionar escolhas aqui.
    """
    f, y, candidato, _ = _validar_entrada(
        df_desenvolvimento, folds_desenvolvimento, config, escolha_oof, proveniencia
    )
    del f  # folds foram validados; todas as linhas fornecidas sao treino final
    cfg = deepcopy(config)
    escolha = deepcopy(escolha_oof)
    x = selecionar_features(df_desenvolvimento, cfg)
    encoder = EncodersPorSufixo(cfg).fit(x)
    x_enc = encoder.transform(x)
    imputador = ImputadorPadronizador(cfg, encoder).fit(x_enc)
    x_pre = imputador.transform(x_enc)
    pca = RedutorPCA(cfg, candidato["reducao"]).fit(x_pre)
    x_red = pca.transform(x_pre)

    indices_rus = selecionar_indices_gmm(
        y, cfg, modo=candidato["modo"], razao=candidato["razao"], fold_validacao=0
    )
    gmm = GMMNaoSupervisionado(cfg, candidato["gmm"]).fit(
        x_red.iloc[indices_rus], registrar_historico_em=registrar_historico_em
    )
    gamma_original = gmm.responsabilidades(x_red)  # TODO o treino; nao apenas RUS
    risco = RiscoHistoricoGaussianas(cfg, candidato["suavizacao"]).fit(
        gamma_original, pd.Series(y, index=x_red.index)
    )
    PreditorProbabilisticoGMM(gmm, risco)  # verifica coerencia interna
    return PacoteFinalPCA(
        encoder=encoder, imputador=imputador, pca=pca, gmm=gmm, risco=risco,
        configuracao=cfg, escolha_oof=escolha, proveniencia=deepcopy(proveniencia),
        n_treino_original=len(df_desenvolvimento), n_ajuste_gmm=len(indices_rus),
    )


def salvar_pacote(pacote: PacoteFinalPCA, pasta_nova: str | Path) -> tuple[Path, Path]:
    """Salva somente em pasta ainda inexistente. Nunca sobrescreve experimentos."""
    if not isinstance(pacote, PacoteFinalPCA) or pacote.versao != VERSAO_PACOTE:
        raise ValueError("Pacote final incompatível.")
    pasta = Path(pasta_nova)
    pasta.mkdir(parents=True, exist_ok=False)
    modelo = pasta / "modelo_final_f1_f9.joblib"
    manifesto = pasta / "manifesto_modelo_final.json"
    try:
        joblib.dump(pacote, modelo, compress=3)
        resumo = {
            "versao": pacote.versao,
            "tipo": "TREINAMENTO_FINAL_F1_F9",
            "teste_f10_executado": False,
            "fracao_base": float(pacote.configuracao["dados"]["fracao_base"]),
            "dimensao_pca": int(pacote.escolha_oof["configuracao"]["reducao"]["n_components"]),
            "id_candidato_oof": pacote.escolha_oof["id_candidato"],
            "limiar_mcc_oof_congelado": pacote.limiar_oof,
            "n_treino_original": pacote.n_treino_original,
            "n_ajuste_gmm": pacote.n_ajuste_gmm,
            "proveniencia": pacote.proveniencia,
            "sha256_modelo_final": _sha256_arquivo(modelo),
        }
        manifesto.write_text(json.dumps(resumo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception:
        # A pasta parcial fica visivel para investigacao; nao apagar nada silenciosamente.
        raise
    return modelo, manifesto


def carregar_pacote_confiavel(pasta: str | Path) -> PacoteFinalPCA:
    """Carregamento apenas de artefatos locais confiaveis (joblib usa pickle)."""
    pasta = Path(pasta)
    modelo = pasta / "modelo_final_f1_f9.joblib"
    manifest = json.loads((pasta / "manifesto_modelo_final.json").read_text(encoding="utf-8"))
    if manifest.get("versao") != VERSAO_PACOTE or _sha256_arquivo(modelo) != manifest.get("sha256_modelo_final"):
        raise RuntimeError("Manifesto ou hash do modelo final incompativel.")
    pacote = joblib.load(modelo)
    if not isinstance(pacote, PacoteFinalPCA) or pacote.versao != VERSAO_PACOTE:
        raise RuntimeError("Pacote final desserializado incompativel.")
    return pacote
