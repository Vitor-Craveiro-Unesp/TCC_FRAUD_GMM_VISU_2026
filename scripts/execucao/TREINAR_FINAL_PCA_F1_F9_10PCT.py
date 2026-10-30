# -*- coding: utf-8 -*-
"""Treinamento final GMM + PCA em F1-F9, com amostra OOF ja consolidada.

Modos:
  --verificar : somente leitura; valida as tres origens e os indices F1-F9.
  --treinar   : ajusta 1D, 2D e 3D e salva NOVOS artefatos, sem acessar F10.

Este programa NAO avalia, prediz nem usa rotulos de F10. Nao refaz OOF,
nao gera galerias e nao edita o YAML central ou as pastas OOF aprovadas.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parents[2]  # scripts/execucao/arquivo.py
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.folds import hash_arquivo
from src.orquestracao_oof import _identificar_folds, _ler_desenvolvimento
from src.treinamento_final_pca import ajustar_final_f1_f9, salvar_pacote
from src.validacao_oof import grade_candidatos_oof

FONTES = {
    1: "ensaio_pca_gmm_20261029_173123",
    2: "ensaio_pca_gmm_20261029_172521",
    3: "ensaio_pca_gmm_20261029_173325",
}
N_ESPERADO = 166_716


def exigir(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise RuntimeError(mensagem)


def ler_json(arquivo: Path) -> dict:
    exigir(arquivo.is_file(), f"Arquivo ausente: {arquivo}")
    obj = json.loads(arquivo.read_text(encoding="utf-8-sig"))
    exigir(isinstance(obj, dict), f"JSON deve conter objeto: {arquivo}")
    return obj


def carregar_candidato(dim: int) -> dict:
    pasta = RAIZ / "outputs" / FONTES[dim]
    config_file = pasta / "configuracao.yaml"
    exigir(config_file.is_file(), f"YAML OOF ausente: {config_file}")
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8-sig"))
    exigir(isinstance(cfg, dict), f"YAML OOF invalido: {config_file}")
    manifest = ler_json(pasta / "manifesto.json")
    escolha = ler_json(pasta / "vencedor" / "metricas" / "melhor_configuracao.json")

    exigir(manifest.get("status") == "concluido_06_03", f"OOF PCA {dim}D incompleto")
    exigir(manifest.get("tipo") == "ensaio", f"Tipo de execucao inesperado PCA {dim}D")
    exigir(int(manifest.get("registros_desenvolvimento", -1)) == N_ESPERADO,
           f"N de desenvolvimento inesperado PCA {dim}D")
    exigir(float(cfg["dados"]["fracao_base"]) == 0.1, f"Fracao diferente de 10% PCA {dim}D")
    exigir(cfg["reducao"]["metodo"].upper() == "PCA", f"Nao e PCA {dim}D")
    candidato = escolha["configuracao"]
    exigir(candidato["reducao"]["n_components"] == dim, f"Dimensao incoerente PCA {dim}D")
    exigir(candidato["gmm"]["n_components"] == 4, f"K diferente de 4 PCA {dim}D")
    exigir(candidato["gmm"]["n_init"] == 5, f"n_init diferente de 5 PCA {dim}D")
    exigir(candidato["gmm"]["covariance_type"] == "full", f"Covariancia nao FULL PCA {dim}D")
    exigir(candidato["modo"] == "RUS" and candidato["razao"] == 16, f"RUS nao 16:1 PCA {dim}D")
    exigir(candidato in grade_candidatos_oof(cfg), f"Candidato fora da grade PCA {dim}D")
    exigir(escolha.get("teste_final_usado") is False, f"Teste final marcado como usado PCA {dim}D")
    exigir(escolha.get("status") == "selecionado_em_ensaio", f"Selecao inesperada PCA {dim}D")
    tau = float(escolha["limiar_mcc_oof"])
    exigir(np.isfinite(tau) and 0 <= tau <= np.nextafter(1., np.inf), f"Limiar invalido PCA {dim}D")

    arq_npz = pasta / "vencedor" / "oof" / "predicoes.npz"
    exigir(arq_npz.is_file(), f"Predicoes OOF ausentes PCA {dim}D")
    with np.load(arq_npz, allow_pickle=False) as z:
        ids = np.asarray(z["linha_original"], dtype=np.int64)
        folds = np.asarray(z["fold"], dtype=np.int64)
        y = np.asarray(z["y_true"], dtype=np.uint8)
        probs = np.asarray(z["prob_fraude"], dtype=np.float64)
        exigir(int(z["fold_esperado"].item()) == 0, f"NPZ nao consolidado PCA {dim}D")
        exigir(str(z["id_candidato"].item()) == escolha["id_candidato"],
               f"NPZ de candidato diferente da selecao PCA {dim}D")
    exigir(len(ids) == N_ESPERADO and np.all(np.diff(ids) > 0),
           f"Indices OOF ausentes/duplicados/desordenados PCA {dim}D")
    exigir(len(folds) == len(ids) == len(y) == len(probs), f"Vetores OOF divergentes PCA {dim}D")
    exigir(np.isin(folds, np.arange(1, 10)).all(), f"F10 no OOF PCA {dim}D")
    exigir(np.isin(y, [0, 1]).all() and np.isfinite(probs).all()
           and ((probs >= 0) & (probs <= 1)).all(), f"Conteudo OOF invalido PCA {dim}D")
    exigir(manifest.get("sha256_parquet") and manifest.get("sha256_folds")
           and manifest.get("sha256_yaml"), f"Hashes de proveniencia ausentes PCA {dim}D")
    return {"dim": dim, "pasta": pasta, "config": cfg, "manifest": manifest,
            "escolha": escolha, "ids": ids, "folds": folds, "y": y,
            "arquivo_npz": arq_npz, "limiar": tau}


def verificar_origens() -> tuple[dict, np.ndarray]:
    fontes = {dim: carregar_candidato(dim) for dim in FONTES}
    referencia = fontes[1]
    for dim, item in fontes.items():
        exigir(np.array_equal(item["ids"], referencia["ids"]),
               f"PCA {dim}D nao compartilha as mesmas observacoes de PCA 1D")
        exigir(np.array_equal(item["folds"], referencia["folds"]),
               f"PCA {dim}D nao compartilha os mesmos folds")
        exigir(np.array_equal(item["y"], referencia["y"]),
               f"PCA {dim}D nao compartilha os mesmos rotulos OOF")
        for campo in ("sha256_parquet", "sha256_folds"):
            exigir(item["manifest"][campo] == referencia["manifest"][campo],
                   f"Hash {campo} diferente PCA {dim}D")
        exigir(item["config"]["dados"]["arquivo"] == referencia["config"]["dados"]["arquivo"],
               f"Arquivo de dados diferente PCA {dim}D")

    parquet = RAIZ / referencia["config"]["dados"]["arquivo"]
    exigir(parquet.is_file(), f"Parquet ausente: {parquet}")
    folds_orig, caminho_folds, hash_parquet = _identificar_folds(
        RAIZ, referencia["config"], parquet
    )
    hash_folds = hash_arquivo(caminho_folds)
    exigir(hash_parquet == referencia["manifest"]["sha256_parquet"],
           "Hash Parquet diverge do manifesto OOF")
    exigir(hash_folds == referencia["manifest"]["sha256_folds"],
           "Hash folds diverge do manifesto OOF")
    ids = referencia["ids"]
    exigir(ids.size and ids.min() >= 0 and ids.max() < len(folds_orig),
           "Indices fora dos limites da base")
    exigir(np.array_equal(folds_orig[ids], referencia["folds"]),
           "Indices dos folds OOF nao coincidem com a divisao original")
    exigir(not np.any(folds_orig[ids] == 10), "F10 entre os indices de desenvolvimento")
    print("=== VERIFICACAO CONCLUIDA, SEM TREINAR ===")
    print(f"Base: {parquet.relative_to(RAIZ)}")
    print(f"Amostra F1-F9 comum: {len(ids)} linhas; F10 excluido")
    for dim in sorted(fontes):
        f = fontes[dim]
        print(f"PCA {dim}D | K4 n_init5 FULL RUS16 | tau OOF={f['limiar']:.17g}")
        print(f"  Origem: {f['pasta'].relative_to(RAIZ)}")
    return fontes, folds_orig


def treinar() -> None:
    fontes, folds_orig = verificar_origens()
    referencia = fontes[1]
    print("\n=== LENDO APENAS LINHAS DE DESENVOLVIMENTO F1-F9 ===", flush=True)
    parquet = RAIZ / referencia["config"]["dados"]["arquivo"]
    dev = _ler_desenvolvimento(parquet, referencia["ids"])
    exigir(len(dev) == N_ESPERADO, "Leitura de F1-F9 incompleta")
    coluna = referencia["config"]["dados"]["coluna_alvo"]
    exigir(np.array_equal(dev[coluna].to_numpy(), referencia["y"]),
           "Rotulos do Parquet divergem do OOF para F1-F9")
    folds_dev = folds_orig[referencia["ids"]]
    del folds_orig

    destino = RAIZ / "outputs" / ("treinamento_final_pca_10pct_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    exigir(not destino.exists(), f"Pasta destino ja existe: {destino}")
    destino.mkdir(parents=False, exist_ok=False)
    print(f"Nova pasta exclusiva: {destino}", flush=True)

    resumo = []
    for dim in sorted(fontes):
        item = fontes[dim]
        print(f"\n=== AJUSTE FINAL PCA {dim}D — F1-F9 ===", flush=True)
        manifest = item["manifest"]
        prov = {"sha256_parquet": manifest["sha256_parquet"],
                "sha256_folds": manifest["sha256_folds"],
                "sha256_yaml": manifest["sha256_yaml"],
                "experimento_oof": str(item["pasta"].relative_to(RAIZ)).replace('\\', '/'),
                "sha256_predicoes_oof": hash_arquivo(item["arquivo_npz"])}
        pacote = ajustar_final_f1_f9(
            dev, folds_dev, item["config"], item["escolha"], prov,
            registrar_historico_em=True,
        )
        arquivo, manifesto = salvar_pacote(pacote, destino / f"PCA_{dim}D")
        resumo.append({"dimensao": dim, "limiar_mcc_oof": pacote.limiar_oof,
                       "modelo": str(arquivo.relative_to(RAIZ)).replace('\\', '/'),
                       "manifesto": str(manifesto.relative_to(RAIZ)).replace('\\', '/')})
        print(f"[SALVO] PCA {dim}D: {arquivo}", flush=True)
        del pacote

    (destino / "RESUMO_MODELOS_FINAIS.json").write_text(
        json.dumps({"tipo": "TREINAMENTO_FINAL_F1_F9_AMOSTRAL_10PCT",
                    "f10_utilizado": False, "modelos": resumo},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[CONCLUIDO] 3 modelos salvos em {destino}; F10 nao utilizado.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modo = parser.add_mutually_exclusive_group(required=True)
    modo.add_argument("--verificar", action="store_true", help="Somente leitura, sem ajuste")
    modo.add_argument("--treinar", action="store_true", help="Treinar e salvar os 3 modelos em pasta nova")
    args = parser.parse_args()
    exigir((RAIZ / "src" / "treinamento_final_pca.py").is_file(),
           "Modulo src/treinamento_final_pca.py ausente")
    if args.verificar:
        verificar_origens()
    else:
        treinar()


if __name__ == "__main__":
    main()
