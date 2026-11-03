# -*- coding: utf-8 -*-
"""V2: refit final de TSNE+GMM 1D sobre desenvolvimento OOF F1..F9.

Planejar: somente artefatos registrados e metadados; nenhum fit ou F10.
Executar: revalida hashes reais e indices de F1..F9, faz fit completo,
serializa em diretorio novo atomico, recarrega e audita identidade.
Nao treina OOF; nao acessa scores ou rotulos F10; nao altera YAML/PCA.
"""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np

from src import orquestracao_oof as orq
from src.ponte_oof_final_tsne import verificar_oof_concluido_tsne
from src.orquestracao_final_tsne import executar_ajuste_final_tsne_pos_oof_em_memoria
from src.treinamento_final_tsne import salvar_pacote, carregar_pacote_confiavel

ROOT = Path(__file__).resolve().parents[2]
TSNE_FOLDER = "ensaio_tsne_gmm_20261102_134606"
PCA_FOLDER = "ensaio_pca_gmm_20261102_132101"
EXPECTED_WINNER = "757edec8b54d68"
EXPECTED_DEV_N = 166716
EXPECTED_F10_N = 18523
EXPECTED_F10_SHA = "0c767c6f55d36cc9620caa75c9605ec30adc65a63846a36e82d17b6dedb85a0c"


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def file_sha(path):
    h = sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def v1_report():
    reports = sorted((ROOT / "outputs").glob("validacao_tsne_f10_v1_*/relatorio_v1.json"),
                     key=lambda p: p.parent.name, reverse=True)
    expected_t = (ROOT / "outputs" / TSNE_FOLDER).resolve()
    expected_p = (ROOT / "outputs" / PCA_FOLDER).resolve()
    for path in reports:
        try:
            r = json_read(path)
            if (r.get("resultado") == "APROVADO"
                    and Path(r["experimentos"]["tsne"]).resolve() == expected_t
                    and Path(r["experimentos"]["pca"]).resolve() == expected_p
                    and r["vencedor_tsne"]["id_candidato"] == EXPECTED_WINNER
                    and r["fonte_oof"]["n_desenvolvimento"] == EXPECTED_DEV_N
                    and r["amostra_f10_comparacao"]["n"] == EXPECTED_F10_N
                    and r["amostra_f10_comparacao"]["sha256_indices"] == EXPECTED_F10_SHA):
                return path, r
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            continue
    raise RuntimeError("Relatorio V1 aprovado, com experimento/candidato/amostra esperados, nao localizado.")


def preflight():
    report_path, r = v1_report()
    folder = ROOT / "outputs" / TSNE_FOLDER
    pre = verificar_oof_concluido_tsne(folder)
    require(pre.plano.dimensao_tsne == 1 and pre.plano.metodo == "TSNE", "Modelo vencedor nao e t-SNE 1D.")
    require(pre.plano.id_candidato_oof == EXPECTED_WINNER, "Vencedor mudou desde V1.")
    require(pre.n_predicoes_oof == EXPECTED_DEV_N, "Numero de registros de desenvolvimento mudou.")
    require(float(pre.plano.limiar_mcc_oof_congelado) == float(r["vencedor_tsne"]["limiar_mcc_oof_congelado"]),
            "Limiar OOF diverge do relatorio aprovado V1.")
    manifest = json_read(folder / "manifesto.json")
    require(manifest.get("status") == "concluido_06_03", "OOF nao esta concluido.")
    require(float(pre.configuracao["dados"]["fracao_base"]) == 0.1, "Amostra diferente dos 10% aprovados.")
    npz = folder / "vencedor" / "oof" / "predicoes.npz"
    require(npz.is_file(), "Arquivo de previsoes OOF ausente.")
    output = folder / "vencedor" / "teste_final"
    require(output.is_dir(), "Diretorio teste_final nao existe no layout OOF.")
    result = {
        "relatorio_v1": report_path, "pasta": folder, "pre": pre,
        "manifesto_oof": manifest, "npz_oof": npz, "destino": output / "modelo_final_f1_f9",
        "diretorio_temporario": output / "modelo_final_f1_f9.incompleto_v2",
    }
    return result


def imprimir(info):
    pre = info["pre"]
    print("=== V2 - FINALIZACAO TSNE+GMM 1D EM F1..F9 ===", flush=True)
    print("V1_RELATORIO_APROVADO=" + str(info["relatorio_v1"]), flush=True)
    print("EXPERIMENTO_TSNE=" + str(info["pasta"]), flush=True)
    print("CANDIDATO_CONGELADO=" + pre.plano.id_candidato_oof, flush=True)
    print("TAU_OOF_CONGELADO=" + format(pre.plano.limiar_mcc_oof_congelado, ".17g"), flush=True)
    print("N_TREINO_DESENVOLVIMENTO=" + str(pre.n_predicoes_oof), flush=True)
    print("DESTINO_MODELO=" + str(info["destino"]), flush=True)
    print("F10_NO_TREINAMENTO=NAO", flush=True)
    print("F10_AVALIADO=NAO", flush=True)
    print("OOF_REEXECUTADO=NAO", flush=True)
    print("PCA_ALTERADO=NAO", flush=True)


def validar_modelo_existente(info, expected_source):
    folder = info["destino"]
    modelo = folder / "modelo_final_f1_f9.joblib"
    manifest_file = folder / "manifesto_modelo_final.json"
    require(modelo.is_file() and manifest_file.is_file(),
            "Pasta final ja existe, mas modelo/manifesto esta incompleto. Preserve e inspecione antes de continuar.")
    m = json_read(manifest_file)
    require(m.get("id_candidato_oof") == EXPECTED_WINNER, "Modelo existente tem outro candidato.")
    require(m.get("n_treino_original") == EXPECTED_DEV_N, "Modelo existente tem tamanho divergente.")
    require(m.get("limiar_mcc_oof_congelado") == info["pre"].plano.limiar_mcc_oof_congelado,
            "Modelo existente tem outro tau.")
    prov = m.get("proveniencia", {})
    for k, v in expected_source.items():
        require(prov.get(k) == v, f"Modelo existente diverge na proveniencia {k}.")
    pacote = carregar_pacote_confiavel(folder)
    require(pacote.n_treino_original == EXPECTED_DEV_N and pacote.limiar_oof == info["pre"].plano.limiar_mcc_oof_congelado,
            "Modelo recarregado nao corresponde ao OOF.")
    del pacote
    gc.collect()
    print("V2_STATUS=JA_CONCLUIDA_MODELO_RECARREGADO_VALIDADO", flush=True)
    return 0


def executar(info):
    output = info["destino"].parent
    temp = info["diretorio_temporario"]
    require(not temp.exists(), "Diretorio temporario V2 ja existe. Nao sobrescrever: " + str(temp))
    lock = ROOT / "outputs" / ".ajuste_final_tsne1d_v2_em_andamento.lock"
    with lock.open("x", encoding="utf-8") as fh:
        fh.write("V2 TSNE: finalizar modelo somente F1..F9.\n")
    try:
        pre = info["pre"]
        manifest = info["manifesto_oof"]
        expected_prov = dict(pre.proveniencia)
        expected_prov["sha256_predicoes_oof"] = file_sha(info["npz_oof"])
        if info["destino"].exists():
            return validar_modelo_existente(info, expected_prov)

        cfg = pre.configuracao
        parquet = ROOT / cfg["dados"]["arquivo"]
        print("[V2] Conferindo hash REAL do Parquet e particao mestre.", flush=True)
        orig_folds, path_folds, digest = orq._identificar_folds(ROOT, cfg, parquet)
        require(digest == manifest["sha256_parquet"], "SHA256 real do Parquet divergiu do OOF.")
        require(orq.hash_arquivo(path_folds) == manifest["sha256_folds"], "SHA256 real dos folds divergiu do OOF.")

        with np.load(info["npz_oof"], allow_pickle=False) as z:
            ids = np.asarray(z["linha_original"], dtype=np.int64)
            folds_oof = np.asarray(z["fold"], dtype=np.uint8)
            y_oof = np.asarray(z["y_true"], dtype=np.uint8)
        require(len(ids) == EXPECTED_DEV_N and np.all(np.diff(ids) > 0), "Indices OOF invalidos.")
        require(bool(np.array_equal(orig_folds[ids], folds_oof)), "Folds mestre divergentes do OOF.")
        require(bool(np.all(folds_oof <= 9)) and set(np.unique(folds_oof)) == set(range(1, 10)),
                "F10 detectado ou folds incompletos no desenvolvimento.")
        print("[V2] Carregando EXCLUSIVAMENTE as linhas F1..F9 selecionadas no OOF.", flush=True)
        df = orq._ler_desenvolvimento(parquet, ids)
        require(len(df) == EXPECTED_DEV_N, "Leitura do desenvolvimento incompleta.")
        require(np.array_equal(df[cfg["dados"]["coluna_alvo"]].to_numpy(dtype=np.uint8), y_oof),
                "Rotulos do Parquet divergentes dos labels OOF.")
        require(np.array_equal(orig_folds[ids], folds_oof), "Indice do desenvolvimento mudou durante a leitura.")
        del orig_folds, y_oof
        gc.collect()

        print("[V2] Ajuste final TSNE+GMM, somente F1..F9 (sem F10).", flush=True)
        resultado = executar_ajuste_final_tsne_pos_oof_em_memoria(
            info["pasta"], df, folds_oof, ids,
            autorizar_ajuste=True, registrar_historico_em=True,
        )
        pacote = resultado.pacote
        pacote.proveniencia["sha256_predicoes_oof"] = expected_prov["sha256_predicoes_oof"]
        require(pacote.escolha_oof["id_candidato"] == EXPECTED_WINNER, "Vencedor alterado no refit.")
        require(pacote.n_treino_original == EXPECTED_DEV_N, "Tamanho do fit final invalido.")
        require(pacote.limiar_oof == pre.plano.limiar_mcc_oof_congelado, "Tau OOF alterado.")
        print("[V2] Salvando modelo na pasta temporaria nova.", flush=True)
        model_path, man_path = salvar_pacote(pacote, temp)
        require(model_path.is_file() and man_path.is_file(), "Modelo temporario nao foi persistido.")
        del df, resultado, pacote
        gc.collect()

        print("[V2] Recarregando modelo persistido e validando integridade.", flush=True)
        carregado = carregar_pacote_confiavel(temp)
        require(carregado.escolha_oof["id_candidato"] == EXPECTED_WINNER, "ID do modelo recarregado mudou.")
        require(carregado.n_treino_original == EXPECTED_DEV_N, "Modelo recarregado tem tamanho errado.")
        require(carregado.limiar_oof == pre.plano.limiar_mcc_oof_congelado, "Modelo recarregado alterou tau OOF.")
        require(carregado.proveniencia["sha256_predicoes_oof"] == expected_prov["sha256_predicoes_oof"],
                "Proveniencia OOF do modelo recarregado mudou.")
        del carregado
        gc.collect()

        require(not info["destino"].exists(), "Destino final surgiu durante o treinamento; bloqueado.")
        os.replace(temp, info["destino"])
        print("MODELO_FINAL_TSNE=" + str(info["destino"] / "modelo_final_f1_f9.joblib"), flush=True)
        print("V2_MODELO_SHA256=" + json_read(info["destino"] / "manifesto_modelo_final.json")["sha256_modelo_final"], flush=True)
        print("V2_STATUS=APROVADA", flush=True)
        print("F10_AVALIADO=NAO", flush=True)
        print("OOF_REEXECUTADO=NAO", flush=True)
        return 0
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    op = parser.add_mutually_exclusive_group(required=True)
    op.add_argument("--planejar", action="store_true")
    op.add_argument("--executar", action="store_true")
    args = parser.parse_args(argv)
    info = preflight()
    imprimir(info)
    if args.planejar:
        print("V2_STATUS=PLANEJAMENTO_APROVADO_SEM_EXECUCAO", flush=True)
        return 0
    return executar(info)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("V2_STATUS=BLOQUEADA", file=sys.stderr)
        print("V2_MOTIVO=" + str(e), file=sys.stderr)
        raise
