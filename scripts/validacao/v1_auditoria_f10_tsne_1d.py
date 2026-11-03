# -*- coding: utf-8 -*-
"""V1: preflight de SOMENTE LEITURA para finalizacao TSNE+GMM 1D.

NAO le Parquet, folds mestre, labels ou scores F10.
Le somente configuracoes e artefatos OOF congelados, metadados e INDICES
F10 ja publicados pela finalizacao PCA 1D (sem ler transacoes).
NAO executa fit/transform, nao altera YAML, PCA, TSNE, OOF ou F10.
Relatorio opcional e gravado APENAS em outputs/validacao_tsne_f10_v1_*.
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

ROOT = Path(__file__).resolve().parents[2]
PCA_PADRAO = "ensaio_pca_gmm_20261102_132101"


def exigir(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise RuntimeError(mensagem)


def ler_json(caminho: Path) -> dict:
    exigir(caminho.is_file(), f"Arquivo obrigatorio ausente: {caminho}")
    valor = json.loads(caminho.read_text(encoding="utf-8-sig"))
    exigir(isinstance(valor, dict), f"JSON deve ser objeto: {caminho}")
    return valor


def ler_yaml(caminho: Path) -> dict:
    exigir(caminho.is_file(), f"YAML ausente: {caminho}")
    valor = yaml.safe_load(caminho.read_text(encoding="utf-8-sig"))
    exigir(isinstance(valor, dict), f"YAML invalido: {caminho}")
    return valor


def normalizar_valor(obj):
    """Normaliza dicionarios e strings de caminho; sem alterar config."""
    if isinstance(obj, dict):
        return {str(k): normalizar_valor(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalizar_valor(x) for x in obj]
    return obj


def dimensao_1d(valor) -> bool:
    """Aceita dimensionamento escalar 1 ou lista singleton [1]."""
    return valor == 1 or valor == [1]


def selecionar_tsne(outputs: Path, pasta: str | None) -> Path:
    if pasta:
        p = Path(pasta)
        if not p.is_absolute():
            p = ROOT / p
        exigir(p.is_dir(), f"Pasta TSNE fornecida nao existe: {p}")
        return p.resolve()
    candidatos = sorted(
        list(outputs.glob("ensaio_tsne_gmm_*")) + list(outputs.glob("experimento_tsne_gmm_*")),
        key=lambda p: p.name, reverse=True,
    )
    for p in candidatos:
        if not p.is_dir():
            continue
        try:
            cfg = ler_yaml(p / "configuracao.yaml")
            m = ler_json(p / "manifesto.json")
            if (cfg["reducao"]["metodo"].upper() == "TSNE"
                    and dimensao_1d(cfg["tsne"]["n_components"])
                    and m.get("status") == "concluido_06_03"
                    and (p / "vencedor" / "oof" / "predicoes.npz").is_file()):
                return p.resolve()
        except (KeyError, OSError, ValueError, RuntimeError, TypeError, yaml.YAMLError):
            continue
    raise RuntimeError("Nenhum experimento TSNE 1D OOF concluido encontrado em outputs/.")


def carregar_oof(pasta: Path, manifesto: dict, escolha: dict) -> dict:
    npz = pasta / "vencedor" / "oof" / "predicoes.npz"
    exigir(npz.is_file(), f"Predicoes OOF nao encontradas: {npz}")
    with np.load(npz, allow_pickle=False) as z:
        campos = {"linha_original", "fold", "y_true", "prob_fraude", "id_candidato", "fold_esperado", "chave_execucao"}
        exigir(campos.issubset(z.files), f"NPZ OOF incompleto: {npz}")
        ids = np.asarray(z["linha_original"], dtype=np.int64)
        folds = np.asarray(z["fold"], dtype=np.uint8)
        ys = np.asarray(z["y_true"], dtype=np.uint8)
        scores = np.asarray(z["prob_fraude"], dtype=np.float64)
        cid = str(z["id_candidato"].item())
        chave = str(z["chave_execucao"].item())
        fold_tipo = int(z["fold_esperado"].item())
    n = ids.size
    exigir(n > 0 and ids.ndim == 1 and folds.shape == (n,) and ys.shape == (n,)
           and scores.shape == (n,), f"Vetores OOF invalidos: {pasta}")
    exigir(bool(np.all(np.diff(ids) > 0) and np.all(ids >= 0)), "Indices OOF nao crescentes e unicos.")
    exigir(set(np.unique(folds).tolist()) == set(range(1, 10)), "OOF nao tem exatamente F1..F9.")
    exigir(bool(np.isin(ys, [0, 1]).all()), "Rotulos OOF nao binarios.")
    exigir(bool(np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all()), "Scores OOF fora de [0,1].")
    exigir(cid == escolha["id_candidato"] and chave == manifesto["chave_execucao"] and fold_tipo == 0,
           "NPZ OOF nao coincide com a escolha/execucao congelada.")
    exigir(n == manifesto["n_desenvolvimento_trabalhado"], "OOF difere do tamanho informado no manifesto.")
    return {"ids": ids, "folds": folds, "y": ys, "n": int(n), "caminho": str(npz)}


def comparar_configuracoes(cp: dict, ct: dict) -> list[str]:
    iguais = [
        ("arquivo_dados", cp["dados"]["arquivo"], ct["dados"]["arquivo"]),
        ("coluna_alvo", cp["dados"]["coluna_alvo"], ct["dados"]["coluna_alvo"]),
        ("coluna_id", cp["dados"].get("coluna_id"), ct["dados"].get("coluna_id")),
        ("colunas_excluir", cp["dados"].get("colunas_excluir", []), ct["dados"].get("colunas_excluir", [])),
        ("fracao_base", float(cp["dados"]["fracao_base"]), float(ct["dados"]["fracao_base"])),
        ("semente", cp["experimento"]["semente"], ct["experimento"]["semente"]),
        ("divisao", cp["divisao"], ct["divisao"]),
        ("validacao_cruzada", cp["validacao_cruzada"], ct["validacao_cruzada"]),
        ("preprocessamento", cp["preprocessamento"], ct["preprocessamento"]),
        ("amostragem", cp["amostragem"], ct["amostragem"]),
    ]
    ok = []
    for chave, a, b in iguais:
        exigir(normalizar_valor(a) == normalizar_valor(b),
               f"Configuracoes PCA/TSNE divergentes no campo {chave}.")
        ok.append(chave)
    return ok


def validar_indices_pca_f10(pasta: Path, ids_dev: np.ndarray) -> dict:
    dest = pasta / "vencedor" / "teste_final"
    caminho = dest / "indices_f10_amostra.npy"
    meta_path = dest / "manifesto_amostra_f10.json"
    exigir(caminho.is_file(),
           f"Indices F10 do PCA ainda nao salvos: {caminho}. "
           "Conclua a etapa de avaliacao PCA antes de estabelecer comparacao pareada.")
    meta = ler_json(meta_path)
    arr = np.load(caminho, allow_pickle=False)
    exigir(arr.ndim == 1 and arr.size > 0 and np.issubdtype(arr.dtype, np.integer),
           "Indices F10 PCA precisam ser vetor inteiro nao vazio.")
    ids = np.asarray(arr, dtype=np.int64)
    exigir(bool(np.all(ids >= 0) and np.all(np.diff(ids) > 0)),
           "Indices F10 PCA precisam ser nao negativos, unicos e crescentes.")
    exigir(len(np.intersect1d(ids, ids_dev)) == 0,
           "Amostra F10 PCA sobrepoe indices OOF de desenvolvimento.")
    exigir(int(meta.get("n", -1)) == len(ids), "Manifesto F10 PCA com tamanho divergente.")
    exigir(float(meta.get("fracao", -1)) == 0.1 and int(meta.get("semente", -1)) == 42,
           "Regra de amostragem PCA F10 difere de 10%/semente42.")
    real_sha = sha256(ids.tobytes()).hexdigest()
    exigir(meta.get("sha256_indices") == real_sha,
           "Hash de indices F10 PCA nao corresponde ao manifesto.")
    return {"n": int(len(ids)), "sha256_indices": real_sha,
            "fonte": str(caminho), "metadados": str(meta_path),
            "pertencimento_f10_no_fold_mestre": "PENDENTE_VERIFICACAO_V3",
            "classe_avaliacao": "EXPLORATORIA_F10_REUTILIZADO",
            "galeria_pca_f10_presente": (dest / "galeria" / "index.html").is_file()}


def auditar(root: Path, tsne_opt: str | None, pca_opt: str | None) -> dict:
    from src.ponte_oof_final_tsne import verificar_oof_concluido_tsne

    out = root / "outputs"
    exigir(out.is_dir(), "Diretorio outputs/ nao encontrado.")
    tsne = selecionar_tsne(out, tsne_opt)
    pca = Path(pca_opt) if pca_opt else out / PCA_PADRAO
    if not pca.is_absolute():
        pca = root / pca
    pca = pca.resolve()
    exigir(pca.is_dir(), f"Experimento PCA de referencia nao encontrado: {pca}")
    exigir(tsne != pca, "Pastas PCA e TSNE coincidem indevidamente.")

    # Ponte oficial verifica todos os contratos OOF do vencedor TSNE.
    ponte = verificar_oof_concluido_tsne(tsne)
    exigir(ponte.plano.metodo == "TSNE" and ponte.plano.dimensao_tsne == 1,
           "Vencedor TSNE nao e 1D.")
    exigir(not ponte.plano.executar_treinamento and not ponte.plano.acessar_f10,
           "Ponte TSNE nao esta em modo de planejamento seguro.")
    ct = ponte.configuracao
    mt = ler_json(tsne / "manifesto.json")
    et = ponte.escolha_oof

    cp = ler_yaml(pca / "configuracao.yaml")
    mp = ler_json(pca / "manifesto.json")
    ep = ler_json(pca / "vencedor" / "metricas" / "melhor_configuracao.json")
    exigir(cp["reducao"]["metodo"].upper() == "PCA" and cp["pca"]["n_components"] == [1],
           "Referencia nao e PCA 1D.")
    exigir(mp.get("status") == "concluido_06_03" and mp.get("versao_layout") == 2,
           "Referencia PCA OOF nao concluida/layout incorreto.")
    exigir(mp.get("tipo") == mt.get("tipo") == "ensaio" and
           float(cp["dados"]["fracao_base"]) == float(ct["dados"]["fracao_base"]) == .1,
           "V1 requer ensaios OOF comparaveis com fracao 10%.")
    exigir(ep.get("teste_final_usado") is False and
           ep["configuracao"]["metodo"] == "PCA" and
           ep["configuracao"]["reducao"]["n_components"] == 1,
           "Escolha PCA OOF invalida ou limiar reotimizado.")

    campos_comparados = comparar_configuracoes(cp, ct)
    exigir(mt["sha256_parquet"] == mp["sha256_parquet"],
           "Hashes declarados do Parquet divergentes entre PCA e TSNE.")
    exigir(mt["sha256_folds"] == mp["sha256_folds"],
           "Hashes declarados dos folds divergentes entre PCA e TSNE.")
    for key in ("sha256_parquet", "sha256_folds"):
        exigir(isinstance(mt[key], str) and len(mt[key]) == 64,
               f"Hash de proveniencia invalido: {key}")

    ooft = carregar_oof(tsne, mt, et)
    oofp = carregar_oof(pca, mp, ep)
    exigir(ooft["n"] == oofp["n"], "Quantidade OOF divergente entre PCA e TSNE.")
    exigir(bool(np.array_equal(ooft["ids"], oofp["ids"])), "F1..F9: IDs diferentes entre PCA/TSNE.")
    exigir(bool(np.array_equal(ooft["folds"], oofp["folds"])), "F1..F9: folds diferentes entre PCA/TSNE.")
    exigir(bool(np.array_equal(ooft["y"], oofp["y"])), "F1..F9: rotulos diferentes entre PCA/TSNE.")
    exigir(ooft["n"] == ponte.n_predicoes_oof, "Vetor TSNE difere da auditoria da ponte OOF.")
    amostra = validar_indices_pca_f10(pca, oofp["ids"])

    return {
        "etapa": "V1_AUDITORIA_PLANEJAMENTO_F10_TSNE1D",
        "resultado": "APROVADO",
        "somente_leitura_de_fontes": True,
        "f10_parquet_lido": False,
        "f10_fold_mestre_lido": False,
        "f10_labels_scores_lidos": False,
        "f10_indices_salvos_pca_lidos": True,
        "treinamento_executado": False,
        "arquivo_modelo_criado": False,
        "experimentos": {"tsne": str(tsne), "pca": str(pca)},
        "fonte_oof": {
            "tipo": mt["tipo"], "fracao_base": float(ct["dados"]["fracao_base"]),
            "n_desenvolvimento": ooft["n"], "folds": list(range(1,10)),
            "ids_rotulos_folds_pca_tsne_identicos": True,
            "sha256_parquet_declarado": mt["sha256_parquet"],
            "sha256_folds_declarado": mt["sha256_folds"],
            "hash_origem_reatestado_no_disco": False,
            "campos_metodologicos_iguais": campos_comparados,
        },
        "vencedor_tsne": {
            "id_candidato": ponte.plano.id_candidato_oof,
            "n_components": 1,
            "reducao": et["configuracao"]["reducao"],
            "gmm": et["configuracao"]["gmm"],
            "amostragem_modo": et["configuracao"]["modo"],
            "amostragem_razao": et["configuracao"]["razao"],
            "limiar_mcc_oof_congelado": ponte.plano.limiar_mcc_oof_congelado,
            "score_oof": et.get("score_oof"),
        },
        "amostra_f10_comparacao": amostra,
        "v2": "Ajuste FINAL somente F1..F9, com verificacao dos hashes reais antes do fit.",
        "v3": "Usar exatamente indices da amostra PCA, verificar fold=10 no vetor mestre e aplicar TSNE OOS sem refit.",
        "graficos_f10_previstos": {
            "preditivos": ["00", "01", "02", "03", "04", "05"],
            "geometria": ["GMM01", "GMM04", "GMM07", "GMM08", "GMM09_parametros_F1_F9", "GMM18_EM_F1_F9"],
            "tsne": ["TSNE13_histograma_F10_OOS", "TSNE14_KL_F1_F9_tempo_OOS"],
            "diagnostico": ["PRE01_F10"],
            "apenas_desenvolvimento": ["TSNE02_trustworthiness", "Continuity_por_perplexity"],
            "excluidos": ["Pearson_TSNE1", "comparacao_candidatos_06", "metricas_por_fold_07"],
        },
        "observacoes": [
            "Avaliacoes F10 sao exploratorias: houve acesso previo a esse fold no projeto.",
            "Esta V1 apenas confere indices salvos do F10 PCA; nao confirma ainda seu pertencimento ao fold mestre.",
            "Hashes de Parquet/folds foram comparados ENTRE manifestos, nao recalculados sobre os arquivos brutos nesta V1.",
            "A V1 nao autoriza nem executa V2, V3 ou qualquer acesso novo ao F10.",
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planejar", action="store_true", help="Somente verificar e imprimir resumo, SEM gravar.")
    ap.add_argument("--registrar", action="store_true", help="Auditar e salvar relatorio somente em outputs/validacao_tsne_f10_v1_*.")
    ap.add_argument("--tsne", type=str, help="Pasta TSNE especifica (opcional; padrao = ultimo TSNE1D concluido).")
    ap.add_argument("--pca", type=str, help=f"Pasta PCA referencia (padrao: outputs/{PCA_PADRAO}).")
    args = ap.parse_args(argv)
    if args.planejar == args.registrar:
        ap.error("Escolha exatamente --planejar OU --registrar.")
    relatorio = auditar(ROOT, args.tsne, args.pca)
    print("\n=== V1 - AUDITORIA DO F10 TSNE+GMM 1D ===", flush=True)
    print("PASTA_TSNE_OOF=" + relatorio["experimentos"]["tsne"])
    print("PASTA_PCA_REFERENCIA=" + relatorio["experimentos"]["pca"])
    print("OOF_TSNE_INTEGRIDADE=OK")
    print("OOF_PCA_INTEGRIDADE=OK")
    print("PCA_TSNE_F1_F9_INDICES_FOLDS_ROTULOS_IDENTICOS=SIM")
    print("PARQUET_E_FOLDS_HASHES_DECLARADOS_IDENTICOS=SIM")
    print("TSNE_CANDIDATO_OOF=" + relatorio["vencedor_tsne"]["id_candidato"])
    print("TSNE_TAU_OOF_CONGELADO={:.7f}".format(relatorio["vencedor_tsne"]["limiar_mcc_oof_congelado"]))
    print("N_DESENVOLVIMENTO=" + str(relatorio["fonte_oof"]["n_desenvolvimento"]))
    print("F10_INDICES_PCA_N=" + str(relatorio["amostra_f10_comparacao"]["n"]))
    print("F10_INDICES_PCA_SHA256=" + relatorio["amostra_f10_comparacao"]["sha256_indices"])
    print("F10_PERTENCIMENTO_FOLD_MESTRE=VERIFICAR_NA_V3")
    print("PCA_F10_GALERIA_PRESENTE=" + str(relatorio["amostra_f10_comparacao"]["galeria_pca_f10_presente"]).upper())
    print("TREINAMENTO_EXECUTADO=NAO")
    print("DADOS_F10_LIDOS=NAO (apenas indices de amostra ja salvos)")
    if args.registrar:
        pasta = ROOT / "outputs" / ("validacao_tsne_f10_v1_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        pasta.mkdir(parents=True, exist_ok=False)
        saida = pasta / "relatorio_v1.json"
        with saida.open("x", encoding="utf-8") as f:
            json.dump(relatorio, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        print("V1_RELATORIO=" + str(saida))
    else:
        print("ARQUIVOS_CRIADOS=NAO")
    print("V1_STATUS=APROVADA")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("V1_STATUS=BLOQUEADA", file=sys.stderr)
        print("V1_MOTIVO=" + str(exc), file=sys.stderr)
        raise
