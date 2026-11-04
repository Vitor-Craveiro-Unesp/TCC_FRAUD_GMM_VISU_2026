"""V2 / t-SNE 1D / Grafico 01: previa SINTETICA, sem alterar PCA.

Reutiliza a curva PR do renderizador OOF homologado em src/visualizacao_oof.py.
NAO treina t-SNE/GMM, nao le Parquet/folds.npy/F10, nao executa OOF.
Escreve apenas em outputs/validacao_tsne_v2_1d/01_curva_precision_recall/.
O PNG tem o MESMO layout do PCA, sem tarjeta de demonstracao sobreposta.
Os CSVs sao testes sinteticos, NAO resultados do experimento oficial.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (average_precision_score, log_loss,
                             matthews_corrcoef, precision_recall_curve)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml"
VISUALIZACAO = ROOT / "src" / "visualizacao_oof.py"
SAIDA = ROOT / "outputs" / "validacao_tsne_v2_1d" / "01_curva_precision_recall"
TAU_SINTETICO = 0.35


def exigir(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise AssertionError(mensagem)


def hash_arquivo(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def gerar_amostra():
    """Reproduz exatamente a amostra do teste V2 G00, seed=20261101."""
    rng = np.random.default_rng(20261101)
    n_por_fold = 40
    folds = np.repeat(np.arange(1, 10, dtype=np.uint8), n_por_fold)
    y = np.zeros(len(folds), dtype=np.uint8)
    for k in range(9):
        local = rng.choice(n_por_fold, size=3, replace=False)
        y[k * n_por_fold + local] = 1
    p = rng.beta(1.8, 10.0, size=len(folds))
    p[y == 1] = rng.beta(2.2, 2.8, size=int(y.sum()))
    p = np.clip(p, 1e-8, 1 - 1e-8)
    exigir(set(np.unique(folds)) == set(range(1, 10)), "Folds invalidos.")
    exigir(not np.any(folds == 10), "F10 encontrado na amostra sintetica.")
    exigir(len(y) == 360 and int(y.sum()) == 27, "Amostra divergente da G00.")
    return y, p, folds


def main() -> None:
    from src import visualizacao_oof as vis

    exigir(CONFIG.is_file() and VISUALIZACAO.is_file(),
           "Arquivos YAML ou renderizador OOF nao encontrados.")
    protegidos = {p: hash_arquivo(p) for p in (CONFIG, VISUALIZACAO)}
    cfg = deepcopy(yaml.safe_load(CONFIG.read_text(encoding="utf-8-sig")))
    exigir(isinstance(cfg, dict), "YAML invalido.")
    # Somente em memoria. Jamais gravar o YAML oficial.
    cfg["reducao"]["metodo"] = "TSNE"
    cfg["tsne"]["n_components"] = 1
    cfg["visualizacao"]["graficos"]["curva_pr"]["habilitar"] = True
    cfg["visualizacao"]["graficos"]["curva_pr"]["modo_eixos"] = "zoom_automatico"
    cfg["visualizacao"]["graficos"]["curva_pr"]["mostrar_ponto_mcc_oof"] = True
    cfg["visualizacao"]["graficos"]["curva_pr"]["mostrar_resumo_operacional"] = True

    y, p, folds = gerar_amostra()
    tau = TAU_SINTETICO  # ponto ficticio FIXO: nao otimizar usando y
    ap = float(average_precision_score(y, p))
    mcc = float(matthews_corrcoef(y, p >= tau))
    ll = float(log_loss(y, p, labels=[0, 1]))
    score = float((ap + (mcc + 1.0)/2.0 + np.exp(-ll))/3.0)
    prevalencia = float(np.mean(y))
    exigir(np.isclose(ap, 0.7096816, atol=5e-8),
           "A amostra mudou em relacao ao Grafico 00.")
    vencedor = pd.Series({"ap": ap, "mcc": mcc, "logloss": ll,
                         "score": score, "n": len(y), "prevalencia": prevalencia})
    escolha = {"id_candidato": "V2_G01_PREVIA_SINTETICA", "limiar_mcc_oof": tau,
               "teste_final_usado": False, "status": "pre_visualizacao_sintetica"}

    prec, rec, _ = precision_recall_curve(y, p)
    ponto = vis._ponto_operacional_pr(y, p, tau)
    limites_x, limites_y, zoom = vis._limites_zoom_pr(cfg, ponto, prevalencia)
    exigir(zoom is True, "A curva precisa exibir zoom e miniatura 0-1.")
    exigir(limites_x[0] <= ponto["recall"] <= limites_x[1] and
           limites_y[0] <= ponto["precision"] <= limites_y[1],
           "Ponto tau* deveria estar visivel no grafico principal.")

    SAIDA.mkdir(parents=True, exist_ok=True)
    salvar_original = vis._salvar
    capturas = []

    def salvar_auditado(fig, destino, *, dpi):
        suptitulo = getattr(fig, "_suptitle", None)
        exigir(suptitulo is not None and suptitulo.get_text() ==
               "Curva Precision–Recall do modelo selecionado em OOF",
               "Titulo diverge do PCA aprovado.")
        textos = [t.get_text() for t in fig.texts]
        textos_esperados = (
            "Resultados da validação OOF",
            f"AP = {ap:.7f}",
            f"Prevalência = {prevalencia:.7f}",
            f"τ* = {tau:.7f}",
            f"Precisão = {ponto['precision']:.7f}",
            f"Recall = {ponto['recall']:.7f}",
        )
        for esperado in textos_esperados:
            exigir(any(esperado in t for t in textos),
                   f"Rodape da curva PR sem o texto: {esperado}")
        exigir(not any("0,50" in t or "0.5000000" in t for t in textos),
               "Rodape repetiu o corte 0,50, removido do padrao aprovado.")
        legendas = [t.get_text() for leg in fig.legends for t in leg.get_texts()]
        exigir("Limiar ótimo do MCC (τ*)" in legendas,
               "Legenda de tau* ausente.")
        exigir("Curva Precision–Recall" in legendas,
               "Legenda da curva ausente.")
        principal = fig.axes[0]
        miniaturas = list(getattr(principal, "child_axes", []))
        exigir(len(miniaturas) == 1, "Miniatura da curva completa nao encontrada.")
        mini = miniaturas[0]
        exigir(np.allclose(mini.get_xlim(), (0, 1)) and
               np.allclose(mini.get_ylim(), (0, 1)),
               "Miniatura nao esta em escala completa 0-1.")
        exigir(np.allclose(principal.get_xlim(), limites_x) and
               np.allclose(principal.get_ylim(), limites_y),
               "Zoom da curva diverge da regra aprovada.")
        f = principal.xaxis.get_major_formatter()
        exigir(f(0.1234567, 0) == "0.1234567", "Precisao dos eixos incorreta.")
        capturas.append(1)
        return salvar_original(fig, destino, dpi=dpi)

    vis._salvar = salvar_auditado
    try:
        vis._plot_pr(cfg, y, p, vencedor, escolha, SAIDA, "png", 180, True)
    finally:
        vis._salvar = salvar_original

    exigir(len(capturas) == 1, "Renderizador PR nao executado exatamente uma vez.")
    imagem = SAIDA / "01_curva_precision_recall.png"
    csv_curva = SAIDA / "01_curva_precision_recall.csv"
    csv_pontos = SAIDA / "01_curva_precision_recall_pontos.csv"
    for arquivo in (imagem, csv_curva, csv_pontos):
        exigir(arquivo.is_file() and arquivo.stat().st_size > 100,
               f"Saida inexistente ou vazia: {arquivo}")
    dados_curva = pd.read_csv(csv_curva, encoding="utf-8-sig")
    exigir(dados_curva.columns.tolist() == ["recall", "precision"],
           "Colunas do CSV PR invalidas.")
    exigir(np.allclose(dados_curva["recall"], rec, atol=1e-12) and
           np.allclose(dados_curva["precision"], prec, atol=1e-12),
           "CSV PR nao corresponde a curva sintetica.")
    operacionais = pd.read_csv(csv_pontos, encoding="utf-8-sig")
    exigir(operacionais["corte"].tolist() == ["MCC OOF", "Fixo 0,50"],
           "Pontos operacionais CSV divergentes.")
    exigir(np.isclose(operacionais.iloc[0]["limiar"], tau) and
           np.isclose(operacionais.iloc[0]["precision"], ponto["precision"]) and
           np.isclose(operacionais.iloc[0]["recall"], ponto["recall"]),
           "Ponto tau* no CSV incorreto.")
    # Marcador persistente para evitar confusao futura com OOF real.
    manifesto = SAIDA / "LEIA_ME_PREVIA_SINTETICA.json"
    dados_manifesto = {
        "tipo": "PREVIA_SINTETICA_NAO_OFICIAL",
        "metodo_simulado": "TSNE", "dimensao_simulada": 1,
        "n": len(y), "n_fraudes": int(y.sum()), "folds": list(range(1, 10)),
        "tau_ficticio_nao_otimizado": tau, "ap_sintetico": ap,
        "f10_real_lido": False, "modelo_treinado": False,
        "codigo_oficial_modificado": False,
        "arquivo_png": imagem.name,
    }
    json_texto = json.dumps(dados_manifesto, ensure_ascii=False,
                            indent=2, allow_nan=False) + "\n"
    if manifesto.exists():
        exigir(manifesto.read_text(encoding="utf-8") == json_texto,
               "Manifesto sintetico ja existente divergente.")
    else:
        manifesto.write_text(json_texto, encoding="utf-8")

    exigir({c: hash_arquivo(c) for c in protegidos} == protegidos,
           "YAML ou modulo oficial foram modificados.")
    print("\n=== V2 / TSNE 1D / GRAFICO 01 - VALIDACAO SINTETICA ===")
    print("METODO_SIMULADO=TSNE | DIMENSAO_SIMULADA=1D")
    print("FOLDS_DESENVOLVIMENTO=1,2,3,4,5,6,7,8,9")
    print(f"N_SINTETICO={len(y)} | FRAUDES_SINTETICAS={int(y.sum())}")
    print(f"AP={ap:.7f} | MCC={mcc:.7f} | LOGLOSS={ll:.7f} | SCORE={score:.7f}")
    print(f"LIMIAR_TAU_FICTICIO={tau:.7f}")
    print("CURVA_COMPLETA_0_A_1=SIM | ZOOM=SIM")
    print("F10_REAL_ACESSADO=NAO")
    print("TREINAMENTO_REAL_EXECUTADO=NAO")
    print("ARQUIVOS_OFICIAIS_ALTERADOS=NAO")
    print(f"PNG_PARA_AVALIAR={imagem}")
    print(f"CSV_CURVA={csv_curva}")
    print(f"CSV_PONTOS={csv_pontos}")
    print("V2_G01_VALIDACAO=APROVADA_TECNICAMENTE_AGUARDANDO_AVALIACAO_VISUAL")


if __name__ == "__main__":
    main()
