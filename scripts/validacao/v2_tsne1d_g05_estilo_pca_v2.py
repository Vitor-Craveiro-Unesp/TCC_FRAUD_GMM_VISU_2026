"""V2 | t-SNE 1D OOF | Grafico 05 | PREVIA SINTETICA NAO OFICIAL.

Reutiliza a visualizacao de distribuicao de probabilidades aprovada para PCA.
A entrada contem 360 observacoes artificiais F1..F9; o tau=0.35 e
FICTICIO e congelado somente para comparar as previas 00..05.
Nao le arquivos de dados, folds reais ou F10 e nao ajusta modelo.
Nao modifica o YAML nem o renderizador oficial.

Rodar a partir da raiz do projeto:
    python -m scripts.validacao.v2_tsne1d_g05_estilo_pca_v2
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
from sklearn.metrics import average_precision_score, matthews_corrcoef, log_loss

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
from src import visualizacao_oof as vis

YAML_OFICIAL = RAIZ / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml"
RENDERIZADOR = RAIZ / "src" / "visualizacao_oof.py"
SAIDA = (
    RAIZ / "outputs" / "validacao_tsne_v2_1d" /
    "05_distribuicao_probabilidades_estilo_pca_v2"
)
NOME = "05_distribuicao_probabilidades"
TAU_FICTICIO = 0.35


def _hash(arquivo: Path) -> str:
    return sha256(arquivo.read_bytes()).hexdigest()


def _base_sintetica():
    # Mesma amostra, semente e ordem dos graficos 00..04.
    rng = np.random.default_rng(20261101)
    folds = np.repeat(np.arange(1, 10, dtype=np.uint8), 40)
    y = np.zeros(360, dtype=np.uint8)
    for k in range(9):
        locais = rng.choice(40, size=3, replace=False)
        y[k * 40 + locais] = 1
    scores = rng.beta(1.8, 10.0, size=360)
    scores[y == 1] = rng.beta(2.2, 2.8, size=int(y.sum()))
    scores = np.clip(scores, 1e-8, 1.0 - 1e-8)
    return folds, y, scores


def main() -> None:
    protegidos = (YAML_OFICIAL, RENDERIZADOR)
    hashes_antes = {str(p): _hash(p) for p in protegidos}
    cfg = deepcopy(yaml.safe_load(YAML_OFICIAL.read_text(encoding="utf-8-sig")))

    # Todas as alteracoes de configuracao permanecem somente em memoria.
    cfg["reducao"]["metodo"] = "TSNE"
    cfg["tsne"]["n_components"] = 1
    op = cfg["visualizacao"]["graficos"]["distribuicao_probabilidades"]
    op.update({
        "habilitar": True,
        "tipo": "histograma",
        "bins": 50,
        "separar_classes": True,
        "normalizar": True,
        "modo_eixo_x": "zoom_automatico",
        "escala_y": "linear",
        "marcar_limiar_oof": True,
        "marcar_limiar_05": False,  # Igual ao PCA: somente a linha tau* na figura.
    })
    cfg["visualizacao"]["padrao"]["formato"] = "png"

    folds, y, scores = _base_sintetica()
    tau = TAU_FICTICIO
    ap = float(average_precision_score(y, scores))
    mcc = float(matthews_corrcoef(y, scores >= tau))
    mcc_05 = float(matthews_corrcoef(y, scores >= 0.5))
    loss = float(log_loss(y, scores, labels=[0, 1]))
    composto = float((ap + (mcc + 1) / 2 + np.exp(-loss)) / 3)
    assert len(folds) == len(y) == len(scores) == 360
    assert set(np.unique(folds)) == set(range(1, 10))
    assert int(y.sum()) == 27
    assert np.isfinite(scores).all() and np.all((0 <= scores) & (scores <= 1))
    assert np.isclose(ap, 0.7096816, atol=5e-8)
    assert np.isclose(mcc, 0.6292825, atol=5e-8)
    assert np.isclose(loss, 0.2298271, atol=5e-8)
    assert np.isclose(composto, 0.7729980, atol=5e-8)

    escolha = {
        "id_candidato": "TSNE_1D_PREVIA_SINTETICA",
        "limiar_mcc_oof": tau,
        "teste_final_usado": False,
    }
    SAIDA.mkdir(parents=True, exist_ok=True)
    # Inspecao programatica do layout: exatamente tres itens na legenda,
    # somente uma linha de corte (tau*). A verificacao acontece antes do save.
    salvar_original = vis._salvar

    def verificar_e_salvar(fig, destino, *, dpi):
        assert len(fig.legends) == 1, "Legenda principal nao encontrada."
        etiquetas = [t.get_text() for t in fig.legends[0].get_texts()]
        assert etiquetas == [
            "Legitima (n=333)".replace("Legitima", "Legítima"),
            "Fraude (n=27)",
            "Limiar ótimo do MCC (τ*)",
        ], f"Legenda diferente do PCA: {etiquetas}"
        assert len(fig.axes) == 1, "Esperado um unico painel de histogramas."
        linhas = fig.axes[0].lines
        assert len(linhas) == 1, "Deve existir apenas a linha do limiar tau*."
        assert np.allclose(np.asarray(linhas[0].get_xdata(), dtype=float), tau), (
            "Linha de limiar diverge de tau*."
        )
        assert not any("0,50" in texto or "fixo" in texto.lower()
                       for texto in etiquetas), "Limiar fixo apareceu na legenda."
        return salvar_original(fig, destino, dpi=dpi)

    vis._salvar = verificar_e_salvar
    try:
        vis._distribuicao(cfg, y, scores, escolha, SAIDA, "png", 180, True)
    finally:
        vis._salvar = salvar_original

    png = SAIDA / (NOME + ".png")
    csv = SAIDA / (NOME + ".csv")
    assert png.is_file() and png.stat().st_size > 0, "PNG ausente ou vazio."
    assert csv.is_file() and csv.stat().st_size > 0, "CSV ausente ou vazio."

    tabela = pd.read_csv(csv, encoding="utf-8-sig")
    assert len(tabela) == 50
    assert (tabela["fim_bin"] > tabela["inicio_bin"]).all()
    assert np.isclose(float(tabela["inicio_bin"].iloc[0]), 0.0)
    assert (tabela["total_legitima"] == 333).all()
    assert (tabela["total_fraude"] == 27).all()
    bordas = np.r_[tabela["inicio_bin"].to_numpy(), tabela["fim_bin"].iloc[-1]]
    esperado_legitimo = np.histogram(scores[y == 0], bins=bordas)[0]
    esperado_fraude = np.histogram(scores[y == 1], bins=bordas)[0]
    np.testing.assert_array_equal(
        tabela["contagem_legitima"].to_numpy(dtype=int), esperado_legitimo
    )
    np.testing.assert_array_equal(
        tabela["contagem_fraude"].to_numpy(dtype=int), esperado_fraude
    )
    np.testing.assert_allclose(
        tabela["frequencia_relativa_legitima"], esperado_legitimo / 333, atol=1e-12
    )
    np.testing.assert_allclose(
        tabela["frequencia_relativa_fraude"], esperado_fraude / 27, atol=1e-12
    )
    assert int(tabela["fora_do_zoom_legitima"].iloc[0]) == 333 - int(sum(esperado_legitimo))
    assert int(tabela["fora_do_zoom_fraude"].iloc[0]) == 27 - int(sum(esperado_fraude))
    assert bordas[-1] >= tau, "Zoom ocultou o limiar ficticio."

    hashes_depois = {str(p): _hash(p) for p in protegidos}
    assert hashes_depois == hashes_antes, "YAML/renderizador oficial modificado."

    meta = {
        "tipo": "PREVIA_SINTETICA_NAO_OFICIAL",
        "metodo_simulado": "TSNE",
        "dimensao_simulada": 1,
        "folds_sinteticos": list(range(1, 10)),
        "n_sintetico": len(y),
        "fraudes_sinteticas": int(y.sum()),
        "limiar_tau_ficticio_congelado": tau,
        "limiar_fixo_auditado_nao_exibido": 0.5,
        "somente_limiar_otimo_na_figura": True,
        "itens_legenda": 3,
        "ap_sintetica": ap,
        "mcc_tau_ficticio": mcc,
        "mcc_050": mcc_05,
        "logloss_sintetico": loss,
        "score_sintetico": composto,
        "bins": 50,
        "normalizacao_por_classe_real": True,
        "f10_real_acessado": False,
        "treinamento_real_executado": False,
        "arquivos_oficiais_alterados": False,
        "nota": "Apenas a linha ficticia tau*=0.35 e mostrada na imagem. "
                "Dados inteiramente sinteticos; tau nao foi otimizado em OOF real.",
    }
    manifesto = SAIDA / "LEIA_ME_PREVIA_SINTETICA.json"
    conteudo = json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if manifesto.exists():
        assert manifesto.read_text(encoding="utf-8") == conteudo, (
            "Manifesto divergente: arquivo anterior preservado, nao sobrescrever."
        )
    else:
        manifesto.write_text(conteudo, encoding="utf-8")

    print("\n=== V2 / TSNE 1D / GRAFICO 05 - DISTRIBUICAO SINTETICA ===")
    print("METODO_SIMULADO=TSNE | DIMENSAO_SIMULADA=1D")
    print("FOLDS_SINTETICOS=1,2,3,4,5,6,7,8,9")
    print(f"N_SINTETICO={len(y)} | FRAUDES_SINTETICAS={int(y.sum())}")
    print(f"AP={ap:.7f} | MCC_TAU_FICTICIO={mcc:.7f} | LOGLOSS={loss:.7f} | SCORE={composto:.7f}")
    print(f"TAU_FICTICIO={tau:.7f} | MCC_050_APENAS_AUDITORIA={mcc_05:.7f}")
    print("LIMIAR_FIXO_050_EXIBIDO=NAO")
    print("LEGENDA_APENAS_LEGITIMA_FRAUDE_TAU_OTIMO=SIM")
    print(f"BINS={len(tabela)} | NORMALIZACAO_POR_CLASSE=SIM | ZOOM=SIM")
    print("F10_REAL_ACESSADO=NAO")
    print("TREINAMENTO_REAL_EXECUTADO=NAO")
    print("ARQUIVOS_OFICIAIS_ALTERADOS=NAO")
    print(f"PNG_PARA_AVALIAR={png}")
    print(f"CSV_VERIFICACAO={csv}")
    print("V2_G05_ESTILO_PCA_V2=APROVADA_TECNICAMENTE_AGUARDANDO_AVALIACAO_VISUAL")


if __name__ == "__main__":
    main()
