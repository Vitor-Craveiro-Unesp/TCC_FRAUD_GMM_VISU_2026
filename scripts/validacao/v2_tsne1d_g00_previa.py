"""V2 t-SNE 1D / G00: previa SINTETICA usando o renderizador OOF existente.

Nao treina GMM ou t-SNE, nao le dataset Parquet, folds.npy nem F10 real.
A unica escrita permitida e em outputs/validacao_tsne_v2_1d/00_painel_inicial/.
A previa nao comprova metricas do experimento: requer aprovacao visual humana.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import average_precision_score, log_loss, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml"
VISUALIZACAO = ROOT / "src" / "visualizacao_oof.py"
SAIDA = ROOT / "outputs" / "validacao_tsne_v2_1d" / "00_painel_inicial"


def arquivo_hash(caminho: Path) -> str:
    return sha256(caminho.read_bytes()).hexdigest()


def exigir(condicao: bool, mensagem: str):
    if not condicao:
        raise AssertionError(mensagem)


def gerar_amostra():
    """Semente local, dados artificiais e nove folds SEM F10."""
    rng = np.random.default_rng(20261101)
    n_por_fold = 40
    folds = np.repeat(np.arange(1, 10, dtype=np.uint8), n_por_fold)
    y = np.zeros(len(folds), dtype=np.uint8)
    for f in range(9):
        local = rng.choice(n_por_fold, size=3, replace=False)
        y[f * n_por_fold + local] = 1
    p = rng.beta(1.8, 10.0, size=len(folds))
    p[y == 1] = rng.beta(2.2, 2.8, size=int(y.sum()))
    p = np.clip(p, 1e-8, 1 - 1e-8)
    exigir(set(np.unique(folds)) == set(range(1, 10)), "Folds sinteticos invalidos.")
    exigir(not np.any(folds == 10), "F10 entrou nos dados sinteticos.")
    exigir(np.isin(y, [0, 1]).all() and 0 < y.sum() < len(y), "Rotulos invalidos.")
    return y, p, folds


def adicionar_tarjeta_sintetica(entrada: Path, saida: Path) -> None:
    """Adiciona aviso FORA do desenho aprovado; preserva PNG original."""
    if saida.exists():
        print(f"[PRESERVADO] {saida.name}")
        return
    with Image.open(entrada) as imagem:
        base = imagem.convert("RGB")
        altura_tarjeta = 84
        nova = Image.new("RGB", (base.width, base.height + altura_tarjeta), "#7b1627")
        nova.paste(base, (0, altura_tarjeta))
        draw = ImageDraw.Draw(nova)
        fonte = ImageFont.load_default()
        textos = [
            "PREVIA SINTETICA - TSNE 1D - GRAFICO 00",
            "NAO SAO DADOS REAIS DO TCC. F10 NAO UTILIZADO.",
        ]
        for i, texto in enumerate(textos):
            draw.text((28, 15 + 31 * i), texto, fill="white", font=fonte)
        nova.save(saida, format="PNG", optimize=True)
    print(f"[GERADO] {saida.name}")


def main():
    from src import visualizacao_oof as vis

    exigir(CONFIG.is_file(), f"YAML esperado nao encontrado: {CONFIG}")
    exigir(VISUALIZACAO.is_file(), f"Modulo visual OOF nao encontrado: {VISUALIZACAO}")
    antes = {p: arquivo_hash(p) for p in (CONFIG, VISUALIZACAO)}
    config = deepcopy(yaml.safe_load(CONFIG.read_text(encoding="utf-8-sig")))
    exigir(isinstance(config, dict), "YAML inesperado.")
    # Alteracoes exclusivamente em RAM; nao editar YAML oficial.
    config["reducao"]["metodo"] = "TSNE"
    config["tsne"]["n_components"] = 1
    config["visualizacao"]["graficos"]["painel_score_composto"]["habilitar"] = True

    y, p, folds = gerar_amostra()
    tau = 0.3500000  # Ficticio e fixo: nao otimizar com rotulos sinteticos.
    ap = float(average_precision_score(y, p))
    mcc = float(matthews_corrcoef(y, p >= tau))
    ll = float(log_loss(y, p, labels=[0, 1]))
    score = float((ap + (mcc + 1.0) / 2.0 + np.exp(-ll)) / 3.0)
    vencedor = pd.Series(dict(ap=ap, mcc=mcc, logloss=ll, score=score))
    escolha = dict(
        id_candidato="V2_G00_SINTETICO",
        limiar_mcc_oof=tau,
        teste_final_usado=False,
        status="ensaio_sintetico_sem_selecao_real",
    )

    verificar = vis._numeros_painel_00(y, p, vencedor, escolha)
    exigir(np.isclose(verificar["score_composto"], score, atol=1e-13), "Score composto divergente.")
    exigir(verificar["n_oof"] == len(folds), "Quantidade OOF sintetica divergente.")
    exigir(np.isclose(verificar["limiar_mcc_oof"], tau), "Tau ficticio modificado.")
    exigir(np.isclose(verificar["mcc_01"], (mcc + 1) / 2), "MCC transformado divergente.")
    exigir(np.isclose(verificar["logloss_01"], np.exp(-ll)), "LogLoss transformado divergente.")

    SAIDA.mkdir(parents=True, exist_ok=True)
    esperado_textos = [
        "PAINEL INICIAL", "INFORMAÇÕES ÚTEIS",
        "AP (AVERAGE PRECISION)", "MCC (MATTHEWS CORRELATION COEFFICIENT)",
        "LOGLOSS", "SCORE COMPOSTO",
        f"{ap:.7f}", f"{(mcc + 1.0) / 2.0:.7f}",
        f"{np.exp(-ll):.7f}", f"{score:.7f}",
        f"AP bruto: {ap:.7f}", f"MCC bruto: {mcc:.7f}",
        f"LogLoss bruto: {ll:.7f}", f"τ*={tau:.7f}",
        "F10 não utilizado",
    ]
    salvar_original = vis._salvar
    capturas = []

    def salvar_auditado(fig, destino, *, dpi):
        textos = [t.get_text() for t in fig.texts]
        for requisito in esperado_textos:
            exigir(any(requisito in texto for texto in textos),
                   f"Texto ausente na figura: {requisito}")
        capturas.append(len(textos))
        return salvar_original(fig, destino, dpi=dpi)

    vis._salvar = salvar_auditado
    try:
        vis._plot_painel_00(
            config, y, p, vencedor, escolha, SAIDA,
            "png", 180, True,
        )
    finally:
        vis._salvar = salvar_original

    exigir(len(capturas) == 1, "Renderizador nao foi invocado exatamente uma vez.")
    original = SAIDA / "00_painel_score_composto.png"
    dados = SAIDA / "00_painel_score_composto.csv"
    previa = SAIDA / "00_painel_score_composto_PREVIA_SINTETICA.png"
    exigir(original.is_file() and original.stat().st_size > 10000, "PNG nao foi criado corretamente.")
    exigir(dados.is_file() and dados.stat().st_size > 100, "CSV nao foi criado corretamente.")
    tabela = pd.read_csv(dados, encoding="utf-8-sig")
    exigir(len(tabela) == 4, "Painel deveria conter quatro linhas de indicadores.")
    exigir(tabela["indicador"].tolist() == ["AP", "MCC", "LogLoss", "Score composto"],
           "Ordem dos indicadores divergente.")
    exigir(np.allclose(tabela["valor_0_a_1"].to_numpy(),
            [ap, (mcc + 1.0) / 2.0, np.exp(-ll), score], atol=1e-12),
           "Os valores CSV divergiram dos indicadores.")
    exigir(np.allclose(tabela["limiar_mcc_oof"], tau), "Limiar no CSV alterado.")
    adicionar_tarjeta_sintetica(original, previa)
    exigir(previa.is_file() and previa.stat().st_size > 10000, "Previa anotada ausente.")
    depois = {p: arquivo_hash(p) for p in antes}
    exigir(antes == depois, "O script alterou o YAML ou visualizacao_oof.py.")
    print("\n=== V2 / TSNE 1D / GRAFICO 00 - VALIDACAO SINTETICA ===")
    print("METODO_SIMULADO=TSNE")
    print("DIMENSAO_SIMULADA=1D")
    print(f"FOLDS_DESENVOLVIMENTO={','.join(map(str, np.unique(folds)))}")
    print(f"N_SINTETICO={len(y)} | FRAUDES_SINTETICAS={int(y.sum())}")
    print(f"AP={ap:.7f} | MCC={mcc:.7f} | LOGLOSS={ll:.7f} | SCORE={score:.7f}")
    print("F10_REAL_ACESSADO=NAO")
    print("TREINAMENTO_REAL_EXECUTADO=NAO")
    print("ARQUIVOS_OFICIAIS_ALTERADOS=NAO")
    print(f"PREVIA_PARA_AVALIAR={previa}")
    print(f"CSV_VERIFICACAO={dados}")
    print("V2_G00_VALIDACAO=APROVADA_TECNICAMENTE_AGUARDANDO_AVALIACAO_VISUAL")


if __name__ == "__main__":
    main()
