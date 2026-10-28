r"""Auditoria de integração PCA 3D sem treinamento e sem F10.

Execute na raiz TCC_EXPERIMENTAL_V3:
    .\.venv\Scripts\python.exe .\auditar_galeria_pca3d.py

1. Localiza os arquivos gerados por validacoes anteriores (sinteticos).
2. Em um novo outputs/validacao_galeria_pca3d_*, monta um conjunto isolado.
3. Verifica completude com o codigo real _verificar_geracoes(...,executar=False).
4. Executa o filtro/copia/index HTML reais de integrar() em fixture isolada.
5. Confere todos os links e escreve relatorio JSON.

Nao altera YAML, src, ensaios ou galerias reais. Nao executa OOF/F10.
"""
from __future__ import annotations

from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import shutil
import sys

# Os arquivos da auditoria sao agrupados por familia, nao por dados reais.
ARQUIVOS = {
    "PCA — Diagnósticos": [
        "11_pca_cotovelo.png",
        "12_pca_projecao_PC1_PC2_PC3.png",
        "PCA_BIPLOT_3D.png",
        "PCA_BIPLOT_3D_INTERATIVO.html",
        "PCA_COEFICIENTES_PC1_PC2_PC3.png",
    ],
    "GMM01 — Densidades": [
        "GMM01_densidade_marginal_3d_PC1.png",
        "GMM01_densidade_marginal_3d_PC2.png",
        "GMM01_densidade_marginal_3d_PC3.png",
        "GMM01_conjunta_corte_PC1_PC2.png",
        "GMM01_conjunta_corte_PC1_PC3.png",
        "GMM01_conjunta_corte_PC2_PC3.png",
        "GMM01_geometria_gaussianas_FULL_3d_95.png",
        "GMM01_densidade_conjunta_PCA3D_INTERATIVA.html",
    ],
    "GMM04 — Dominância": [
        "GMM04_dominancia_3d.png",
        "GMM04_dominancia_3d_INTERATIVA.html",
    ],
    "GMM07 — Sensibilidade": [
        "GMM07_sensibilidade_3d_comparativo.png",
        "GMM07_sensibilidade_3d_comparativo_INTERATIVA.html",
    ],
    "GMM08 — Mapa de Risco": [
        "GMM08_mapa_risco_3d_comparativo.png",
        "GMM08_mapa_risco_3d_comparativo_INTERATIVA.html",
    ],
    "GMM09 — Parâmetros FULL": ["09_painel_parametros_gaussianos.png"],
    "GMM18 — Convergência EM": ["GMM18_convergencia_em_inicializacoes.png"],
}
NECESSARIOS = [item for itens in ARQUIVOS.values() for item in itens]
assert len(NECESSARIOS) == len(set(NECESSARIOS)) == 21


class LeitorLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        props = dict(attrs)
        if tag == "a" and props.get("href"):
            self.links.append(("href", props["href"]))
        elif tag == "img" and props.get("src"):
            self.links.append(("src", props["src"]))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for bloco in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def localizar_fontes(outputs: Path) -> dict[str, Path]:
    candidatos: dict[str, list[Path]] = {a: [] for a in NECESSARIOS}
    # Apenas validacoes isoladas; nao consultar ensaios reais ou F10.
    for pasta in outputs.iterdir():
        if not (pasta.is_dir() and pasta.name.startswith("validacao_")):
            continue
        if pasta.name.startswith("validacao_galeria_pca3d_"):
            continue
        for arquivo in pasta.rglob("*"):
            if arquivo.is_file() and arquivo.name in candidatos:
                # Nunca escolher um item da galeria anterior em lugar da fonte.
                if "galeria_teste" in arquivo.parts or "galeria" in arquivo.parts:
                    continue
                if "PCA_1D" in arquivo.parts or "PCA_2D" in arquivo.parts:
                    continue
                candidatos[arquivo.name].append(arquivo)
    selecionados: dict[str, Path] = {}
    for nome, lista in candidatos.items():
        if lista:
            selecionados[nome] = max(
                lista,
                key=lambda p: (
                    int(p.parent.name == "PCA_3D" and "integracao" in p.parent.parent.name),
                    p.stat().st_mtime_ns,
                ),
            )
    return selecionados


def auditar():
    raiz = Path.cwd().resolve()
    if not (raiz / "src" / "visualizacao_integrada.py").is_file():
        raise RuntimeError("Execute na raiz TCC_EXPERIMENTAL_V3 (src/ ausente).")
    if not (raiz / "outputs").is_dir():
        raise RuntimeError("Diretorio outputs/ nao encontrado.")

    from src import visualizacao_integrada as integ
    from src import visualizacao_geometrica as geo
    from src import visualizacao_oof as oof
    from src import estrutura_saida as layout
    import pandas as pd
    import yaml

    config_path = raiz / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml"
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes.decode("utf-8-sig"))
    if ".html" not in integ.FORMATS:
        raise RuntimeError("Integrador nao aceita HTML em FORMATS.")
    if str(config["visualizacao"]["padrao"].get("formato", "png")).lower() != "png":
        raise RuntimeError("Este teste foi projetado para PNG + HTML, YAML em outro formato.")

    fontes = localizar_fontes(raiz / "outputs")
    faltantes = [nome for nome in NECESSARIOS if nome not in fontes]
    raiz_nova = (raiz / "outputs" /
                 ("validacao_galeria_pca3d_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    raiz_nova.mkdir(parents=True, exist_ok=False)
    geodir = layout.imagens_geometria(raiz_nova)
    oofdir = layout.imagens_oof(raiz_nova)
    geodir.mkdir(parents=True, exist_ok=True)
    oofdir.mkdir(parents=True, exist_ok=True)
    relatorio: dict = {
        "tipo": "AUDITORIA_SINTETICA_GALERIA_PCA3D_NAO_OOF",
        "total_esperado": len(NECESSARIOS),
        "total_encontrado": len(fontes),
        "faltantes": faltantes,
        "fontes": {nome: str(caminho) for nome, caminho in fontes.items()},
        "f10_utilizado": False,
        "executou_modelos": False,
        "config_original_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    for nome, origem in fontes.items():
        shutil.copy2(origem, geodir / nome)

    # Verificar regressao: arquivos de outra dimensao NAO entram no HTML.
    # Criar PNGs de teste separados dos arquivos reais e so na fixture.
    for legado in ("GMM04_dominancia_2d.png", "PCA_BIPLOT_2D.png",
                   "GMM08_mapa_risco_2d_comparativo.png", "LEGADO_999.png"):
        (geodir / legado).write_bytes(b"LEGADO_TESTE_NAO_E_IMAGEM")

    # Os placeholders OOF existem APENAS para verificar a logica de completude
    # (que usa is_file). Nao representam predicoes e nunca sao copiados na galeria.
    oof_ativos = [n for n in oof.ARQUIVOS if oof._ativado(config, n)]
    for nome in oof_ativos:
        (oofdir / (oof.ARQUIVOS[nome] + ".png")).write_bytes(b"SO_TESTE_EXISTENCIA_OOF")

    escolha = {
        "id_candidato": "PREVIA_SINTETICA_NAO_OOF",
        "score_oof": 0.0,
        "teste_final_usado": False,
        "configuracao": {
            "metodo": "PCA",
            "reducao": {"n_components": 3},
            "gmm": {"n_components": 3},
        },
    }
    config_teste = dict(config)
    config_teste["pca"] = dict(config["pca"], n_components=[3])
    with patch.object(layout, "imagens_oof", return_value=oofdir), \
         patch.object(layout, "imagens_geometria", return_value=geodir):
        od, gd, ativos_oof, ativos_geo, pronto_oof, pronto_geo = integ._verificar_geracoes(
            raiz_nova, config_teste, escolha, executar=False
        )
    relatorio["completude_oficial"] = {"oof": bool(pronto_oof), "geometria": bool(pronto_geo)}
    relatorio["tipos_ativos"] = list(ativos_geo)
    esperado_ativo = {
        "densidades_marginais", "regioes_dominancia", "painel_parametros_gaussianos",
        "pca_cotovelo", "pca_projecoes", "pca_biplot", "mapa_risco_fronteira",
        "sensibilidade_fronteira", "convergencia_em",
    }
    inativos = sorted(esperado_ativo - set(ativos_geo))
    relatorio["tipos_inativos_no_yaml"] = inativos

    # Executa INTEGRAR REAL, com entrada simulada e _verificar_geracoes
    # bloqueada para nao reexecutar 07.01/07.02. A filtragem/copia/HTML sao reais.
    (raiz_nova / "manifesto.json").write_text(
        json.dumps({"status": "concluido_06_03", "tipo": "ensaio"}),
        encoding="utf-8",
    )
    nao_gerar = (oofdir, geodir, [], list(esperado_ativo), True, True)
    with patch.object(oof, "_ler_resultados", return_value=(config_teste, pd.DataFrame(), escolha, None, None)), \
         patch.object(integ, "_verificar_geracoes", return_value=nao_gerar):
        result = integ.integrar(raiz_nova, planejar=False, complementar_2d=False)

    index = Path(result["galeria_html"])
    pagina = index.read_text(encoding="utf-8")
    # Selo visivel para nao confundir esta previa com validacao OOF real.
    pagina = pagina.replace(
        "Vencedor OOF: PREVIA_SINTETICA_NAO_OOF | PCA | Score = 0.000000",
        "PREVIA SINTETICA — PCA 3D; ARQUIVOS DE VALIDACOES DIFERENTES — NAO OOF REAL",
    ).replace(
        "Ensaio amostral: resultados exploratorios. Sem F10.",
        "PREVIA SINTETICA: arquivos de validacoes independentes; sem OOF ou F10 real.",
    ).replace(
        "Modelo reajustado em F1–F9. Esta pasta NAO contem medidas do F10.",
        "Imagens sinteticas independentes para testar a galeria; nao correspondem a um unico modelo.",
    ).replace(
        "Mapa complementar 2D nao e o modelo vencedor nem altera o ranking OOF.",
        "NENHUM experimento foi executado nesta auditoria; somente copia e verificacao de links.",
    )
    index.write_text(pagina, encoding="utf-8")

    leitor = LeitorLinks()
    leitor.feed(pagina)
    links_quebrados = []
    for _, endereco in leitor.links:
        if endereco.startswith(("http://", "https://", "#", "data:", "javascript:")):
            continue
        if not (index.parent / endereco.split("#")[0].split("?")[0]).is_file():
            links_quebrados.append(endereco)
    copiados = {Path(p).name for p in result["arquivos"]}
    nao_copiados = sorted(set(NECESSARIOS) - copiados)
    indevidos = sorted(set(copiados) - set(NECESSARIOS))
    relatorio.update({
        "index_html": str(index),
        "copiados": len(copiados),
        "arquivos_ausentes_na_galeria": nao_copiados,
        "arquivos_indevidos_na_galeria": indevidos,
        "links_quebrados": links_quebrados,
        "total_links_html": len(leitor.links),
        "yaml_preservado": hashlib.sha256(config_path.read_bytes()).hexdigest() == relatorio["config_original_sha256"],
    })
    aprovado = (
        not faltantes and not nao_copiados and not indevidos and not links_quebrados
        and pronto_geo and pronto_oof and not inativos and relatorio["yaml_preservado"]
        and len(copiados) == 21
    )
    relatorio["auditoria_aprovada"] = aprovado
    (raiz_nova / "auditoria_galeria_pca3d.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== AUDITORIA DA GALERIA PCA 3D (SINTETICA) ===")
    print(f"Imagens e HTMLs encontrados: {len(fontes)}/{len(NECESSARIOS)}")
    print(f"Copiados na galeria pelo integrador: {len(copiados)}/{len(NECESSARIOS)}")
    print(f"Completude oficial em fixture isolada: OOF={pronto_oof}; geometria={pronto_geo}")
    if faltantes:
        print("FALTANTES NAS VALIDACOES:", ", ".join(faltantes))
    if inativos:
        print("TIPOS INATIVOS NO YAML:", ", ".join(inativos))
    if nao_copiados:
        print("FALTANTES NA GALERIA:", ", ".join(nao_copiados))
    if indevidos:
        print("INDEVIDOS NA GALERIA:", ", ".join(indevidos))
    if links_quebrados:
        print("LINKS QUEBRADOS:", ", ".join(links_quebrados))
    print("YAML preservado:", relatorio["yaml_preservado"])
    print("F10 nao utilizado; nao executou OOF nem geometria real.")
    print("GALERIA=" + str(index))
    print("RELATORIO=" + str(raiz_nova / "auditoria_galeria_pca3d.json"))
    print("RESULTADO=" + ("APROVADO" if aprovado else "PENDENTE"))
    return 0 if aprovado else 2


if __name__ == "__main__":
    try:
        sys.exit(auditar())
    except Exception as exc:
        print("ERRO NA AUDITORIA:", type(exc).__name__ + ":", exc, file=sys.stderr)
        sys.exit(1)
