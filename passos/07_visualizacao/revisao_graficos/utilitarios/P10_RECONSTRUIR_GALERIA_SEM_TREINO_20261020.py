"""P10: criar cópia revisada de galeria histórica, apenas a partir de PNGs existentes.

Executar na raiz do repositório TCC_EXPERIMENTAL_V3:
    python _transferencia_ias_1/P10_RECONSTRUIR_GALERIA_SEM_TREINO_20261020.py --verificar
    python _transferencia_ias_1/P10_RECONSTRUIR_GALERIA_SEM_TREINO_20261020.py --aplicar

Não importa módulos de machine learning, não ajusta modelos, não executa F10 e não
modifica os arquivos originais do ensaio. Exige confirmação de todos os ativos.
"""
from __future__ import annotations

import argparse
import copy
import html
import json
from pathlib import Path
import shutil

PASTA_ENSAIO = "ensaio_pca_gmm_20261017_195252"
PAINEL = Path("oof/00_painel_score_composto.png")
DESTINO_NOME = "visualizacao_consolidada_P10_revisada"
SECOES = {
    "oof": ("00–06 — Painel e validação OOF", "Predições OOF do experimento histórico."),
    "geometria_desenvolvimento": (
        "Geometria do desenvolvimento — ensaio histórico",
        "Artefatos do desenvolvimento F1–F9; não são avaliação F10.",
    ),
    "complementar_2d": (
        "16 — Diagnóstico 2D complementar",
        "Candidato complementar; não é o vencedor oficial.",
    ),
}


def _path_rel_seguro(nome: str) -> Path:
    relativo = Path(nome)
    if (relativo.is_absolute() or ".." in relativo.parts or
            len(relativo.parts) != 2 or relativo.parts[0] not in SECOES or
            relativo.suffix.lower() != ".png"):
        raise ValueError(f"Referência inesperada no manifesto: {nome!r}")
    return relativo


def _criar_html(arquivos: list[Path], manifesto: dict) -> str:
    cards = []
    for secao, (titulo, descricao) in SECOES.items():
        itens = [p for p in arquivos if p.parts[0] == secao]
        if not itens:
            continue
        cards.append(f'<section><h2>{html.escape(titulo)}</h2><p>{html.escape(descricao)}</p><div class="grade">')
        for p in itens:
            ref = html.escape(p.as_posix(), quote=True)
            nome = html.escape(p.stem.replace("_", " "))
            cards.append(
                f'<article><a href="{ref}" target="_blank" rel="noopener">'
                f'<img src="{ref}" loading="lazy" alt="{nome}">'
                f'<div>{nome}</div></a></article>'
            )
        cards.append("</div></section>")
    vencedor = html.escape(str(manifesto["id_vencedor"]))
    pontuacao = html.escape(str(manifesto["score_vencedor_oof"]))
    css = """body{font:16px system-ui,sans-serif;background:#f5f7fa;color:#172236;max-width:1320px;margin:auto;padding:24px}
section{margin-top:36px}.grade{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:20px}
article{background:white;border:1px solid #d7dee6;border-radius:10px;padding:12px;box-shadow:0 2px 8px #0001}
article img{width:100%;aspect-ratio:4/3;object-fit:contain}article a{color:#233a58;text-decoration:none}
.aviso{padding:12px;background:#e8edf4;border-left:4px solid #355877;border-radius:4px}"""
    return ("<!doctype html>\n<html lang=\"pt-BR\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>TCC — Galeria histórica revisada P10</title>"
            f"<style>{css}</style></head><body>"
            "<h1>Passo 07 — Galeria histórica revisada (P10)</h1>"
            f"<p class=\"aviso\">Vencedor OOF: {vencedor} | score OOF: {pontuacao}. "
            "Este HTML reproduz somente as imagens já existentes e acrescenta o Painel 00. "
            "O manifesto histórico e a galeria original foram preservados. F10 não utilizado.</p>"
            + "".join(cards) + "</body></html>\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--verificar", action="store_true", help="Somente verificar, sem escrever")
    grupo.add_argument("--aplicar", action="store_true", help="Criar galeria revisada em pasta nova")
    args = parser.parse_args()

    projeto = Path.cwd()
    ensaio = projeto / "outputs" / PASTA_ENSAIO
    anterior = ensaio / "visualizacao_consolidada"
    manifesto_antigo = anterior / "manifesto_integracao_07_03.json"
    painel_origem = ensaio / "oof" / "graficos_gerados" / PAINEL.name
    saida = ensaio / DESTINO_NOME

    if not manifesto_antigo.is_file():
        raise FileNotFoundError(f"Manifesto histórico não encontrado: {manifesto_antigo}")
    antigo = json.loads(manifesto_antigo.read_text(encoding="utf-8-sig"))
    if antigo.get("f10_utilizado") is not False or antigo.get("tipo_experimento") != "ensaio":
        raise ValueError("Manifesto não identifica um ensaio sem F10; operação bloqueada.")
    if antigo.get("id_vencedor") != "0d74604010afa3":
        raise ValueError("ID do vencedor difere do manifesto revisado; operação bloqueada.")
    anterior_lista = antigo.get("arquivos")
    if not isinstance(anterior_lista, list) or len(anterior_lista) != 14:
        raise ValueError("O manifesto histórico não possui os 14 artefatos esperados.")
    if PAINEL.as_posix() in anterior_lista or PAINEL.name in antigo.get("tipos_oof", []):
        raise ValueError("Painel 00 já consta do manifesto; não reaplicar.")
    if antigo.get("numero_imagens") != len(anterior_lista):
        raise ValueError("Quantidade de imagens do manifesto não confere.")

    imagens = [PAINEL] + [_path_rel_seguro(nome) for nome in anterior_lista]
    for rel in imagens:
        origem = painel_origem if rel == PAINEL else anterior / rel
        if not origem.is_file():
            raise FileNotFoundError(f"Imagem de origem não encontrada: {origem}")
    print(f"Ensaio: {ensaio}")
    print(f"Galeria original: {anterior}")
    print(f"Painel 00 localizado: {painel_origem}")
    print(f"Imagens originais: {len(anterior_lista)}; imagens na revisão: {len(imagens)}")
    print(f"Destino separado: {saida}")
    print("Treino, geometria, F10 e arquivos originais: NÃO serão executados ou alterados.")
    if args.verificar:
        print("VERIFICAÇÃO OK — nenhuma escrita efetuada.")
        return
    if saida.exists():
        raise FileExistsError(f"A pasta de revisão já existe; não sobrescrever: {saida}")

    tmp = ensaio / f".{DESTINO_NOME}_tmp"
    if tmp.exists():
        raise FileExistsError(f"Pasta temporária preexistente; inspecione: {tmp}")
    tmp.mkdir()
    try:
        for rel in imagens:
            origem = painel_origem if rel == PAINEL else anterior / rel
            destino = tmp / rel
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origem, destino)
        (tmp / "index.html").write_text(_criar_html(imagens, antigo), encoding="utf-8")
        novo = copy.deepcopy(antigo)
        novo["tipos_oof"] = [PAINEL.stem] + antigo["tipos_oof"]
        novo["arquivos"] = [r.as_posix() for r in imagens]
        novo["numero_imagens"] = len(imagens)
        novo["revisao_p10"] = {
            "tipo": "galeria_historica_reconstruida_sem_treino",
            "fonte": str(manifesto_antigo),
            "painel_adicionado": PAINEL.as_posix(),
            "originais_preservados": True,
            "f10_utilizado": False,
        }
        novo["galeria_html"] = str(saida / "index.html")
        (tmp / "manifesto_integracao_07_03.json").write_text(
            json.dumps(novo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        tmp.rename(saida)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f"REVISÃO P10 CRIADA: {saida / 'index.html'}")
    print("A revisão é independente do HTML histórico. Confira visualmente o Painel 00.")


if __name__ == "__main__":
    main()
