"""Passo 02.02: criar e persistir 10 folds estratificados.

Executar na raiz: python src/folds.py
Nao faz encoding, reducao, amostragem ou treinamento.
"""

from hashlib import sha256
from pathlib import Path
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold
import yaml


RAIZ = Path(__file__).resolve().parents[1]
YAML = RAIZ / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml"
RAIZ_DIVISOES = RAIZ / "data" / "splits"


def hash_arquivo(caminho):
    resumo = sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def carregar_alvo(caminho, coluna):
    """Carrega somente o alvo, em lotes, preservando a ordem das linhas."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(caminho)
    if coluna not in parquet.schema_arrow.names:
        raise ValueError(f"Coluna-alvo ausente: {coluna}")

    valores = []
    for lote in parquet.iter_batches(columns=[coluna], batch_size=100_000):
        arr = lote.column(0)
        if arr.null_count:
            raise ValueError("A coluna-alvo contém valores ausentes.")
        valores.append(np.asarray(arr.to_numpy(zero_copy_only=False)))

    if not valores:
        raise ValueError("A base está vazia.")
    return np.concatenate(valores)


def criar_folds(y, total, semente, embaralhar):
    """Atribui um inteiro de 1 a 'total' a cada linha, sem perder a ordem."""
    validador = StratifiedKFold(
        n_splits=total,
        shuffle=embaralhar,
        random_state=semente if embaralhar else None,
    )
    folds = np.zeros(len(y), dtype=np.uint16)
    X_dummy = np.zeros((len(y), 1), dtype=np.uint8)
    for numero, (_, indices_validacao) in enumerate(
        validador.split(X_dummy, y), start=1
    ):
        folds[indices_validacao] = numero
    return folds


def conferir_folds(folds, y, total):
    """Impede guardar divisões incompletas ou não estratificadas."""
    if len(folds) != len(y) or np.any(folds < 1) or np.any(folds > total):
        raise ValueError("Atribuições de folds inválidas.")

    tabela = []
    classes = np.unique(y)
    contagens_por_classe = {str(c): [] for c in classes}

    for k in range(1, total + 1):
        selecionado = folds == k
        quantidade = int(selecionado.sum())
        if quantidade == 0:
            raise ValueError(f"Fold {k} vazio.")
        contagem = {str(c): int(np.count_nonzero(y[selecionado] == c)) for c in classes}
        for classe, n in contagem.items():
            contagens_por_classe[classe].append(n)
        tabela.append({"fold": k, "total": quantidade, "classes": contagem})

    for classe, valores in contagens_por_classe.items():
        if max(valores) - min(valores) > 1:
            raise ValueError(f"Distribuição da classe {classe} não estratificada.")

    return tabela


def main():
    if not YAML.is_file():
        raise FileNotFoundError(f"YAML oficial não encontrado: {YAML}")
    with YAML.open(encoding="utf-8-sig") as arquivo:
        config = yaml.safe_load(arquivo)

    dados = config["dados"]
    divisao = config["divisao"]
    total = int(divisao["total_folds"])
    teste = int(divisao["qual_fold_teste"])
    if total < 3 or not (1 <= teste <= total):
        raise ValueError("Número de folds ou fold de teste inválido.")
    if divisao["quantidade_folds_teste"] != 1:
        raise ValueError("Este projeto reserva exatamente um fold para teste.")
    if divisao["estratificada"] is not True:
        raise ValueError("Este passo implementa divisão estratificada.")
    if dados.get("fracao_base", 1.0) != 1.0:
        raise ValueError("Neste passo, fracao_base deve ser 1.0 (base integral).")

    caminho = RAIZ / dados["arquivo"]
    if not caminho.is_file():
        raise FileNotFoundError(f"Parquet não encontrado: {caminho}")

    y_original = carregar_alvo(caminho, dados["coluna_alvo"])
    classes, frequencias = np.unique(y_original, return_counts=True)
    positiva = dados["classe_positiva"]
    if len(classes) != 2 or positiva not in classes:
        raise ValueError("Alvo deve conter exatamente duas classes, incluindo a positiva.")
    if min(frequencias) < total:
        raise ValueError("Há menos exemplos de uma classe que o número de folds.")
    y = (y_original == positiva).astype(np.uint8)

    parametros = {
        "caminho_relativo": dados["arquivo"],
        "coluna_alvo": dados["coluna_alvo"],
        "classe_positiva": positiva,
        "total_folds": total,
        "fold_teste": teste,
        "estratificada": True,
        "embaralhar": bool(divisao["embaralhar"]),
        "semente": int(config["experimento"]["semente"]),
        "sha256_parquet": hash_arquivo(caminho),
        "total_linhas": len(y),
    }

    # Cada base e configuração de divisão tem sua própria pasta reutilizável.
    chave = sha256(json.dumps(parametros, sort_keys=True).encode()).hexdigest()[:12]
    pasta_saida = RAIZ_DIVISOES / f"{caminho.stem}_{chave}"
    arquivo_folds = pasta_saida / "folds.npy"
    arquivo_manifesto = pasta_saida / "manifesto_folds.json"

    presentes = (arquivo_folds.exists(), arquivo_manifesto.exists())
    if any(presentes):
        if not all(presentes):
            raise RuntimeError("Divisão anterior incompleta em data/splits. Verifique os arquivos.")
        manifesto = json.loads(arquivo_manifesto.read_text(encoding="utf-8"))
        if manifesto.get("parametros") != parametros:
            raise RuntimeError(
                "A divisão salva tem parâmetros incompatíveis. "
                "Verifique os arquivos antes de continuar."
            )
        folds = np.load(arquivo_folds, allow_pickle=False)
        if manifesto.get("sha256_folds") != hash_arquivo(arquivo_folds):
            raise RuntimeError("Arquivo folds.npy foi alterado ou corrompido.")
        situacao = "REUTILIZADOS (sem recriar)"
    else:
        folds = criar_folds(y, total, parametros["semente"], parametros["embaralhar"])
        situacao = "CRIADOS"

    tabela = conferir_folds(folds, y, total)
    if not any(presentes):
        pasta_saida.mkdir(parents=True, exist_ok=True)
        np.save(arquivo_folds, folds, allow_pickle=False)
        manifesto = {
            "parametros": parametros,
            "sha256_folds": hash_arquivo(arquivo_folds),
            "contagem_por_fold": tabela,
            "regra_cv": "Para cada F1..F9: valida nele, treina nos outros oito; F10 nunca entra.",
        }
        arquivo_manifesto.write_text(
            json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"\nPASSO 02.02 — FOLDS {situacao}")
    print("Fold | Tipo       | Linhas     | Legítimas  | Fraudes")
    print("-" * 60)
    for linha in tabela:
        c = linha["classes"]
        papel = "TESTE FINAL" if linha["fold"] == teste else "DESENVOLV."
        print(
            f"F{linha['fold']:02d}  | {papel:<10} | {linha['total']:>10,} | "
            f"{c['0']:>10,} | {c['1']:>7,}"
        )
    print(f"\nArquivos: {arquivo_folds.relative_to(RAIZ)}; "
          f"{arquivo_manifesto.relative_to(RAIZ)}")
    print("[OK] Cada linha pertence a um único fold.")
    print("[OK] Proporção das classes preservada em todos os folds.")
    print(f"[OK] F{teste} reservado para teste final.")
    print("PASSO 02.02 — VALIDAÇÃO APROVADA")


if __name__ == "__main__":
    main()
