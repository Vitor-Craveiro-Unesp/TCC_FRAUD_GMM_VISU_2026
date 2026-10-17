"""Passo 02.01: auditoria descritiva, somente leitura, sem ajustar modelos.

Execute na raiz do projeto: python src/data.py
"""

from collections import Counter
from datetime import datetime
from pathlib import Path
import json

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


RAIZ = Path(__file__).resolve().parents[1]
CONFIG = RAIZ / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml"
PASTA_RELATORIO = RAIZ / "passos" / "02_dados" / "01_auditoria_da_base"
TAMANHO_LOTE = 100_000


def contar_nulos(parquet, indice, nome):
    """Usa metadados; lê uma coluna somente se não houver contagem registrada."""
    totais = []
    for grupo in range(parquet.num_row_groups):
        estatisticas = parquet.metadata.row_group(grupo).column(indice).statistics
        if estatisticas is None or not estatisticas.has_null_count:
            totais = None
            break
        totais.append(estatisticas.null_count)

    if totais is not None:
        return int(sum(totais))
    return sum(
        lote.column(0).null_count
        for lote in parquet.iter_batches(batch_size=TAMANHO_LOTE, columns=[nome])
    )


def amostra_coluna(parquet, nome, limite=20_000):
    """Lê só a primeira amostra, sem carregar o Parquet inteiro."""
    lote = next(parquet.iter_batches(batch_size=limite, columns=[nome]), None)
    return [] if lote is None else lote.column(0).to_pylist()


def auditar_base():
    if not CONFIG.is_file():
        raise FileNotFoundError(f"YAML oficial não encontrado: {CONFIG}")

    with CONFIG.open(encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Arquivo YAML vazio ou inválido.")

    dados = config["dados"]
    arquivo = RAIZ / dados["arquivo"]
    alvo = dados["coluna_alvo"]
    identificador = dados["coluna_id"]
    positiva = dados["classe_positiva"]

    if not arquivo.is_file():
        raise FileNotFoundError(f"Base não encontrada: {arquivo}")

    parquet = pq.ParquetFile(arquivo)
    esquema = parquet.schema_arrow
    nomes = esquema.names
    for coluna in (alvo, identificador):
        if coluna not in nomes:
            raise ValueError(f"Coluna obrigatória ausente: {coluna}")

    variaveis = []
    for i, nome in enumerate(nomes):
        campo = esquema.field(i)
        variaveis.append({
            "nome": nome,
            "tipo": str(campo.type),
            "nulos": contar_nulos(parquet, i, nome),
        })

    # Frequências do alvo: leitura sequencial em lotes, sem DataFrame completo.
    contagem_alvo = Counter()
    for lote in parquet.iter_batches(batch_size=TAMANHO_LOTE, columns=[alvo]):
        frequencias = pc.value_counts(lote.column(0)).to_pylist()
        for item in frequencias:
            chave = "<NULO>" if item["values"] is None else str(item["values"])
            contagem_alvo[chave] += int(item["counts"])

    n = parquet.metadata.num_rows
    n_fraude = contagem_alvo.get(str(positiva), 0)

    # Diagnóstico parcial de identificadores: não presume unicidade global.
    ids_amostra = amostra_coluna(parquet, identificador, limite=50_000)
    ids_nao_nulos = [x for x in ids_amostra if x is not None]
    repeticoes_amostra = len(ids_nao_nulos) - len(set(ids_nao_nulos))

    # Categorias observadas somente em pequena amostra: inspeção, não encoding.
    resumo_categoricas = {}
    for coluna in variaveis:
        nome = coluna["nome"]
        tipo = esquema.field(nome).type
        if nome in (alvo, identificador):
            continue
        if pa.types.is_string(tipo) or pa.types.is_large_string(tipo) or pa.types.is_dictionary(tipo):
            valores = amostra_coluna(parquet, nome)
            diferentes = {str(v) for v in valores if v is not None}
            resumo_categoricas[nome] = {
                "observacoes_amostra": len(valores),
                "categorias_distintas_na_amostra": len(diferentes),
                "exemplos": sorted(diferentes)[:8],
            }

    suspeitas_tempo = [
        nome for nome in nomes
        if any(x in nome.upper() for x in ("DATE", "TIME", "YEAR", "MONTH", "DAY", "HOUR", "WEEK"))
    ]
    suspeitas_entidade = [
        nome for nome in nomes
        if nome != identificador and any(x in nome.upper() for x in ("CARD", "ACCOUNT", "CUSTOMER", "CLIENT", "NAME", "USER", "ID"))
    ]

    resultado = {
        "gerado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
        "arquivo": str(arquivo.relative_to(RAIZ)),
        "tamanho_arquivo_mib": round(arquivo.stat().st_size / (1024 ** 2), 2),
        "quantidade_registros": n,
        "quantidade_variaveis": len(nomes),
        "row_groups": parquet.num_row_groups,
        "alvo": alvo,
        "classe_positiva_configurada": positiva,
        "distribuicao_alvo": dict(contagem_alvo),
        "proporcao_positiva": round(n_fraude / n, 8) if n else None,
        "identificador": identificador,
        "ids_inspecionados_amostra": len(ids_nao_nulos),
        "ids_repetidos_amostra": repeticoes_amostra,
        "variaveis": variaveis,
        "categoricas_amostra": resumo_categoricas,
        "possiveis_variaveis_temporais": suspeitas_tempo,
        "possiveis_identificadores_entidade": suspeitas_entidade,
        "ressalvas": [
            "Inspeção descritiva: não foram ajustados encoders, PCA, t-SNE ou GMM.",
            "Unicidade de ID avaliada somente em amostra; não comprova ausência de dependência entre transações.",
            "Amostras de categorias não representam cardinalidade completa.",
            "Antes de definir os folds, avaliar dependência temporal e agrupamentos por entidade.",
            "Não utilizar o conjunto de teste final para escolher hiperparâmetros ou transformações.",
        ],
    }

    linhas = [
        "=== PASSO 02.01 — AUDITORIA DESCRITIVA ===",
        f"Arquivo: {resultado['arquivo']}",
        f"Tamanho: {resultado['tamanho_arquivo_mib']} MiB",
        f"Registros: {n:,}",
        f"Variáveis: {len(nomes)} | Row groups: {parquet.num_row_groups}",
        "",
        "=== DISTRIBUIÇÃO DO ALVO ===",
    ]
    for classe, qtd in contagem_alvo.items():
        linhas.append(f"Classe {classe}: {qtd:,} ({100 * qtd / n:.4f}%)")
    linhas += [
        "",
        "=== COLUNAS (TIPO E AUSÊNCIAS) ===",
        *[f"{v['nome']} | {v['tipo']} | nulos={v['nulos']:,}" for v in variaveis],
        "",
        "=== IDENTIFICADOR (SOMENTE AMOSTRA) ===",
        f"Coluna: {identificador}",
        f"IDs não nulos verificados: {len(ids_nao_nulos):,}",
        f"Repetições encontradas na amostra: {repeticoes_amostra:,}",
        "",
        "=== CATEGORIAS: CARDINALIDADE NA AMOSTRA ===",
    ]
    for nome, info in resumo_categoricas.items():
        linhas.append(
            f"{nome}: {info['categorias_distintas_na_amostra']} distintas na amostra; "
            f"exemplos={info['exemplos']}"
        )
    linhas += [
        "",
        "=== CANDIDATAS A VERIFICAÇÃO DE DEPENDÊNCIA ===",
        f"Temporais: {suspeitas_tempo}",
        f"Entidades: {suspeitas_entidade}",
        "",
        "ATENÇÃO: termos no nome não comprovam vazamento; exigem revisão.",
        "A escolha entre divisão estratificada, temporal ou por grupos será feita depois.",
        "PASSO 02.01 — EXECUÇÃO DA AUDITORIA FINALIZADA; AGUARDANDO REVISÃO.",
    ]

    PASTA_RELATORIO.mkdir(parents=True, exist_ok=True)
    (PASTA_RELATORIO / "auditoria.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    texto = "\n".join(linhas) + "\n"
    (PASTA_RELATORIO / "saida_auditoria.txt").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    auditar_base()
