from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# 1. CONFIGURACAO
# =====================================================

RAIZ = Path(__file__).resolve().parent

ENTRADA = RAIZ / "data" / "raw" / "dataset_raw.parquet"

PASTA_SAIDA = RAIZ / "data" / "whav"

SAIDA = PASTA_SAIDA / "dataset_raw_whav.parquet"

RAIO_TERRA_KM = 6371.0

COLUNAS_GEO = [
    "SEND_LAT_REGISTER",
    "SEND_LONG_REGISTER",
    "RECEIVE_LAT",
    "RECEIVE_LONG",
]

NOVA_COLUNA = "DISTANCIA_GEOGRAFICA[HAV]"

# =====================================================
# 2. LEITURA E VALIDACAO
# =====================================================

if not ENTRADA.exists():
    raise FileNotFoundError(
        f"Arquivo original nao encontrado: {ENTRADA}"
    )

df = pd.read_parquet(ENTRADA)

faltantes = [
    coluna for coluna in COLUNAS_GEO
    if coluna not in df.columns
]

if faltantes:
    raise ValueError(
        f"Colunas geograficas ausentes: {faltantes}"
    )

if NOVA_COLUNA in df.columns:
    raise ValueError(
        f"A coluna {NOVA_COLUNA} ja existe."
    )

n_linhas_original = len(df)
n_colunas_original = len(df.columns)

# =====================================================
# 3. PREPARACAO DAS COORDENADAS
# =====================================================

coordenadas = df[COLUNAS_GEO].apply(
    pd.to_numeric,
    errors="coerce"
)

# Coordenadas devem estar em graus decimais validos
validas = (
    coordenadas["SEND_LAT_REGISTER"].between(-90, 90)
    & coordenadas["SEND_LONG_REGISTER"].between(-180, 180)
    & coordenadas["RECEIVE_LAT"].between(-90, 90)
    & coordenadas["RECEIVE_LONG"].between(-180, 180)
)

# Substituicao temporaria apenas para o calculo
# Coordenadas invalidas receberao NaN no resultado
coords = coordenadas.where(validas, 0.0)

# =====================================================
# 4. CONVERSAO PARA RADIANOS
# =====================================================

phi1 = np.radians(
    coords["SEND_LAT_REGISTER"].to_numpy()
)

lambda1 = np.radians(
    coords["SEND_LONG_REGISTER"].to_numpy()
)

phi2 = np.radians(
    coords["RECEIVE_LAT"].to_numpy()
)

lambda2 = np.radians(
    coords["RECEIVE_LONG"].to_numpy()
)

# =====================================================
# 5. FORMULA DE HAVERSINE
# =====================================================

delta_phi = phi2 - phi1
delta_lambda = lambda2 - lambda1

a = (
    np.sin(delta_phi / 2.0) ** 2
    + np.cos(phi1)
    * np.cos(phi2)
    * np.sin(delta_lambda / 2.0) ** 2
)

# Protecao contra erros numericos
a = np.clip(a, 0.0, 1.0)

distancia_km = (
    2.0
    * RAIO_TERRA_KM
    * np.arcsin(np.sqrt(a))
)

# Coordenadas ausentes ou invalidas: NaN
distancia_km[~validas.to_numpy()] = np.nan

# =====================================================
# 6. TRANSFORMACAO DO DATASET
# =====================================================

# Substituir as quatro coordenadas por uma distancia
df_transformado = df.drop(columns=COLUNAS_GEO)

df_transformado[NOVA_COLUNA] = distancia_km

# =====================================================
# 7. SALVAMENTO
# =====================================================

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)

df_transformado.to_parquet(
    SAIDA,
    engine="pyarrow",
    compression="snappy",
    index=False
)

# =====================================================
# 8. VALIDACAO DO ARQUIVO GERADO
# =====================================================

df_verificacao = pd.read_parquet(SAIDA)

assert len(df_verificacao) == n_linhas_original

assert (
    len(df_verificacao.columns)
    == n_colunas_original - 3
)

assert NOVA_COLUNA in df_verificacao.columns

assert all(
    coluna not in df_verificacao.columns
    for coluna in COLUNAS_GEO
)

# =====================================================
# 9. RELATORIO
# =====================================================

print("\n" + "=" * 55)
print("TRANSFORMACAO HAVERSINE CONCLUIDA")
print("=" * 55)

print(f"Observacoes: {n_linhas_original:,}")
print(f"Variaveis originais: {n_colunas_original}")
print(f"Variaveis finais: {len(df_verificacao.columns)}")

print(f"\nNova variavel: {NOVA_COLUNA}")

print(
    f"Distancias validas: "
    f"{int(np.isfinite(distancia_km).sum()):,}"
)

print(
    f"Distancias ausentes/invalidas: "
    f"{int(np.isnan(distancia_km).sum()):,}"
)

print("\nEstatisticas da distancia (km):")
print(df_transformado[NOVA_COLUNA].describe())

print(f"\nArquivo salvo em:\n{SAIDA}")

print("\nVALIDACAO: OK")
