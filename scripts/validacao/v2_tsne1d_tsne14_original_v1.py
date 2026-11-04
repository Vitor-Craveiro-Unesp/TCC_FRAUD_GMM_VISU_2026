# -*- coding: utf-8 -*-
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import json
import time

import numpy as np
import pandas as pd
import yaml

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from src.reducao_tsne import RedutorTSNE
from src import visualizacao_tsne_pdf as tsne_pdf

RAIZ = Path.cwd()

SAIDA = (
    RAIZ / "outputs/validacao_tsne_v2_1d"
    / "TSNE14_diagnostico_original_v1"
)

PNG = SAIDA / "14_tsne_diagnostico.png"
CSV = SAIDA / "14_tsne_diagnostico.csv"

# 01 - PROTEGER ARQUIVOS OFICIAIS
protegidos = [
    RAIZ / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml",
    RAIZ / "src/reducao_tsne.py",
    RAIZ / "src/visualizacao_tsne_pdf.py",
    RAIZ / "src/visualizacao_geometrica.py",
    RAIZ / "src/visualizacao_oof.py",
]

def hash_arquivo(p):
    return sha256(p.read_bytes()).hexdigest()

antes = {
    str(p): hash_arquivo(p)
    for p in protegidos
}

config = deepcopy(
    yaml.safe_load(
        protegidos[0].read_text(encoding="utf-8-sig")
    )
)

config["reducao"]["metodo"] = "TSNE"
config["experimento"]["semente"] = 42
config["execucao"]["n_jobs"] = 2

parametros = {
    "perplexity": 10,
    "n_components": 1,
    "initialization": "pca",
    "metric": "euclidean",
    "learning_rate": "auto",
    "early_exaggeration": 12.0,
    "n_iter": 300,
    "theta": 0.5,
}

config["tsne"].update(parametros)

# Configuracao da visualizacao exclusivamente em memoria.
op = config["visualizacao"]["graficos"]["tsne_diagnostico"]

op["mostrar_kl_final"] = True
op["mostrar_tempo_execucao"] = True

# 02 - DADOS EXCLUSIVAMENTE SINTETICOS
X, y = make_classification(
    n_samples=1200,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    n_repeated=0,
    n_clusters_per_class=2,
    weights=[0.92, 0.08],
    class_sep=1.2,
    random_state=42,
)

X = StandardScaler().fit_transform(X)

features = pd.DataFrame(
    X,
    columns=[
        f"FEATURE_{i:02d}"
        for i in range(1, 13)
    ],
)

assert features.shape == (1200, 12)
assert np.isfinite(features.to_numpy()).all()

# 03 - AJUSTE TSNE REAL APENAS EM DADOS ARTIFICIAIS
inicio = time.perf_counter()

redutor = RedutorTSNE(
    config,
    parametros,
).fit(features)

coordenadas = redutor.transform_treino()

tempo = float(time.perf_counter() - inicio)

assert coordenadas.shape == (1200, 1)
assert list(coordenadas.columns) == ["TSNE1"]
assert np.isfinite(coordenadas.to_numpy()).all()
assert np.isfinite(tempo) and tempo >= 0

kl_bruto = redutor.kl_divergence_

kl = (
    None
    if kl_bruto is None
    else float(kl_bruto)
)

if kl is not None:
    assert np.isfinite(kl)
    assert kl >= 0

# 04 - DADOS EXIGIDOS PELO RENDERIZADOR
dados = {
    "redutor": redutor,
    "coord": coordenadas,
    "tempo_reducao": tempo,
}

if SAIDA.exists():
    raise FileExistsError(
        "Saida TSNE14 ja existente: " + str(SAIDA)
    )

SAIDA.mkdir(parents=True, exist_ok=False)

# 05 - MESMO RENDERIZADOR OFICIAL
resultado = tsne_pdf.diagnostico_kl(
    config,
    dados,
    SAIDA,
    "png",
    180,
    True,
    origem="simulado",
)

# 06 - VALIDACOES
assert PNG.is_file(), "PNG nao gerado."
assert CSV.is_file(), "CSV nao gerado."

assert PNG.stat().st_size > 1000
assert CSV.stat().st_size > 0

tabela = pd.read_csv(CSV)

assert len(tabela) == 1

registro = tabela.iloc[0]

assert int(registro["n_componentes"]) == 1
assert int(registro["n_desenvolvimento"]) == 1200
assert float(registro["perplexity"]) == 10.0

assert np.isclose(
    float(registro["tempo_reducao_s"]),
    tempo,
    rtol=1e-9,
    atol=1e-9,
)

if kl is not None:
    assert np.isclose(
        float(registro["kl_final"]),
        kl,
        rtol=1e-9,
        atol=1e-9,
    )
else:
    assert pd.isna(registro["kl_final"])

assert not bool(registro["historico_disponivel"])
assert resultado["historico_disponivel"] is False

assert "SINTÉTICA" in str(registro["origem"]).upper()

# 07 - MANIFESTO
manifesto = {
    "grafico": "TSNE14",
    "tipo": "PREVIA_SINTETICA_NAO_OFICIAL",
    "metodo": "TSNE",
    "dimensao": 1,
    "n_sintetico": 1200,
    "n_features": 12,
    "perplexity": 10,
    "initialization": "pca",
    "metric": "euclidean",
    "n_iter": 300,
    "kl_final": kl,
    "kl_disponivel": kl is not None,
    "tempo_reducao_segundos": tempo,
    "historico_kl_por_iteracao_disponivel": False,
    "renderizador": "src.visualizacao_tsne_pdf.diagnostico_kl",
    "treinamento_tsne_sintetico": True,
    "treinamento_tsne_dados_reais": False,
    "oof_real_executado": False,
    "f10_real_acessado": False,
}

(SAIDA / "manifesto_previa.json").write_text(
    json.dumps(
        manifesto,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

# 08 - GARANTIR PRESERVACAO DO PROJETO
depois = {
    str(p): hash_arquivo(p)
    for p in protegidos
}

assert antes == depois, "Arquivo oficial alterado."

# 09 - RESULTADOS
print()
print("=== V2 / TSNE 1D / TSNE14 ===")
print("RENDERIZADOR_OFICIAL=SIM")
print("DIMENSAO=1D")
print("N_SINTETICO=1200")
print("PERPLEXITY=10")
print("INICIALIZACAO=pca")

if kl is not None:
    print(f"KL_FINAL={kl:.7f}")
else:
    print("KL_FINAL=INDISPONIVEL")

print(f"TEMPO_REDUCAO={tempo:.7f}s")
print("HISTORICO_KL_INVENTADO=NAO")
print("TREINAMENTO_SOMENTE_SINTETICO=SIM")
print("DADOS_REAIS_UTILIZADOS=NAO")
print("OOF_REAL_EXECUTADO=NAO")
print("F10_REAL_ACESSADO=NAO")
print("ARQUIVOS_OFICIAIS_ALTERADOS=NAO")
print(f"PNG_PARA_AVALIAR={PNG}")
print(f"CSV_PARA_AVALIAR={CSV}")
print("V2_TSNE14_VALIDACAO=AGUARDANDO_APROVACAO_VISUAL")