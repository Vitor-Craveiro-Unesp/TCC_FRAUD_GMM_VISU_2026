"""Parte 7/7 — homologacao integrada e regressao TSNE + GMM.

Somente dados sinteticos; nunca le Parquet, folds reais ou F10 real.
Realiza tres pequenos ajustes, um por dimensao, autorizados explicitamente
dentro dos fixtures de teste. Nao integra CLI de producao nem executa OOF.
Todos os artefatos sao criados no tmp_path do pytest.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from src.validacao_oof import grade_candidatos_oof
from src.orquestracao_final_tsne import (
    planejar_orquestracao_final_tsne,
    validar_entrada_f1_f9_contra_oof,
    executar_ajuste_final_tsne_pos_oof_em_memoria,
)
from src.ponte_oof_final_tsne import verificar_oof_concluido_tsne
from src.treinamento_final_tsne import (
    PacoteFinalTSNE,
    salvar_pacote,
    carregar_pacote_confiavel,
)
from src.avaliacao_final_tsne import avaliar_f10_em_memoria

ROOT = Path(__file__).resolve().parents[1]
PROTEGIDOS = [
    "config_TCC_EXPERIMENTAL_V3_atualizado.yaml",
    "src/reducao.py",
    "src/treinamento_final_pca.py",
    "src/avaliacao_final_pca.py",
    "src/orquestracao_oof.py",
]


def _sha(caminho: Path) -> str:
    return sha256(caminho.read_bytes()).hexdigest()


def _hash_arvore(caminho: Path) -> dict:
    return {
        str(p.relative_to(caminho)): _sha(p)
        for p in caminho.rglob("*") if p.is_file()
    }


def _canon(obj) -> str:
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False
    )


@pytest.fixture(scope="module", autouse=True)
def preservar_pca_e_yaml():
    """Falhar se este modulo de testes alterar arquivos homologados."""
    antes = {nome: _sha(ROOT / nome) for nome in PROTEGIDOS}
    yield
    depois = {nome: _sha(ROOT / nome) for nome in PROTEGIDOS}
    assert depois == antes, "Regressao: YAML ou arquivos PCA/orquestrador foram alterados."


@pytest.fixture(scope="module", params=(1, 2, 3), ids=("1D", "2D", "3D"))
def caso(request, tmp_path_factory):
    dim = request.param
    cfg = deepcopy(yaml.safe_load(
        (ROOT / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").read_text(
            encoding="utf-8-sig"
        )
    ))
    cfg["reducao"]["metodo"] = "TSNE"
    cfg["tsne"]["n_components"] = dim
    cfg["tsne"]["initialization"] = ["random"]
    cfg["tsne"]["perplexity"] = [10]
    cfg["dados"]["fracao_base"] = 0.1
    cfg["execucao"]["n_jobs"] = 1
    cfg["gmm"]["n_components"] = [4]
    cfg["gmm"]["covariance_type"] = "full"
    cfg["gmm"]["n_init"] = 5
    cfg["amostragem"]["modo"] = "RUS"
    cfg["amostragem"]["razao_negativos_positivos"] = 16
    cfg["risco_historico"]["suavizacao"] = 0.0

    grade = list(grade_candidatos_oof(cfg))
    assert len(grade) == 1
    candidato = grade[0]
    cid = sha256(_canon(candidato).encode("utf-8")).hexdigest()[:14]

    rng = np.random.default_rng(42)
    folds = np.repeat(np.arange(1, 10, dtype=np.uint8), 40)
    n = len(folds)
    y = np.zeros(n, dtype=np.uint8)
    for k in range(9):
        y[k * 40:k * 40 + 2] = 1

    v = rng.normal(size=n)
    df = pd.DataFrame({
        "TARGET_OMEGA": y,
        "NID_ALPHA": np.arange(n, dtype=np.int64) + 500,
        "TRANS_VALUE": v + 0.45 * y,
        "TRANS_NUM_CARD_FEWF": rng.choice(["R", "S", "T"], size=n),
        "SEND_GENDER_BE": rng.choice(["F", "M"], size=n),
        "TRANS_MONTH_SIN": np.sin(np.arange(n) * 2 * np.pi / 12),
        "TRANS_MONTH_COS": np.cos(np.arange(n) * 2 * np.pi / 12),
        "VALOR_CONTINUO": rng.normal(size=n) + 0.5 * y,
    })
    dados_copia = df.copy(deep=True)
    folds_copia = folds.copy()
    config_copia = deepcopy(cfg)
    linhas_originais = np.arange(n, dtype=np.int64) + 12000
    escolha = {
        "id_candidato": cid,
        "configuracao": candidato,
        "limiar_mcc_oof": 0.3,
        "status": "selecionado_em_ensaio",
        "teste_final_usado": False,
    }
    manifesto = {
        "status": "concluido_06_03",
        "versao_layout": 2,
        "tipo": "ensaio",
        "chave_execucao": "OOF_SINTETICO_PARTE7",
        "candidatos_concluidos": [cid],
        "n_desenvolvimento_trabalhado": n,
        "sha256_parquet": "HASH_FICTICIO_SEM_PARQUET",
        "sha256_folds": "HASH_FICTICIO_SEM_FOLDS",
        "sha256_yaml": "HASH_FICTICIO_CONFIG",
    }
    metricas = {
        "id_candidato": cid,
        "configuracao": candidato,
        "limiar_mcc": 0.3,
        "n": n,
    }
    exp = tmp_path_factory.mktemp(f"oof_tsne_sintetico_{dim}d")
    (exp / "vencedor" / "metricas").mkdir(parents=True)
    (exp / "vencedor" / "oof").mkdir()
    (exp / "configuracao.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    (exp / "manifesto.json").write_text(
        json.dumps(manifesto), encoding="utf-8"
    )
    (exp / "vencedor" / "metricas" / "melhor_configuracao.json").write_text(
        json.dumps(escolha), encoding="utf-8"
    )
    (exp / "vencedor" / "metricas" / "metricas.json").write_text(
        json.dumps(metricas), encoding="utf-8"
    )
    np.savez_compressed(
        exp / "vencedor" / "oof" / "predicoes.npz",
        linha_original=linhas_originais,
        fold=folds,
        y_true=y,
        prob_fraude=rng.uniform(0, 1, n),
        chave_execucao=np.array(manifesto["chave_execucao"]),
        id_candidato=np.array(cid),
        fold_esperado=np.array(0, dtype=np.uint8),
    )

    # Ajuste real pequeno APENAS nos dados sinteticos e com autorizacao explicita.
    antes_oof = _hash_arvore(exp)
    item = executar_ajuste_final_tsne_pos_oof_em_memoria(
        exp, df, folds, linhas_originais,
        autorizar_ajuste=True,
        registrar_historico_em=False,
    )
    assert _hash_arvore(exp) == antes_oof
    pd.testing.assert_frame_equal(df, dados_copia)
    np.testing.assert_array_equal(folds, folds_copia)
    assert cfg == config_copia

    return {
        "dim": dim, "exp": exp, "df": df,
        "folds": folds, "ids": linhas_originais,
        "cfg": cfg, "escolha": escolha,
        "pacote": item.pacote, "item": item,
        "hash_oof": antes_oof,
    }


def test_01_planejamento_congelado(caso):
    p = planejar_orquestracao_final_tsne(caso["exp"])
    assert p.plano.metodo == "TSNE"
    assert p.plano.dimensao_tsne == caso["dim"]
    assert p.plano.executar_treinamento is False
    assert p.plano.acessar_f10 is False
    assert p.plano.fold_teste_reservado == 10
    assert p.plano.folds_treino == tuple(range(1, 10))
    assert p.plano.limiar_mcc_oof_congelado == 0.3
    assert p.n_predicoes_oof == len(caso["df"])


def test_02_ponte_oof_e_alinhamento(caso):
    p = verificar_oof_concluido_tsne(caso["exp"])
    assert p.escolha_oof["id_candidato"] == caso["escolha"]["id_candidato"]
    assert p.configuracao["reducao"]["metodo"] == "TSNE"
    validado = validar_entrada_f1_f9_contra_oof(
        caso["exp"], caso["df"], caso["folds"], caso["ids"]
    )
    assert validado.n_predicoes_oof == len(caso["df"])


def test_03_pacote_e_risco_historico(caso):
    pacote = caso["pacote"]
    assert isinstance(pacote, PacoteFinalTSNE)
    assert pacote.configuracao["reducao"]["metodo"] == "TSNE"
    assert pacote.n_treino_original == len(caso["df"])
    assert pacote.n_ajuste_gmm == 18 + 16 * 18
    assert pacote.risco.n_treino_ == len(caso["df"])
    assert pacote.limiar_oof == 0.3
    assert pacote.escolha_oof["teste_final_usado"] is False
    assert pacote.tsne.modelo_.negative_gradient_method == (
        "bh" if caso["dim"] == 3 else "auto"
    )


def test_04_geometria_pos_oos(caso):
    pacote = caso["pacote"]
    z_anterior = np.asarray(pacote.tsne.embedding_).copy()
    cache = pacote.tsne.transform_treino().copy()
    x = caso["df"].drop(columns=["TARGET_OMEGA"]).iloc[:19].copy()
    x.index = pd.Index(range(9000, 9019))
    s1 = pacote.prever_sem_rotulos(x)
    s2 = pacote.prever_sem_rotulos(x)
    np.testing.assert_allclose(s1, s2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(pacote.tsne.embedding_), z_anterior, rtol=0, atol=1e-6
    )
    pd.testing.assert_frame_equal(pacote.tsne.transform_treino(), cache)
    assert s1.index.equals(x.index) and s1.between(0, 1).all()
    assert np.isfinite(s1.to_numpy()).all()


def test_05_inferencia_sem_rotulos(caso):
    pacote = caso["pacote"]
    entrada = caso["df"].drop(columns=["TARGET_OMEGA"]).iloc[:10].copy()
    entrada.index = pd.Index(range(50000, 50010))
    scores = pacote.prever_sem_rotulos(entrada)
    assert scores.index.equals(entrada.index)
    assert scores.between(0, 1).all()
    with pytest.raises(ValueError, match="Remova o alvo"):
        pacote.prever_sem_rotulos(caso["df"].iloc[:10])


def test_06_manifesto_serializacao_e_recarrega(caso, tmp_path):
    pacote = caso["pacote"]
    entrada = caso["df"].drop(columns=["TARGET_OMEGA"]).iloc[:17].copy()
    esperado = pacote.prever_sem_rotulos(entrada)
    destino = tmp_path / "modelo_tsne"
    modelo, manifesto = salvar_pacote(pacote, destino)
    meta = json.loads(manifesto.read_text(encoding="utf-8"))
    assert meta["metodo"] == "TSNE"
    assert meta["dimensao_tsne"] == caso["dim"]
    assert meta["teste_f10_executado"] is False
    assert meta["limiar_mcc_oof_congelado"] == 0.3
    assert meta["n_treino_original"] == len(caso["df"])
    assert _sha(modelo) == meta["sha256_modelo_final"]
    recarregado = carregar_pacote_confiavel(destino)
    obtido = recarregado.prever_sem_rotulos(entrada)
    np.testing.assert_allclose(obtido, esperado, rtol=0, atol=1e-6)
    with pytest.raises(FileExistsError):
        salvar_pacote(pacote, destino)


def test_07_hash_detecta_corrupcao(caso, tmp_path):
    destino = tmp_path / "modelo_tamper"
    modelo, _ = salvar_pacote(caso["pacote"], destino)
    with modelo.open("ab") as stream:
        stream.write(b"MODIFICACAO_SINTETICA")
    with pytest.raises(RuntimeError, match="hash"):
        carregar_pacote_confiavel(destino)


def test_08_avaliacao_f10_inteiramente_sintetico(caso):
    """F10 eh APENAS identificador sintético; F10 real nunca eh carregado."""
    rng = np.random.default_rng(9200 + caso["dim"])
    n = 24
    y = np.zeros(n, dtype=np.uint8)
    y[:4] = 1
    base = pd.DataFrame({
        "TARGET_OMEGA": y,
        "NID_ALPHA": np.arange(n, dtype=np.int64) + 99000,
        "TRANS_VALUE": rng.normal(size=n) + 0.3 * y,
        "TRANS_NUM_CARD_FEWF": rng.choice(["R", "S", "T"], n),
        "SEND_GENDER_BE": rng.choice(["F", "M"], n),
        "TRANS_MONTH_SIN": np.sin(np.arange(n) * 2 * np.pi / 12),
        "TRANS_MONTH_COS": np.cos(np.arange(n) * 2 * np.pi / 12),
        "VALOR_CONTINUO": rng.normal(size=n),
    })
    dados_copia = base.copy(deep=True)
    tabela, resumo = avaliar_f10_em_memoria(
        caso["pacote"],
        base,
        np.full(n, 10, dtype=np.uint8),
        np.arange(n, dtype=np.int64) + 950000,
    )
    assert len(tabela) == n
    assert tabela["fold"].eq(10).all()
    assert tabela["prob_fraude"].between(0, 1).all()
    assert resumo["metodo"] == "TSNE"
    assert resumo["dimensao_tsne"] == caso["dim"]
    assert "dimensao_pca" not in resumo
    assert resumo["limiar_otimizado_no_f10"] is False
    assert resumo["limiar_mcc_oof_congelado"] == 0.3
    assert set(resumo["comparacao_limiares"]) == {"tau_0_50", "tau_mcc_oof"}
    assert np.array_equal(
        tabela["predicao_tau_0_50"].to_numpy(),
        (tabela["prob_fraude"].to_numpy() >= 0.5).astype(np.uint8),
    )
    assert np.array_equal(
        tabela["predicao_tau_mcc_oof"].to_numpy(),
        (tabela["prob_fraude"].to_numpy() >= 0.3).astype(np.uint8),
    )
    pd.testing.assert_frame_equal(base, dados_copia)


def test_09_f10_no_treino_bloqueado(caso, monkeypatch):
    from src import orquestracao_final_tsne as modulo
    folds = caso["folds"].copy()
    folds[-1] = 10

    def proibido(*args, **kwargs):
        raise AssertionError("Nenhum ajuste deveria iniciar com F10.")
    with monkeypatch.context() as m:
        m.setattr(modulo, "executar_ajuste_final_tsne_em_memoria", proibido)
        with pytest.raises(ValueError, match="F10 proibido"):
            modulo.executar_ajuste_final_tsne_pos_oof_em_memoria(
                caso["exp"], caso["df"], folds, caso["ids"],
                autorizar_ajuste=True,
            )


def test_10_sem_autorizacao_nao_ajusta(caso, monkeypatch):
    from src import orquestracao_final_tsne as modulo
    def proibido(*args, **kwargs):
        raise AssertionError("Nao deve ajustar sem autorizacao.")
    with monkeypatch.context() as m:
        m.setattr(modulo, "validar_entrada_f1_f9_contra_oof", proibido)
        with pytest.raises(PermissionError, match="autorizar_ajuste"):
            modulo.executar_ajuste_final_tsne_pos_oof_em_memoria(
                caso["exp"], caso["df"], caso["folds"], caso["ids"]
            )


def test_11_rotulos_adulterados_bloqueados(caso):
    alterado = caso["df"].copy()
    alterado.loc[0, "TARGET_OMEGA"] ^= 1
    with pytest.raises(ValueError, match="Rotulos divergentes"):
        validar_entrada_f1_f9_contra_oof(
            caso["exp"], alterado, caso["folds"], caso["ids"]
        )


def test_12_ids_adulterados_bloqueados(caso):
    ids = caso["ids"].copy()
    ids[0] -= 1
    with pytest.raises(ValueError, match="Linhas originais divergentes"):
        validar_entrada_f1_f9_contra_oof(
            caso["exp"], caso["df"], caso["folds"], ids
        )


def test_13_cli_planeja_sem_escrever(caso):
    antes = _hash_arvore(caso["exp"])
    cmd = [
        sys.executable, "-m", "src.orquestracao_final_tsne",
        "--planejar", "--experimento", str(caso["exp"])
    ]
    resultado = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "SOMENTE PLANEJAMENTO" in resultado.stdout
    assert "treinamento_executado=NAO" in resultado.stdout
    assert _hash_arvore(caso["exp"]) == antes


def test_14_cli_nao_aceita_executar(caso):
    cmd = [
        sys.executable, "-m", "src.orquestracao_final_tsne",
        "--planejar", "--executar", "--experimento", str(caso["exp"])
    ]
    resultado = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert resultado.returncode != 0
    assert "unrecognized arguments" in resultado.stderr


def test_15_oof_sintetico_integro(caso):
    assert _hash_arvore(caso["exp"]) == caso["hash_oof"]
