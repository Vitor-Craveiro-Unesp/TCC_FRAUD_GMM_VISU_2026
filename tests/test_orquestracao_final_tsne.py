"""Parte 6: orquestracao separada; OOF inteiramente sintetico F1-F9.

Nao le dados reais, Parquet, manifestos reais nem F10. Um ajuste minimo
real exercita SOMENTE os dados sinteticos do fixture.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import src.orquestracao_final_tsne as modulo
from src.validacao_oof import grade_candidatos_oof


def _canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@pytest.fixture
def caso(tmp_path):
    raiz = Path(__file__).resolve().parents[1]
    cfg = deepcopy(yaml.safe_load(
        (raiz / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").read_text(encoding="utf-8-sig")
    ))
    cfg["reducao"]["metodo"] = "TSNE"
    cfg["tsne"]["n_components"] = 1
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
    candidato = next(grade_candidatos_oof(cfg))
    cid = sha256(_canon(candidato).encode("utf-8")).hexdigest()[:14]

    rng = np.random.default_rng(42)
    folds = np.repeat(np.arange(1, 10, dtype=np.uint8), 40)
    n = len(folds)
    y = np.zeros(n, dtype=np.uint8)
    for k in range(9):
        y[k * 40:k * 40 + 2] = 1
    ruido = rng.normal(size=n)
    df = pd.DataFrame({
        "TARGET_OMEGA": y,
        "NID_ALPHA": np.arange(n, dtype=np.int64) + 500,
        "TRANS_VALUE": ruido + 0.45 * y,
        "TRANS_NUM_CARD_FEWF": rng.choice(["R", "S", "T"], n),
        "SEND_GENDER_BE": rng.choice(["F", "M"], n),
        "TRANS_MONTH_SIN": np.sin(np.arange(n) * 2 * np.pi / 12),
        "TRANS_MONTH_COS": np.cos(np.arange(n) * 2 * np.pi / 12),
        "VALOR_CONTINUO": rng.normal(size=n) + 0.5 * y,
    })
    linhas = np.arange(n, dtype=np.int64) + 12000
    escolha = {
        "id_candidato": cid, "configuracao": candidato,
        "limiar_mcc_oof": 0.3, "status": "selecionado_em_ensaio",
        "teste_final_usado": False,
    }
    manifesto = {
        "status": "concluido_06_03", "versao_layout": 2,
        "tipo": "ensaio", "chave_execucao": "CHAVE_OOF_SINTETICA",
        "candidatos_concluidos": [cid], "n_desenvolvimento_trabalhado": n,
        "sha256_parquet": "HASH_FICTICIO_SEM_PARQUET",
        "sha256_folds": "HASH_FICTICIO_SEM_FOLDS_REAIS",
        "sha256_yaml": "HASH_FICTICIO_CONFIG_EM_MEMORIA",
    }
    metricas = {
        "id_candidato": cid, "configuracao": candidato,
        "limiar_mcc": 0.3, "n": n,
    }
    exp = tmp_path / "OOF_TSNE_SINTETICO"
    (exp / "vencedor" / "metricas").mkdir(parents=True)
    (exp / "vencedor" / "oof").mkdir()
    (exp / "configuracao.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    (exp / "manifesto.json").write_text(json.dumps(manifesto), encoding="utf-8")
    (exp / "vencedor" / "metricas" / "melhor_configuracao.json").write_text(
        json.dumps(escolha), encoding="utf-8"
    )
    (exp / "vencedor" / "metricas" / "metricas.json").write_text(
        json.dumps(metricas), encoding="utf-8"
    )
    np.savez_compressed(
        exp / "vencedor" / "oof" / "predicoes.npz",
        linha_original=linhas, fold=folds, y_true=y,
        prob_fraude=rng.uniform(0, 1, n),
        chave_execucao=np.array(manifesto["chave_execucao"]),
        id_candidato=np.array(cid), fold_esperado=np.array(0, dtype=np.uint8),
    )
    return exp, df, folds, linhas


def _arquivos(exp):
    return {str(x.relative_to(exp)): sha256(x.read_bytes()).hexdigest()
            for x in exp.rglob("*") if x.is_file()}


def test_planejamento_apenas_tsne(caso):
    exp, _, _, _ = caso
    p = modulo.planejar_orquestracao_final_tsne(exp)
    assert p.plano.metodo == "TSNE" and p.plano.dimensao_tsne == 1
    assert p.plano.executar_treinamento is False
    assert p.plano.acessar_f10 is False
    assert p.plano.fold_teste_reservado == 10
    assert p.n_predicoes_oof == 360


def test_planejamento_sem_modificar_artefatos(caso):
    exp, _, _, _ = caso
    antes = _arquivos(exp)
    modulo.planejar_orquestracao_final_tsne(exp)
    assert _arquivos(exp) == antes


def test_cli_somente_planejar(caso, capsys):
    exp, _, _, _ = caso
    assert modulo.main(["--planejar", "--experimento", str(exp)]) == 0
    saida = capsys.readouterr().out
    assert "treinamento_executado=NAO" in saida
    assert "F10=RESERVADO" in saida
    assert "dimensao=1D" in saida


def test_cli_nao_aceita_execucao(caso):
    exp, _, _, _ = caso
    with pytest.raises(SystemExit) as erro:
        modulo.main(["--executar", "--experimento", str(exp)])
    assert erro.value.code == 2


def test_cli_exige_experimento(caso):
    with pytest.raises(SystemExit) as erro:
        modulo.main(["--planejar"])
    assert erro.value.code == 2


def test_cli_nao_aceita_acesso_f10(caso):
    exp, _, _, _ = caso
    with pytest.raises(SystemExit) as erro:
        modulo.main(["--planejar", "--experimento", str(exp), "--f10"])
    assert erro.value.code == 2


def test_rejeita_pca(caso):
    exp, _, _, _ = caso
    p = exp / "configuracao.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["reducao"]["metodo"] = "PCA"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="PCA proibido"):
        modulo.planejar_orquestracao_final_tsne(exp)


def test_rejeita_pasta_ausente(caso):
    exp, _, _, _ = caso
    with pytest.raises(FileNotFoundError):
        modulo.planejar_orquestracao_final_tsne(exp / "inexistente")


def test_sem_autorizacao_nega_antes_de_qualquer_io(monkeypatch, caso):
    exp, df, folds, linhas = caso
    def proibido(*_args, **_kwargs):
        raise AssertionError("Sem autorizacao nao pode sequer ler o OOF")
    monkeypatch.setattr(modulo, "validar_entrada_f1_f9_contra_oof", proibido)
    with pytest.raises(PermissionError, match="autorizar_ajuste=True"):
        modulo.executar_ajuste_final_tsne_pos_oof_em_memoria(exp, df, folds, linhas)


def test_autorizacao_precisa_ser_booleano_true(caso):
    exp, df, folds, linhas = caso
    with pytest.raises(PermissionError):
        modulo.executar_ajuste_final_tsne_pos_oof_em_memoria(
            exp, df, folds, linhas, autorizar_ajuste=1
        )


def test_valida_entrada_congruente(caso):
    exp, df, folds, linhas = caso
    pre = modulo.validar_entrada_f1_f9_contra_oof(exp, df, folds, linhas)
    assert pre.n_predicoes_oof == len(df)


def test_f10_sintetico_bloqueado(caso):
    exp, df, folds, linhas = caso
    f = folds.copy()
    f[-1] = 10
    with pytest.raises(ValueError, match="F10 proibido"):
        modulo.validar_entrada_f1_f9_contra_oof(exp, df, f, linhas)


def test_folds_divergentes_bloqueados(caso):
    exp, df, folds, linhas = caso
    f = folds.copy()
    f[0], f[40] = f[40], f[0]
    with pytest.raises(ValueError, match="Atribuicao de folds"):
        modulo.validar_entrada_f1_f9_contra_oof(exp, df, f, linhas)


def test_ids_originais_divergentes_bloqueados(caso):
    exp, df, folds, linhas = caso
    ids = linhas.copy()
    ids[0] -= 1
    with pytest.raises(ValueError, match="Linhas originais divergentes"):
        modulo.validar_entrada_f1_f9_contra_oof(exp, df, folds, ids)


def test_ids_repetidos_bloqueados(caso):
    exp, df, folds, linhas = caso
    ids = linhas.copy()
    ids[2] = ids[1]
    with pytest.raises(ValueError, match="unicas"):
        modulo.validar_entrada_f1_f9_contra_oof(exp, df, folds, ids)


def test_rotulos_divergentes_bloqueados(caso):
    exp, df, folds, linhas = caso
    d = df.copy(deep=True)
    d.loc[0, "TARGET_OMEGA"] ^= 1
    with pytest.raises(ValueError, match="Rotulos divergentes"):
        modulo.validar_entrada_f1_f9_contra_oof(exp, d, folds, linhas)


def test_alvo_ausente_bloqueado(caso):
    exp, df, folds, linhas = caso
    with pytest.raises(ValueError, match="Coluna alvo"):
        modulo.validar_entrada_f1_f9_contra_oof(exp, df.drop(columns=["TARGET_OMEGA"]), folds, linhas)


def test_dataframe_reordenado_bloqueado(caso):
    exp, df, folds, linhas = caso
    d = df.iloc[::-1].copy()
    with pytest.raises(ValueError, match="RangeIndex"):
        modulo.validar_entrada_f1_f9_contra_oof(exp, d, folds, linhas)


def test_quantidade_linhas_divergente_bloqueada(caso):
    exp, df, folds, linhas = caso
    with pytest.raises(ValueError, match="Quantidade de linhas"):
        modulo.validar_entrada_f1_f9_contra_oof(exp, df.iloc[:-1].copy(), folds[:-1], linhas[:-1])


def test_folds_float_bloqueados(caso):
    exp, df, folds, linhas = caso
    with pytest.raises(ValueError, match="SOMENTE F1-F9"):
        modulo.validar_entrada_f1_f9_contra_oof(exp, df, folds.astype(float), linhas)


def test_integracao_rota_sem_ajuste_mockado(monkeypatch, caso):
    exp, df, folds, linhas = caso
    chamado = []
    sentinela = object()
    def falso_executor(df_recebido, folds_recebidos, config, escolha, origem, *, registrar_historico_em):
        chamado.append((df_recebido, folds_recebidos, config, escolha, origem, registrar_historico_em))
        return sentinela
    monkeypatch.setattr(modulo, "executar_ajuste_final_tsne_em_memoria", falso_executor)
    out = modulo.executar_ajuste_final_tsne_pos_oof_em_memoria(
        exp, df, folds, linhas, autorizar_ajuste=True
    )
    assert out is sentinela and len(chamado) == 1
    assert chamado[0][0] is df and chamado[0][1] is folds
    assert chamado[0][3]["limiar_mcc_oof"] == 0.3
    assert chamado[0][3]["teste_final_usado"] is False
    assert chamado[0][5] is False


def test_ajuste_sintetico_integrado_sem_persistencia(caso):
    exp, df, folds, linhas = caso
    antes = _arquivos(exp)
    d_antes = df.copy(deep=True)
    f_antes = folds.copy()
    resultado = modulo.executar_ajuste_final_tsne_pos_oof_em_memoria(
        exp, df, folds, linhas, autorizar_ajuste=True, registrar_historico_em=False
    )
    assert resultado.pacote.configuracao["reducao"]["metodo"] == "TSNE"
    assert resultado.pacote.n_treino_original == len(df)
    assert resultado.pacote.risco.n_treino_ == len(df)
    assert resultado.pacote.limiar_oof == 0.3
    assert resultado.plano.acessar_f10 is False
    entradas = df.drop(columns=["TARGET_OMEGA"]).iloc[:15].copy()
    entradas.index = pd.Index(range(9000, 9015))
    p = resultado.pacote.prever_sem_rotulos(entradas)
    assert p.index.equals(entradas.index)
    assert np.isfinite(p.to_numpy()).all() and p.between(0, 1).all()
    assert _arquivos(exp) == antes
    pd.testing.assert_frame_equal(df, d_antes)
    np.testing.assert_array_equal(folds, f_antes)
