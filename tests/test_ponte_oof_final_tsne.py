"""Parte 5: contrato de leitura OOF concluido, 100% sintetico, SEM treino/F10 real."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ponte_oof_final_tsne import verificar_oof_concluido_tsne
from src.validacao_oof import grade_candidatos_oof


def canon(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@pytest.fixture
def pasta(tmp_path):
    raiz = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((raiz / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").read_text(encoding="utf-8-sig"))
    cfg = deepcopy(cfg)
    cfg["reducao"]["metodo"] = "TSNE"
    cfg["tsne"]["n_components"] = 3
    cfg["tsne"]["initialization"] = ["random"]
    cfg["dados"]["fracao_base"] = 0.1
    cfg["execucao"]["n_jobs"] = 1
    cfg["gmm"]["n_components"] = [4]
    cfg["gmm"]["covariance_type"] = "full"
    cfg["gmm"]["n_init"] = 5
    cfg["amostragem"]["modo"] = "RUS"
    cfg["amostragem"]["razao_negativos_positivos"] = 16
    escolha_candidato = next(grade_candidatos_oof(cfg))
    cid = sha256(canon(escolha_candidato).encode("utf-8")).hexdigest()[:14]
    escolha = {"id_candidato": cid, "configuracao": escolha_candidato,
               "limiar_mcc_oof": 0.3, "status": "selecionado_em_ensaio",
               "teste_final_usado": False}
    metricas = {"id_candidato": cid, "configuracao": escolha_candidato,
                "limiar_mcc": 0.3, "n": 36}
    manifesto = {"status": "concluido_06_03", "versao_layout": 2,
                 "tipo": "ensaio", "chave_execucao": "HASH_EXECUCAO_SINTETICA",
                 "candidatos_concluidos": [cid], "n_desenvolvimento_trabalhado": 36,
                 "sha256_parquet": "SINTETICO_PARQUET", "sha256_folds": "SINTETICO_FOLDS",
                 "sha256_yaml": "SINTETICO_YAML"}
    m = tmp_path / "vencedor" / "metricas"
    m.mkdir(parents=True)
    o = tmp_path / "vencedor" / "oof"
    o.mkdir()
    (tmp_path / "configuracao.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (tmp_path / "manifesto.json").write_text(json.dumps(manifesto), encoding="utf-8")
    (m / "melhor_configuracao.json").write_text(json.dumps(escolha), encoding="utf-8")
    (m / "metricas.json").write_text(json.dumps(metricas), encoding="utf-8")
    np.savez_compressed(o / "predicoes.npz", linha_original=np.arange(36, dtype=np.int64),
                        fold=np.repeat(np.arange(1, 10, dtype=np.uint8), 4),
                        y_true=np.tile([0, 0, 0, 1], 9).astype(np.uint8),
                        prob_fraude=np.linspace(0.01, 0.98, 36),
                        chave_execucao=np.array(manifesto["chave_execucao"]),
                        id_candidato=np.array(cid), fold_esperado=np.array(0, dtype=np.uint8))
    return tmp_path


def arquivo_json(pasta, rel, f):
    p = pasta / rel
    obj = json.loads(p.read_text(encoding="utf-8"))
    f(obj)
    p.write_text(json.dumps(obj), encoding="utf-8")


def arquivo_npz(pasta, f):
    p = pasta / "vencedor" / "oof" / "predicoes.npz"
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    f(d)
    np.savez_compressed(p, **d)


def test_aceita_oof_concluido_sintetico(pasta):
    pre = verificar_oof_concluido_tsne(pasta)
    assert pre.plano.dimensao_tsne == 3
    assert pre.plano.acessar_f10 is False
    assert pre.plano.executar_treinamento is False
    assert pre.plano.limiar_mcc_oof_congelado == 0.3
    assert pre.n_predicoes_oof == 36
    assert pre.proveniencia["sha256_parquet"] == "SINTETICO_PARQUET"
    assert pre.escolha_oof["teste_final_usado"] is False


def test_nenhum_arquivo_criado(pasta):
    antes = sorted(x.relative_to(pasta) for x in pasta.rglob("*") if x.is_file())
    verificar_oof_concluido_tsne(pasta)
    depois = sorted(x.relative_to(pasta) for x in pasta.rglob("*") if x.is_file())
    assert antes == depois


def test_rejeita_pasta_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        verificar_oof_concluido_tsne(tmp_path / "nao_existe")


def test_rejeita_oof_incompleto(pasta):
    arquivo_json(pasta, "manifesto.json", lambda a: a.update(status="em_andamento"))
    with pytest.raises(ValueError, match="nao concluido"):
        verificar_oof_concluido_tsne(pasta)


def test_rejeita_layout_antigo(pasta):
    arquivo_json(pasta, "manifesto.json", lambda a: a.update(versao_layout=1))
    with pytest.raises(ValueError, match="layout 2"):
        verificar_oof_concluido_tsne(pasta)


def test_rejeita_pca(pasta):
    p = pasta / "configuracao.yaml"
    c = yaml.safe_load(p.read_text(encoding="utf-8"))
    c["reducao"]["metodo"] = "PCA"
    p.write_text(yaml.safe_dump(c), encoding="utf-8")
    with pytest.raises(ValueError, match="PCA proibido"):
        verificar_oof_concluido_tsne(pasta)


def test_rejeita_vencedor_nao_concluido(pasta):
    arquivo_json(pasta, "manifesto.json", lambda a: a.update(candidatos_concluidos=[]))
    with pytest.raises(ValueError, match="candidatos OOF"):
        verificar_oof_concluido_tsne(pasta)


def test_rejeita_id_falso(pasta):
    arquivo_json(pasta, "vencedor/metricas/melhor_configuracao.json", lambda a: a.update(id_candidato="falso"))
    with pytest.raises(ValueError, match="ID do vencedor"):
        verificar_oof_concluido_tsne(pasta)


def test_rejeita_limite_alterado(pasta):
    arquivo_json(pasta, "vencedor/metricas/metricas.json", lambda a: a.update(limiar_mcc=0.7))
    with pytest.raises(ValueError, match="Limiar MCC"):
        verificar_oof_concluido_tsne(pasta)


def test_rejeita_scores_nao_finitos(pasta):
    def f(d):
        d["prob_fraude"][0] = np.nan
    arquivo_npz(pasta, f)
    with pytest.raises(ValueError, match="probabilidades OOF invalidos"):
        verificar_oof_concluido_tsne(pasta)


def test_rejeita_f10_sintetico(pasta):
    def f(d):
        d["fold"][0] = 10
    arquivo_npz(pasta, f)
    with pytest.raises(ValueError, match="F10 proibido"):
        verificar_oof_concluido_tsne(pasta)


def test_rejeita_cache_outro_candidato(pasta):
    def f(d):
        d["id_candidato"] = np.array("outro")
    arquivo_npz(pasta, f)
    with pytest.raises(ValueError, match="Cache do vencedor"):
        verificar_oof_concluido_tsne(pasta)


def test_rejeita_contagem_predicoes(pasta):
    arquivo_json(pasta, "vencedor/metricas/metricas.json", lambda a: a.update(n=34))
    with pytest.raises(ValueError, match="Quantidade de previsoes"):
        verificar_oof_concluido_tsne(pasta)


def test_rejeita_oof_amostral_marcado_oficial(pasta):
    arquivo_json(pasta, "manifesto.json", lambda a: a.update(tipo="oficial"))
    with pytest.raises(ValueError, match="amostral"):
        verificar_oof_concluido_tsne(pasta)


def test_rejeita_linhas_duplicadas(pasta):
    def f(d):
        d["linha_original"][2] = d["linha_original"][1]
    arquivo_npz(pasta, f)
    with pytest.raises(ValueError, match="Linhas e vetores"):
        verificar_oof_concluido_tsne(pasta)


def test_rejeita_hash_ausente(pasta):
    arquivo_json(pasta, "manifesto.json", lambda a: a.update(sha256_yaml=""))
    with pytest.raises(ValueError, match="sha256_yaml"):
        verificar_oof_concluido_tsne(pasta)
