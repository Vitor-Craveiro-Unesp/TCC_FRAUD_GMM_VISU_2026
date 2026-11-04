"""Parte 3: roteamento TSNE puro. Nao realiza treinamento nem abre F10."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.roteamento_final_tsne import planejar_finalizacao_tsne
from src.validacao_oof import grade_candidatos_oof


@pytest.fixture
def caso():
    # YAML somente leitura; qualquer alteracao e feita na copia em memoria.
    raiz = Path(__file__).resolve().parents[1]
    with (raiz / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").open(encoding="utf-8-sig") as arq:
        config = yaml.safe_load(arq)
    config = deepcopy(config)
    config["reducao"]["metodo"] = "TSNE"
    config["tsne"]["n_components"] = 2
    config["tsne"]["initialization"] = ["random"]
    c = next(grade_candidatos_oof(config))
    escolha = {
        "id_candidato": "tsne_sintetico",
        "configuracao": c,
        "limiar_mcc_oof": 0.3,
        "status": "selecionado_em_ensaio",
        "teste_final_usado": False,
    }
    return config, escolha


@pytest.mark.parametrize("dim", (1, 2, 3))
def test_dimensoes_roteadas_sem_execucao(caso, dim):
    cfg, escolha = caso
    cfg["tsne"]["n_components"] = dim
    escolha["configuracao"] = next(grade_candidatos_oof(cfg))
    cfg_antes, escolha_antes = deepcopy(cfg), deepcopy(escolha)
    plano = planejar_finalizacao_tsne(cfg, escolha)
    assert plano.metodo == "TSNE"
    assert plano.dimensao_tsne == dim
    assert plano.folds_treino == tuple(range(1, 10))
    assert plano.fold_teste_reservado == 10
    assert plano.modulo_treinamento == "src.treinamento_final_tsne"
    assert plano.modulo_avaliacao == "src.avaliacao_final_tsne"
    assert plano.executar_treinamento is False and plano.acessar_f10 is False
    assert plano.limiar_mcc_oof_congelado == 0.3
    assert cfg == cfg_antes and escolha == escolha_antes


def test_nao_aceita_pca(caso):
    cfg, escolha = caso
    cfg["reducao"]["metodo"] = "PCA"
    with pytest.raises(ValueError, match="exclusivo TSNE"):
        planejar_finalizacao_tsne(cfg, escolha)


def test_nao_aceita_candidato_pca(caso):
    cfg, escolha = caso
    escolha["configuracao"]["metodo"] = "PCA"
    with pytest.raises(ValueError, match="Candidato deve ser TSNE"):
        planejar_finalizacao_tsne(cfg, escolha)


def test_nao_aceita_f10_usado(caso):
    cfg, escolha = caso
    escolha["teste_final_usado"] = True
    with pytest.raises(ValueError, match="F10"):
        planejar_finalizacao_tsne(cfg, escolha)


def test_nao_aceita_status_invalido(caso):
    cfg, escolha = caso
    escolha["status"] = "em_andamento"
    with pytest.raises(ValueError, match="status valido"):
        planejar_finalizacao_tsne(cfg, escolha)


@pytest.mark.parametrize("tau", (float('nan'), float('inf'), -0.1, 1.1, True, None))
def test_rejeita_limiar_invalido(caso, tau):
    cfg, escolha = caso
    escolha["limiar_mcc_oof"] = tau
    with pytest.raises(ValueError, match="Limiar"):
        planejar_finalizacao_tsne(cfg, escolha)


def test_rejeita_candidato_fora_grade(caso):
    cfg, escolha = caso
    escolha["configuracao"]["reducao"]["perplexity"] = 99
    with pytest.raises(ValueError, match="grade TSNE"):
        planejar_finalizacao_tsne(cfg, escolha)


def test_aceita_k_dinamico_e_rejeita_vencedor_gmm_fora_grade(caso):
    cfg, escolha = caso
    cfg["gmm"]["n_components"] = [3]
    escolha["configuracao"] = next(grade_candidatos_oof(cfg))
    plano = planejar_finalizacao_tsne(cfg, escolha)
    assert plano.id_candidato_oof == escolha["id_candidato"]
    assert escolha["configuracao"]["gmm"]["n_components"] == 3
    # Um vencedor de K diferente do YAML congelado continua proibido.
    escolha["configuracao"] = deepcopy(escolha["configuracao"])
    escolha["configuracao"]["gmm"]["n_components"] = 4
    with pytest.raises(ValueError, match="grade TSNE"):
        planejar_finalizacao_tsne(cfg, escolha)


def test_aceita_rus_dinamico_e_rejeita_vencedor_rus_fora_grade(caso):
    cfg, escolha = caso
    cfg["amostragem"]["razao_negativos_positivos"] = 15
    escolha["configuracao"] = next(grade_candidatos_oof(cfg))
    plano = planejar_finalizacao_tsne(cfg, escolha)
    assert plano.fold_teste_reservado == 10
    assert escolha["configuracao"]["razao"] == 15
    # Alterar apenas o candidato depois da selecao torna-o incompativel.
    escolha["configuracao"] = deepcopy(escolha["configuracao"])
    escolha["configuracao"]["razao"] = 16
    with pytest.raises(ValueError, match="grade TSNE"):
        planejar_finalizacao_tsne(cfg, escolha)


def test_rejeita_divisao_sem_f10_reservado(caso):
    cfg, escolha = caso
    cfg["divisao"]["qual_fold_teste"] = 9
    with pytest.raises(ValueError, match="F10"):
        planejar_finalizacao_tsne(cfg, escolha)


def test_pca_automatico_nao_muda_rota_tsne(caso):
    cfg, escolha = caso
    cfg["execucao"]["finalizacao_pca1d_automatica"] = True
    plano = planejar_finalizacao_tsne(cfg, escolha)
    assert plano.modulo_treinamento.endswith("_tsne")
    assert plano.modulo_avaliacao.endswith("_tsne")
