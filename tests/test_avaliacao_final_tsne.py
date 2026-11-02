"""Regressoes sintéticas para avaliacao TSNE; nenhum Parquet/F10 real.

Execucao isolada: .venv\\Scripts\\python.exe -m pytest tests/test_avaliacao_final_tsne.py -q
"""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from src.avaliacao_final_tsne import (
    avaliar_scores_com_limiar_oof_congelado,
    avaliar_f10_em_memoria,
)
from src.treinamento_final_tsne import ajustar_final_f1_f9
from src.validacao_oof import grade_candidatos_oof


def test_metricas_de_exemplo_sem_ajustar_limiar():
    y = pd.Series([0, 0, 1, 1], index=[4, 5, 6, 7])
    s = pd.Series([0.10, 0.40, 0.60, 0.90], index=y.index)
    m = avaliar_scores_com_limiar_oof_congelado(y, s, limiar_mcc_oof=0.3)
    assert m["limiar_otimizado_no_f10"] is False
    assert m["limiar_mcc_oof_congelado"] == 0.3
    assert m["comparacao_limiares"]["tau_0_50"]["matriz_confusao"]["contagens"] == [[2, 0], [0, 2]]
    assert m["comparacao_limiares"]["tau_mcc_oof"]["matriz_confusao"]["contagens"] == [[1, 1], [0, 2]]
    assert m["ap"] == pytest.approx(1.0)
    assert 0 <= m["score_composto_limiar_oof"] <= 1
    assert m["logloss"] >= 0


@pytest.mark.parametrize("tau", [float('nan'), float('inf'), -0.1, 1.1, True])
def test_rejeita_limiar_invalido(tau):
    with pytest.raises(ValueError, match="Limiar"):
        avaliar_scores_com_limiar_oof_congelado([0, 1], [0.2, 0.8], limiar_mcc_oof=tau)


def test_rejeita_indices_trocados():
    y = pd.Series([0, 1], index=[11, 12])
    s = pd.Series([0.1, 0.9], index=[12, 11])
    with pytest.raises(ValueError, match="Indices"):
        avaliar_scores_com_limiar_oof_congelado(y, s, limiar_mcc_oof=0.5)


@pytest.mark.parametrize("y,s", [([0, 0], [0.2, 0.5]), ([0, 1], [0.1, float('nan')]), ([0, 1], [0.1, 1.2])])
def test_rejeita_scores_ou_rotulos_invalidos(y, s):
    with pytest.raises(ValueError):
        avaliar_scores_com_limiar_oof_congelado(y, s, limiar_mcc_oof=0.5)


@pytest.fixture(scope="module")
def pacote_e_dados():
    raiz = Path(__file__).resolve().parents[1]
    cfg = deepcopy(yaml.safe_load((raiz / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf-8-sig')))
    cfg['reducao']['metodo'] = 'TSNE'
    cfg['dados']['fracao_base'] = 0.1
    cfg['tsne']['n_components'] = 2
    cfg['tsne']['initialization'] = ['random']
    cfg['tsne']['perplexity'] = [10]
    cfg['gmm']['n_components'] = [4]
    cfg['gmm']['covariance_type'] = 'full'
    cfg['gmm']['n_init'] = 5
    cfg['amostragem']['modo'] = 'RUS'
    cfg['amostragem']['razao_negativos_positivos'] = 16
    cfg['risco_historico']['suavizacao'] = 0.0
    cfg['execucao']['n_jobs'] = 1
    rng = np.random.default_rng(42)
    folds = np.repeat(np.arange(1, 10, dtype=np.uint8), 50)
    n = len(folds)
    y = np.zeros(n, dtype=np.uint8)
    for f in range(9):
        y[50*f:50*f+2] = 1
    v = rng.normal(size=n)
    df = pd.DataFrame({
        'TARGET_OMEGA': y,
        'NID_ALPHA': np.arange(n, dtype=np.int64),
        'TRANS_VALUE': v + y*.45,
        'TRANS_NUM_CARD_FEWF': rng.choice(['R','S','T'], n),
        'SEND_GENDER_BE': rng.choice(['F','M'],n),
        'TRANS_MONTH_SIN': np.sin(np.arange(n)*2*np.pi/12),
        'TRANS_MONTH_COS': np.cos(np.arange(n)*2*np.pi/12),
        'VALOR_CONTINUO': rng.normal(size=n),
    })
    c = next(grade_candidatos_oof(cfg))
    escolha = {'id_candidato': 'sintetico_2d', 'configuracao': c, 'limiar_mcc_oof': .3,
               'status':'selecionado_em_ensaio', 'teste_final_usado': False}
    origem = {'sha256_parquet': 'SINTETICO', 'sha256_folds': 'SINTETICO',
              'sha256_yaml': 'SINTETICO', 'experimento_oof':'SINTETICO'}
    pacote = ajustar_final_f1_f9(df, folds, cfg, escolha, origem, registrar_historico_em=False)
    m = 30
    risco = rng.normal(size=m)
    df_fake = pd.DataFrame({
        'TARGET_OMEGA': np.r_[np.zeros(25,dtype=np.uint8), np.ones(5,dtype=np.uint8)],
        'NID_ALPHA': np.arange(10000,10000+m,dtype=np.int64),
        'TRANS_VALUE': risco,
        'TRANS_NUM_CARD_FEWF': rng.choice(['R','S','T'], m),
        'SEND_GENDER_BE': rng.choice(['F','M'],m),
        'TRANS_MONTH_SIN': np.sin(np.arange(m)*2*np.pi/12),
        'TRANS_MONTH_COS': np.cos(np.arange(m)*2*np.pi/12),
        'VALOR_CONTINUO': rng.normal(size=m),
    })
    return pacote, df_fake


def test_avaliacao_com_lote_f10_totalmente_sintetico(pacote_e_dados):
    pacote, df = pacote_e_dados
    antes = df.copy(deep=True)
    tabela, resultado = avaliar_f10_em_memoria(
        pacote, df, np.full(len(df),10,dtype=np.uint8), np.arange(100000,100000+len(df))
    )
    assert resultado['metodo'] == 'TSNE'
    assert resultado['dimensao_tsne'] == 2
    assert resultado['limiar_mcc_oof_congelado'] == .3
    assert resultado['limiar_otimizado_no_f10'] is False
    assert resultado['id_candidato_oof'] == 'sintetico_2d'
    assert len(tabela)==len(df)
    assert (tabela['fold']==10).all()
    assert (tabela['prob_fraude'].between(0,1)).all()
    assert np.isfinite(tabela['prob_fraude']).all()
    np.testing.assert_array_equal(tabela['predicao_tau_0_50'], (tabela['prob_fraude']>=.5).astype(np.uint8))
    np.testing.assert_array_equal(tabela['predicao_tau_mcc_oof'], (tabela['prob_fraude']>=.3).astype(np.uint8))
    pd.testing.assert_frame_equal(df, antes)


def test_avaliacao_nao_entrega_rotulos_ao_preditor(pacote_e_dados, monkeypatch):
    pacote, df = pacote_e_dados
    original = pacote.prever_sem_rotulos
    recebido = []
    def espiado(X):
        recebido.append(list(X.columns))
        assert 'TARGET_OMEGA' not in X
        return original(X)
    monkeypatch.setattr(pacote, 'prever_sem_rotulos', espiado)
    a, _ = avaliar_f10_em_memoria(pacote, df, np.full(len(df),10), np.arange(10000,10000+len(df)))
    df2=df.copy()
    df2['TARGET_OMEGA'] = 1-df2['TARGET_OMEGA']
    b, _ = avaliar_f10_em_memoria(pacote, df2, np.full(len(df2),10), np.arange(10000,10000+len(df2)))
    np.testing.assert_array_equal(a['prob_fraude'], b['prob_fraude'])
    assert len(recebido)==2


def test_bloqueia_fold_fora_de_f10_sintetico(pacote_e_dados):
    pacote, df = pacote_e_dados
    folds = np.full(len(df), 10, dtype=np.uint8)
    folds[-1] = 9
    with pytest.raises(ValueError, match="folds"):
        avaliar_f10_em_memoria(pacote, df, folds, np.arange(len(df)))


def test_bloqueia_ids_repetidos_ou_desordenados(pacote_e_dados):
    pacote, df = pacote_e_dados
    ids = np.arange(len(df))
    ids[-1] = ids[-2]
    with pytest.raises(ValueError, match="Indices"):
        avaliar_f10_em_memoria(pacote, df, np.full(len(df),10), ids)


def test_bloqueia_escolha_nao_congelada(pacote_e_dados):
    pacote, df = pacote_e_dados
    original = pacote.escolha_oof['teste_final_usado']
    try:
        pacote.escolha_oof['teste_final_usado'] = True
        with pytest.raises(ValueError, match="congelada"):
            avaliar_f10_em_memoria(pacote, df, np.full(len(df),10), np.arange(len(df)))
    finally:
        pacote.escolha_oof['teste_final_usado'] = original
