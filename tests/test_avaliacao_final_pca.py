"""Etapa 5: apenas dados sinteticos; nenhuma leitura do F10 real."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import matthews_corrcoef, average_precision_score, log_loss

from src.avaliacao_final_pca import (
    avaliar_scores_com_limiar_oof_congelado,
    avaliar_f10_em_memoria,
)


class PacoteSintetico:
    def __init__(self, scores, tau=0.55):
        self.valores = np.asarray(scores, dtype=float)
        self.configuracao = {
            'dados': {'coluna_alvo': 'TARGET_OMEGA'},
            'divisao': {'qual_fold_teste': 10},
        }
        self.escolha_oof = {
            'teste_final_usado': False,
            'status': 'selecionado_em_ensaio',
            'id_candidato': 'sintetico',
            'configuracao': {'reducao': {'n_components': 2}},
            'limiar_mcc_oof': tau,
        }
        self.chamadas_predicao = 0

    def prever_sem_rotulos(self, x):
        assert 'TARGET_OMEGA' not in x.columns
        self.chamadas_predicao += 1
        return pd.Series(self.valores, index=x.index, name='prob_fraude')


def _exemplo():
    y = pd.Series([0, 1, 0, 1, 0, 1], dtype='uint8')
    p = pd.Series([0.01, 0.20, 0.50, 0.60, 0.70, 0.90])
    return y, p


def test_metricas_e_score_congelado():
    y, p = _exemplo()
    r = avaliar_scores_com_limiar_oof_congelado(y, p, limiar_mcc_oof=0.55)
    assert r['limiar_otimizado_no_f10'] is False
    assert r['limiar_mcc_oof_congelado'] == 0.55
    assert r['ap'] == pytest.approx(average_precision_score(y, p))
    assert r['logloss'] == pytest.approx(log_loss(y, p, labels=[0, 1]))
    mcc = matthews_corrcoef(y, (p >= 0.55).astype('uint8'))
    assert r['mcc_limiar_oof'] == pytest.approx(mcc)
    assert r['score_composto_limiar_oof'] == pytest.approx(
        (r['ap'] + (mcc + 1) / 2 + np.exp(-r['logloss'])) / 3
    )
    assert r['comparacao_limiares']['tau_mcc_oof']['limiar'] == 0.55
    assert r['comparacao_limiares']['tau_0_50']['limiar'] == 0.5
    json.dumps(r, allow_nan=False)


def test_matrizes_e_percentuais_somam_cem():
    y, p = _exemplo()
    r = avaliar_scores_com_limiar_oof_congelado(y, p, limiar_mcc_oof=0.55)
    for item in r['comparacao_limiares'].values():
        cm = item['matriz_confusao']
        assert sum(sum(linha) for linha in cm['contagens']) == len(y)
        for linha in cm['percentuais_por_classe_real']:
            assert sum(linha) == pytest.approx(100.0)
        assert sum(sum(linha) for linha in cm['percentuais_do_total']) == pytest.approx(100.0)


def test_scores_e_rotulos_desalinhados_bloqueados():
    y, p = _exemplo()
    with pytest.raises(ValueError, match='Indices'):
        avaliar_scores_com_limiar_oof_congelado(y, p.reindex([1, 0, 2, 3, 4, 5]), limiar_mcc_oof=0.4)
    with pytest.raises(ValueError, match='Series'):
        avaliar_scores_com_limiar_oof_congelado(y, p.to_numpy(), limiar_mcc_oof=0.4)


def test_score_invalido_e_limiar_invalido_bloqueados():
    y, p = _exemplo()
    with pytest.raises(ValueError, match='Scores'):
        avaliar_scores_com_limiar_oof_congelado(y, p * 20, limiar_mcc_oof=0.4)
    with pytest.raises(ValueError, match='Limiar'):
        avaliar_scores_com_limiar_oof_congelado(y, p, limiar_mcc_oof=np.nan)
    with pytest.raises(ValueError, match='Limiar'):
        avaliar_scores_com_limiar_oof_congelado(y, p, limiar_mcc_oof=True)


def test_limiar_nao_e_reotimizado_mesmo_quando_um_outro_seria_melhor():
    y, p = _exemplo()
    r = avaliar_scores_com_limiar_oof_congelado(y, p, limiar_mcc_oof=0.99)
    assert r['limiar_mcc_oof_congelado'] == 0.99
    assert r['mcc_limiar_oof'] == 0.0
    assert r['comparacao_limiares']['tau_mcc_oof']['positivos_preditos'] == 0


def test_inferencia_sintetica_remove_alvo_e_preserva_ordem():
    y, p = _exemplo()
    pacote = PacoteSintetico(p)
    df = pd.DataFrame({'TARGET_OMEGA': y, 'VALOR': [2, 3, 4, 5, 6, 7]})
    ids = np.arange(101, 107)
    tabela, r = avaliar_f10_em_memoria(pacote, df, np.full(len(df), 10), ids)
    assert pacote.chamadas_predicao == 1
    assert tabela['linha_original'].tolist() == ids.tolist()
    assert np.array_equal(tabela['prob_fraude'].to_numpy(), p.to_numpy())
    assert (tabela['fold'] == 10).all()
    assert r['dimensao_pca'] == 2
    assert r['limiar_otimizado_no_f10'] is False


def test_folds_que_nao_sao_f10_bloqueados_antes_de_prever():
    y, p = _exemplo()
    pacote = PacoteSintetico(p)
    df = pd.DataFrame({'TARGET_OMEGA': y, 'VALOR': [2, 3, 4, 5, 6, 7]})
    with pytest.raises(ValueError, match='Somente linhas'):
        avaliar_f10_em_memoria(pacote, df, np.array([10, 10, 9, 10, 10, 10]), np.arange(6))
    assert pacote.chamadas_predicao == 0


def test_indices_duplicados_bloqueados_antes_de_prever():
    y, p = _exemplo()
    pacote = PacoteSintetico(p)
    df = pd.DataFrame({'TARGET_OMEGA': y, 'VALOR': [2, 3, 4, 5, 6, 7]})
    with pytest.raises(ValueError, match='linhas_originais'):
        avaliar_f10_em_memoria(pacote, df, np.full(6, 10), [0, 1, 1, 3, 4, 5])
    assert pacote.chamadas_predicao == 0
