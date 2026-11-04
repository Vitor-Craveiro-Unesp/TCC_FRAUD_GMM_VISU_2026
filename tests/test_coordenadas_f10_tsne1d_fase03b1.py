"""Fase 3B1: coordenadas OOS da mesma inferencia e hashes de F10.

Testes sinteticos, sem uso de openTSNE nem leitura da base real.
"""
from types import MethodType

import numpy as np
import pandas as pd
import pytest

from src.avaliacao_final_tsne import avaliar_f10_em_memoria
from src.treinamento_final_tsne import PacoteFinalTSNE


def _pacote():
    p = object.__new__(PacoteFinalTSNE)
    p.configuracao = {'reducao': {'metodo': 'TSNE'}, 'divisao': {'qual_fold_teste': 10},
                       'dados': {'coluna_alvo': 'TARGET_OMEGA'}}
    p.escolha_oof = {'teste_final_usado': False, 'status': 'selecionado_em_ensaio',
                     'id_candidato': 'candidato-sintetico', 'limiar_mcc_oof': 0.4,
                     'configuracao': {'metodo': 'TSNE', 'reducao': {'n_components': 1}}}
    return p


def _dados():
    return pd.DataFrame({'X': [3.0, 5.0, 7.0, 9.0],
                         'TARGET_OMEGA': [0, 1, 0, 1]})


def test_coords_scores_mesma_inferencia_sem_alvo():
    p = _pacote()
    chamadas = []
    def inferir(self, entrada):
        assert 'TARGET_OMEGA' not in entrada
        chamadas.append(1)
        return (pd.Series([0.1, 0.8, 0.2, 0.7],index=entrada.index),
                pd.DataFrame({'TSNE1': [0.01, 0.25, 0.3, 0.8]},index=entrada.index))
    p.prever_com_coordenadas_sem_rotulos=MethodType(inferir,p)
    p.prever_sem_rotulos=MethodType(lambda *_: (_ for _ in ()).throw(AssertionError('inferiu duas vezes')),p)
    tab, met, coord = avaliar_f10_em_memoria(p,_dados(),np.full(4,10),np.arange(50,54),retornar_coordenadas=True)
    assert chamadas == [1]
    assert tab['linha_original'].tolist() == coord['linha_original'].tolist() == [50,51,52,53]
    assert coord['TSNE1'].tolist() == [0.01,0.25,0.3,0.8]
    assert met['limiar_otimizado_no_f10'] is False


def test_coords_ordem_invalida_bloqueada():
    p = _pacote()
    def inferir(self, entrada):
        return (pd.Series([0.1,0.8,0.2,0.7],index=entrada.index),
                pd.DataFrame({'TSNE1':[0.1,0.2,0.3,0.4]},index=[3,2,1,0]))
    p.prever_com_coordenadas_sem_rotulos=MethodType(inferir,p)
    with pytest.raises(RuntimeError,match='Coordenadas'):
        avaliar_f10_em_memoria(p,_dados(),np.full(4,10),np.arange(50,54),retornar_coordenadas=True)


def test_api_anterior_compatibilidade():
    p = _pacote()
    def inferir(self, entrada):
        return pd.Series([0.1,0.8,0.2,0.7],index=entrada.index)
    p.prever_sem_rotulos=MethodType(inferir,p)
    resultado=avaliar_f10_em_memoria(p,_dados(),np.full(4,10),np.arange(50,54))
    assert len(resultado)==2
