"""Guardas da simulacao de instalacao limpa: nunca treina nem acessa F10 real."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import yaml

from src import finalizacao_automatica_pca1d as pca1d
from src import orquestracao_oof as orq

BASE = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load((BASE / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf-8-sig'))


@pytest.mark.parametrize('fracao', [0.01, 0.03, 0.1])
def test_pca_aceita_fracoes_desenvolvimento_independentes_de_f10(fracao):
    c = deepcopy(_config())
    c['reducao']['metodo'] = 'PCA'
    c['dados']['fracao_base'] = fracao
    assert pca1d.parametros_ok(c)
    assert orq.validar_operacao(c)


@pytest.mark.parametrize('fracao', [0, -0.1, 1.1, 0.02])
def test_pca_finalizador_rejeita_fracoes_nao_homologadas(fracao):
    c = _config()
    c['dados']['fracao_base'] = fracao
    assert not pca1d.parametros_ok(c)


def test_amostra_f10_10pct_independente_da_fracao_de_desenvolvimento(monkeypatch, tmp_path):
    # 9 folds de desenvolvimento e F10 com 200 legitimas e 20 fraudes.
    folds = np.array([1,2,3,4,5,6,7,8,9] + [10]*220, dtype=np.uint8)
    y = np.array([0]*9 + [0]*200 + [1]*20, dtype=np.uint8)
    monkeypatch.setattr(orq, '_ler_alvo', lambda parquet, coluna: y)
    results=[]
    for frac in (0.01, 0.03, 0.1):
        state={'orig_folds':folds, 'parquet':tmp_path/'nao_ler.parquet',
               'config':{'dados':{'coluna_alvo':'TARGET_OMEGA', 'fracao_base':frac},
                         'divisao':{'fracao_amostra_f10':0.10, 'qual_fold_teste':10},
                         'experimento':{'semente':42}}}
        folder=tmp_path/f'frac_{frac}'
        ids=pca1d.selecionar_amostra_f10(state,folder)
        assert len(ids)==22
        assert np.all(folds[ids]==10)
        assert (y[ids]==1).sum()==2
        results.append(ids)
    assert np.array_equal(results[0],results[1])
    assert np.array_equal(results[1],results[2])
