"""Testes sintéticos puros; NUNCA leem Parquet ou modelos reais."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/execucao/EXTRAIR_COORDENADAS_PCA1D_F10.py'
spec = importlib.util.spec_from_file_location('extrator_pca1d', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_indices_alinhados():
    ids = np.array([9, 15, 40], dtype=np.int64)
    p = pd.DataFrame({'linha_original': ids, 'fold': [10, 10, 10]})
    assert np.array_equal(m.validar_indices(ids, p, 3), ids)


def test_bloqueio_indices_trocados():
    with pytest.raises(RuntimeError, match='nao coincidem'):
        m.validar_indices(np.array([9, 15, 40]),
                          pd.DataFrame({'linha_original':[9, 40, 15], 'fold':[10]*3}), 3)


def test_bloqueio_folds_ou_repeticoes():
    with pytest.raises(RuntimeError):
        m.validar_indices(np.array([9, 9, 40]),
                          pd.DataFrame({'linha_original':[9, 9, 40], 'fold':[10]*3}), 3)
    with pytest.raises(RuntimeError):
        m.validar_indices(np.array([9, 15, 40]),
                          pd.DataFrame({'linha_original':[9, 15, 40], 'fold':[9, 10, 10]}), 3)


class FakeEncoder:
    colunas_ = ['VALOR_FEWF', 'QUANTIA']
    def transform(self, X):
        return X.copy()

class FakeImputador:
    padronizadas_ = ['VALOR_FEWF', 'QUANTIA']
    class Scaler:
        mean_ = np.array([0.5, 3.0])
        scale_ = np.array([1.0, 2.0])
    scaler_ = Scaler()
    def transform(self, X):
        return pd.DataFrame({'VALOR_FEWF': X['VALOR_FEWF'] - 0.5,
                             'QUANTIA': (X['QUANTIA'] - 3.0) / 2.0}, index=X.index)

class FakePCA:
    def transform(self, X):
        return pd.DataFrame({'PC1': X['VALOR_FEWF'] + X['QUANTIA']}, index=X.index)

class FakePacote:
    encoder = FakeEncoder()
    imputador = FakeImputador()
    pca = FakePCA()
    configuracao = {'dados': {'coluna_alvo': 'TARGET_OMEGA','coluna_id': 'NID_ALPHA','colunas_excluir': []},
                    'preprocessamento': {'encodings': {'frequency': {'sufixo': '_FEWF'}}}}


def test_transform_sem_fit():
    d = pd.DataFrame({'VALOR_FEWF':[1.0,2.0,3.0], 'QUANTIA':[5.0,7.0,9.0]})
    z, pcs = m.aplicar_transformadores_congelados(FakePacote(), d)
    assert list(pcs.columns) == ['PC1']
    assert pcs['PC1'].tolist() == [1.5, 3.5, 5.5]
    assert z.shape == (3,2)


def test_proibe_target_e_muda_ordem():
    d = pd.DataFrame({'VALOR_FEWF':[1.0,2.0], 'QUANTIA':[5.0,7.0]})
    with pytest.raises(RuntimeError, match='diferente'):
        m.aplicar_transformadores_congelados(FakePacote(), d[['QUANTIA','VALOR_FEWF']])
    with pytest.raises(RuntimeError):
        m.aplicar_transformadores_congelados(FakePacote(), d.assign(TARGET_OMEGA=1))


def test_resumo_exato():
    d = pd.DataFrame({'VALOR_FEWF':[1.0,2.0,3.0], 'QUANTIA':[5.0,7.0,9.0]})
    z, _ = m.aplicar_transformadores_congelados(FakePacote(), d)
    r = m.resumo_preprocessamento(FakePacote(), d, z)
    assert len(r) == 2
    assert r.loc[1,'media_z_f10'] == pytest.approx(2.0)
    assert r.loc[0,'media_treino_f1_f9_codificada'] == 0.5
