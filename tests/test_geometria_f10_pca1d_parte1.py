"""Verifica invariantes sintéticas; sem Parquet, folds ou dados F10 reais."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/execucao/GERAR_GEOMETRIA_F10_PCA1D_PARTE1.py'
spec = importlib.util.spec_from_file_location('pca1d_geometria_f10', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def dados():
    coords = pd.DataFrame({'linha_original':[12, 14, 18], 'PC1':[-1.0, .0, 1.0]})
    p = [.1,.6,.9]
    preds = pd.DataFrame({'linha_original':[12,14,18], 'fold':[10]*3,
                          'y_true':[0,0,1], 'prob_fraude':p,
                          'predicao_tau_0_50':[0,1,1],
                          'predicao_tau_mcc_oof':[1,1,1]})
    return coords,preds


def test_alinhamento_aceito():
    c,p = dados()
    m.validar_alinhamento(c,p,3)
    m.validar_cortes(p,.05)


def test_rejeita_desordem_ou_troca():
    c,p=dados()
    c.loc[0,'linha_original']=99
    with pytest.raises(RuntimeError):
        m.validar_alinhamento(c,p,3)


def test_rejeita_fold_9_e_nan():
    c,p=dados();p.loc[1,'fold']=9
    with pytest.raises(RuntimeError):m.validar_alinhamento(c,p,3)
    c,p=dados();c.loc[1,'PC1']=np.nan
    with pytest.raises(RuntimeError):m.validar_alinhamento(c,p,3)


def test_rejeita_reotimizacao_implícita_ou_corte_trocado():
    _,p=dados()
    with pytest.raises(RuntimeError):m.validar_cortes(p,.5)


def test_csv_f10_usa_id_real_e_nao_nome_treino(tmp_path):
    file=tmp_path/'12_pca_projecao_PC1.csv'
    pd.DataFrame({'PC1':[-1.,0.], 'classe_real':[0,1],
                  'risco_estimado_treino':[.01,.4], 'variancia_explicada_PC1':[.2,.2]}).to_csv(file,index=False)
    m.corrigir_csv_projecao(file,np.array([50,51]))
    out=pd.read_csv(file)
    assert list(out.columns)[0]=='linha_original_f10'
    assert 'prob_fraude_f10' in out and 'risco_estimado_treino' not in out


def test_impede_sobrescrever(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'FIGURAS',tmp_path)
    monkeypatch.setattr(m,'MANIFESTO',tmp_path/'novo.json')
    monkeypatch.setattr(m,'GALERIA',tmp_path/'nova.html')
    (tmp_path/'12_pca_projecao_PC1.png').write_bytes(b'existente')
    with pytest.raises(RuntimeError,match='existente'):
        m.checar_destino('png')


def test_html_revisao_origens():
    page=m.html_revisao(18523,.010524627848033968,{})
    assert '18,523' in page and 'F1–F9' in page and 'F10' in page
    for nome in m.NOMES:
        assert nome+'.png' in page
