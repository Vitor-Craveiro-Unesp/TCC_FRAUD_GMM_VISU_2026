# -*- coding: utf-8 -*-
"""Testes inteiramente sintéticos; nunca usam o F10/Parquet reais."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.execucao import GERAR_PRE01_F10_PCA1D as m


def tabela(n=8):
    return pd.DataFrame([{
        'feature':'TRANS_VALUE', 'tipo':'numerica','n_f10':n,
        'media_treino_f1_f9_codificada':10., 'escala_treino_f1_f9':2.,
        'media_z_f10':.5, 'desvio_z_f10':1.2, 'variancia_z_f10':1.44,
        'drift_media_em_escala_treino':.5, 'percentual_ausentes_brutos_f10':12.5,
    },{
        'feature':'SEND_JOB_FEWF', 'tipo':'FEWF', 'n_f10':n,
        'media_treino_f1_f9_codificada':.2, 'escala_treino_f1_f9':.1,
        'media_z_f10':-.15, 'desvio_z_f10':.8, 'variancia_z_f10':.64,
        'drift_media_em_escala_treino':-.15, 'percentual_ausentes_brutos_f10':0.,
    }])


def geometria(n=8):
    return pd.DataFrame({'linha_original':range(100,100+n),'PC1':np.linspace(-2,3,n)})


def test_reconstroi_media_variancia_sem_ajustar():
    rows=m.enriquecer(tabela())
    assert rows.loc[0,'media_f10_codificada_imputada'] == pytest.approx(11.)
    assert rows.loc[0,'variancia_f10_codificada_imputada'] == pytest.approx(5.76)
    assert rows.loc[0,'desvio_f10_codificado_imputado'] == pytest.approx(2.4)
    assert rows.loc[0,'razao_dispersao_f10_treino'] == pytest.approx(1.2)
    assert rows.loc[0,'drift_log_dispersao'] == pytest.approx(np.log(1.2))


def test_rejeita_coordenadas_fora_ordem():
    c=geometria()
    c.loc[4,'linha_original']=c.loc[3,'linha_original']
    with pytest.raises(RuntimeError, match='duplicados'):
        m.validar_tabelas(tabela(),c,geometria()[['linha_original']],8)


def test_rejeita_variancia_incoerente():
    t=tabela()
    t.loc[0,'variancia_z_f10']=3.
    with pytest.raises(RuntimeError, match='inconsistentes'):
        m.validar_tabelas(t,geometria(),geometria()[['linha_original']],8)


def test_histograma_conserva_n_inclusive_variancia_zero():
    c=geometria()
    assert m.histograma_pc1(c)['contagem_f10'].sum()==len(c)
    c['PC1']=2.0
    assert m.histograma_pc1(c)['contagem_f10'].sum()==len(c)


def fake_root(tmp_path:Path):
    p=m.caminhos(tmp_path)
    for d in (p['geo'],p['modelo'],p['ensaio']/'vencedor/metricas'):
        d.mkdir(parents=True,exist_ok=True)
    g=geometria()
    g.to_csv(p['geo']/'coordenadas_PC1_f10.csv',index=False)
    tabela().to_csv(p['geo']/'resumo_padronizacao_f10.csv',index=False)
    p['teste'].mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'linha_original':g['linha_original'],'y_true':[0]*8}).to_csv(p['teste']/'predicoes_f10.csv',index=False)
    (p['teste']/'metricas_f10.json').write_text('{"n":8}',encoding='utf-8')
    job=p['modelo']/'modelo_final_f1_f9.joblib'
    job.write_bytes(b'FAKE-JOBLIB-FOR-SYNTHETIC-TESTS-NOT-TO-LOAD')
    fh=m.sha256_arquivo(job)
    gh={f:m.sha256_arquivo(p['geo']/f) for f in
        ('coordenadas_PC1_f10.csv','resumo_padronizacao_f10.csv')}
    (p['geo']/'manifesto_geometria_f10.json').write_text(json.dumps({
        'dimensao':1,'tipo':'EXTRACAO_DESCRITIVA_F10_JA_AVALIADO','n':8,
        'parquet_target_lido':False,'novo_treinamento':False,
        'nova_inferencia_probabilidades':False,'nova_avaliacao_f10':False,
        'limiar_mcc_oof_congelado':.01,'sha256_modelo_final':fh,
        'original_f10':'outputs/f10_ficticio/manifesto_avaliacao_f10.json',
        'arquivos':gh
    }),encoding='utf-8')
    (p['modelo']/'manifesto_modelo_final.json').write_text(json.dumps({
        'dimensao_pca':1,'n_treino_original':166716,
        'id_candidato_oof':'teste','limiar_mcc_oof_congelado':.01,
        'sha256_modelo_final':fh}),encoding='utf-8')
    (p['ensaio']/'vencedor/metricas/melhor_configuracao.json').write_text(
        '{"id_candidato":"teste","limiar_mcc_oof":0.01}', encoding='utf-8')
    (p['teste']/'manifesto_espelho_graficos_f10.json').write_text(json.dumps({
        'n':8,'limiar_mcc_congelado_oof':.01,
        'predicoes_sha256':m.sha256_arquivo(p['teste']/'predicoes_f10.csv'),
        'metricas_sha256':m.sha256_arquivo(p['teste']/'metricas_f10.json')
    }),encoding='utf-8')
    return p


def test_confere_sem_escrever(tmp_path):
    fake_root(tmp_path)
    result=m.conferir(tmp_path)
    assert result['n']==8
    assert not result['paths']['saida'].exists()


def test_rejeita_resumo_adulterado(tmp_path):
    p=fake_root(tmp_path)
    with (p['geo']/'resumo_padronizacao_f10.csv').open('ab') as f:
        f.write(b'\n')
    with pytest.raises(RuntimeError,match='alterado'):
        m.conferir(tmp_path)


def test_html_aproxima_estilo_sem_inventar_folds():
    pytest.importorskip('plotly')
    coords=geometria()
    t=m.enriquecer(tabela())
    h=m.histograma_pc1(coords)
    html=m.html_pre01(t,h,coords.PC1.to_numpy(),{'n_f10':8,'tau_star_oof':.01})
    assert 'PRE01 — Diagnóstico completo da padronização (F10)' in html
    assert 'não pode ser repetido no F10' in html
    assert 'table-summary' in html and 'plot-pca' in html
    assert 'μ<sub>F10</sub>' in html
    assert 'F10 não há “ajustes 8F”' in html
    assert 'Não foi feito novo ajuste PCA' in html


def test_gravacao_uma_vez_sem_sobrescrever(tmp_path):
    pytest.importorskip('plotly')
    fake_root(tmp_path)
    state=m.conferir(tmp_path,exige_destino_livre=True)
    saida=m.gerar(state)
    assert saida.is_file()
    assert (saida.parent/m.TABELA_ARQ).is_file()
    assert (saida.parent/m.HIST_ARQ).is_file()
    with pytest.raises(RuntimeError,match='preservada'):
        m.conferir(tmp_path,exige_destino_livre=True)


def test_modo_preflight_nunca_carrega_modelo(tmp_path,monkeypatch):
    import joblib
    fake_root(tmp_path)
    monkeypatch.setattr(joblib,'load',lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('joblib.load proibido')))
    m.conferir(tmp_path)
