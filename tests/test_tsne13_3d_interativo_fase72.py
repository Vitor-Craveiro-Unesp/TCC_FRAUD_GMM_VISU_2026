# -*- coding: utf-8 -*-
"""Testes do renderizador 3D isolado — nao carrega Parquet real nem F10 real."""
from __future__ import annotations

import json
import re
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from src.visualizacao_tsne3d_interativa import (
    PNG, HTML, CSV, MANIFESTO, _amostra_estratificada, gerar_tsne13_3d,
)


def _dados(n=120, fraude=8):
    rng = np.random.default_rng(42)
    coords = pd.DataFrame(rng.normal(size=(n, 3)), columns=['TSNE1', 'TSNE2', 'TSNE3'])
    coords.insert(0, 'linha_original', np.arange(500, 500+n))
    y = np.array([1]*fraude + [0]*(n-fraude))
    p = rng.uniform(0.005, 0.05, n)
    gamma = rng.dirichlet([1, 2, 3, 4], n)
    return coords, y, p, gamma


def test_amostra_visual_preserva_fraudes_raras_e_deterministica():
    _, y, _, _ = _dados(120, 8)
    ix = _amostra_estratificada(y, max_pontos=50, semente=42)
    assert len(ix) == 50
    assert y[ix].sum() == 8
    assert np.array_equal(ix, _amostra_estratificada(y, max_pontos=50, semente=42))
    assert np.all(np.diff(ix) > 0)


def test_amostra_fraudes_abundantes_e_caso_uniclasse():
    y = np.array([1]*90+[0]*30)
    ix = _amostra_estratificada(y, max_pontos=50, semente=42)
    assert len(ix) == 50 and np.unique(y[ix]).size == 2
    assert len(_amostra_estratificada(np.zeros(120), max_pontos=50, semente=42)) == 50


def test_render_sintetico_offline_modos_manifesto(tmp_path):
    z,y,p,gamma = _dados()
    saida = gerar_tsne13_3d(z, y, tmp_path, scores=p, limiar_oof=.0123,
                             responsabilidades=gamma, origem='simulado', max_pontos=50, dpi=80)
    assert set(saida) == {PNG,HTML,CSV,MANIFESTO}
    assert (tmp_path/PNG).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    pagina=(tmp_path/HTML).read_text(encoding='utf-8')
    assert 'scatter3d' in pagina and 'Plotly.newPlot' in pagina and 'PREVIA SINTETICA' in pagina
    assert not re.search(r'<script[^>]+\bsrc\s*=', pagina, flags=re.IGNORECASE)
    assert 'Gaussiana dominante' in pagina and 'Risco estimado' in pagina
    assert 'Predicao tau* OOF' in pagina and 'TSNE3' in pagina
    tabela=pd.read_csv(tmp_path/CSV)
    assert len(tabela)==50 and tabela['classe_real'].sum()==8
    assert tabela['linha_original'].min() >= 500
    man=json.loads((tmp_path/MANIFESTO).read_text(encoding='utf-8'))
    assert man['n_exibidos']==50 and man['n_fraudes_exibidas']==8
    assert man['modelo_retreinado'] is False
    for nome, digest in man['sha256_arquivos'].items():
        assert sha256((tmp_path/nome).read_bytes()).hexdigest()==digest


def test_f10_exige_fracao_e_nao_reajusta(tmp_path):
    z,y,p,_ = _dados(26,4)
    with pytest.raises(ValueError, match='fracao'):
        gerar_tsne13_3d(z,y,tmp_path,scores=p,origem='f10',dpi=80)
    saida=gerar_tsne13_3d(z,y,tmp_path,scores=p,origem='f10',
                          fracao_f10=.1,dpi=80)
    assert '10.00%' in saida[HTML].read_text(encoding='utf-8')
    assert 'F10 exploratorio' in saida[HTML].read_text(encoding='utf-8')
    m=json.loads(saida[MANIFESTO].read_text(encoding='utf-8'))
    assert m['f10_exploratorio'] and m['fracao_f10_dentro_do_fold']==.1
    assert m['limiar_oof_congelado'] is None


def test_nao_sobrescreve_saida(tmp_path):
    z,y,p,_ = _dados(20,2)
    gerar_tsne13_3d(z,y,tmp_path,scores=p,origem='simulado',dpi=80)
    antes=(tmp_path/HTML).read_bytes()
    with pytest.raises(ValueError, match='sobrescrever'):
        gerar_tsne13_3d(z,y,tmp_path,scores=p,origem='simulado',dpi=80)
    assert (tmp_path/HTML).read_bytes() == antes


@pytest.mark.parametrize('problema', ['coordenadas','scores','tau','gamma','classes'])
def test_rejeita_entradas_invalidas(tmp_path, problema):
    z,y,p,gamma = _dados(20,2)
    if problema=='coordenadas':
        z=z.drop(columns=['TSNE3'])
    elif problema=='scores':
        p[0]=1.2
    elif problema=='tau':
        tau=1.2
    elif problema=='gamma':
        gamma[0,:]=0
    elif problema=='classes':
        y[0]=2
    if problema != 'tau': tau=.05
    with pytest.raises(ValueError):
        gerar_tsne13_3d(z,y,tmp_path,scores=p,limiar_oof=tau,
                        responsabilidades=gamma,origem='simulado',dpi=80)


def test_sem_scores_oferece_somente_modo_classe(tmp_path):
    z,y,_,_=_dados(20,2)
    saida=gerar_tsne13_3d(z,y,tmp_path,origem='simulado',dpi=80)
    m=json.loads(saida[MANIFESTO].read_text(encoding='utf-8'))
    assert m['modos_disponiveis']==['Classe real']


def test_nao_muda_vetores_de_entrada(tmp_path):
    z,y,p,g=_dados(20,2)
    antes=(z.copy(deep=True),y.copy(),p.copy(),g.copy())
    gerar_tsne13_3d(z,y,tmp_path,scores=p,responsabilidades=g,
                    origem='simulado',dpi=80)
    pd.testing.assert_frame_equal(z,antes[0])
    assert np.array_equal(y,antes[1]) and np.array_equal(p,antes[2])
    assert np.array_equal(g,antes[3])
