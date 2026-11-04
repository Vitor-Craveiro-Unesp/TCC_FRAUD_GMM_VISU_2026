"""Geometria F10: integridade, alinhamento, modelo congelado e quatro figuras."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.folds import hash_arquivo
from src import visualizacao_geometrica_final_tsne1d as geo
from tests.test_visualizacao_final_tsne1d_fase03a import preparar


def contexto(tmp_path):
    exp,t=preparar(tmp_path)
    orig=pd.read_csv(t/'predicoes_f10.csv')
    ids=orig['linha_original'].to_numpy(dtype=np.int64)
    coords=pd.DataFrame({'linha_original':ids,'TSNE1':np.linspace(-2.,2.,len(ids))})
    coords.to_csv(t/'coordenadas_TSNE1_f10.csv',index=False,float_format='%.17g')
    marker=json.loads((t/'manifesto_finalizacao_tsne.json').read_text())
    marker['sha256_coordenadas']=hash_arquivo(t/'coordenadas_TSNE1_f10.csv')
    (t/'manifesto_finalizacao_tsne.json').write_text(json.dumps(marker))
    return exp,t,orig


def test_nao_aceita_coordenada_alterada(tmp_path):
    exp,t,_=contexto(tmp_path)
    (t/'coordenadas_TSNE1_f10.csv').write_text('corrupcao')
    with pytest.raises(RuntimeError,match='Coordenadas F10 alteradas'):
        geo._fontes(exp)


def test_nao_aceita_coordenada_permutada(tmp_path):
    exp,t,_=contexto(tmp_path)
    f=t/'coordenadas_TSNE1_f10.csv'
    coords=pd.read_csv(f)
    coords=coords.iloc[::-1]
    coords.to_csv(f,index=False)
    marker=json.loads((t/'manifesto_finalizacao_tsne.json').read_text())
    marker['sha256_coordenadas']=hash_arquivo(f)
    (t/'manifesto_finalizacao_tsne.json').write_text(json.dumps(marker))
    with pytest.raises(RuntimeError,match='desalinhados'):
        geo._fontes(exp)


def test_renderizacao_quatro_pngs_e_retomada(tmp_path,monkeypatch):
    exp,t,orig=contexto(tmp_path)
    from src import treinamento_final_tsne
    from src import probabilidade
    from src import visualizacao_gmm01,visualizacao_gmm04,visualizacao_gmm07,visualizacao_geometrica
    from src.visualizacao_geometrica_final_tsne1d import _fontes
    _,cfg,met,marker,z,y,p,tau=_fontes(exp)

    class FakeGMM:
        colunas_=['TSNE1']
        modelo_=SimpleNamespace(covariance_type='full')
        def responsabilidades(self,x):
            assert list(x.columns)==['TSNE1']
            return np.column_stack((np.asarray(x['TSNE1']),np.ones(len(x))))
    pacote=SimpleNamespace(configuracao=cfg,escolha_oof={'id_candidato':'fakeseguro'},
                           limiar_oof=tau,gmm=FakeGMM(),risco=SimpleNamespace(riscos_=[0.1,0.4]),
                           tsne=object())
    monkeypatch.setattr(treinamento_final_tsne,'carregar_pacote_confiavel',lambda _:pacote)
    monkeypatch.setattr(probabilidade,'calcular_probabilidades',lambda gamma,riscos:p.copy())
    called=[]
    def mock_png(nome):
        def f(*args,**kwargs):
            dest=next(x for x in args if isinstance(x,Path) and x.name=='.visualizacao_geometrica_incompleta')
            (dest/nome).write_bytes(b'\x89PNG\r\n\x1a\nSIMULADO')
            called.append(nome)
        return f
    monkeypatch.setattr(visualizacao_gmm01,'gerar',mock_png(geo.PNG_GEOMETRIA[0]))
    monkeypatch.setattr(visualizacao_gmm04,'gerar',mock_png(geo.PNG_GEOMETRIA[1]))
    monkeypatch.setattr(visualizacao_gmm07,'gerar',mock_png(geo.PNG_GEOMETRIA[2]))
    monkeypatch.setattr(visualizacao_geometrica,'_fronteira',mock_png(geo.PNG_GEOMETRIA[3]))
    origem=hash_arquivo(t/'predicoes_f10.csv')
    path=geo.gerar(exp)
    assert path.name=='visualizacao_geometrica'
    assert set(called)==set(geo.PNG_GEOMETRIA)
    assert geo.gerar(exp)==path
    assert hash_arquivo(t/'predicoes_f10.csv')==origem
    man=json.loads((path/'manifesto_geometria_f10.json').read_text())
    assert man['galeria_f10_completa'] is False
    assert man['sem_novas_predicoes'] is True
    (path/geo.PNG_GEOMETRIA[0]).write_bytes(b'ALTERADO')
    with pytest.raises(RuntimeError,match='alterada'):
        geo.gerar(exp)


def test_sem_coordenadas_b1_nao_habilita_geometria(tmp_path):
    exp,t=preparar(tmp_path)
    with pytest.raises(RuntimeError,match='Fase 3B.1 ausente'):
        geo._fontes(exp)
