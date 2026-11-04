"""Fase 3B.2B: graficos diagnosticos e galeria 12/12 com dados sinteticos."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.folds import hash_arquivo
from src import visualizacao_final_completa_tsne1d as final
from src import visualizacao_final_tsne1d as pred
from src import visualizacao_geometrica_final_tsne1d as geo
from tests.test_geometria_final_tsne1d_fase03b2a import contexto


def simular(tmp_path, monkeypatch):
    pasta, teste, scores = contexto(tmp_path)
    _,cfg,met,marco,z,y,p,tau = geo._fontes(pasta)
    ids=scores['linha_original'].to_numpy(dtype=np.int64)
    gamma=np.tile([.1,.2,.3,.4],(len(ids),1))
    modelo=SimpleNamespace(
        tsne=SimpleNamespace(kl_divergence_=0.32, parametros={'n_components':1}),
        gmm=SimpleNamespace(modelo_=SimpleNamespace(n_components=4)),
        config=cfg)
    monkeypatch.setattr(final,'_fontes_imutaveis',lambda _: (
        teste,cfg,met,marco,z['TSNE1'].to_numpy(),y,p,tau,ids,modelo,gamma))
    monkeypatch.setattr(final,'_resumo_pre01',lambda *args: (
        pd.DataFrame([{'feature':'VAR1','tipo':'padronizada','media_transformada_f10':.2,
                       'desvio_transformado_f10':1.1,'media_treino_f1_f9_codificada':2.0,
                       'escala_treino_f1_f9':.5,'percentual_ausentes_brutos_f10':0.0}]),1))
    return pasta, teste, marco


def test_diagnosticos_reais_sinteticos_e_retomada(tmp_path,monkeypatch):
    pasta,teste,marco=simular(tmp_path,monkeypatch)
    destino=final.gerar_diagnosticos(pasta)
    assert destino.name=='diagnosticos_tsne'
    assert (destino/'13_tsne_mapa_TSNE1.png').read_bytes().startswith(b'\x89PNG')
    assert (destino/'14_tsne_diagnostico.png').read_bytes().startswith(b'\x89PNG')
    html=(destino/final.PRE01).read_text(encoding='utf-8')
    assert 'Plotly.newPlot' in html and len(html)>100000
    assert final.gerar_diagnosticos(pasta)==destino
    (destino/'14_tsne_diagnostico.csv').write_text('ALTERADO')
    with pytest.raises(RuntimeError,match='alterado'):
        final.gerar_diagnosticos(pasta)


def test_galeria_12_pags_com_proveniencia(tmp_path,monkeypatch):
    pasta,teste,marco=simular(tmp_path,monkeypatch)
    caminho_pred=pred.gerar(pasta)
    caminho_geo=teste/'visualizacao_geometrica';caminho_geo.mkdir()
    for nome in geo.PNG_GEOMETRIA:
        (caminho_geo/nome).write_bytes(b'\x89PNG\r\n\x1a\nGEOMETRIA_SIMULADA')
    monkeypatch.setattr(geo,'gerar',lambda _:caminho_geo)
    destino=final.gerar_galeria(pasta)
    man=json.loads((destino/'manifesto_galeria_f10.json').read_text())
    assert len(man['sha256_arquivos'])==14
    assert len(list((destino/'figuras').glob('*.png')))==12
    assert man['sha256_predicoes']==marco['sha256_predicoes']
    html=(destino/'index.html').read_text(encoding='utf-8')
    assert '1.00%' in html and '10.00% do fold' in html
    assert '../../visualizacao/galeria/index.html' in html
    assert final.gerar_galeria(pasta)==destino
    (destino/'figuras'/pred.PNG_FILES[0]).write_bytes(b'CORROMPIDO')
    with pytest.raises(RuntimeError,match='alterada/ausente'):
        final.gerar_galeria(pasta)


def test_nao_usa_contagem_fixa_de_experimento_antigo():
    import inspect
    src=inspect.getsource(final)
    assert '18523' not in src and '18.523' not in src
    assert 'v6.preflight' not in src
    assert 'F10_exploratorio_reutilizado' not in src


def test_cli_retomada_galeria_f10_sem_retreino(tmp_path,monkeypatch):
    import yaml
    from src import orquestracao_oof as orq
    exp=tmp_path/'outputs'/'ensaio_tsne_gmm_teste'
    exp.mkdir(parents=True)
    cfg={'reducao':{'metodo':'TSNE'},'tsne':{'n_components':[1]}}
    (exp/'configuracao.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    chamados=[]
    monkeypatch.setattr(final,'gerar_galeria',lambda p:chamados.append(Path(p)))
    assert orq.main(['--concluir-galeria-f10','--experimento',str(exp)],raiz=tmp_path)==exp
    assert chamados==[exp]
