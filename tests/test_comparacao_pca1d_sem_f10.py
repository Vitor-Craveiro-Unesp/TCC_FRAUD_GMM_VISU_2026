"""Somente caminhos ficticios. Nenhum arquivo real de F10 e lido."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import pytest

MODULE = Path(__file__).resolve().parents[1] / 'scripts' / 'execucao' / 'TESTE_1D_OOF_VS_F10.py'
spec = importlib.util.spec_from_file_location('comparador_testado',MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_identificar_graficos_por_nome(tmp_path):
    vis = tmp_path/'vencedor'/'visualizacao'/'oof'
    vis.mkdir(parents=True)
    for n in ['00_painel_score_composto.png','01_curva_precision_recall.png',
              '18_distribuicao_scores_por_classe.png','05_matriz_confusao_dupla.png',
              'extra_geom.png']:
        (vis/n).write_bytes(b'PNG simulado')
    pngs, pares, modos=m.identificar_imagens_oof(tmp_path)
    assert len(pngs)==5
    assert pares['painel'].name=='00_painel_score_composto.png'
    assert pares['pr'].name=='01_curva_precision_recall.png'
    assert 'distribuicao' in pares['scores'].name
    assert 'confusao' in pares['cm'].name
    assert 'aproxima' in modos['cm']


def test_sem_equivalente_disponivel(tmp_path):
    vis=tmp_path/'vencedor'/'visualizacao'/'oof'
    vis.mkdir(parents=True)
    (vis/'somente_geometria.png').write_bytes(b'fake')
    files, pares, modes=m.identificar_imagens_oof(tmp_path)
    assert all(v is None for v in pares.values())
    assert 'sem equivalente' in modes['pr']


def test_comparacao_html_somente_figuras_salvas(tmp_path,monkeypatch):
    oof=tmp_path/'outputs'/'ensaio_pca_gmm_fake'
    gal=oof/'vencedor'/'visualizacao'/'galeria';gal.mkdir(parents=True)
    (gal/'index.html').write_text('<html>OOF</html>')
    (oof/'vencedor'/'metricas').mkdir(parents=True)
    (oof/'vencedor'/'metricas'/'melhor_configuracao.json').write_text(json.dumps({'limiar_mcc_oof':.01}))
    vis=oof/'vencedor'/'visualizacao'/'oof';vis.mkdir()
    (vis/'01_curva_precision_recall.png').write_bytes(b'figure')
    f10=tmp_path/'outputs'/'galeria_final_f10_fake'; (f10/'PCA_1D').mkdir(parents=True)
    (f10/'index.html').write_text('<html>F10</html>')
    for n in m.F10_NOMES.values():
        (f10/'PCA_1D'/n).write_bytes(b'figure')
    results=tmp_path/'outputs'/'avaliacao_final_fake';(results/'PCA_1D').mkdir(parents=True)
    (results/'PCA_1D'/'metricas_f10.json').write_text(json.dumps({'limiar_mcc_oof_congelado':.01}))
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'OOF_ANTIGO',oof)
    monkeypatch.setattr(m,'F10_FIGURAS',f10)
    monkeypatch.setattr(m,'F10_RESULTADOS',results)
    dest=tmp_path/'outputs'/'comparacao_fake'
    pagina,resumo=m.html_comparacao(dest,oof,True)
    text=pagina.read_text()
    assert 'Nenhum resultado F10 foi recalculado' in text
    assert '01_curva_precision_recall.png' in text
    assert '04_matrizes_confusao_050_vs_tau_oof.png' in text
    assert resumo['f10_reexecutado'] is False
    assert resumo['pca_1d_figuras_oof_encontradas']==1
    assert (dest/'manifesto_comparacao.json').is_file()


def test_conferir_oof_bloqueia_f10_utilizado(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'OOF_ANTIGO',tmp_path/'antigo')
    origem=tmp_path/'antigo';origem.mkdir()
    (origem/'manifesto.json').write_text(json.dumps({'sha256_parquet':'x','sha256_folds':'y'}))
    novo=tmp_path/'novo';(novo/'vencedor'/'metricas').mkdir(parents=True)
    (novo/'manifesto.json').write_text(json.dumps({'status':'concluido_06_03','tipo':'ensaio','registros_desenvolvimento':166716,'sha256_parquet':'x','sha256_folds':'y'}))
    (novo/'vencedor'/'metricas'/'melhor_configuracao.json').write_text(json.dumps({'status':'selecionado_em_ensaio','teste_final_usado':True}))
    (novo/'configuracao.yaml').write_text('pca:\n  n_components: [1]\ngmm:\n  n_components: [4]\n  n_init: 5\ndados:\n  fracao_base: 0.1\n')
    with pytest.raises(RuntimeError,match='utilizado F10'):
        m.conferir_oof(novo,166716)
