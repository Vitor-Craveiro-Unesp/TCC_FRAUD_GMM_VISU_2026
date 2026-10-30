"""Testes sinteticos sem treino, Parquet ou acesso ao F10 real."""
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def m(tmp_path,monkeypatch):
    src=tmp_path/'src'; src.mkdir()
    (src/'__init__.py').write_text('')
    shutil.copy2(ROOT/'PACOTE_PCA3D/finalizacao_automatica_pca3d.py',
                 src/'finalizacao_automatica_pca3d.py')
    # Biblioteca comum PCA 1D congelada
    src_origem=Path.cwd()/'src/finalizacao_automatica_pca1d.py'
    if not src_origem.is_file():src_origem=ROOT/'tests/fixtures_pca3d/finalizacao_automatica_pca1d.py'
    shutil.copy2(src_origem,src/'finalizacao_automatica_pca1d.py')
    for nome in list(sys.modules):
        if nome=='src' or nome.startswith('src.finalizacao_automatica_'):
            monkeypatch.delitem(sys.modules,nome,raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    mod=importlib.import_module('src.finalizacao_automatica_pca3d')
    yield mod
    for nome in list(sys.modules):
        if nome=='src' or nome.startswith('src.finalizacao_automatica_'):
            monkeypatch.delitem(sys.modules,nome,raising=False)


def test_hist_pc2_e_pc3_preservam_total(m):
    z=pd.DataFrame({'PC2':np.linspace(-4,6,201),'PC3':np.linspace(-.1,.1,201)})
    for nome in ('PC2','PC3'):
        h=m.histograma_pc(z,nome)
        assert len(h)==60
        assert h['contagem_f10'].sum()==201
        assert f'{nome.lower()}_centro' in h.columns


def test_hist_pc3_constante(m):
    assert m.histograma_pc(pd.DataFrame({'PC3':np.ones(17)}),'PC3')['contagem_f10'].sum()==17


def test_hist_rejeita_nan_e_dimensao(m):
    with pytest.raises(RuntimeError,match='PC3 invalida'):
        m.histograma_pc(pd.DataFrame({'PC3':[0,np.nan]}),'PC3')
    with pytest.raises(RuntimeError,match='Componente PCA invalida'):
        m.histograma_pc(pd.DataFrame({'PCA4':[1,2]}),'PCA4')


def test_galeria_18_figuras_e_5_html(m):
    nomes=[k for k,_ in m.GRAFICOS]
    htmls=[k for k,_ in m.HTMLS_INTERATIVOS]
    assert len(nomes)==len(set(nomes))==18
    assert len(htmls)==len(set(htmls))==5
    assert nomes[:6]==[k for k,_ in m.comum.GRAFICOS[:6]]
    assert 'PCA_OBSERVACOES_PC1_PC2_PC3.png' in nomes
    assert 'GMM08_mapa_risco_3d_comparativo.png' in nomes
    assert 'GMM07_sensibilidade_3d_comparativo_INTERATIVA.html' in htmls
    assert 'GMM01_geometria_gaussianas_FULL_3d_95_INTERATIVA.html' in htmls


def test_correcao_legendas_puramente_visual(m):
    from matplotlib.figure import Figure
    f=Figure()
    f.text(.5,.5,'Observações do desenvolvimento; F10 não utilizado')
    m._normalizar_rotulos_figura(f)
    texto=f.texts[0].get_text()
    assert 'Observações F10' in texto
    assert 'não utilizado' not in texto


def test_html_corrige_origem_e_preserva_estrutura(tmp_path,m):
    p=tmp_path/'GMM07_sensibilidade_3d_comparativo_INTERATIVA.html'
    p.write_text('<html><h2>Observações do desenvolvimento</h2><p>F10 não utilizado.</p></html>')
    m._normalizar_html_interativo(tmp_path)
    h=p.read_text()
    assert '<html>' in h and 'Observações F10' in h and 'F10 exploratório' in h


def test_instalador_nao_sobrepoe_1d_2d(tmp_path):
    p=tmp_path
    (p/'src').mkdir(); (p/'PACOTE_PCA3D').mkdir()
    shutil.copy2(ROOT/'INSTALAR_AUTOMACAO_PCA3D.py',p/'INSTALAR_AUTOMACAO_PCA3D.py')
    shutil.copy2(ROOT/'PACOTE_PCA3D/finalizacao_automatica_pca3d.py',
                 p/'PACOTE_PCA3D/finalizacao_automatica_pca3d.py')
    fixtures=ROOT/'tests/fixtures_pca3d'
    orig=(fixtures/'orquestracao_oof_pca2d.py').read_text()
    cfg=(fixtures/'configuracao_referencia_pca2d.yaml').read_text()
    (p/'src/orquestracao_oof.py').write_text(orig)
    (p/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').write_text(cfg)
    shutil.copy2(fixtures/'finalizacao_automatica_pca1d.py',p/'src/finalizacao_automatica_pca1d.py')
    shutil.copy2(fixtures/'finalizacao_automatica_pca2d.py',p/'src/finalizacao_automatica_pca2d.py')
    def run(*a):return subprocess.run(a,cwd=p,text=True,capture_output=True)
    assert run('git','init','-q').returncode==0
    assert run('git','-c','user.name=Test','-c','user.email=test@example.com','commit','--allow-empty','-m','base').returncode==0
    assert run('git','switch','-c','feature/pca3d-automacao-oof-f10').returncode==0
    originals=[(p/k).read_bytes() for k in ('src/finalizacao_automatica_pca1d.py','src/finalizacao_automatica_pca2d.py')]
    teste=run(sys.executable,'INSTALAR_AUTOMACAO_PCA3D.py','--verificar')
    assert teste.returncode==0,teste.stdout+teste.stderr
    assert not (p/'src/finalizacao_automatica_pca3d.py').exists()
    atual=run(sys.executable,'INSTALAR_AUTOMACAO_PCA3D.py','--instalar')
    assert atual.returncode==0,atual.stdout+atual.stderr
    after=[(p/k).read_bytes() for k in ('src/finalizacao_automatica_pca1d.py','src/finalizacao_automatica_pca2d.py')]
    assert after==originals
    arq=(p/'src/orquestracao_oof.py').read_text()
    assert 'from src.finalizacao_automatica_pca3d import concluir_ensaio' in arq
    assert 'from src.finalizacao_automatica_pca3d import validar_automacao' in arq
    c=yaml.safe_load((p/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text())
    assert c['pca']['n_components']==[3]
    segundo=run(sys.executable,'INSTALAR_AUTOMACAO_PCA3D.py','--instalar')
    assert segundo.returncode!=0


def test_galeria_gera_links_e_manifesto_sem_dados_reais(tmp_path,m,monkeypatch):
    from types import SimpleNamespace
    folder=tmp_path/'outputs/ensaio_pca_gmm_sintetico'
    d=folder/'vencedor/teste_final'
    figs=d/'graficos_gerados';figs.mkdir(parents=True)
    for nome,_ in m.GRAFICOS:
        (figs/nome).write_bytes(b'png_de_teste_sintetico')
    for nome,_ in m.HTMLS_INTERATIVOS:
        (figs/nome).write_text('<html>teste sintetico</html>',encoding='utf-8')
    pre=d/'diagnosticos/PRE01_F10/PRE01_F10_diagnostico_padronizacao_completo.html'
    pre.parent.mkdir(parents=True)
    pre.write_text('<html>PRE01</html>',encoding='utf-8')
    for path in (d/'modelo_final_f1_f9/modelo_final_f1_f9.joblib',d/'predicoes_f10.csv',d/'metricas_f10.json'):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'dados_sinteticos')
    pacote=SimpleNamespace(limiar_oof=.07,n_treino_original=900)
    metricas=dict(ap=.04,mcc_limiar_oof=.12,logloss=.1,score_composto_limiar_oof=.5)
    m.galeria_final(folder,np.arange(50),pacote,metricas)
    html=(d/'galeria/index.html').read_text(encoding='utf-8')
    assert 'PRE01 F10 interativo' in html
    assert 'coordenadas_PC1_PC2_PC3_f10.csv' in html
    assert 'GMM08_mapa_risco_3d_comparativo_INTERATIVA.html' in html
    assert 'GMM01_geometria_gaussianas_FULL_3d_95_INTERATIVA.html' in html
    meta=json.loads((d/'manifesto_teste_completo_pca3d_f10.json').read_text())
    assert meta['limiar_otimizado_no_f10'] is False
    assert len(meta['graficos_sha256'])==18
    assert len(meta['interativos_sha256'])==5
    with pytest.raises(FileExistsError):
        m.galeria_final(folder,np.arange(50),pacote,metricas)
