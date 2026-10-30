"""Testes sem Parquet, sem F10 e sem treinamento real."""
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

ROOT_PACOTE=Path(__file__).resolve().parents[1]


def modulo(tmp_path,monkeypatch):
    s=tmp_path/'src';s.mkdir(exist_ok=True)
    (s/'__init__.py').write_text('')
    origem=Path(__file__).resolve().parents[1]/'PACOTE_PCA2D/finalizacao_automatica_pca2d.py'
    shutil.copy2(origem,s/'finalizacao_automatica_pca2d.py')
    # Os testes executados no repositorio real ja possuem a dependencia base.
    # Na fixture copiamos apenas a versao do pacote de referencia quando fornecida.
    base=Path.cwd()/'src/finalizacao_automatica_pca1d.py'
    if not base.is_file():
        base=ROOT_PACOTE/'tests/fixtures/finalizacao_automatica_pca1d.py'
    if base.is_file():
        shutil.copy2(base,s/'finalizacao_automatica_pca1d.py')
    else:
        pytest.skip('Nucleo PCA1D ausente.')
    for k in list(sys.modules):
        if k == 'src' or k.startswith('src.finalizacao_automatica_'):
            monkeypatch.delitem(sys.modules,k,raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    m=importlib.import_module('src.finalizacao_automatica_pca2d')
    yield m
    for k in list(sys.modules):
        if k == 'src' or k.startswith('src.finalizacao_automatica_'):
            monkeypatch.delitem(sys.modules,k,raising=False)


@pytest.fixture
def m(tmp_path,monkeypatch):
    yield from modulo(tmp_path,monkeypatch)


def test_histograma_2d_conserva_n(m):
    x=pd.DataFrame({'PC2':np.linspace(-3,5,201)})
    h=m.histograma_pc2(x,60)
    assert h['contagem_f10'].sum()==len(x)
    assert len(h)==60
    assert h.columns.to_list()==['pc2_inicio','pc2_fim','pc2_centro','contagem_f10']


def test_histograma_pc2_constante(m):
    h=m.histograma_pc2(pd.DataFrame({'PC2':np.ones(17)}))
    assert h.contagem_f10.sum()==17


def test_histograma_pc2_rejeita_nan(m):
    with pytest.raises(RuntimeError,match='PC2 invalida'):
        m.histograma_pc2(pd.DataFrame({'PC2':[0,np.nan]}))


def test_nomes_graficos_2d(m):
    nomes=[x for x,_ in m.GRAFICOS]
    assert len(nomes)==len(set(nomes))==15
    assert 'GMM08_mapa_risco_2d_comparativo.png' in nomes
    assert 'GMM07_sensibilidade_2d_comparativo.png' in nomes
    assert 'GMM01_densidade_conjunta_superficie_PCA2D.png' in nomes
    assert not any('1d' in n.lower() for n in nomes)


def test_pre01_2d_puro(m):
    assert m.histograma_pc2(pd.DataFrame({'PC2':np.arange(10)}))['contagem_f10'].sum()==10


def test_instalador_gera_roteamento_e_yaml_sem_retreinar(tmp_path):
    # Teste em REPOSITORIO FALSO, nao toca na raiz do projeto.
    script=ROOT_PACOTE/'INSTALAR_AUTOMACAO_PCA2D.py'
    p=tmp_path
    (p/'src').mkdir()
    (p/'PACOTE_PCA2D').mkdir()
    shutil.copy2(script,p/script.name)
    shutil.copy2(ROOT_PACOTE/'PACOTE_PCA2D/finalizacao_automatica_pca2d.py',
                 p/'PACOTE_PCA2D/finalizacao_automatica_pca2d.py')
    fixtures=ROOT_PACOTE/'tests/fixtures'
    shutil.copy2(fixtures/'orquestracao_oof_pca1d.py',p/'src/orquestracao_oof.py')
    shutil.copy2(fixtures/'finalizacao_automatica_pca1d.py',p/'src/finalizacao_automatica_pca1d.py')
    shutil.copy2(fixtures/'configuracao_referencia.yaml',p/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml')
    def run(*args):
        return subprocess.run(args,cwd=p,text=True,capture_output=True)
    assert run('git','init','-q').returncode==0
    assert run('git','-c','user.email=test@example.com','-c','user.name=Test','commit','--allow-empty','-m','base').returncode==0
    assert run('git','switch','-c','feature/pca2d-automacao-oof-f10').returncode==0
    old=(p/'src/orquestracao_oof.py').read_bytes()
    check=run(sys.executable,script.name,'--verificar')
    assert check.returncode==0, check.stderr+check.stdout
    assert old==(p/'src/orquestracao_oof.py').read_bytes()
    assert not (p/'src/finalizacao_automatica_pca2d.py').exists()
    install=run(sys.executable,script.name,'--instalar')
    assert install.returncode==0, install.stderr+install.stdout
    assert (p/'src/finalizacao_automatica_pca2d.py').is_file()
    config=yaml.safe_load((p/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text())
    assert config['pca']['n_components']==[2]
    source=(p/'src/orquestracao_oof.py').read_text()
    assert 'from src.finalizacao_automatica_pca2d import concluir_ensaio' in source
    assert 'from src.finalizacao_automatica_pca1d import concluir_ensaio' in source
    assert 'from src.finalizacao_automatica_pca2d import validar_automacao' in source
    second=run(sys.executable,script.name,'--instalar')
    assert second.returncode!=0, 'Instalacao repetida deveria ser bloqueada.'
