"""Passo 06.04: ensaios sem Parquet real (sem baixar/alterar dados)."""
import copy
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
import yaml

from src import preflight_oof as p

ROOT = Path(__file__).resolve().parents[1]


def cfg():
    return yaml.safe_load((ROOT/p.YAML_OFICIAL).read_text(encoding='utf-8-sig'))


def test_estimativa_ram_positiva_e_dependente_de_linhas():
    a=p._estimar_memoria(10000,700,22)
    b=p._estimar_memoria(20000,700,22)
    assert b['dataframe_estimado_bytes']==2*a['dataframe_estimado_bytes']
    assert b['memoria_recomendada_bytes']>a['memoria_recomendada_bytes']
    assert a['matriz_features_estimadas']==40


@pytest.mark.parametrize('args',[(0,200,22),(100,0,22),(100,150,1)])
def test_estimar_memoria_entrada_invalida(args):
    with pytest.raises(ValueError):p._estimar_memoria(*args)


def test_disponibilidade_ram_insuficiente_gera_aviso():
    res=p._avaliar_recursos(p.GIB,6*p.GIB,10*p.GIB,50*1024**2)
    assert res['avisos'] and res['bloqueios']==[]


def test_memoria_desconhecida_nao_significa_aprovada():
    res=p._avaliar_recursos(None,6*p.GIB,10*p.GIB,50*1024**2)
    assert res['avisos']


def test_disco_insuficiente_bloqueia():
    res=p._avaliar_recursos(64*p.GIB,p.GIB,10*p.GIB,50*1024**2)
    assert res['bloqueios']


def test_disco_suficiente_sem_bloqueio():
    res=p._avaliar_recursos(64*p.GIB,8*p.GIB,10*p.GIB,50*1024**2)
    assert res['bloqueios']==[]


def _fake_pyarrow(monkeypatch, linhas):
    class FalsoLote:
        def __init__(self, df):self.df=df;self.num_rows=len(df)
        def to_pandas(self):return self.df
    class FalsoParquet:
        def __init__(self, arq):
            self.metadata=types.SimpleNamespace(num_rows=len(linhas),num_columns=3)
        def iter_batches(self,batch_size=4096):
            for start in range(0,len(linhas),batch_size):
                yield FalsoLote(linhas.iloc[start:start+batch_size])
    parq=types.ModuleType('pyarrow.parquet')
    parq.ParquetFile=FalsoParquet
    fake=types.ModuleType('pyarrow')
    fake.__path__=[]
    fake.parquet=parq
    monkeypatch.setitem(sys.modules,'pyarrow',fake)
    monkeypatch.setitem(sys.modules,'pyarrow.parquet',parq)


def test_amostra_nao_contem_fold10(monkeypatch,tmp_path):
    df=pd.DataFrame({'NID_ALPHA':np.arange(250), 'TRANS_VALOR':np.arange(250),
                     'TARGET_OMEGA':np.zeros(250)})
    folds=np.tile(np.arange(1,11),25)
    _fake_pyarrow(monkeypatch,df)
    n,b=p._amostra_desenvolvimento(tmp_path/'dummy.parquet',folds,max_linhas=180)
    assert n==180 and b>0
    # Formato do lote incorpora 1/10 F10, excluido antes de adicionar ao DataFrame.


def test_amostra_limitada(monkeypatch,tmp_path):
    df=pd.DataFrame({'X':np.arange(600)})
    folds=np.repeat(np.arange(1,11),60)
    _fake_pyarrow(monkeypatch,df)
    n,_=p._amostra_desenvolvimento(tmp_path/'dummy.parquet',folds,max_linhas=120)
    assert n==120


def test_amostra_muito_pequena_falha():
    with pytest.raises(ValueError,match='pequena'):
        p._amostra_desenvolvimento('nao_existe',np.zeros(5),max_linhas=30)


def _montar_fake_projeto(tmp_path,monkeypatch,fracao=1.0, ram=64*p.GIB, disco=8*p.GIB):
    config=cfg()
    config['dados']['fracao_base']=fracao
    (tmp_path/p.YAML_OFICIAL).write_text(yaml.safe_dump(config),encoding='utf-8')
    original=tmp_path/'data'/'raw'/'dataset_raw.parquet'
    original.parent.mkdir(parents=True)
    original.write_bytes(b'dataset_sintetico')
    folds=np.repeat(np.arange(1,11),30)
    fold_file=tmp_path/'data'/'splits'/'fake'/'folds.npy'
    fold_file.parent.mkdir(parents=True)
    np.save(fold_file,folds)
    monkeypatch.setattr(p,'_identificar_folds',lambda *a: (folds,fold_file,'hash_teste'))
    df=pd.DataFrame({'NID_ALPHA':np.arange(300),
                     'TARGET_OMEGA':np.tile([0,1],150),
                     'TRANS_VALUE':np.arange(300,dtype=float)})
    _fake_pyarrow(monkeypatch,df)
    monkeypatch.setattr(p,'_recursos_sistema',lambda: (64*p.GIB,ram))
    monkeypatch.setattr(p,'_destino_armazenamento',lambda *a: (tmp_path/'outputs',disco))
    monkeypatch.setattr(p,'_estado_git',lambda *a:'7506620')
    return config


def test_diagnostico_realiza_apenas_leituras(tmp_path,monkeypatch):
    _montar_fake_projeto(tmp_path,monkeypatch)
    before={str(f.relative_to(tmp_path)):f.read_bytes() for f in tmp_path.rglob('*') if f.is_file()}
    diag=p.diagnosticar(tmp_path,amostra=180)
    after={str(f.relative_to(tmp_path)):f.read_bytes() for f in tmp_path.rglob('*') if f.is_file()}
    assert before==after
    assert diag['situacao']=='AGUARDANDO_REVISAO_MANUAL'
    assert not diag['iniciou_treinamento'] and not diag['teste_f10_utilizado_em_fit']
    assert diag['desenvolvimento_linhas']==270
    assert diag['f10_linhas']==30
    assert diag['quantidade_candidatos']==4 and diag['total_rodadas']==36
    assert diag['git_commit']=='7506620'


def test_preflight_impede_fracao_base_inferior_a_um(tmp_path,monkeypatch):
    _montar_fake_projeto(tmp_path,monkeypatch,fracao=.5)
    with pytest.raises(ValueError,match='fracao_base'):
        p.diagnosticar(tmp_path,amostra=180)


def test_preflight_bloqueia_disco_insuficiente(tmp_path,monkeypatch):
    _montar_fake_projeto(tmp_path,monkeypatch,disco=p.GIB)
    r=p.diagnosticar(tmp_path,amostra=180)
    assert r['situacao']=='BLOQUEADO' and r['bloqueios']


def test_preflight_avisa_ram_baixa(tmp_path,monkeypatch):
    _montar_fake_projeto(tmp_path,monkeypatch,ram=p.GIB)
    r=p.diagnosticar(tmp_path,amostra=180)
    assert r['situacao']=='AGUARDANDO_REVISAO_MANUAL'
    assert any('RAM' in a for a in r['avisos'])


def test_preflight_preserva_configuracao_e_nao_inicia_oof(tmp_path,monkeypatch):
    _montar_fake_projeto(tmp_path,monkeypatch)
    from src import orquestracao_oof as o
    monkeypatch.setattr(o,'executar_grade',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Nunca iniciar modelo')))
    r=p.main(['--amostra','180','--json'],raiz=tmp_path)
    assert r['total_rodadas']==36 and r['iniciou_treinamento'] is False
    assert not (tmp_path/'outputs').exists()


def test_preflight_obriga_yaml_oficial(tmp_path):
    with pytest.raises(FileNotFoundError,match='YAML oficial'):
        p.diagnosticar(tmp_path)


def test_json_serializavel(tmp_path,monkeypatch):
    _montar_fake_projeto(tmp_path,monkeypatch)
    r=p.diagnosticar(tmp_path,amostra=180)
    assert json.loads(json.dumps(r))['teste_f10_utilizado_em_fit'] is False


def test_formatacao_gib():
    assert p._formatar_gib(None)=='indisponivel'
    assert p._formatar_gib(2*p.GIB)=='2.00 GiB'
