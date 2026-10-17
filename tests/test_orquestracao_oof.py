"""Testes sintéticos 06.03: arquivos/cache, retomada, grade, F10 e métricas."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.orquestracao_oof import (
    validar_operacao, _chave, _chave_execucao, _salvar_npz, _ler_npz,
    _verificar_tabela, _salvar_json, _selecionar_posicoes,
    _abrir_execucao, _finalizar, executar_grade, planejar,
)

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def config():
    cfg=yaml.safe_load((ROOT/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf-8'))
    cfg['reducao']['metodo']='PCA'
    cfg['pca']['n_components']=[2]
    cfg['gmm']['n_components']=[1]
    cfg['dados']['fracao_base']=1.0
    cfg['amostragem']['modo']='NORMAL'
    cfg['execucao']['n_jobs']=1
    return cfg

@pytest.fixture
def dados():
    rng=np.random.default_rng(42)
    n=30*10
    f=np.repeat(np.arange(1,11),30).astype(np.uint16)
    y=np.tile(np.r_[np.zeros(24,dtype=np.uint8),np.ones(6,dtype=np.uint8)],10)
    df=pd.DataFrame({
        'NID_ALPHA':np.arange(n),
        'TARGET_OMEGA':y,
        'SEND_GENDER_BE': np.where(rng.random(n)<0.5,'F','M'),
        'SEND_JOB_FEWF':np.where(rng.random(n)<0.5,'a','b'),
        'TRANS_YEAR_BE':np.where(rng.random(n)<0.5,2019,2020),
        'TRANS_VALUE':rng.normal(size=n),
        'TRANS_WEEK_OHEWI':rng.choice(['Mon','Tue','Wed'],size=n),
    })
    idx=np.flatnonzero(f!=10)
    return df.iloc[idx].reset_index(drop=True),f[idx],idx,f,y

@pytest.mark.parametrize('campo', ['usar_cache','retomar','proteger_resultados'])
def test_booleanos_invalidos_sao_rejeitados(config,campo):
    config['execucao'][campo]='true'
    with pytest.raises(ValueError,match=campo):validar_operacao(config)

@pytest.mark.parametrize('jobs', [0,-2,True,'-1'])
def test_jobs_invalidos(config,jobs):
    config['execucao']['n_jobs']=jobs
    with pytest.raises(ValueError,match='n_jobs'):validar_operacao(config)

@pytest.mark.parametrize('fracao', [0,-0.1,1.5,True])
def test_fracao_invalida(config,fracao):
    config['dados']['fracao_base']=fracao
    with pytest.raises(ValueError,match='fracao_base'):validar_operacao(config)

def test_retomada_exige_cache(config):
    config['execucao']['usar_cache']=False
    with pytest.raises(ValueError,match='cache'):validar_operacao(config)

def test_diretorio_explicito_exige_retomada(config):
    config['execucao']['retomar']=False
    config['execucao']['diretorio_retomada']='qualquer'
    with pytest.raises(ValueError,match='retomar'):validar_operacao(config)

def test_protecao_obrigatoria(config):
    config['execucao']['proteger_resultados']=False
    with pytest.raises(ValueError,match='proteger_resultados'):validar_operacao(config)

def test_operacao_oficial(config):
    assert validar_operacao(config)['retomar'] is True


def test_grade_real_do_yaml_total_quatro():
    cfg=yaml.safe_load((ROOT/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf8'))
    cfg['reducao']['metodo']='PCA'
    cfg['pca']['n_components']=[2,3]
    cfg['gmm']['n_components']=[1,2]
    cfg['amostragem']['modo']='NORMAL'
    cfg['risco_historico']['suavizacao']=0.0
    from src.validacao_oof import grade_candidatos_oof
    c=list(grade_candidatos_oof(cfg))
    assert len(c)==4
    assert {v['gmm']['n_components'] for v in c}=={1,2}
    assert {v['reducao']['n_components'] for v in c}=={2,3}


def test_chave_deterministica(config):
    assert _chave(config)==_chave(deepcopy(config))
    assert _chave_execucao(config,'a','b',ensaio=True,por_fold=10)!=_chave_execucao(config,'a','b',ensaio=False,por_fold=10)


def test_selecao_exclui_f10(dados,config):
    df,fdev,orig,f,y=dados
    pos=_selecionar_posicoes(f,y,config,ensaio=True,por_fold=10)
    assert len(pos)==90
    assert set(f[pos])==set(range(1,10))
    assert (f[pos]!=10).all()
    for k in range(1,10):
        assert np.any(y[pos[f[pos]==k]]==1)
        assert np.any(y[pos[f[pos]==k]]==0)


def test_selecao_com_fracao(dados,config):
    df,fdev,orig,f,y=dados
    config['dados']['fracao_base']=0.5
    pos=_selecionar_posicoes(f,y,config,ensaio=False)
    assert len(pos)==9*15
    assert (f[pos]<10).all()


def test_cache_roundtrip(tmp_path):
    df=pd.DataFrame({'linha_original':[3,6],'fold':[2,2],'y_true':[0,1], 'prob_fraude':[0.1,0.9]})
    arq=tmp_path/'a.npz'
    _salvar_npz(arq,df,chave_execucao='A',id_candidato='B',fold=2)
    pd.testing.assert_frame_equal(df.astype({'linha_original':'int64','fold':'uint8','y_true':'uint8'}),
                                  _ler_npz(arq,chave_execucao='A',id_candidato='B',fold=2))
    assert not (tmp_path/'a.npz.incompleto').exists()
    with pytest.raises(RuntimeError,match='incompatível|incompativel'):
        _ler_npz(arq,chave_execucao='X',id_candidato='B',fold=2)


def test_verificacao_captura_alteracao_de_rotulo(dados):
    df,fdev,orig,f,y=dados
    t=pd.DataFrame({'linha_original':np.array([0,1]), 'fold':fdev[:2],
        'y_true':[1,0], 'prob_fraude':[0.5,0.3]})
    with pytest.raises(RuntimeError,match='Rotulos'):
        _verificar_tabela(t,indices_esperados=np.array([0,1]),folds_trabalho=fdev,y_trabalho=df['TARGET_OMEGA'].to_numpy())


def test_criar_e_retomar_manifesto(tmp_path,config):
    chave='ab123'
    pasta,m,retomada=_abrir_execucao(tmp_path,config,chave,ensaio=True)
    assert not retomada
    assert (pasta/'configuracao.yaml').exists()
    assert (pasta/'oof'/'cache').exists()
    assert (pasta/'teste_final'/'graficos_gerados').exists()
    pasta2,m2,ret2=_abrir_execucao(tmp_path,config,chave,ensaio=True)
    assert not retomada and ret2 and pasta2==pasta
    assert m2['chave_execucao']==chave


def test_retomada_incompativel_rejeitada(tmp_path,config):
    pasta,m,_=_abrir_execucao(tmp_path,config,'OK',ensaio=True)
    config['execucao']['diretorio_retomada']=str(pasta)
    with pytest.raises(RuntimeError,match='incompativel|incompatível'):
        _abrir_execucao(tmp_path,config,'NAO',ensaio=True)


def test_resultado_final_nao_e_sobrescrito(tmp_path,config):
    pasta,m,_=_abrir_execucao(tmp_path,config,'AA',ensaio=True)
    final=_finalizar(pasta,m,config)
    assert final.exists() and not pasta.exists()
    assert final.name.startswith('ensaio_pca_gmm_')
    assert json.loads((final/'manifesto.json').read_text())['status']=='concluido_06_03'


def test_planejar_nao_cria_resultados(config,capsys):
    assert planejar(config)==1
    saida=capsys.readouterr().out
    assert 'F10' in saida and 'Nenhum modelo' in saida


def test_execucao_grid_integral_e_retomada(tmp_path,config,dados,monkeypatch):
    df,fdev,orig,f,y=dados
    pasta,manifest,_=_abrir_execucao(tmp_path,config,'TESTE123',ensaio=False)
    result=executar_grade(df,fdev,orig,f,config,pasta,manifest,ensaio=False)
    assert result['status']=='selecionado_em_oof_integral'
    assert result['teste_final_usado'] is False
    assert len(list((pasta/'oof'/'cache').glob('*.npz')))==9
    assert len(list((pasta/'oof'/'predicoes').glob('*.npz')))==1
    assert len(list((pasta/'oof'/'metricas').glob('*.json')))==1
    assert len(pd.read_csv(pasta/'oof'/'resultados_grade.csv'))==1
    assert not list((pasta/'teste_final').rglob('*.npz'))

    # Os resultados existentes precisam ser reaproveitados sem tocar o motor.
    from src import orquestracao_oof as o
    monkeypatch.setattr(o,'prever_fold_oof',lambda *a,**k: (_ for _ in ()).throw(AssertionError('Nao deve treinar!')))
    segundo=executar_grade(df,fdev,orig,f,config,pasta,manifest,ensaio=False)
    assert segundo['id_candidato']==result['id_candidato']


def test_execucao_amostral_nao_gera_metricas_oficiais(tmp_path,config,dados):
    df,fdev,orig,f,y=dados
    selecionados=np.flatnonzero(np.arange(len(df))%2==0)
    parcial=df.iloc[selecionados].reset_index(drop=True)
    pasta,m,_=_abrir_execucao(tmp_path,config,'AMOSTRA',ensaio=True)
    result=executar_grade(parcial,fdev[selecionados],orig[selecionados],f,config,pasta,m,ensaio=True)
    assert result['status']=='selecionado_em_ensaio'
    meta=json.loads(next((pasta/'oof'/'metricas').glob('*.json')).read_text())
    assert meta['tipo_avaliacao']=='OOF_AMOSTRAL_NAO_OFICIAL'


def test_cache_corrompido_nao_e_silenciado(tmp_path,config,dados):
    df,fdev,orig,f,y=dados
    pasta,m,_=_abrir_execucao(tmp_path,config,'TESTE_CORROMPIDO',ensaio=False)
    from src.validacao_oof import grade_candidatos_oof
    cid=_chave(next(grade_candidatos_oof(config)))[:14]
    cache=pasta/'oof'/'cache'/f'{cid}_F01.npz'
    cache.write_text('nao e npz')
    with pytest.raises(Exception):
        executar_grade(df,fdev,orig,f,config,pasta,m,ensaio=False)
    assert cache.read_text()=='nao e npz'


def test_retomada_apos_interrupcao_no_meio_de_um_candidato(tmp_path,config,dados,monkeypatch):
    df,fdev,orig,f,y=dados
    pasta,manifest,_=_abrir_execucao(tmp_path,config,'QUEDA',ensaio=False)
    from src import orquestracao_oof as o
    original=o.prever_fold_oof
    visitas=[]
    def falhar_no_segundo_fold(*args):
        k=args[-1]
        visitas.append(k)
        if k==2: raise RuntimeError('Interrupcao simulada')
        return original(*args)
    monkeypatch.setattr(o,'prever_fold_oof',falhar_no_segundo_fold)
    with pytest.raises(RuntimeError,match='Interrupcao simulada'):
        executar_grade(df,fdev,orig,f,config,pasta,manifest,ensaio=False)
    assert visitas==[1,2]
    assert len(list((pasta/'oof'/'cache').glob('*.npz')))==1

    def retomar_sem_recalcular_f1(*args):
        if args[-1]==1: raise AssertionError('F1 ja existia no cache')
        return original(*args)
    monkeypatch.setattr(o,'prever_fold_oof',retomar_sem_recalcular_f1)
    resultado=executar_grade(df,fdev,orig,f,config,pasta,manifest,ensaio=False)
    assert resultado['status']=='selecionado_em_oof_integral'
    assert len(list((pasta/'oof'/'cache').glob('*.npz')))==9
