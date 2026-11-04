"""Fase 2: contratos seguros t-SNE 1D sem tocar base real ou F10 real."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest
import yaml

from src import finalizacao_automatica_tsne1d as fin
from src import orquestracao_oof as orq


@pytest.fixture
def cfg():
    p = Path(__file__).resolve().parents[1]/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    c = deepcopy(yaml.safe_load(p.read_text(encoding='utf-8-sig')))
    c['reducao']['metodo']='TSNE'
    c['tsne']['n_components']=[1]
    c['execucao']['finalizacao_pca1d_automatica']=True
    return c


@pytest.mark.parametrize('f', [0.01, 0.03, 0.1])
def test_fracoes_validas(cfg,f):
    cfg['dados']['fracao_base']=f
    assert fin.validar_automacao(cfg) is True


@pytest.mark.parametrize('f', [0.02,0.0,-1,1.0])
def test_fracoes_nao_homologadas(cfg,f):
    cfg['dados']['fracao_base']=f
    with pytest.raises(ValueError,match='1%, 3% ou 10%'):
        fin.validar_automacao(cfg)


def test_2d_bloqueado(cfg):
    cfg['tsne']['n_components']=[2]
    with pytest.raises(ValueError,match='2D/3D'):
        fin.validar_automacao(cfg)


def test_pca_bloqueado(cfg):
    cfg['reducao']['metodo']='PCA'
    with pytest.raises(ValueError, match='nao aceita PCA'):
        fin.validar_automacao(cfg)


def test_amostragem_10pct_por_classe_sem_entrar_em_f1f9(cfg):
    folds=np.repeat(np.arange(1,11,dtype=np.uint8), 100)
    y=np.tile(np.r_[np.ones(10,dtype=np.uint8),np.zeros(90,dtype=np.uint8)],10)
    ids=fin._amostra_f10(folds,y,config=cfg)
    assert np.all(folds[ids]==10)
    assert len(ids)==11 # 2 fraudes pelo minimo; 9 legitimas
    assert int(y[ids].sum())==2
    assert np.array_equal(ids, fin._amostra_f10(folds,y,config=cfg))


def test_amostragem_10pct_exact_stratified(cfg):
    folds=np.repeat(np.arange(1,11,dtype=np.uint8), 200)
    y=np.tile(np.r_[np.ones(50,dtype=np.uint8),np.zeros(150,dtype=np.uint8)],10)
    ids=fin._amostra_f10(folds,y,config=cfg)
    assert len(ids)==20 and y[ids].sum()==5


def test_planejar_tsne_1d_autorizado_sem_criar_saidas(tmp_path,cfg,capsys):
    cfg['dados']['fracao_base']=0.01
    (tmp_path/orq.ARQUIVO_CONFIG).write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    assert orq.main(['--planejar'], raiz=tmp_path) is None
    stdout=capsys.readouterr().out
    assert 'ETAPA FINAL TSNE 1D' in stdout
    assert '12 PNGs e PRE01 offline' in stdout
    assert not (tmp_path/'outputs').exists()


def test_retomada_f10_routing_tsne(monkeypatch,tmp_path,cfg):
    d=tmp_path/'outputs'/'ensaio_tsne_gmm_x'
    d.mkdir(parents=True)
    (d/'configuracao.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    chamados=[]
    monkeypatch.setattr(fin,'concluir_ensaio', lambda pasta,raiz=None: chamados.append((Path(pasta),Path(raiz))))
    orq.main(['--concluir-teste-final','--experimento',str(d)],raiz=tmp_path)
    assert chamados==[(d,tmp_path)]


def test_modelo_e_f10_nao_abertos_se_experimento_fora_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(fin,'verificar_oof_concluido_tsne',lambda *_: (_ for _ in ()).throw(AssertionError('nao deve ler')))
    with pytest.raises(ValueError,match='dentro de outputs'):
        fin.concluir_ensaio(tmp_path/'fora',raiz=tmp_path)


def test_fluxo_modelo_antes_f10_sem_duplicar(monkeypatch, tmp_path, cfg):
    """Integra o F10 com fakes estritos; nao depende de openTSNE/parquet real."""
    from types import SimpleNamespace
    from src import finalizacao_automatica_pca1d as pca
    base=tmp_path
    exp=base/'outputs'/'ensaio_tsne_gmm_sintetico'
    (exp/'vencedor'/'oof').mkdir(parents=True)
    (exp/'vencedor'/'teste_final').mkdir()
    (exp/'configuracao.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    (exp/'manifesto.json').write_text(json.dumps({'sha256_parquet':'HASH_PQ', 'sha256_folds':'HASH_FOLD'}),encoding='utf-8')
    ids_dev=np.arange(0,90,dtype=np.int64)
    folds=np.repeat(np.arange(1,11,dtype=np.uint8),10)
    y=np.tile(np.r_[np.ones(2,dtype=np.uint8),np.zeros(8,dtype=np.uint8)],10)
    import pandas as pd
    df_dev=pd.DataFrame({'TARGET_OMEGA':y[ids_dev]})
    np.savez_compressed(exp/'vencedor/oof/predicoes.npz', linha_original=ids_dev,
                        fold=folds[ids_dev], y_true=y[ids_dev])
    escolha={'id_candidato':'abc','limiar_mcc_oof':0.25}
    pre=SimpleNamespace(configuracao=cfg,escolha_oof=escolha,
                        proveniencia={'sha256_parquet':'HASH_PQ','sha256_folds':'HASH_FOLD'})
    pacote=SimpleNamespace(configuracao=cfg,escolha_oof=escolha,
                            proveniencia=pre.proveniencia)
    ordem=[]
    monkeypatch.setattr(fin,'verificar_oof_concluido_tsne',lambda _: pre)
    monkeypatch.setattr(orq,'_identificar_folds',lambda *args: (folds,base/'folds.npy','HASH_PQ'))
    from hashlib import sha256
    def hash_mock(p):
        p=Path(p)
        return 'HASH_FOLD' if p.name in ('folds.npy', 'predicoes.npz') else sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(fin,'hash_arquivo',hash_mock)
    monkeypatch.setattr(orq,'_ler_desenvolvimento', lambda *args: df_dev)
    monkeypatch.setattr(orq,'_ler_alvo',lambda *args: (ordem.append('LER F10'),y)[1])

    def refit(*args,**kwargs):
        assert 'LER F10' not in ordem
        ordem.append('REFIT F1F9')
        return SimpleNamespace(pacote=pacote)
    def persiste(resultado,pasta):
        ordem.append('MODELO SALVO')
        pasta.mkdir()
        (pasta/'modelo_final_f1_f9.joblib').write_bytes(b'MODELO SINTETICO')
        return pasta/'modelo_final_f1_f9.joblib',pasta/'manifesto_modelo_final.json'
    monkeypatch.setattr(fin,'executar_ajuste_final_tsne_em_memoria',refit)
    monkeypatch.setattr(fin,'persistir_finalizacao_controlada',persiste)
    monkeypatch.setattr(pca,'ler_linhas_f10',lambda _,ids: pd.DataFrame({'TARGET_OMEGA':y[ids]}))

    def avalia(pacote_recebido, df, folds_f10, ids, *, retornar_coordenadas=False):
        assert retornar_coordenadas is True
        assert ordem==['REFIT F1F9','MODELO SALVO','LER F10']
        assert np.all(folds_f10==10)
        return (pd.DataFrame({'linha_original':ids,'prob_fraude':np.repeat(.5,len(ids))}),
                {'ap':.2,'mcc':.1},
                pd.DataFrame({'linha_original':ids,'TSNE1':np.arange(len(ids),dtype=float)}))
    monkeypatch.setattr(fin,'avaliar_f10_em_memoria',avalia)
    assert fin.concluir_ensaio(exp,raiz=base)==exp
    assert (exp/'vencedor/teste_final/metricas_f10.json').is_file()
    coord = pd.read_csv(exp/'vencedor/teste_final/coordenadas_TSNE1_f10.csv')
    assert np.array_equal(coord['linha_original'].to_numpy(), fin._amostra_f10(folds,y,config=cfg))
    status=json.loads((exp/'vencedor/teste_final/manifesto_finalizacao_tsne.json').read_text())
    assert status['galeria_f10_concluida'] is False
    assert ordem==['REFIT F1F9','MODELO SALVO','LER F10']
    assert fin.concluir_ensaio(exp,raiz=base)==exp
    assert ordem==['REFIT F1F9','MODELO SALVO','LER F10']
