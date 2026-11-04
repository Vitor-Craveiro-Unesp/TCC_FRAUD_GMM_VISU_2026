"""PRE01 OOF do t-SNE: testagem sem ajuste de modelos ou acesso F10."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import pre01_oof_tsne1d as mod
from src import pre01_v2_diagnostico as pre
from src import orquestracao_oof as orq
from src import preprocessing as prep


def origem_sintetica(tmp_path, monkeypatch):
    root = tmp_path
    exp = root / 'outputs' / 'ensaio_tsne_gmm_novo'
    vis = exp / 'vencedor/visualizacao'
    gallery = vis / 'galeria/index.html'
    gallery.parent.mkdir(parents=True)
    gallery.write_text('<!doctype html><html><body><h1>OOF TSNE</h1></body></html>', encoding='utf-8')
    oof = exp / 'vencedor/oof/predicoes.npz'
    oof.parent.mkdir(parents=True)
    ids = np.arange(9, dtype=np.int64)
    folds = np.arange(1, 10, dtype=np.uint8)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    np.savez(oof, linha_original=ids, fold=folds, y_true=y)
    cfg = {'reducao': {'metodo': 'TSNE'}, 'tsne': {'n_components': 1},
           'dados': {'arquivo': 'data/raw/base.parquet', 'coluna_alvo': 'TARGET_OMEGA'},
           'experimento': {'semente': 42}, 'gmm': {'n_components': [4], 'n_init': 5}}
    (exp / 'configuracao.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    (exp / 'manifesto.json').write_text(json.dumps({
        'status':'concluido_06_03', 'visualizacoes':'concluidas', 'tipo':'ensaio',
        'registros_desenvolvimento':9,'sha256_parquet':'HASHBASE','sha256_folds':'HASHFOLDS',
    }), encoding='utf-8')
    all_folds = np.concatenate([folds, [10, 10, 10]]).astype(np.uint8)
    base = pd.DataFrame({'TARGET_OMEGA': np.concatenate([y, [1,1,1]]),
                         'VAR': np.arange(12, dtype=float)})
    chamada = []
    def identificar_folds(_root, _cfg, _arq):
        chamada.append('validou_folds')
        return all_folds, root/'folds.npy', 'HASHBASE'
    def ler_desenvolvimento(_pq, indices):
        assert np.array_equal(indices, np.arange(9))
        chamada.append('leu_F1_F9')
        return base.iloc[indices].copy()
    def calcular(X, f, c):
        assert len(X) == 9 and set(f.tolist()) == set(range(1,10))
        chamada.append('calculou_PRE01_sem_F10')
        return (pd.DataFrame([{'feature':'VAR','media':.0}]),
                pd.DataFrame([{'fold':1,'desvio':1.0}]),
                pd.DataFrame([{'fold':1}]),
                pd.DataFrame([{'pc1':.4}]))
    def gerar_html(*args):
        assert args[-1]['f10_usado'] is False
        assert args[-1]['metodo_do_experimento']=='TSNE'
        assert args[-1]['k']==4 and args[-1]['n_init']==5
        return '<html>PRE01 completo: F1-F9</html>'
    monkeypatch.setattr(orq, '_identificar_folds', identificar_folds)
    monkeypatch.setattr(mod, 'hash_arquivo', lambda p: 'HASHFOLDS' if Path(p).name=='folds.npy' else __import__('hashlib').sha256(Path(p).read_bytes()).hexdigest())
    monkeypatch.setattr(orq, '_ler_desenvolvimento', ler_desenvolvimento)
    monkeypatch.setattr(prep, 'selecionar_features', lambda df, cfg: df[['VAR']])
    monkeypatch.setattr(pre, 'calc', calcular)
    monkeypatch.setattr(pre, 'build_html', gerar_html)
    return root, exp, gallery, chamada


def test_integracao_pre01_tsne1d_completa_e_idempotente(tmp_path, monkeypatch):
    root, exp, gallery, chamada = origem_sintetica(tmp_path, monkeypatch)
    assert mod.integrar_pre01_oof(exp, raiz=root) == gallery
    assert chamada == ['validou_folds','leu_F1_F9','calculou_PRE01_sem_F10']
    html = gallery.read_text(encoding='utf-8')
    assert html.count(pre.NEW_MARKER)==1
    assert 'Diagnóstico completo da padronização (HTML interativo)' in html
    assert 'oof/'+pre.FILENAME in html
    for caminho in (exp/'vencedor/visualizacao/oof', exp/'vencedor/visualizacao/galeria/oof'):
        assert {p.name for p in caminho.iterdir()} == set(mod.ARQUIVOS)
        assert 'F1-F9' in (caminho / pre.FILENAME).read_text(encoding='utf-8')
    assert not (exp/'vencedor/teste_final').exists()
    assert mod.integrar_pre01_oof(exp, raiz=root)==gallery
    assert chamada == ['validou_folds','leu_F1_F9','calculou_PRE01_sem_F10']
    assert gallery.read_text(encoding='utf-8') == html


def test_bloqueia_saida_parcial_sem_reescrever(tmp_path, monkeypatch):
    root, exp, gallery, chamada = origem_sintetica(tmp_path, monkeypatch)
    path=exp/'vencedor/visualizacao/oof'/pre.FILENAME
    path.parent.mkdir(parents=True)
    path.write_text('ANTIGO',encoding='utf-8')
    with pytest.raises(RuntimeError,match='parcialmente instalado'):
        mod.integrar_pre01_oof(exp,raiz=root)
    assert path.read_text(encoding='utf-8')=='ANTIGO'
    assert chamada==[]


def test_bloqueia_f10_entre_indices_antes_de_calcular(tmp_path,monkeypatch):
    root,exp,gallery,chamada=origem_sintetica(tmp_path,monkeypatch)
    oof=exp/'vencedor/oof/predicoes.npz'
    with np.load(oof) as z:
        ids=np.asarray(z['linha_original']); f=np.asarray(z['fold']); y=np.asarray(z['y_true'])
    ids[-1]=9
    np.savez(oof, linha_original=ids,fold=f,y_true=y)
    with pytest.raises(RuntimeError,match='F10 ou mudou de fold'):
        mod.integrar_pre01_oof(exp,raiz=root)
    assert 'calculou_PRE01_sem_F10' not in chamada
    assert not (exp/'vencedor/visualizacao/oof'/pre.FILENAME).exists()


def test_outras_dimensoes_impedidas_pelo_pacote_1d(tmp_path,monkeypatch):
    root, exp, _, _=origem_sintetica(tmp_path,monkeypatch)
    f=exp/'configuracao.yaml'
    c=yaml.safe_load(f.read_text());c['tsne']['n_components']=2
    f.write_text(yaml.safe_dump(c))
    with pytest.raises(RuntimeError,match='t-SNE 1D'):
        mod.integrar_pre01_oof(exp, raiz=root)


def test_hook_integrado_no_orquestrador():
    import inspect
    fonte=inspect.getsource(orq.main)
    assert "from src.pre01_oof_tsne1d import integrar_pre01_oof" in fonte
    assert "integrar_pre01_oof(final, raiz=raiz)" in fonte
