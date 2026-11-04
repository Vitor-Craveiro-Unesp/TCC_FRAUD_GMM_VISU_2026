"""Fase 4.4: fracao de F10 no YAML; testes sem dados reais ou fit."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest
import yaml

from src.amostragem_f10 import fracao_amostra_f10, selecionar_f10_estratificado
from src import finalizacao_automatica_pca1d as pca
from src import finalizacao_automatica_tsne1d as tsne
from src import orquestracao_oof as orq
from src import visualizacao_final_tsne1d as vis


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf-8-sig'))


def _sintetico():
    folds = np.array([1,2,3,4,5,6,7,8,9] + [10]*220, dtype=np.uint8)
    y = np.array([0]*9 + [0]*200 + [1]*20, dtype=np.uint8)
    return folds, y


@pytest.mark.parametrize('metodo,dim', [('PCA',1),('PCA',2),('PCA',3),
                                       ('TSNE',1),('TSNE',2),('TSNE',3)])
@pytest.mark.parametrize('fracao', [0.01,0.03,0.10,0.50,1.0])
def test_metodos_e_dimensoes_usam_mesma_regra_fracao(config, metodo,dim,fracao):
    """TSNE3D somente no contrato de amostragem; finalizador nao liberado."""
    c = deepcopy(config)
    c['reducao']['metodo'] = metodo
    c['pca']['n_components'] = [dim]
    c['tsne']['n_components'] = dim
    c['divisao']['fracao_amostra_f10'] = fracao
    assert orq.validar_operacao(c)
    folds,y = _sintetico()
    ids = selecionar_f10_estratificado(folds, y, c)
    assert np.all(folds[ids]==10)
    assert np.all(np.diff(ids)>0)
    assert int((y[ids]==0).sum()) == max(2,round(fracao*200))
    assert int((y[ids]==1).sum()) == max(2,round(fracao*20))
    assert np.array_equal(ids, selecionar_f10_estratificado(folds,y,c))


@pytest.mark.parametrize('invalido', [None, '0.10', False, True, 0.0,-0.01,1.01,float('nan'),float('inf')])
def test_rejeita_fracoes_invalidas_antes_de_treinar(config,invalido):
    config['divisao']['fracao_amostra_f10'] = invalido
    with pytest.raises(ValueError,match='fracao_amostra_f10'):
        fracao_amostra_f10(config)
    with pytest.raises(ValueError,match='fracao_amostra_f10'):
        orq.validar_operacao(config)


def test_falta_no_yaml_rejeitada_explicitamente(config):
    del config['divisao']['fracao_amostra_f10']
    with pytest.raises(ValueError,match='Informe divisao.fracao_amostra_f10'):
        orq.validar_operacao(config)


def test_10_pct_preserva_amostra_historica_deterministica(config):
    folds,y=_sintetico()
    config['divisao']['fracao_amostra_f10'] = .1
    ids=selecionar_f10_estratificado(folds,y,config)
    rng=np.random.default_rng(42)
    f10=np.flatnonzero(folds==10)
    esperado=np.sort(np.concatenate([
        rng.choice(f10[y[f10]==cl], size=max(2,round(.1*len(f10[y[f10]==cl]))),replace=False)
        for cl in (0,1)
    ]))
    assert np.array_equal(ids,esperado)


def test_pca_1d_cache_rejeita_frac_divergente_sem_reabrir_f10(config,monkeypatch,tmp_path):
    folds,y=_sintetico()
    config['dados']['fracao_base']=.01
    config['pca']['n_components']=[1]
    config['divisao']['fracao_amostra_f10']=.03
    monkeypatch.setattr(orq,'_ler_alvo',lambda *args: y)
    estado={'config':config, 'orig_folds':folds, 'parquet':tmp_path/'fake.parquet'}
    ids=pca.selecionar_amostra_f10(estado,tmp_path)
    t=tmp_path/'vencedor/teste_final';t.mkdir(parents=True)
    np.save(t/'indices_f10_amostra.npy',ids)
    (t/'manifesto_amostra_f10.json').write_text(json.dumps({'fracao':.03}),encoding='utf-8')
    monkeypatch.setattr(orq,'_ler_alvo',lambda *args: (_ for _ in ()).throw(AssertionError('releitura F10')))
    assert np.array_equal(pca.selecionar_amostra_f10(estado,tmp_path),ids)
    config['divisao']['fracao_amostra_f10']=.1
    with pytest.raises(RuntimeError,match='Fracao F10 salva'):
        pca.selecionar_amostra_f10(estado,tmp_path)


def test_tsne_1d_ler_e_gerar_fracao_configurada(config):
    config['reducao']['metodo']='TSNE'
    config['tsne']['n_components']=[1]
    config['divisao']['fracao_amostra_f10']=.03
    assert tsne.validar_automacao(config)
    folds,y=_sintetico()
    assert len(tsne._amostra_f10(folds,y,config=config))==8


def test_tsne_2d_tem_amostragem_oficial_e_1d_nao_e_2d(config):
    config['reducao']['metodo']='TSNE'
    config['tsne']['n_components']=[2]
    config['divisao']['fracao_amostra_f10']=.1
    assert len(selecionar_f10_estratificado(*_sintetico(),config))==22
    with pytest.raises(ValueError,match='2D/3D'):
        tsne.validar_automacao(config)
    from src.finalizacao_automatica_tsne2d import validar_automacao
    assert validar_automacao(config) is True


def test_textos_e_manifestos_usam_fracao_dinamica():
    from pathlib import Path
    p=Path(__file__).resolve().parents[1]/'src'
    for nome in ('finalizacao_automatica_pca1d.py', 'finalizacao_automatica_pca2d.py',
                 'finalizacao_automatica_pca3d.py', 'visualizacao_final_tsne1d.py',
                 'visualizacao_geometrica_final_tsne1d.py','visualizacao_final_completa_tsne1d.py'):
        txt=(p/nome).read_text(encoding='utf-8-sig')
        assert 'fracao_amostra_f10' in txt, nome
    assert 'FRACAO_F10 = 0.1' not in (p/'finalizacao_automatica_tsne1d.py').read_text()


def test_rotulos_pca_f10_sao_dinamicos_sem_estado_global(config,monkeypatch):
    from src import visualizacao_oof as vo
    config['divisao']['fracao_amostra_f10'] = .03
    old=vo._salvar
    with pca.rotulos_graficos_f10(config):
        assert vo._salvar is not old
    assert vo._salvar is old
