"""P08 — testes sinteticos dos Biplots PCA; nenhum ensaio/F10 utilizado."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.decomposition import PCA

from src import visualizacao_pca_biplot as b
from src import visualizacao_geometrica as geo
from src import visualizacao_integrada as integ


@pytest.fixture
def config():
    p = Path(__file__).resolve().parents[1] / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    cfg = yaml.safe_load(p.read_text(encoding='utf-8-sig'))
    cfg['visualizacao']['padrao']['dpi'] = 55
    cfg['visualizacao']['geometria']['pontos_maximos_amostra'] = 70
    return cfg


def dado(dim, *, whiten=False, constante=False):
    rng = np.random.default_rng(37)
    n = 155
    base = rng.normal(size=(n, 5))
    base[:, 1] = .7 * base[:, 0] + .3 * base[:, 1]
    if constante:
        base[:, 4] = 3.0
    x = pd.DataFrame(base, columns=['FeatureA','FeatureB','FeatureC','FeatureD','FeatureE'],
                     index=pd.Index(np.arange(300, 300+n), name='linha'))
    modelo = PCA(n_components=dim, svd_solver='full', whiten=whiten).fit(x.to_numpy(dtype=float))
    coords = pd.DataFrame(modelo.transform(x.to_numpy(dtype=float)),
                          columns=[f'PC{i+1}' for i in range(dim)],index=x.index)
    red = SimpleNamespace(modelo_=modelo, colunas_=list(x.columns))
    y = np.array([0]*130 + [1]*25)
    return {'pre':x,'coord':coords,'redutor':red,'y':y}


@pytest.mark.parametrize('dim', [1,2,3])
@pytest.mark.parametrize('whiten', [False,True])
def test_correlacoes_reais_equivalentes_pearson(dim, whiten):
    dados = dado(dim,whiten=whiten,constante=True)
    t = b.tabela_correlacoes(dados)
    assert len(t) == 5
    assert t['constante'].sum() == 1
    assert np.allclose(t.loc[t['constante'],[f'corr_PC{i+1}' for i in range(dim)]], 0)
    for j in range(4):
        for i in range(dim):
            observado = np.corrcoef(dados['pre'].iloc[:,j],dados['coord'].iloc[:,i])[0,1]
            assert np.isclose(t.loc[j,f'corr_PC{i+1}'],observado,atol=2e-6)


@pytest.mark.parametrize('dim', [1,2,3])
def test_gera_png_csv_desenvolvimento_dimensao_correta(tmp_path, config, dim):
    d = dado(dim)
    info = b.gerar(config,d,tmp_path,'png',55,True)
    assert info['dimensoes'] == dim
    assert info['n_obs_total'] == 155
    assert info['n_obs_amostra'] == 70
    assert info['origem'] == 'desenvolvimento_F1_F9'
    assert info['f10_utilizado'] is False
    png = tmp_path/f'PCA_BIPLOT_{dim}D.png'
    assert png.is_file() and png.stat().st_size>1000
    csv = pd.read_csv(tmp_path/f'PCA_BIPLOT_{dim}D_correlacoes.csv',encoding='utf-8-sig')
    assert len(csv)==d['pre'].shape[1]
    sampled = pd.read_csv(tmp_path/f'PCA_BIPLOT_{dim}D_scores_amostra.csv',encoding='utf-8-sig')
    assert len(sampled)==70
    assert list(sampled.columns[:2]) == ['indice_linha_desenvolvimento','PC1']
    assert set(sampled['classe_real']) == {0,1}
    # Arquivos de ensaios anteriores nao sao sobrescritos.
    conteudo = png.read_bytes()
    b.gerar(config,d,tmp_path,'png',55,True)
    assert png.read_bytes()==conteudo


@pytest.mark.parametrize('erro', ['eixo','indice','redutor','rotulo'])
def test_recusa_entrada_incompatível(erro):
    d = dado(2)
    if erro=='eixo': d['coord'].columns = ['PC1','PC4']
    if erro=='indice': d['coord'].index += 1
    if erro=='redutor': d.pop('redutor')
    if erro=='rotulo': d['y']=np.array([0]*155)
    if erro=='rotulo': d['y'][0]=3
    with pytest.raises(ValueError):
        b.tabela_correlacoes(d)


def test_despachante_yml_e_galeria(config, tmp_path, monkeypatch):
    assert geo._ativo(config, 'pca_biplot') is True
    assert 'pca_biplot' in geo.TIPOS
    assert integ.GEO_PREFIXOS['pca_biplot']=='PCA_BIPLOT_'
    # TSNE nunca inclui o Biplot; PCA sim.
    from src import visualizacao_oof as oof
    cfg = deepcopy(config)
    cfg['visualizacao']['graficos']['pca_biplot']['habilitar']=False
    assert geo._ativo(cfg, 'pca_biplot') is False
    d = dado(2)
    b.gerar(config,d,tmp_path,'png',55,True)
    assert (tmp_path/'PCA_BIPLOT_2D.png').is_file()


def test_numero_indefinido_nao_muda_numeracao_aprovada():
    assert 'pca_biplot' not in geo.NUM
    assert geo.NUM['pca_cotovelo']==11
    assert geo.NUM['pca_projecoes']==12


def test_galeria_07_03_inclui_biplot_2d_sem_retreinar(tmp_path, config, monkeypatch):
    import json
    from src import visualizacao_oof as oof
    cfg = deepcopy(config)
    for chave in oof.ARQUIVOS:
        cfg['visualizacao']['graficos'][chave]['habilitar'] = chave == 'curva_pr'
    for chave in geo.TIPOS:
        cfg['visualizacao']['graficos'][chave]['habilitar'] = chave == 'pca_biplot'
    escolhido = {'id_candidato':'teste_sintetico','score_oof':.42,
                 'teste_final_usado':False,
                 'configuracao':{'metodo':'PCA','reducao':{'n_components':2}}}
    exp = tmp_path/'ensaio_sintetico'; exp.mkdir()
    (exp/'manifesto.json').write_text(json.dumps({'status':'concluido_06_03','tipo':'ensaio'}))
    a=exp/'vencedor'/'visualizacao'/'oof';bpath=exp/'vencedor'/'visualizacao'/'geometria_desenvolvimento'
    a.mkdir(parents=True);bpath.mkdir(parents=True)
    (a/'01_curva_precision_recall.png').write_bytes(b'png exemplo')
    (bpath/'PCA_BIPLOT_2D.png').write_bytes(b'png biplot')
    (a/'manifesto_graficos_oof_07_01.json').write_text(json.dumps({'id_vencedor':'teste_sintetico','usa_f10':False}))
    (bpath/'manifesto_geometria_07_02.json').write_text(json.dumps({'id_vencedor':'teste_sintetico','f10_utilizado':False,'origem':'desenvolvimento_F1_F9'}))
    monkeypatch.setattr(oof, '_ler_resultados', lambda *_: (cfg,None,escolhido,np.array([]),np.array([])))
    monkeypatch.setattr(geo, 'executar', lambda *_: pytest.fail('Nao deveria retreinar'))
    resumo = integ.integrar(exp, complementar_2d=False)
    assert resumo['numero_imagens'] == 2
    assert 'pca_biplot' in resumo['tipos_geometricos']
    assert (exp/'vencedor'/'visualizacao'/'galeria'/'geometria_desenvolvimento'/'PCA_BIPLOT_2D.png').is_file()
    html = (exp/'vencedor'/'visualizacao'/'galeria'/'index.html').read_text(encoding='utf-8')
    assert 'PCA_BIPLOT_2D.png' in html
    assert resumo['f10_utilizado'] is False
