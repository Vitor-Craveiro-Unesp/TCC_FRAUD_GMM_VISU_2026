"""Verificacao integrada de 02 / 13-3D / 14 sem F10.

O teste visual 02 usa um redutor fake com embedding PCA: a API de
producao usa RedutorTSNE openTSNE real, sem substituicao na pipeline.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src import visualizacao_geometrica as geo
from src import visualizacao_integrada as integ
from src import visualizacao_tsne_pdf as viz


@pytest.fixture
def simulado():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
          'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf-8-sig'))
    cfg['reducao']['metodo'] = 'TSNE'
    cfg['visualizacao']['padrao']['figsize'] = [11, 7]
    cfg['visualizacao']['geometria']['pontos_maximos_amostra'] = 85
    cfg['visualizacao']['graficos']['tsne_trustworthiness'].update(
        perplexidades=[8, 12], tamanho_amostra_maximo=100, vizinhos_trustworthiness=5)
    rng = np.random.default_rng(1313)
    n = 160
    y = rng.binomial(1, .2, size=n)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f'X{i}' for i in range(5)])
    # Ajuste verdadeiro do sklearn TSNE apenas no dataset SIMULADO.
    Z = TSNE(n_components=3, perplexity=12, init='pca', max_iter=300,
             random_state=1313, learning_rate='auto').fit_transform(X)
    par = dict(perplexity=12.0, n_components=3, metric='euclidean',
               initialization='pca', learning_rate='auto',
               early_exaggeration=12.0, n_iter=300, theta=.5)
    red = SimpleNamespace(parametros=par, kl_divergence_=.8365)
    data = dict(pre=X, coord=pd.DataFrame(Z, columns=['TSNE1','TSNE2','TSNE3']),
                y=y, redutor=red, scores=np.full(n,.2),
                limiar_oof=.16, tempo_reducao=2.35,
                gamma=pd.DataFrame(np.column_stack([np.full(n,.6),np.full(n,.4)])))
    return cfg, data


def test_13_3d_verdadeiro_saida_galeria_e_csv(simulado, tmp_path, monkeypatch):
    cfg, dados = simulado
    capturado=[]
    salvar=geo._guardar
    def espiar(fig, destino, nome, ext, dpi):
        if nome.endswith('TSNE1_TSNE2_TSNE3'):
            capturado.append((fig.axes[0].name, fig.axes[0].get_zlabel(),
                              [t.get_text() for t in fig.texts]))
        return salvar(fig,destino,nome,ext,dpi)
    monkeypatch.setattr(geo,'_guardar',espiar)
    geo._projecoes(cfg,dados,tmp_path,'png',95,True,'TSNE')
    nome='13_tsne_mapa_TSNE1_TSNE2_TSNE3'
    assert (tmp_path/f'{nome}.png').is_file()
    assert (tmp_path/f'{nome}.csv').is_file()
    assert len(pd.read_csv(tmp_path/f'{nome}.csv')) == 85
    assert len(capturado)==1 and capturado[0][0]=='3d'
    assert 't-SNE 3' in capturado[0][1]
    assert any('F10' in t for t in capturado[0][2])
    assert not list(tmp_path.glob('13_tsne_mapa_TSNE1_TSNE2.png'))
    assert integ.GEO_PREFIXOS['tsne_mapa']=='13_tsne_mapa_'


def test_13_bloqueia_dimensao_falsa(simulado,tmp_path):
    cfg, dados=simulado
    dados['redutor'].parametros['n_components']=2
    with pytest.raises(ValueError,match='3D'):
        viz.projecao_3d(cfg,dados,tmp_path,'png',75,False,
                        cfg['visualizacao']['graficos']['tsne_mapa'])


def test_02_diagnostico_local_em_amostra_reprodutivel(simulado,tmp_path,monkeypatch):
    cfg,dados=simulado
    from src import reducao_tsne
    fit_calls=[]
    class RedutorParaTeste:
        def __init__(self, config, parametros):
            self.parametros=parametros
        def fit(self, X):
            fit_calls.append((self.parametros['perplexity'], len(X)))
            # Teste de contrato, sem necessidade de openTSNE no runner.
            zz=PCA(n_components=3).fit_transform(X)
            self.coordenadas=pd.DataFrame(zz,index=X.index,
                                           columns=['TSNE1','TSNE2','TSNE3'])
            return self
        def transform_treino(self):
            return self.coordenadas
    monkeypatch.setattr(reducao_tsne,'RedutorTSNE',RedutorParaTeste)
    # Selecionada=12; somente 8 e 12; nao altera redutor original.
    out=viz.trustworthiness_perplexity(cfg,dados,tmp_path,'png',95,True)
    assert out['n_ajustes']==2
    assert fit_calls==[(8.0,100),(12.0,100)]
    assert (tmp_path/'TSNE02_trustworthiness_vs_perplexity.png').exists()
    df=pd.read_csv(tmp_path/'TSNE02_trustworthiness_vs_perplexity.csv')
    assert df['perplexity'].tolist()==[8,12]
    assert df['trustworthiness'].between(0,1).all()
    assert df['n_amostra'].tolist()==[100,100]
    assert dados['redutor'].parametros['perplexity']==12
    assert 'tsne_trustworthiness' in geo.TIPOS
    assert integ.GEO_PREFIXOS['tsne_trustworthiness']=='TSNE02_trustworthiness_vs_perplexity'
    assert geo._ativo(cfg,'tsne_trustworthiness')


def test_02_bloqueia_amostra_pequena(simulado,tmp_path):
    cfg,dados=simulado
    cfg['visualizacao']['graficos']['tsne_trustworthiness']['vizinhos_trustworthiness']=60
    with pytest.raises(ValueError,match='k deve ser menor'):
        viz.trustworthiness_perplexity(cfg,dados,tmp_path,'png',75,False)


def test_14_kl_e_tempo_real_sem_historico_inventado(simulado,tmp_path,monkeypatch):
    cfg,dados=simulado
    original=geo._guardar
    visto=[]
    def espiar(fig,destino,nome,ext,dpi):
        if nome=='14_tsne_diagnostico':
            textos=[t.get_text() for ax in fig.axes for t in ax.texts]
            textos += [t.get_text() for t in fig.texts]
            visto.extend(textos)
        return original(fig,destino,nome,ext,dpi)
    monkeypatch.setattr(geo,'_guardar',espiar)
    meta=viz.diagnostico_kl(cfg,dados,tmp_path,'png',100,True,origem='simulado')
    assert meta['kl_disponivel'] and not meta['historico_disponivel']
    csv=pd.read_csv(tmp_path/'14_tsne_diagnostico.csv')
    assert csv.loc[0,'kl_final']==pytest.approx(.8365)
    assert csv.loc[0,'tempo_reducao_s']==pytest.approx(2.35)
    assert not bool(csv.loc[0,'historico_disponivel'])
    assert any('Histórico de KL por iteração: indisponível' in t for t in visto)
    assert (tmp_path/'14_tsne_diagnostico.png').is_file()


def test_14_indisponibilidade_kl_nao_inventada(simulado,tmp_path):
    cfg,dados=simulado
    dados['redutor'].kl_divergence_=float('nan')
    meta=geo._tsne_diagnostico(cfg,dados,tmp_path,'png',75,True)
    assert not meta['kl_disponivel']
    assert pd.isna(pd.read_csv(tmp_path/'14_tsne_diagnostico.csv').loc[0,'kl_final'])


def test_02_openTSNE_real_se_disponivel(simulado, tmp_path, monkeypatch):
    """Smoke real; opcional apenas no runner que nao possui openTSNE."""
    pytest.importorskip('openTSNE')
    cfg, dados = simulado
    cfg['visualizacao']['graficos']['tsne_trustworthiness'].update(
        perplexidades=[8, 12], tamanho_amostra_maximo=85,
        vizinhos_trustworthiness=5)
    # Duas execucoes reais de openTSNE, mesmas features preprocessadas.
    resultado = viz.trustworthiness_perplexity(cfg, dados, tmp_path,
                                                'png', 80, True)
    assert resultado['n_ajustes'] == 2
    tabela = pd.read_csv(tmp_path/'TSNE02_trustworthiness_vs_perplexity.csv')
    assert tabela['trustworthiness'].between(0, 1).all()
