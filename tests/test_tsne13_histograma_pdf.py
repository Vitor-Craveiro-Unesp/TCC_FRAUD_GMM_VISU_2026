"""t-SNE 13 1D: prévias numéricas sintéticas e testes sem F10.

Para estas prévias, sklearn.manifold.TSNE é usado SOMENTE para produzir
coordenadas sintéticas. Produção usa RedutorTSNE(openTSNE) já ajustado.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.manifold import TSNE
from src import visualizacao_geometrica as geo


@pytest.fixture(scope='module')
def exemplo():
    rng = np.random.default_rng(1313)
    n = 560
    y = np.r_[np.zeros(455, dtype=int), np.ones(105, dtype=int)]
    rng.shuffle(y)
    X = np.column_stack([
        rng.normal(loc=y * .55, scale=.9),
        rng.normal(loc=y * -.65, scale=.85),
        rng.normal(loc=y * .35, scale=1.0),
        rng.normal(scale=1.25, size=n),
    ])
    # t-SNE de verdade, ajustado SEM usar y como entrada.
    perplexidade = 27.0
    tsne = TSNE(n_components=1, perplexity=perplexidade, init='pca',
                max_iter=320, random_state=1313, learning_rate='auto')
    z = tsne.fit_transform(X)
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
                          'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf-8-sig'))
    cfg['visualizacao']['padrao']['figsize'] = [10, 6]
    cfg['visualizacao']['geometria']['pontos_maximos_amostra'] = 200
    cfg['visualizacao']['graficos']['tsne_mapa']['mostrar_contornos_gaussianas'] = False
    riscos = np.clip(.12 + .06 * np.tanh(z[:, 0] / 10) + .07 * y, .001, .95)
    p_g = np.clip(.5 + .24 * np.tanh(z[:, 0] / 8), .01, .99)
    redutor = SimpleNamespace(parametros={
        'n_components': 1, 'perplexity': perplexidade, 'metric': 'euclidean',
        'learning_rate': 'auto', 'initialization': 'pca',
    })
    dados = {'coord': pd.DataFrame(z, columns=['TSNE1']),
             'redutor': redutor, 'y': y,
             'gamma': pd.DataFrame(np.column_stack([p_g, 1-p_g])),
             'scores': riscos, 'limiar_oof': .2}
    return cfg, dados


def test_tsne13_histograma_1d_pdf_simulado(tmp_path, exemplo, monkeypatch):
    cfg, dados = exemplo
    orig = geo._guardar
    capturas = []
    def capturar(fig, destino, nome, ext, dpi):
        if nome == '13_tsne_mapa_TSNE1':
            ax = fig.axes[0]
            capturas.append((ax.name, ax.get_xlabel(), ax.get_ylabel(),
                             ax.get_title(), len(ax.patches),
                             [t.get_text() for t in fig.texts],
                             bool(ax.get_legend())))
        return orig(fig, destino, nome, ext, dpi)
    monkeypatch.setattr(geo, '_guardar', capturar)
    geo._tsne_histograma_1d(cfg, dados, tmp_path, 'png', 95, True,
                            cfg['visualizacao']['graficos']['tsne_mapa'],
                            origem='simulado')
    assert len(capturas) == 1
    nome, xlabel, ylabel, titulo, barras, textos, legenda = capturas[0]
    assert nome == 'rectilinear'
    assert 't-SNE 1' in xlabel and 'variância explicada' in xlabel
    assert 'Densidade' in ylabel and 'Histograma' in titulo
    assert barras >= 36 and legenda
    assert 'PRÉVIA SINTÉTICA' in '\n'.join(textos)
    assert any('F10 não utilizado' in t for t in textos)
    assert (tmp_path/'13_tsne_mapa_TSNE1.png').exists()
    tabela = pd.read_csv(tmp_path/'13_tsne_mapa_TSNE1.csv')
    assert len(tabela) == len(dados['y'])
    assert np.allclose(tabela['TSNE1'], dados['coord']['TSNE1'])
    assert set(tabela['classe_real']) == {0, 1}
    assert set(tabela['gaussiana_dominante']).issubset({1, 2})
    assert not tabela.isna().any().any()


@pytest.mark.parametrize('modo', ['classe_real','classe_predita','risco','gaussiana_dominante'])
def test_tsne13_1d_modos_conforme_yaml(tmp_path, exemplo, modo):
    cfg, dados = exemplo
    cfg = yaml.safe_load(yaml.safe_dump(cfg))
    cfg['visualizacao']['graficos']['tsne_mapa']['colorir_por'] = modo
    geo._projecoes(cfg, dados, tmp_path, 'png', 65, True, 'TSNE')
    assert (tmp_path/'13_tsne_mapa_TSNE1.png').is_file()
    assert (tmp_path/'13_tsne_mapa_TSNE1.csv').is_file()
    assert not list(tmp_path.glob('13_tsne_mapa_TSNE1_TSNE*.png'))


def test_tsne13_nao_inventa_eixos(tmp_path, exemplo):
    cfg, dados = exemplo
    dados = dict(dados)
    dados['redutor'] = SimpleNamespace(parametros={'n_components': 2})
    with pytest.raises(ValueError, match='ajustado para 1 dimensao'):
        geo._tsne_histograma_1d(cfg, dados, tmp_path, 'png', 60, False,
                                cfg['visualizacao']['graficos']['tsne_mapa'])
    assert not list(tmp_path.iterdir())


def test_tsne13_valida_scores_e_classe(tmp_path, exemplo):
    cfg, original = exemplo
    dados = dict(original)
    dados['scores'] = np.full(len(original['y']), 1.1)
    with pytest.raises(ValueError, match='Riscos desalinhados'):
        geo._tsne_histograma_1d(cfg, dados, tmp_path, 'png', 60, False,
                                cfg['visualizacao']['graficos']['tsne_mapa'])
