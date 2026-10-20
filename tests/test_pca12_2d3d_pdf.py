"""PCA 12: visualizacoes 2D/3D com PCA realmente ajustado em dados simulados.

Apenas desenvolvimento simulado. Nao busca F10, parquet ou resultados OOF.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.decomposition import PCA
from src import visualizacao_geometrica as geo


def _caso(d):
    rng = np.random.default_rng(1200 + d)
    y = np.array([0] * 850 + [1] * 150, dtype=int)
    rng.shuffle(y)
    X = np.column_stack([
        rng.normal(loc=y * .95, scale=1.1, size=len(y)),
        rng.normal(loc=y * -.75, scale=.85, size=len(y)),
        rng.normal(loc=y * .35, scale=.65, size=len(y)),
        rng.normal(scale=.20, size=len(y)),
    ])
    pca = PCA(n_components=d).fit(X)
    Z = pca.transform(X)
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
                          'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf-8-sig'))
    cfg['visualizacao']['padrao']['figsize'] = [10, 6]
    cfg['visualizacao']['graficos']['pca_projecoes']['mostrar_contornos_gaussianas'] = False
    cfg['visualizacao']['geometria']['pontos_maximos_amostra'] = 700
    gam1 = 1 / (1 + np.exp(-Z[:, 0]))
    dados = {
        'coord': pd.DataFrame(Z, columns=[f'PC{i+1}' for i in range(d)]),
        'redutor': SimpleNamespace(modelo_=pca),
        'y': y,
        'gamma': pd.DataFrame(np.column_stack([gam1, 1-gam1])),
        'scores': rng.uniform(.01, .5, len(y)),
        'limiar_oof': .2,
    }
    return cfg, dados


@pytest.mark.parametrize('d', [2, 3])
def test_pca12_projecao_pdf_sintetica(d, tmp_path, monkeypatch):
    cfg, data = _caso(d)
    anterior = geo._guardar
    figuras = []
    def capturar(fig, destino, nome, ext, dpi):
        figuras.append((fig, nome))
        return anterior(fig, destino, nome, ext, dpi)
    monkeypatch.setattr(geo, '_guardar', capturar)
    geo._pca_projecao_2d3d(cfg, data, tmp_path, 'png', 95, True,
                           cfg['visualizacao']['graficos']['pca_projecoes'],
                           origem='simulado')
    nome = '12_pca_projecao_PC1_PC2' + ('_PC3' if d == 3 else '')
    assert (tmp_path / f'{nome}.png').is_file()
    fig, nome_obtido = figuras[0]
    assert nome == nome_obtido
    ax = fig.axes[0]
    assert ('3d' in str(ax.name).lower()) == (d == 3)
    for j in range(d):
        eixo = [ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel() if d == 3 else ''][j]
        assert f'PC{j+1}' in eixo
        assert f'{data["redutor"].modelo_.explained_variance_ratio_[j]:.1%}' in eixo
    assert 'PRÉVIA SINTÉTICA' in '\n'.join(t.get_text() for t in fig.texts)
    tabela = pd.read_csv(tmp_path / f'{nome}.csv')
    assert len(tabela) == len(data['y'])
    assert tabela['exibido_na_imagem'].sum() == 700
    for j in range(d):
        assert np.allclose(tabela[f'PC{j+1}'], data['coord'].iloc[:, j])
    assert not tabela.isna().any().any()


@pytest.mark.parametrize('d', [2, 3])
def test_pca12_2d_3d_nao_inventa_componentes(d, tmp_path):
    cfg, data = _caso(d)
    pca_verdadeiro = data['redutor'].modelo_
    data['redutor'].modelo_ = PCA(n_components=1).fit(
        np.column_stack([data['coord'].to_numpy(), np.arange(len(data['y'])) / 1000]))
    with pytest.raises(ValueError, match='Dimensão dos scores diverge'):
        geo._pca_projecao_2d3d(cfg, data, tmp_path, 'png', 60, False,
                               cfg['visualizacao']['graficos']['pca_projecoes'])
    data['redutor'].modelo_ = pca_verdadeiro


@pytest.mark.parametrize('d', [2, 3])
def test_pca12_2d_3d_no_dispatcher(d, tmp_path):
    cfg, data = _caso(d)
    geo._projecoes(cfg, data, tmp_path, 'png', 60, False, 'PCA')
    nome = '12_pca_projecao_PC1_PC2' + ('_PC3' if d == 3 else '')
    assert (tmp_path / f'{nome}.png').exists()


@pytest.mark.parametrize('modo', ['classe_real', 'classe_predita', 'risco', 'gaussiana_dominante'])
def test_pca12_2d_3d_modos(modo, tmp_path):
    cfg, data = _caso(2)
    cfg['visualizacao']['graficos']['pca_projecoes']['colorir_por'] = modo
    geo._pca_projecao_2d3d(cfg, data, tmp_path, 'png', 60, True,
                           cfg['visualizacao']['graficos']['pca_projecoes'],
                           origem='simulado')
    assert (tmp_path / '12_pca_projecao_PC1_PC2.png').exists()
