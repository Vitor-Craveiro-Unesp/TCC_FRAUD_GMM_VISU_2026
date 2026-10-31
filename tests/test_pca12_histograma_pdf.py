"""PCA 12: histograma 1D conforme o catálogo, testes 100% sintéticos."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.decomposition import PCA
from src import visualizacao_geometrica as geo


def _dados():
    rng = np.random.default_rng(1206)
    y = np.array([0] * 1080 + [1] * 120, dtype=int)
    x = np.column_stack([
        rng.normal(loc=y * 1.0, scale=1.4),
        rng.normal(scale=.6, size=len(y)),
        rng.normal(scale=.3, size=len(y)),
    ])
    pca = PCA(n_components=1).fit(x)
    coord = pca.transform(x)
    gamma = np.column_stack([1/(1 + np.exp(-coord[:, 0])),
                             1 - 1/(1 + np.exp(-coord[:, 0]))])
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
                          'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf-8-sig'))
    cfg['visualizacao']['padrao']['figsize'] = [10, 5.8]
    data = {
        'coord': pd.DataFrame(coord, columns=['PC1']),
        'redutor': SimpleNamespace(modelo_=pca),
        'y': y,
        'gamma': pd.DataFrame(gamma),
        'scores': rng.uniform(.01, .45, size=len(y)),
        'limiar_oof': .15,
    }
    return cfg, data


def test_pca12_histograma_1d_e_csv_completo(tmp_path, monkeypatch):
    cfg, data = _dados()
    original = geo._guardar
    capturas = []
    def interceptar(fig, dest, nome, ext, dpi):
        if nome == '12_pca_projecao_PC1':
            ax = fig.axes[0]
            capturas.append({
                'bins': len(ax.patches),
                'legenda': ax.get_legend() is not None,
                'xlabel': ax.get_xlabel(),
                'ylabel': ax.get_ylabel(),
                'titulo': ax.get_title(),
                'texto': [t.get_text() for t in fig.texts],
            })
        return original(fig, dest, nome, ext, dpi)
    monkeypatch.setattr(geo, '_guardar', interceptar)
    geo._pca_histograma_1d(cfg, data, tmp_path, 'png', 80, True,
                          cfg['visualizacao']['graficos']['pca_projecoes'],
                          origem='simulado')
    assert len(capturas) == 1
    cap = capturas[0]
    assert cap['bins'] >= 36
    assert 'Histograma' in cap['titulo']
    assert 'PC1' in cap['xlabel'] and f"{data['redutor'].modelo_.explained_variance_ratio_[0]:.7f}" in cap['xlabel']
    assert 'Densidade' in cap['ylabel']
    assert 'variância explicada' in cap['xlabel'].lower()
    assert cap['legenda']
    assert (tmp_path / '12_pca_projecao_PC1.png').exists()
    df = pd.read_csv(tmp_path / '12_pca_projecao_PC1.csv')
    assert len(df) == len(data['y'])  # nenhuma subamostragem distorcendo o histograma
    assert np.allclose(df['PC1'], data['coord']['PC1'])
    assert np.allclose(df['variancia_explicada_PC1'],
                       data['redutor'].modelo_.explained_variance_ratio_[0])
    assert set(df['classe_real']) == {0, 1}


@pytest.mark.parametrize('modo', ['classe_real', 'classe_predita', 'risco', 'gaussiana_dominante'])
def test_pca12_todos_modos_configuraveis(modo, tmp_path):
    cfg, data = _dados()
    cfg['visualizacao']['graficos']['pca_projecoes']['colorir_por'] = modo
    geo._projecoes(cfg, data, tmp_path, 'png', 65, False, 'PCA')
    assert (tmp_path / '12_pca_projecao_PC1.png').exists()
    assert not list(tmp_path.glob('12_pca_projecao_PC1_PC2*.png'))


def test_pca12_pcs_2d_comportamento_preservado(tmp_path):
    cfg, data = _dados()
    # PCA de 2 dimensoes efetivamente ajustado; nunca inventar PC2.
    rng = np.random.default_rng(420)
    x = np.column_stack([data['coord']['PC1'].to_numpy(),
                         rng.normal(size=len(data['y'])),
                         rng.normal(scale=.25, size=len(data['y']))])
    pca = PCA(n_components=2).fit(x)
    data['redutor'].modelo_ = pca
    data['coord'] = pd.DataFrame(pca.transform(x), columns=['PC1', 'PC2'])
    # Sem modelo GMM neste teste: desativa somente os contornos opcionais.
    cfg['visualizacao']['graficos']['pca_projecoes']['mostrar_contornos_gaussianas'] = False
    # O modo 2D continua produzindo pontos, sem modificar t-SNE ou GMM.
    geo._projecoes(cfg, data, tmp_path, 'png', 65, True, 'PCA')
    assert (tmp_path / '12_pca_projecao_PC1_PC2.png').exists()
