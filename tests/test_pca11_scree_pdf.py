"""Grafico PCA 11: verificacoes sinteticas, sem acessar F10."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from src import visualizacao_geometrica as geo


def _dados_e_config():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
                          'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf-8-sig'))
    rng = np.random.default_rng(53)
    X = rng.normal(size=(120, 9))
    X[:, 0] *= 3
    X[:, 1] *= 2
    return cfg, {'pre': pd.DataFrame(X, columns=[f'f{i}' for i in range(9)]),
                 'coord': pd.DataFrame(X[:, :1], columns=['PC1'])}


def test_pca11_barras_linha_marcacoes_e_csv(tmp_path, monkeypatch):
    cfg, data = _dados_e_config()
    capturas = []
    guardar = geo._guardar
    def espiar(fig, destino, nome, ext, dpi):
        if nome == '11_pca_cotovelo':
            capturas.append({'n_barras':len(fig.axes[0].patches),
                'n_linhas':len(fig.axes[1].lines),
                'titulos':[t.get_text() for t in fig.texts],
                'eixos':len(fig.axes)})
        return guardar(fig, destino, nome, ext, dpi)
    monkeypatch.setattr(geo, '_guardar', espiar)
    geo._scree(cfg,data,tmp_path,'png',75,True,origem='simulado')
    assert (tmp_path/'11_pca_cotovelo.png').is_file()
    tab = pd.read_csv(tmp_path/'11_pca_cotovelo.csv')
    assert len(tab) == 9
    assert np.allclose(np.cumsum(tab['variancia_individual']),tab['variancia_acumulada'])
    assert 'variancia_absoluta' in tab.columns
    assert capturas[0]['n_barras'] == 9
    assert capturas[0]['eixos'] == 2
    assert capturas[0]['n_linhas'] >= 4  # curva + 80%, 90%, 95%
    assert any('C(1)' in txt and 'C(2)' in txt and 'C(3)' in txt for txt in capturas[0]['titulos'])


def test_pca11_sem_pc3_mostra_somente_marcas_existentes(tmp_path):
    cfg, data = _dados_e_config()
    data['pre'] = data['pre'].iloc[:, :2]
    geo._scree(cfg,data,tmp_path,'png',60,True,origem='simulado')
    tabela = pd.read_csv(tmp_path/'11_pca_cotovelo.csv')
    assert list(tabela['componente']) == [1, 2]
    assert np.isclose(tabela['variancia_acumulada'].iloc[-1], 1, atol=1e-7)


def test_pca11_vencedor_nao_alterado(tmp_path):
    cfg, data = _dados_e_config()
    antes = data['coord'].copy(deep=True)
    geo._scree(cfg,data,tmp_path,'png',60,True,origem='simulado')
    pd.testing.assert_frame_equal(data['coord'], antes)
