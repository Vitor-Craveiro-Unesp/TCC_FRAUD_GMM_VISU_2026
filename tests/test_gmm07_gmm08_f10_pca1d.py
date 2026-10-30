# -*- coding: utf-8 -*-
"""Testes sinteticos, sem Parquet, F10 real, modelos locais ou escrita no projeto."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

ARQUIVO = Path(__file__).resolve().parents[1] / 'scripts/execucao/GERAR_GMM07_GMM08_F10_PCA1D.py'
spec = importlib.util.spec_from_file_location('gmm_f10_p2', ARQUIVO)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def dados_sinteticos(n=40, tau=.15):
    xy = np.linspace(-3, 3, n)
    p = np.linspace(.001, .3, n)
    y = np.array(([0]*30)+([1]*10), dtype=np.int8) if n == 40 else np.zeros(n, dtype=np.int8)
    preds = pd.DataFrame({'linha_original': np.arange(n)*3+1, 'fold': 10,
                          'y_true':y, 'prob_fraude':p,
                          'predicao_tau_0_50': (p>=.5).astype(int),
                          'predicao_tau_mcc_oof': (p>=tau).astype(int)})
    coords = pd.DataFrame({'linha_original':preds['linha_original'], 'PC1':xy})
    met = {'limiar_otimizado_no_f10': False, 'limiar_mcc_oof_congelado':tau,
           'mcc_limiar_oof':0.13,
           'comparacao_limiares':{'tau_0_50':{'limiar':.5,'mcc':0.0},
                                  'tau_mcc_oof':{'limiar':tau,'mcc':.13}}}
    return coords,preds,met


def test_indices_f10_alinhados():
    coords,preds,_ = dados_sinteticos()
    mod.comparar_ids(coords,preds,len(preds))


def test_indices_f10_duplicados_sao_bloqueados():
    coords,preds,_ = dados_sinteticos()
    coords.loc[1,'linha_original']=coords.loc[0,'linha_original']
    with pytest.raises(RuntimeError, match='desordenados'):
        mod.comparar_ids(coords,preds,len(preds))


def test_indices_f10_desalinhados_sao_bloqueados():
    coords,preds,_ = dados_sinteticos()
    coords.loc[0,'linha_original']=0
    with pytest.raises(RuntimeError, match='divergentes'):
        mod.comparar_ids(coords,preds,len(preds))


def test_mcc_usa_os_numeros_ja_salvos():
    _,preds,met=dados_sinteticos()
    assert mod.cortes_fixos(preds,met,.15)==(0,.13)


def test_reotimizacao_f10_proibida():
    _,preds,met=dados_sinteticos()
    met['limiar_otimizado_no_f10']=True
    with pytest.raises(RuntimeError, match='otimizado'):
        mod.cortes_fixos(preds,met,.15)


def test_limiar_congelado_nao_pode_mudar():
    _,preds,met=dados_sinteticos()
    with pytest.raises(RuntimeError,match='divergente'):
        mod.cortes_fixos(preds,met,.14)


def test_textos_oof_trocados_sem_mudar_geometria():
    fig,ax=plt.subplots()
    ax.set_title('CORTE FIXO\nMCC OOF = 0.1300000')
    ax.set_xlabel('LIMIAR ÓTIMO DO MCC')
    ax.text(.4,.2,'B — Corte ótimo MCC OOF τ* = 0.15')
    mod.rotulos_para_f10(fig)
    t = '\n'.join(x.get_text() for x in fig.findobj(match=matplotlib.text.Text))
    assert 'MCC F10' in t
    assert 'MCC OOF =' not in t
    assert 'LIMIAR τ* CONGELADO (OOF)' in t
    assert 'B — Limiar τ* congelado OOF' in t
    plt.close(fig)


def test_wrappers_restauram_mesmo_apos_erro():
    classe = SimpleNamespace(_avaliar_cortes_oof=lambda a,b:(9,9), _guardar=lambda *a:None)
    gmm = SimpleNamespace(_rotulo_mcc=lambda x:('original','original'))
    oldm, oldg, oldr=classe._avaliar_cortes_oof,classe._guardar,gmm._rotulo_mcc
    with pytest.raises(RuntimeError,match='erro intencional'):
        with mod.rotinas_de_desenho_f10(classe,gmm,.15,0,.13):
            assert classe._avaliar_cortes_oof({'origem':'f10_amostral_congelado'},.15)==(0,.13)
            assert gmm._rotulo_mcc({'origem':'f10_amostral_congelado'})[0]=='MCC F10'
            raise RuntimeError('erro intencional')
    assert classe._avaliar_cortes_oof is oldm
    assert classe._guardar is oldg
    assert gmm._rotulo_mcc is oldr


class ModeloSintetico:
    covariance_type='full'
    means_=np.array([[-1.4],[1.3]])
    covariances_=np.array([[[.65]],[[.85]]])
    weights_=np.array([.7,.3])
    def predict_proba(self, X):
        x=np.asarray(X, dtype=float).reshape(-1)
        a=self.weights_ * np.exp(-.5*(x[:,None]-self.means_.reshape(1,-1))**2 / self.covariances_.reshape(1,-1)) / np.sqrt(2*np.pi*self.covariances_.reshape(1,-1))
        return a/a.sum(axis=1,keepdims=True)


def test_geracao_integrada_sintetica_visualizador_original(tmp_path):
    # Sem dados reais e sem uso de modelos do usuario. Executa as DUAS rotinas
    # de desenho originais via substituicao temporaria em memoria.
    # Dependencias de ajuste nao exercitadas neste teste: stubs isolados.
    from types import ModuleType
    if not (Path(__file__).resolve().parents[1]/'src/reducao_tsne.py').exists():
        st=ModuleType('src.reducao_tsne')
        st.RedutorTSNE=type('RedutorTSNE', (), {})
        st.grade_tsne=lambda *a, **k: []
        sys.modules['src.reducao_tsne']=st
    if not (Path(__file__).resolve().parents[1]/'src/gmm_convergencia_em.py').exists():
        st=ModuleType('src.gmm_convergencia_em')
        st.GaussianMixtureComHistorico=type('GaussianMixtureComHistorico', (), {})
        sys.modules['src.gmm_convergencia_em']=st
    try:
        from src import visualizacao_geometrica as geo
        from src import visualizacao_gmm07 as gmm07
    except ModuleNotFoundError:
        pytest.skip('Projeto src completo nao esta instalado neste ambiente sintético.')
    config=yaml.safe_load((Path(__file__).resolve().parent/'fixtures_gmm_f10_pca1d.yaml').read_text(encoding='utf-8'))
    x=np.linspace(-3,3,40)
    y=np.array([0]*30+[1]*10,dtype=np.int8)
    gm=SimpleNamespace(modelo_=ModeloSintetico(),colunas_=['PC1'],saidas_=['G1','G2'])
    rh=SimpleNamespace(riscos_=pd.Series([.01,.15]))
    data={'coord':pd.DataFrame({'PC1':x}), 'y':y, 'scores':np.linspace(.005,.15,len(x)), 'gmm':gm, 'rh':rh}
    with mod.rotinas_de_desenho_f10(geo,gmm07,.15,.01,.13):
        gmm07.gerar(config,data,{'limiar_mcc_oof':.15},tmp_path,'png',90,True,
                    avaliacao={'origem':'f10_amostral_congelado'})
        geo._fronteira(config,data,{'limiar_mcc_oof':.15},tmp_path,'png',90,True,
                     avaliacao={'origem':'f10_amostral_congelado'})
    assert all((tmp_path/name).is_file() for name in mod.ARQUIVOS_GERADOS)
    df=pd.read_csv(tmp_path/'GMM07_metricas_comparativas.csv')
    assert list(df['origem_mcc']) == ['f10_amostral_congelado']*2
