"""Contrato t-SNE 2D oficial; sem Parquet real ou treino completo."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src import orquestracao_oof as orq
from src import finalizacao_automatica_tsne1d as fin1
from src import finalizacao_automatica_tsne2d as fin2
from src import pre01_oof_tsne2d as pre2
from src import visualizacao_final_completa_tsne2d as vis2
from src import pre01_v2_diagnostico as pre
from src.folds import hash_arquivo


@pytest.fixture
def cfg2d():
    yaml_file = Path(__file__).resolve().parents[1] / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    c = deepcopy(yaml.safe_load(yaml_file.read_text(encoding='utf-8')))
    c['reducao']['metodo'] = 'TSNE'
    c['tsne']['n_components'] = 2
    c['dados']['fracao_base'] = .01
    c['divisao']['fracao_amostra_f10'] = .10
    c['execucao']['finalizacao_pca1d_automatica'] = True
    return c


@pytest.mark.parametrize('fracao', [.01, .03, .10])
def test_validator_autoriza_2d_todas_fracoes_homologadas(cfg2d, fracao):
    cfg2d['dados']['fracao_base'] = fracao
    assert fin2.validar_automacao(cfg2d) is True


@pytest.mark.parametrize('dim', [1, [1], 3, [3]])
def test_validator_recusa_outros_ndim(cfg2d, dim):
    cfg2d['tsne']['n_components'] = dim
    with pytest.raises(ValueError, match='somente dimensao 2'):
        fin2.validar_automacao(cfg2d)


def test_guardas_tambem_preservam_1d_e_bloqueiam_3d(cfg2d):
    with pytest.raises(ValueError, match='2D/3D'):
        fin1.validar_automacao(cfg2d)
    cfg2d['tsne']['n_components'] = 1
    assert fin1.validar_automacao(cfg2d)
    cfg2d['tsne']['n_components'] = 3
    with pytest.raises(ValueError):
        fin2.validar_automacao(cfg2d)


def test_planejar_2d_sem_criar_outputs(tmp_path, cfg2d, capsys):
    (tmp_path / orq.ARQUIVO_CONFIG).write_text(yaml.safe_dump(cfg2d, allow_unicode=True), encoding='utf-8')
    assert orq.main(['--planejar'], raiz=tmp_path) is None
    text = capsys.readouterr().out
    assert 'Metodo: TSNE; candidatos: 2' in text
    assert 'total de ajustes GMM: 18' in text
    assert 'ETAPA FINAL TSNE 2D' in text
    assert '15 PNGs e PRE01 offline' in text
    assert '10.00%' in text
    assert not (tmp_path / 'outputs').exists()


def test_planejar_3d_suportado_sem_criar_saidas(tmp_path, cfg2d, capsys):
    cfg2d['tsne']['n_components'] = 3
    (tmp_path / orq.ARQUIVO_CONFIG).write_text(yaml.safe_dump(cfg2d), encoding='utf-8')
    assert orq.main(['--planejar'], raiz=tmp_path) is None
    assert 'ETAPA FINAL TSNE 3D' in capsys.readouterr().out
    assert not (tmp_path / 'outputs').exists()


def test_amostragem_2d_obedece_yaml_nao_constante(cfg2d):
    folds = np.repeat(np.arange(1, 11, dtype=np.uint8), 200)
    y = np.tile(np.r_[np.ones(50, dtype=np.uint8), np.zeros(150, dtype=np.uint8)], 10)
    amostra10 = fin2._amostra_f10(folds, y, config=cfg2d)
    assert len(amostra10) == 20 and y[amostra10].sum() == 5
    assert np.all(folds[amostra10] == 10)
    cfg2d['divisao']['fracao_amostra_f10'] = .20
    amostra20 = fin2._amostra_f10(folds, y, config=cfg2d)
    assert len(amostra20) == 40 and y[amostra20].sum() == 10
    assert np.array_equal(amostra20, fin2._amostra_f10(folds, y, config=cfg2d))


def test_roteamento_retomadas_somente_2d(tmp_path, cfg2d, monkeypatch):
    pasta = tmp_path / 'outputs' / 'ensaio_tsne_gmm_2d'
    pasta.mkdir(parents=True)
    (pasta/'configuracao.yaml').write_text(yaml.safe_dump(cfg2d), encoding='utf-8')
    chamadas = []
    monkeypatch.setattr(fin2, 'concluir_ensaio', lambda p, raiz=None: chamadas.append(('F10', Path(p))))
    monkeypatch.setattr(vis2, 'gerar_galeria', lambda p: chamadas.append(('GALERIA', Path(p))))
    assert orq.main(['--concluir-teste-final', '--experimento', str(pasta)], raiz=tmp_path) == pasta
    assert orq.main(['--concluir-galeria-f10', '--experimento', str(pasta)], raiz=tmp_path) == pasta
    assert chamadas == [('F10',pasta), ('GALERIA',pasta)]


def test_pre01_2d_completo_sem_f10(tmp_path, monkeypatch):
    from tests.test_pre01_oof_tsne1d_fase45 import origem_sintetica
    root, exp, galeria, chamadas = origem_sintetica(tmp_path, monkeypatch)
    c = yaml.safe_load((exp/'configuracao.yaml').read_text(encoding='utf-8'))
    c['tsne']['n_components'] = 2
    (exp/'configuracao.yaml').write_text(yaml.safe_dump(c), encoding='utf-8')
    monkeypatch.setattr(pre2, 'hash_arquivo', lambda p:
                        'HASHFOLDS' if Path(p).name == 'folds.npy' else hash_arquivo(p))
    assert pre2.integrar_pre01_oof(exp, raiz=root) == galeria
    assert chamadas == ['validou_folds', 'leu_F1_F9', 'calculou_PRE01_sem_F10']
    man = json.loads((exp/'vencedor/visualizacao/oof'/pre.META).read_text(encoding='utf-8'))
    assert man['dimensao_experimento'] == 2
    assert man['f10_usado'] is False
    assert pre2.integrar_pre01_oof(exp, raiz=root) == galeria
    assert not (exp/'vencedor/teste_final').exists()


def _fontes_falsas(tmp_path, cfg2d):
    pasta = tmp_path / 'outputs' / 'ensaio_tsne_gmm_novo'
    teste = pasta / 'vencedor' / 'teste_final'
    modelo = teste / 'modelo_final_f1_f9'
    modelo.mkdir(parents=True)
    (modelo/'modelo_final_f1_f9.joblib').write_bytes(b'MODELO SINTETICO')
    (pasta/'configuracao.yaml').write_text(yaml.safe_dump(cfg2d, allow_unicode=True), encoding='utf-8')
    ids = np.arange(950, 970, dtype=np.int64)
    y = np.array([0,1]*10, dtype=np.uint8)
    p = np.full(20, .20)
    tau = .25
    df = pd.DataFrame({
        'linha_original': ids, 'fold': np.repeat(10,20), 'y_true': y,
        'prob_fraude': p, 'predicao_tau_0_50': (p>=.5).astype(int),
        'predicao_tau_mcc_oof': (p>=tau).astype(int),
    })
    df.to_csv(teste/'predicoes_f10.csv', index=False)
    pd.DataFrame({'linha_original': ids,
                  'TSNE1': np.linspace(-2,2,20), 'TSNE2': np.linspace(1,-1,20)}).to_csv(
        teste/'coordenadas_TSNE1_TSNE2_f10.csv', index=False)
    with (teste/'indices_f10_amostra.npy').open('wb') as f:
        np.save(f, ids)
    met = __import__('src.avaliacao_final_tsne', fromlist=['avaliar_scores_com_limiar_oof_congelado']).avaliar_scores_com_limiar_oof_congelado(
        y,p,limiar_mcc_oof=tau)
    met.update(fracao_f10_dentro_do_fold=.10, id_candidato_oof='candidato2d')
    (teste/'metricas_f10.json').write_text(json.dumps(met), encoding='utf-8')
    marker = {'etapa':'F10_AVALIADO_GALERIA_PENDENTE','dimensao':2,'metodo':'TSNE',
              'id_candidato_oof':'candidato2d', 'fracao_f10_dentro_do_fold':.10}
    for caminho, chave in (
        ('predicoes_f10.csv','sha256_predicoes'),
        ('coordenadas_TSNE1_TSNE2_f10.csv','sha256_coordenadas'),
        ('metricas_f10.json','sha256_metricas'),
        ('indices_f10_amostra.npy','sha256_indices_arquivo'),
        ('modelo_final_f1_f9/modelo_final_f1_f9.joblib','sha256_modelo')):
        marker[chave] = hash_arquivo(teste / caminho)
    (teste/'manifesto_finalizacao_tsne.json').write_text(json.dumps(marker), encoding='utf-8')
    return pasta, teste, marker


def test_fonte_visual_2d_valida_hash_e_scores(monkeypatch, tmp_path, cfg2d):
    pasta, teste, marco = _fontes_falsas(tmp_path,cfg2d)
    gamma = pd.DataFrame(np.tile([.25]*4, (20,1)), columns=['G1','G2','G3','G4'])
    modelo = SimpleNamespace(configuracao=cfg2d,
        escolha_oof={'id_candidato':'candidato2d'}, limiar_oof=.25,
        tsne=SimpleNamespace(parametros={'n_components':2}),
        gmm=SimpleNamespace(colunas_=['TSNE1','TSNE2'],modelo_=SimpleNamespace(covariance_type='full'),
                            responsabilidades=lambda z: gamma),
        risco=SimpleNamespace(riscos_=pd.Series([.2]*4,index=['G1','G2','G3','G4'])))
    monkeypatch.setattr(vis2, 'carregar_pacote_confiavel', lambda *_: modelo)
    saida = vis2._fontes(pasta)
    assert len(saida[4]) == 20 and saida[7].shape == (20,2) and saida[-1].shape == (20,4)
    cfg2d['divisao']['fracao_amostra_f10'] = .03
    (pasta/'configuracao.yaml').write_text(yaml.safe_dump(cfg2d), encoding='utf-8')
    with pytest.raises(RuntimeError,match='fracao do YAML'):
        vis2._fontes(pasta)


def test_detecta_corrompimento_fonte_visual(tmp_path, cfg2d):
    pasta, teste, _ = _fontes_falsas(tmp_path,cfg2d)
    (teste/'coordenadas_TSNE1_TSNE2_f10.csv').write_text('ALTERADO', encoding='utf-8')
    with pytest.raises(RuntimeError, match='congelado F10 nao existe ou mudou'):
        vis2._fontes(pasta)


def test_galeria_15_png_links_e_protecao(tmp_path, monkeypatch, cfg2d):
    pasta, teste, marco = _fontes_falsas(tmp_path,cfg2d)
    vis = teste/'visualizacao_f10_2d'
    (vis/'PRE01_F10').mkdir(parents=True)
    for name in vis2.PNG_ESPERADOS:
        (vis/name).write_bytes(b'\x89PNG\r\n\x1a\nPNG_SINTETICO')
    (vis/vis2.PRE01).write_text('<html>PRE01 F10 interativo</html>',encoding='utf-8')
    fake = (teste,cfg2d,marco,
            {'ap':.02,'mcc_limiar_oof':0.0,'logloss':.2},
            np.arange(20),np.array([0,1]*10),np.full(20,.2),
            pd.DataFrame({'TSNE1':np.arange(20),'TSNE2':np.arange(20)}),
            .25,object(),object())
    monkeypatch.setattr(vis2, '_fontes', lambda _: fake)
    monkeypatch.setattr(vis2, 'gerar_visuais', lambda _: vis)
    destino = vis2.gerar_galeria(pasta)
    html = (destino/'index.html').read_text(encoding='utf-8')
    assert html.count('<img loading="lazy"') == 15
    assert 'Diagnostico completo da padronizacao (HTML interativo)' in html
    assert '10.00% do fold' in html
    assert '../../visualizacao/galeria/index.html' in html
    assert vis2.gerar_galeria(pasta) == destino
    (destino/'figuras'/vis2.PNG_ESPERADOS[0]).write_bytes(b'ALTERADO')
    with pytest.raises(RuntimeError, match='alterada'):
        vis2.gerar_galeria(pasta)


def test_render2d_nunca_hardcode_fracao_antiga():
    import inspect
    from src import visualizacao_tsne_2d
    fonte = inspect.getsource(visualizacao_tsne_2d.gerar_projecao_2d)
    assert 'F10 real (3%)' not in fonte
    assert 'fracao_amostra_f10(config)' in fonte


def test_pre01_html_2d_e_offline_sem_fit(monkeypatch, cfg2d, tmp_path):
    from src import visualizacao_final_completa_tsne1d as vis1
    def resumo_sem_fit(*args):
        return (pd.DataFrame([{'feature':'VAR_A', 'tipo':'padronizada',
                               'media_transformada_f10':.15,'desvio_transformado_f10':.91,
                               'media_treino_f1_f9_codificada':2.0,'escala_treino_f1_f9':.8},
                              {'feature':'VAR_B', 'tipo':'preservada_sem_padronizar',
                               'media_transformada_f10':.4,'desvio_transformado_f10':.5,
                               'media_treino_f1_f9_codificada':None,'escala_treino_f1_f9':None}]), 1)
    monkeypatch.setattr(vis1, '_resumo_pre01', resumo_sem_fit)
    z = pd.DataFrame({'TSNE1':[0.0,1.0], 'TSNE2':[2.0,3.0]})
    html, tab = vis2._pre01_html(tmp_path, tmp_path, cfg2d,
                                 np.array([10,12]), np.array([0,1]), z, object())
    assert len(html)>100000 and 'Plotly.newPlot' in html
    assert 'TSNE1' in html and 'TSNE2' in html
    assert 'Nenhum fit' in html and 'sem novo ajuste' not in html
    assert tab.shape[0]==2


def test_renderizacao_real_2d_gmm_full_com_dados_sinteticos(monkeypatch, tmp_path, cfg2d):
    """Smoke real dos 15 PNGs, sem Parquet, F10 real, OOF ou retreino de t-SNE."""
    from sklearn.mixture import GaussianMixture
    from src.probabilidade import calcular_probabilidades
    from src.avaliacao_final_tsne import avaliar_scores_com_limiar_oof_congelado
    import matplotlib.pyplot as plt

    cfg2d['visualizacao']['padrao']['dpi'] = 65
    cfg2d['visualizacao']['graficos']['densidades_marginais']['pontos_grade_2d'] = 80
    for nome in ('regioes_dominancia','sensibilidade_fronteira','mapa_risco_fronteira'):
        cfg2d['visualizacao']['graficos'][nome]['resolucao_2d'] = 35
    rng = np.random.default_rng(73)
    x = rng.normal(size=(220,2))
    x[:60] += [2.0,1.2]
    x[130:] += [-1.4,1.6]
    z = pd.DataFrame(x,columns=['TSNE1','TSNE2'])
    modelo_gmm = GaussianMixture(n_components=4,covariance_type='full',n_init=1,random_state=42).fit(x)
    class GMMTeste:
        modelo_ = modelo_gmm
        colunas_ = ['TSNE1','TSNE2']
        saidas_ = ['G1','G2','G3','G4']
        def responsabilidades(self, dataframe):
            return pd.DataFrame(modelo_gmm.predict_proba(dataframe.to_numpy()),
                                index=dataframe.index,columns=self.saidas_)
    gm = GMMTeste()
    rh = SimpleNamespace(riscos_=pd.Series([.02,.03,.08,.21],index=gm.saidas_))
    gamma = gm.responsabilidades(z)
    p = calcular_probabilidades(gamma,rh.riscos_).to_numpy()
    y = (np.arange(len(x)) % 7 == 0).astype(np.uint8)
    tau = .08
    met = avaliar_scores_com_limiar_oof_congelado(y,p,limiar_mcc_oof=tau)
    pacote = SimpleNamespace(gmm=gm,risco=rh,
                             tsne=SimpleNamespace(kl_divergence_=.234,
                                                  parametros={'perplexity':10,'metric':'euclidean','n_components':2}))
    exp = tmp_path/'outputs'/'ensaio_tsne_gmm_sintetico'
    teste = exp/'vencedor/teste_final'
    teste.mkdir(parents=True)
    ids = np.arange(len(y),dtype=np.int64)
    marco = {k:'HASH' for k in ('sha256_modelo','sha256_predicoes','sha256_coordenadas','sha256_metricas')}
    monkeypatch.setattr(vis2,'_fontes',lambda _p:
                        (teste,cfg2d,marco,met,ids,y,p,z,tau,pacote,gamma))
    monkeypatch.setattr(vis2,'_pre01_html',lambda *a:
                        ('<html>PRE01 sintetico</html>',pd.DataFrame({'feature':['X']})))
    try:
        fontes = vis2.gerar_visuais(exp)
        assert {f.name for f in fontes.glob('*.png')} == set(vis2.PNG_ESPERADOS)
        assert vis2.gerar_visuais(exp) == fontes
        galeria = vis2.gerar_galeria(exp)
        assert (galeria/'index.html').is_file()
        assert len(list((galeria/'figuras').glob('*.png'))) == 15
    finally:
        plt.close('all')
