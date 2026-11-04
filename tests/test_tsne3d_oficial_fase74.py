# -*- coding: utf-8 -*-
"""Fase 7.4: contratos oficiais e renderizacao sintetica, sem dados F10 reais."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.mixture import GaussianMixture

from src import orquestracao_oof as orq
from src import finalizacao_automatica_tsne3d as fin3
from src import finalizacao_automatica_tsne2d as fin2
from src import pre01_oof_tsne3d as pre3
from src import visualizacao_final_completa_tsne3d as vis3
from src import visualizacao_final_completa_tsne1d as vis1
from src.avaliacao_final_tsne import avaliar_scores_com_limiar_oof_congelado
from src.folds import hash_arquivo
from src.probabilidade import calcular_probabilidades


@pytest.fixture
def cfg3d():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
                          'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf-8-sig'))
    cfg['dados']['fracao_base'] = .01
    cfg['divisao']['fracao_amostra_f10'] = .10
    cfg['reducao']['metodo'] = 'TSNE'
    cfg['tsne']['n_components'] = 3
    cfg['execucao']['finalizacao_pca1d_automatica'] = True
    return cfg


@pytest.mark.parametrize('frac', [.01, .03, .10])
def test_validator_concorda_com_fracoes_de_desenvolvimento(cfg3d, frac):
    cfg3d['dados']['fracao_base'] = frac
    assert fin3.validar_automacao(cfg3d)


def test_tres_dimensoes_e_um_validador_distinto(cfg3d):
    for invalido in (1,2,[2],[1],4):
        cfg3d['tsne']['n_components'] = invalido
        with pytest.raises(ValueError, match='dimensao 3'):
            fin3.validar_automacao(cfg3d)
    cfg3d['tsne']['n_components'] = 3
    with pytest.raises(ValueError):
        fin2.validar_automacao(cfg3d)


@pytest.mark.parametrize('valor', [0, 1.01, -0.01, None])
def test_validator_recusa_fracao_f10_invalida(cfg3d, valor):
    cfg3d['divisao']['fracao_amostra_f10'] = valor
    with pytest.raises((TypeError, ValueError)):
        fin3.validar_automacao(cfg3d)


def test_planejar_tsne3d_18_ajustes_sem_criar_saida(cfg3d, tmp_path, capsys):
    (tmp_path/orq.ARQUIVO_CONFIG).write_text(yaml.safe_dump(cfg3d),encoding='utf-8')
    assert orq.main(['--planejar'],raiz=tmp_path) is None
    msg = capsys.readouterr().out
    assert 'Metodo: TSNE; candidatos: 2' in msg
    assert 'total de ajustes GMM: 18' in msg
    assert 'ETAPA FINAL TSNE 3D' in msg and '12 PNGs e PRE01 offline' in msg
    assert '10.00%' in msg
    assert not (tmp_path/'outputs').exists()


def test_f10_amostra_obedece_yaml_direto(cfg3d):
    folds = np.repeat(np.arange(1, 11, dtype=np.uint8), 200)
    y = np.tile(np.r_[np.ones(50,dtype=np.uint8),np.zeros(150,dtype=np.uint8)], 10)
    indices = fin3._amostra_f10(folds, y, config=cfg3d)
    assert len(indices) == 20 and int(y[indices].sum()) == 5 and (folds[indices] == 10).all()
    cfg3d['divisao']['fracao_amostra_f10'] = .2
    novos = fin3._amostra_f10(folds, y, config=cfg3d)
    assert len(novos) == 40 and int(y[novos].sum()) == 10


def test_roteamento_retomadas_3d_sem_treino(cfg3d, monkeypatch, tmp_path):
    pasta = tmp_path/'outputs'/'ensaio_tsne3d'
    pasta.mkdir(parents=True)
    (pasta/'configuracao.yaml').write_text(yaml.safe_dump(cfg3d),encoding='utf-8')
    chamou=[]
    monkeypatch.setattr(fin3,'concluir_ensaio',lambda p, raiz=None: chamou.append(('F10',Path(p))))
    monkeypatch.setattr(vis3,'gerar_galeria',lambda p: chamou.append(('GALERIA',Path(p))))
    assert orq.main(['--concluir-teste-final','--experimento',str(pasta)],raiz=tmp_path) == pasta
    assert orq.main(['--concluir-galeria-f10','--experimento',str(pasta)],raiz=tmp_path) == pasta
    assert chamou == [('F10',pasta),('GALERIA',pasta)]


def test_pre01_dim3_sem_f10(cfg3d, monkeypatch, tmp_path):
    from tests.test_pre01_oof_tsne1d_fase45 import origem_sintetica
    from src import pre01_v2_diagnostico as pre
    root, exp, galeria, chamadas = origem_sintetica(tmp_path, monkeypatch)
    (exp/'configuracao.yaml').write_text(yaml.safe_dump(cfg3d),encoding='utf-8')
    monkeypatch.setattr(pre3, 'hash_arquivo', lambda p:
                        'HASHFOLDS' if Path(p).name == 'folds.npy' else hash_arquivo(p))
    assert pre3.integrar_pre01_oof(exp, raiz=root) == galeria
    assert chamadas == ['validou_folds','leu_F1_F9','calculou_PRE01_sem_F10']
    man = json.loads((exp/'vencedor/visualizacao/oof'/pre.META).read_text(encoding='utf-8'))
    assert man['dimensao_experimento'] == 3 and man['f10_usado'] is False
    assert pre3.integrar_pre01_oof(exp, raiz=root) == galeria


def _fonte_sintetica(tmp_path,cfg3d):
    pasta=tmp_path/'outputs'/'ensaio_tsne3d'
    teste=pasta/'vencedor'/'teste_final'
    (teste/'modelo_final_f1_f9').mkdir(parents=True)
    (teste/'modelo_final_f1_f9/modelo_final_f1_f9.joblib').write_bytes(b'MODELO SINTETICO')
    (pasta/'configuracao.yaml').write_text(yaml.safe_dump(cfg3d),encoding='utf-8')
    ids=np.arange(1000, 1060, dtype=np.int64)
    y=np.array([0]*50+[1]*10,dtype=np.uint8)
    rng=np.random.default_rng(44)
    coords=rng.normal(size=(len(y),3))
    modelo=GaussianMixture(n_components=4,covariance_type='full',n_init=1,random_state=42).fit(coords)
    gamma=pd.DataFrame(modelo.predict_proba(coords),columns=['G1','G2','G3','G4'])
    risco=pd.Series([.02,.08,.25,.6],index=gamma.columns)
    p=calcular_probabilidades(gamma,risco).to_numpy(dtype=float)
    tau=.15
    dados=pd.DataFrame({'linha_original':ids, 'fold':10,'y_true':y,
                        'prob_fraude':p,'predicao_tau_0_50':(p>=.5).astype(int),
                        'predicao_tau_mcc_oof':(p>=tau).astype(int)})
    dados.to_csv(teste/'predicoes_f10.csv',index=False,float_format='%.17g')
    z=pd.DataFrame(coords,columns=['TSNE1','TSNE2','TSNE3'])
    z.insert(0,'linha_original',ids)
    z.to_csv(teste/'coordenadas_TSNE1_TSNE2_TSNE3_f10.csv',index=False,float_format='%.17g')
    with (teste/'indices_f10_amostra.npy').open('wb') as f:
        np.save(f, ids)
    met=avaliar_scores_com_limiar_oof_congelado(y,p,limiar_mcc_oof=tau)
    met.update(fracao_f10_dentro_do_fold=.1,id_candidato_oof='c3')
    (teste/'metricas_f10.json').write_text(json.dumps(met),encoding='utf-8')
    marker={'etapa':'F10_AVALIADO_GALERIA_PENDENTE','metodo':'TSNE','dimensao':3,
            'id_candidato_oof':'c3','fracao_f10_dentro_do_fold':.1}
    for nome,campo in [
        ('predicoes_f10.csv','sha256_predicoes'),
        ('coordenadas_TSNE1_TSNE2_TSNE3_f10.csv','sha256_coordenadas'),
        ('metricas_f10.json','sha256_metricas'),
        ('indices_f10_amostra.npy','sha256_indices_arquivo'),
        ('modelo_final_f1_f9/modelo_final_f1_f9.joblib','sha256_modelo')]:
        marker[campo]=hash_arquivo(teste/nome)
    (teste/'manifesto_finalizacao_tsne.json').write_text(json.dumps(marker),encoding='utf-8')
    pacote=SimpleNamespace(configuracao=cfg3d,
        escolha_oof={'id_candidato':'c3'},limiar_oof=tau,
        tsne=SimpleNamespace(parametros={'n_components':3,'perplexity':10},kl_divergence_=.24),
        gmm=SimpleNamespace(colunas_=['TSNE1','TSNE2','TSNE3'],modelo_=modelo,
                            responsabilidades=lambda coord: pd.DataFrame(modelo.predict_proba(coord.to_numpy()),columns=gamma.columns)),
        risco=SimpleNamespace(riscos_=risco))
    return pasta, teste, marker, ids, y, p, z.loc[:,['TSNE1','TSNE2','TSNE3']],tau,pacote,gamma,met


def test_fonte_3d_congelada_e_hash(cfg3d,monkeypatch,tmp_path):
    pasta,teste,marker,ids,y,p,z,tau,pacote,gamma,met=_fonte_sintetica(tmp_path,cfg3d)
    monkeypatch.setattr(vis3,'carregar_pacote_confiavel',lambda *_: pacote)
    fonte=vis3._fontes(pasta)
    assert len(fonte[4]) == len(y) and fonte[7].shape == (len(y),3)
    cfg3d['divisao']['fracao_amostra_f10']=.03
    (pasta/'configuracao.yaml').write_text(yaml.safe_dump(cfg3d),encoding='utf-8')
    with pytest.raises(RuntimeError,match='fracao do YAML'):
        vis3._fontes(pasta)


def test_fonte_3d_rejeita_coordenadas_alteradas(cfg3d,tmp_path):
    pasta,teste,*_=_fonte_sintetica(tmp_path,cfg3d)
    (teste/'coordenadas_TSNE1_TSNE2_TSNE3_f10.csv').write_text('ALTERADO',encoding='utf-8')
    with pytest.raises(RuntimeError,match='congelado F10 nao existe ou mudou'):
        vis3._fontes(pasta)


def test_pre01_html_3d_offline(cfg3d,monkeypatch,tmp_path):
    def resumo(*args):
        return pd.DataFrame([{'feature':'VAR1','tipo':'padronizada',
            'media_transformada_f10':.1,'desvio_transformado_f10':.9,
            'media_treino_f1_f9_codificada':0,'escala_treino_f1_f9':1}]),1
    monkeypatch.setattr(vis1,'_resumo_pre01',resumo)
    z=pd.DataFrame({'TSNE1':[1,2],'TSNE2':[3,4],'TSNE3':[5,6]})
    pagina,tab=vis3._pre01_html(tmp_path,tmp_path,cfg3d,
                                np.array([1,2]),np.array([0,1]),z,object())
    assert len(pagina)>100000 and 'scatter3d' in pagina and 'TSNE3' in pagina
    assert 'Nenhum fit' in pagina and 't-SNE 3D' in pagina


def test_renderizacao_3d_e_galeria_12_png_5_html(cfg3d,monkeypatch,tmp_path):
    """Smoke real dos renderizadores GMM FULL, TSNE13, TSNE14 e galeria em dados sintéticos."""
    pasta,teste,marker,ids,y,p,z,tau,pacote,gamma,met=_fonte_sintetica(tmp_path,cfg3d)
    cfg3d['visualizacao']['padrao']['dpi']=72
    (pasta/'configuracao.yaml').write_text(yaml.safe_dump(cfg3d),encoding='utf-8')
    monkeypatch.setattr(vis3,'_fontes',lambda *_: (teste,cfg3d,marker,met,ids,y,p,z,tau,pacote,gamma))
    monkeypatch.setattr(vis3,'_pre01_html',lambda *a: ('<html>PRE01 sintetico</html>',pd.DataFrame({'feature':['X']})))
    monkeypatch.setattr(vis3,'_oof_congelado',lambda *_: (y,p))
    vis=vis3.gerar_visuais(pasta)
    assert {f.name for f in vis.glob('*.png')}==set(vis3.PNG_ESPERADOS)
    assert {f.name for f in vis.glob('*.html')}==set(vis3.HTML_INTERATIVOS)
    assert vis3.gerar_visuais(pasta)==vis
    galeria=vis3.gerar_galeria(pasta)
    index=(galeria/'index.html').read_text(encoding='utf-8')
    assert index.count('<img loading="lazy"')==12
    assert index.count('class="link"')==5
    assert '../../visualizacao/galeria/index.html' in index
    assert len(list((galeria/'interativos').glob('*.html')))==5
    assert vis3.gerar_galeria(pasta)==galeria
    (galeria/'interativos'/vis3.HTML_INTERATIVOS[0]).write_text('CORROMPIDO',encoding='utf-8')
    with pytest.raises(RuntimeError,match='alterada'):
        vis3.gerar_galeria(pasta)


def test_saida_3d_nao_deve_mexer_em_configs_pca(cfg3d):
    assert cfg3d['pca']['n_components']
    assert cfg3d['reducao']['metodo']=='TSNE'


def test_tsne13_html_oof_a_partir_de_csv_sem_f10(cfg3d,monkeypatch,tmp_path):
    from src import galeria_interativa_oof_tsne3d as oof3
    raiz=tmp_path
    exp=raiz/'outputs'/'ensaio_tsne3d'
    galeria=exp/'vencedor'/'visualizacao'/'galeria'
    fonte=exp/'vencedor'/'visualizacao'/'geometria_desenvolvimento'
    galeria.mkdir(parents=True)
    fonte.mkdir(parents=True)
    (galeria/'index.html').write_text('<html><body><h1>OOF</h1></body></html>', encoding='utf-8')
    (exp/'configuracao.yaml').write_text(yaml.safe_dump(cfg3d),encoding='utf-8')
    rng=np.random.default_rng(42)
    df=pd.DataFrame(rng.normal(size=(30,3)),columns=['TSNE1','TSNE2','TSNE3'])
    df['classe_real']=(np.arange(len(df))%7==0).astype(int)
    df['risco_estimado_treino']=np.linspace(.01,.4,len(df))
    df['gaussiana_dominante']=(np.arange(len(df))%4)+1
    df.to_csv(fonte/'13_tsne_mapa_TSNE1_TSNE2_TSNE3.csv',index=False)
    monkeypatch.setattr(oof3,'verificar_oof_concluido_tsne',
                        lambda *_: SimpleNamespace(escolha_oof={'limiar_mcc_oof':.2}))
    index=oof3.integrar_tsne13_oof(exp,raiz=raiz)
    assert index==galeria/'index.html'
    assert oof3.MARCA in index.read_text(encoding='utf-8')
    assert (galeria/oof3.LINK).is_file()
    assert oof3.integrar_tsne13_oof(exp,raiz=raiz)==index
    assert not (exp/'vencedor/teste_final').exists()
