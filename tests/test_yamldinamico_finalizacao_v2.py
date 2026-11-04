"""Regressao cientifica leve: 6 modalidades, K variavel, F10 congelado.

Nao abre Parquet nem results reais; nao usa rede nem SLURM. As integracoes
usam dados sinteticos e F10 SINTETICO, nao os dados reservados do TCC.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import importlib.util

import numpy as np
import pandas as pd
import pytest
import yaml

from src import orquestracao_oof as orq
from src.amostragem_f10 import selecionar_f10_estratificado
from src.contrato_finalizacao_dinamica import (
    validar_configuracao_finalizacao, validar_vencedor, validar_viabilidade_oof,
)
from src.validacao_oof import grade_candidatos_oof


class TSNESinteticoSemBiblioteca:
    """DOUBLE DE TESTE: exercita integração sem executar openTSNE.

    NAO valida algoritmo t-SNE nem geometria real do embedding.
    """
    def __init__(self, config, parametros):
        self.parametros = parametros
        self.d = int(parametros['n_components'])
        self.modelo_ = SimpleNamespace(negative_gradient_method='bh' if self.d == 3 else 'auto')
    def _coordenadas(self, X):
        a = np.asarray(X, dtype=float)
        return pd.DataFrame(a[:, :self.d].copy(), index=X.index,
                            columns=[f'TSNE{i+1}' for i in range(self.d)])
    def fit(self, X):
        self.treino_ = self._coordenadas(X)
        return self
    def transform_treino(self):
        return self.treino_.copy()
    def transform(self, X):
        return self._coordenadas(X)


def cfg_base(metodo='PCA', dim=1):
    base = Path(__file__).resolve().parents[1]
    cfg = deepcopy(yaml.safe_load((base/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf-8-sig')))
    cfg['reducao']['metodo'] = metodo
    cfg['pca' if metodo == 'PCA' else 'tsne']['n_components'] = [dim]
    cfg['tsne']['initialization'] = ['random']
    cfg['dados']['fracao_base'] = .05
    cfg['divisao']['fracao_amostra_f10'] = .25
    cfg['gmm']['n_components'] = [3, 4, 5]
    cfg['gmm']['n_init'] = 3
    cfg['gmm']['max_iter'] = 35
    cfg['amostragem']['razao_negativos_positivos'] = 10
    cfg['experimento']['semente'] = 81
    cfg['execucao']['n_jobs'] = 1
    cfg['execucao']['finalizacao_pca1d_automatica'] = True
    return cfg


def dados_sinteticos():
    rng = np.random.default_rng(121)
    folds = np.repeat(np.arange(1, 10, dtype=np.uint8), 25)
    n = len(folds)
    y = np.zeros(n, dtype=np.uint8)
    y[::25] = 1
    def frame(quantidade, rotulos, seed):
        rg = np.random.default_rng(seed)
        return pd.DataFrame({
            'TARGET_OMEGA': rotulos,
            'NID_ALPHA': np.arange(quantidade),
            'TRANS_VALUE': rg.normal(size=quantidade) + 1.0*rotulos,
            'TRANS_NUM_CARD_FEWF': rg.choice(['A','B','C'], size=quantidade),
            'SEND_GENDER_BE': rg.choice(['F','M'], size=quantidade),
            'TRANS_MONTH_SIN': np.sin(np.arange(quantidade)*np.pi/6),
            'TRANS_MONTH_COS': np.cos(np.arange(quantidade)*np.pi/6),
            'VALOR_CONTINUO': rg.normal(size=quantidade) + .8*rotulos,
        })
    df=frame(n,y,121)
    y10 = np.zeros(36, dtype=np.uint8)
    y10[::9] = 1
    f10=frame(36,y10,122)
    return df, folds, f10


def escolha(cfg, k=5):
    c = next(c for c in grade_candidatos_oof(cfg) if c['gmm']['n_components']==k)
    ident = sha256(json.dumps(c,sort_keys=True,ensure_ascii=False,
                  separators=(',',':')).encode('utf-8')).hexdigest()[:14]
    return {'id_candidato':ident,'configuracao':c,'limiar_mcc_oof':0.24,
            'status':'selecionado_em_ensaio','teste_final_usado':False}


@pytest.mark.parametrize('metodo,dim', [('PCA',1),('PCA',2),('PCA',3),('TSNE',1),('TSNE',2),('TSNE',3)])
def test_seis_modalidades_grade_historica_e_dinamica(metodo,dim):
    cfg=cfg_base(metodo,dim)
    grade=validar_configuracao_finalizacao(cfg,metodo=metodo,dimensao=dim)
    assert len(grade)==3
    assert [c['gmm']['n_components'] for c in grade]==[3,4,5]
    assert all(c['gmm']['n_init']==3 and c['razao']==10 for c in grade)
    eleito=escolha(cfg,5)
    assert validar_vencedor(cfg,eleito,metodo=metodo,dimensao=dim)['gmm']['n_components']==5
    cfg['gmm']['n_components']=[4]
    cfg['gmm']['n_init']=5
    cfg['amostragem']['razao_negativos_positivos']=16
    cfg['experimento']['semente']=42
    cfg['dados']['fracao_base']=.01
    assert len(validar_configuracao_finalizacao(cfg,metodo=metodo,dimensao=dim))==1


@pytest.mark.parametrize('metodo,dim', [('PCA',1),('PCA',2),('PCA',3),('TSNE',1),('TSNE',2),('TSNE',3)])
def test_integracao_ajuste_final_f10_e_manifesto_sinteticos(tmp_path,monkeypatch,metodo,dim):
    cfg=cfg_base(metodo,dim)
    cfg['gmm']['n_components']=[3,5]
    eleito=escolha(cfg,5)
    df, folds, f10=dados_sinteticos()
    validar_viabilidade_oof(folds,df['TARGET_OMEGA'].to_numpy(),cfg)
    prov={'sha256_parquet':'SINTETICO','sha256_folds':'SINTETICO',
          'sha256_yaml':'SINTETICO','experimento_oof':'SINTETICO_NAO_REAL'}
    if metodo=='PCA':
        from src.treinamento_final_pca import ajustar_final_f1_f9, salvar_pacote
        from src.avaliacao_final_pca import avaliar_f10_em_memoria
    else:
        from src import treinamento_final_tsne
        from src.treinamento_final_tsne import ajustar_final_f1_f9, salvar_pacote
        from src.avaliacao_final_tsne import avaliar_f10_em_memoria
        if importlib.util.find_spec('openTSNE') is None:
            monkeypatch.setattr(treinamento_final_tsne, 'RedutorTSNE', TSNESinteticoSemBiblioteca)
    # Integra refit REAL, soft risk e avaliacao F10 sintetica. OOF ja foi
    # selecionado em uma grade sintética; nenhum F10 participa da selecao.
    pacote=ajustar_final_f1_f9(df,folds,cfg,eleito,prov,registrar_historico_em=False)
    assert pacote.gmm.modelo_.n_components==5
    assert pacote.gmm.modelo_.n_init==3
    assert pacote.gmm.modelo_.covariance_type=='full'
    assert pacote.risco.n_treino_==len(df)
    assert pacote.n_ajuste_gmm < len(df)
    assert pacote.configuracao['amostragem']['razao_negativos_positivos']==10
    assert pacote.configuracao['experimento']['semente']==81
    assert pacote.limiar_oof==.24
    models=tmp_path/f'{metodo}{dim}D'
    _model, man=salvar_pacote(pacote,models)
    registro=json.loads(man.read_text(encoding='utf-8'))
    assert registro['gmm_parametros_vencedor']['n_components']==5
    assert registro['gmm_parametros_vencedor']['n_init']==3
    assert registro['razao_rus_vencedora']==10
    assert registro['semente_experimento']==81
    ids=np.arange(2000,2036,dtype=np.int64)
    ft=np.repeat(10,36).astype(np.uint8)
    if metodo=='TSNE':
        tab, met, coords = avaliar_f10_em_memoria(pacote,f10,ft,ids,retornar_coordenadas=True)
        assert coords.shape==(36,dim+1)
    else:
        tab, met = avaliar_f10_em_memoria(pacote,f10,ft,ids)
    assert met['limiar_mcc_oof_congelado']==.24
    assert met['limiar_otimizado_no_f10'] is False
    assert met['id_candidato_oof']==eleito['id_candidato']
    assert np.all(tab.fold.to_numpy()==10)
    assert np.array_equal(tab.predicao_tau_mcc_oof.to_numpy(),
                          (tab.prob_fraude.to_numpy()>=.24).astype('uint8'))
    with pytest.raises(ValueError,match='F10 proibido'):
        fb=folds.copy();fb[-1]=10
        ajustar_final_f1_f9(df,fb,cfg,eleito,prov,registrar_historico_em=False)


@pytest.mark.parametrize('campo,valor,match',[
    ('dados.fracao_base',0,'fracao_base'),
    ('dados.fracao_base',1.01,'fracao_base'),
    ('dados.fracao_base',float('nan'),'fracao_base'),
    ('divisao.fracao_amostra_f10',0,'fracao_amostra_f10'),
    ('gmm.n_components',[0],'n_components'),
    ('gmm.n_init',0,'n_init'),
    ('gmm.covariance_type','diag','covariance_type'),
    ('experimento.semente',-1,'semente'),
    ('experimento.semente',2**32,'semente'),
    ('amostragem.razao_negativos_positivos',0,'Razao RUS'),
    ('pca.n_components',[1,2],'uma unica dimensao'),
])
def test_invalido_bloqueado_sem_treinamento(campo,valor,match):
    cfg=cfg_base()
    objeto=cfg
    *pais, chave=campo.split('.')
    for p in pais:objeto=objeto[p]
    objeto[chave]=valor
    with pytest.raises(ValueError,match=match):
        validar_configuracao_finalizacao(cfg)


def test_vencedor_fora_grade_ou_f10_usado_bloqueado():
    cfg=cfg_base()
    s=escolha(cfg,5)
    s['teste_final_usado']=True
    with pytest.raises(ValueError,match='F10'):
        validar_vencedor(cfg,s,metodo='PCA')
    s=escolha(cfg,5)
    s['configuracao']['gmm']['n_components']=7
    with pytest.raises(ValueError,match='grade'):
        validar_vencedor(cfg,s,metodo='PCA')


def test_amostra_f10_fracao_semente_e_folds_invariantes():
    cfg=cfg_base()
    folds=np.repeat(np.arange(1,11,dtype=np.uint8),100)
    y=np.tile(np.r_[np.ones(12,dtype=np.uint8),np.zeros(88,dtype=np.uint8)],10)
    ids1=selecionar_f10_estratificado(folds,y,cfg)
    assert np.all(folds[ids1]==10)
    assert np.array_equal(ids1,selecionar_f10_estratificado(folds,y,cfg))
    cfg2=deepcopy(cfg);cfg2['divisao']['fracao_amostra_f10']=.6
    ids2=selecionar_f10_estratificado(folds,y,cfg2)
    assert len(ids2)>len(ids1)
    assert np.all(folds[ids2]==10)
    cfg2['experimento']['semente']=45
    ids3=selecionar_f10_estratificado(folds,y,cfg2)
    assert not np.array_equal(ids2,ids3)


def test_preflight_amostra_inviavel_k_rejeita_antes_fit():
    cfg=cfg_base()
    cfg['gmm']['n_components']=[400]
    df, folds,_=dados_sinteticos()
    with pytest.raises(ValueError,match='GMM K=400'):
        validar_viabilidade_oof(folds,df.TARGET_OMEGA,cfg)


def test_folds_semente_alterada_sem_recriar_automaticamente(tmp_path):
    cfg=cfg_base()
    arquivo=tmp_path/'data/raw/dataset_raw.parquet'
    arquivo.parent.mkdir(parents=True)
    arquivo.write_bytes(b'APENAS_HASH_SINTETICO_NAO_E_PARQUET')
    div=tmp_path/'data/splits/um';div.mkdir(parents=True)
    f=np.repeat(np.arange(1,11,dtype=np.uint8),5)
    np.save(div/'folds.npy',f)
    meta={'parametros':{'sha256_parquet':sha256(arquivo.read_bytes()).hexdigest(),
           'caminho_relativo':cfg['dados']['arquivo'],'semente':42,
           'coluna_alvo':cfg['dados']['coluna_alvo'],'classe_positiva':1,
           'total_folds':10,'fold_teste':10,'estratificada':True,'embaralhar':True},
          'sha256_folds':sha256((div/'folds.npy').read_bytes()).hexdigest()}
    (div/'manifesto_folds.json').write_text(json.dumps(meta),encoding='utf-8')
    with pytest.raises(RuntimeError,match='divisao compativel'):
        orq._identificar_folds(tmp_path,cfg,arquivo) # seed=81, manifesto seed=42
    assert (div/'folds.npy').is_file() and meta['sha256_folds']==sha256((div/'folds.npy').read_bytes()).hexdigest()


def test_plot_tsne3d_k_3_e_k_5_sem_exigir_quatro():
    from src.visualizacao_tsne3d_interativa import _dados_validos, _figura_html
    rng=np.random.default_rng(112)
    n=20;z=pd.DataFrame(rng.normal(size=(n,3)),columns=['TSNE1','TSNE2','TSNE3'])
    y=np.tile([0,1],n//2)
    scores=np.linspace(.01,.98,n)
    for k in (3,5):
        gamma=rng.dirichlet(np.ones(k),size=n)
        dados=_dados_validos(z,y,scores=scores,limiar_oof=.4,responsabilidades=gamma)
        assert dados[4].shape==(n,k)
        fig, modos=_figura_html(*dados[:6],origem='desenvolvimento',fracao_f10=None,tipo_id=dados[6])
        nomes=[trace.name for trace in fig.data]
        assert len([x for x in nomes if x.startswith('G')])<=k
        assert 'Gaussiana dominante' in modos


def test_ooof_grade_sintetica_seleciona_k5_sem_f10(tmp_path,monkeypatch):
    cfg=cfg_base('PCA',1)
    df,folds,_=dados_sinteticos()
    # A grade usa scores artificiais deterministas; nao treina modelos nesta etapa.
    def fake_predicoes(df_,f_,config_,candidato,fold):
        assert not np.any(f_==10)
        ix=np.flatnonzero(f_==fold)
        y=df_.iloc[ix].TARGET_OMEGA.to_numpy()
        p=np.where(y==1,.96,.04) if candidato['gmm']['n_components']==5 else np.repeat(.35,len(ix))
        return pd.DataFrame({'linha_original':ix,'fold':np.full(len(ix),fold),
                             'y_true':y,'prob_fraude':p})
    monkeypatch.setattr(orq,'prever_fold_oof',fake_predicoes)
    # O seletor nao recebe F10 e escreve SOMENTE numa pasta sintetica temporaria.
    exp, manifest,_=orq._abrir_execucao(tmp_path,cfg,'CHAVE_TESTE',ensaio=True,nova_execucao=True)
    resultado=orq.executar_grade(df,folds,np.arange(len(df)),np.r_[folds,np.repeat(10,36)],
                                 cfg,exp,manifest,ensaio=True)
    assert resultado['configuracao']['gmm']['n_components']==5
    assert resultado['teste_final_usado'] is False
    assert resultado['limiar_mcc_oof'] <= 1
    assert validar_vencedor(cfg,resultado,metodo='PCA',dimensao=1)['gmm']['n_components']==5


@pytest.mark.skipif(importlib.util.find_spec('openTSNE') is None,
                    reason='openTSNE nao instalado: validacao real do embedding pendente')
def test_tsne_verdadeiro_em_base_sintetica():
    from src.reducao_tsne import RedutorTSNE
    cfg=cfg_base('TSNE',3)
    x=pd.DataFrame(np.random.default_rng(55).normal(size=(40,5)))
    model=RedutorTSNE(cfg,{'n_components':3,'perplexity':10,'initialization':'random',
                           'n_iter':300,'learning_rate':'auto',
                           'early_exaggeration':12.,'theta':.5,'metric':'euclidean'})
    emb=model.fit(x).transform_treino()
    assert emb.shape==(40,3)

@pytest.mark.parametrize('dim,k',[(1,3),(1,5),(2,3),(2,5),(3,3),(3,5)])
def test_gmm01_gmm04_full_e_paleta_k_variavel_sintetico(tmp_path, dim, k):
    """Renderizacao PCA GMM01/GMM04 1D-3D sem carregar dataset real."""
    from types import SimpleNamespace
    from sklearn.mixture import GaussianMixture
    from src import visualizacao_gmm01 as g01, visualizacao_gmm04 as g04
    from src.cores_gaussianas import cor_gaussiana
    cfg=cfg_base('PCA',dim)
    cfg['visualizacao']['graficos']['densidades_marginais'].update(
        pontos_curva=110,pontos_grade_2d=80)
    cfg['visualizacao']['graficos']['regioes_dominancia'].update(
        resolucao_1d=110,resolucao_2d=30,pontos_scatter=40)
    rng=np.random.default_rng(k*100+dim)
    centros=np.linspace(-2,2,k)
    X=np.concatenate([rng.normal(loc=c, scale=.75, size=(25,dim)) for c in centros])
    modelo=GaussianMixture(n_components=k,covariance_type='full', n_init=1,
                           random_state=42,reg_covar=1e-4).fit(X)
    nomes=[f'PC{i+1}' for i in range(dim)]
    colunas=[f'G{i+1}' for i in range(k)]
    gmm=SimpleNamespace(modelo_=modelo,colunas_=nomes,saidas_=colunas)
    data={'gmm':gmm,'coord':pd.DataFrame(X,columns=nomes),
          'y':np.zeros(len(X),dtype=np.uint8)}
    g01.gerar(cfg,data,tmp_path,'png',55,True)
    assert (tmp_path/f'GMM01_densidade_marginal_{dim}d_PC1.png').is_file()
    c=pd.read_csv(tmp_path/f'GMM01_densidade_marginal_{dim}d_PC1.csv')
    assert all(f'contribuicao_G{i+1}' in c.columns for i in range(k))
    assert np.allclose(c['densidade_mistura'],
                       c[[f'contribuicao_G{i+1}' for i in range(k)]].sum(axis=1))
    g04.gerar(cfg,data,tmp_path,'png',55,True)
    assert (tmp_path/f'GMM04_dominancia_{dim}d.png').is_file()
    assert len({cor_gaussiana(i) for i in range(k)})==k
