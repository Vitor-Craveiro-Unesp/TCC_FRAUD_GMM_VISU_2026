"""Passo 07.02: testes sinteticos; nenhum teste consulta dados F10."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import matthews_corrcoef
from src import visualizacao_geometrica as m

@pytest.fixture(scope='module')
def cfg():
    path=Path(__file__).resolve().parents[1]/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    c=yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    c['visualizacao']['padrao']['figsize']=[5,3]
    c['visualizacao']['padrao']['dpi']=60
    c['visualizacao']['geometria']['pontos_maximos_amostra']=150
    c['visualizacao']['graficos']['mapa_risco_fronteira']['resolucao_1d']=60
    c['visualizacao']['graficos']['mapa_risco_fronteira']['resolucao_2d']=30
    return c

@pytest.fixture(scope='module')
def sint(cfg):
    rng=np.random.default_rng(321)
    n=450
    x1=rng.normal(size=n);x2=rng.normal(size=n)
    y=(rng.random(n)<(.1+ .04*(x1>0))).astype(np.uint8)
    df=pd.DataFrame({'TARGET_OMEGA':y,'NID_ALPHA':np.arange(n),
         'VALOR':x1,'V2':x2,'RECEIVE_LOC_FEWF':rng.choice(['fraud_x','fraud_y'],n),
         'TRANS_HOUR_SIN':np.sin(x1),'TRANS_HOUR_COS':np.cos(x1),
         'RECIVE_CATEGORY_OHEWI':rng.choice(['a','b','c'],n),
         'SEND_GENDER_BE':rng.choice(['F','M'],n)})
    pca=copy.deepcopy(cfg)
    pca['pca']['n_components']=[1,2,3]
    can={'metodo':'PCA', 'modo':'NORMAL','razao':None,
         'reducao':{'n_components':1,'whiten':False,'svd_solver':'auto','iterated_power':'auto'},
         'gmm':{'n_components':4,'covariance_type':'full','reg_covar':1e-6,
                'max_iter':300,'n_init':2,'tol':.001,'init_params':'kmeans'},
         'suavizacao':0.0}
    escolha={'id_candidato':'modelo_sint','configuracao':can,'limiar_mcc_oof':.12,'teste_final_usado':False}
    return df,pca,escolha


def test_vencedor_pca_1d_treinado_sem_grid(sint):
    df,c,e=sint
    r=m._treinar_vencedor(df,c,e)
    assert r['coord'].shape==(len(df),1)
    assert r['gamma'].shape==(len(df),4)
    assert np.allclose(r['gamma'].sum(axis=1),1,atol=1e-6)
    assert abs(r['rh'].massas_.sum()-len(df))<.001
    assert len(r['rh'].riscos_)==4
    assert r['n_treino_gmm']==len(df)
    assert 0<r['rh'].prevalencia_treino_<1


def test_quatro_tipos_pca_com_painel_salvos(sint,tmp_path):
    df,c,e=sint
    data=m._treinar_vencedor(df,c,e)
    ext,dpi,csv=m._estilo(c)
    m._painel_parametros_gaussianos(c,data,tmp_path,ext,dpi,csv)
    m._scree(c,data,tmp_path,ext,dpi,csv)
    m._projecoes(c,data,tmp_path,ext,dpi,csv,'PCA')
    m._fronteira(c,data,e,tmp_path,ext,dpi,csv,
                avaliacao={"y":data["y"],"p":data["scores"]})
    m._relatorio(c,data,tmp_path)
    imagens=sorted(tmp_path.glob('*.png'))
    assert len(imagens)==4,imagens
    assert (tmp_path/'relatorio_gaussianas_desenvolvimento.csv').exists()
    assert (tmp_path/'relatorio_gaussianas_desenvolvimento.json').exists()
    assert len(pd.read_csv(tmp_path/'09_painel_parametros_gaussianos.csv'))==4
    assert len(pd.read_csv(tmp_path/'GMM08_malha_PC1.csv'))==60
    assert (tmp_path/'GMM08_mapa_risco_1d_comparativo.png').exists()
    assert not (tmp_path/'GMM08_mapa_risco_1d_fixo.png').exists()
    assert not (tmp_path/'GMM08_mapa_risco_1d_mcc.png').exists()
    assert 'F10' in (tmp_path/'relatorio_gaussianas_desenvolvimento.md').read_text(encoding='utf-8')


def test_mapa_gmm08_2d_comparativo_unico(sint, tmp_path, monkeypatch):
    df, c, e = sint
    d = copy.deepcopy(e)
    d['configuracao']['reducao']['n_components'] = 2
    data = m._treinar_vencedor(df, c, d)
    capturas = []
    original = m._guardar

    def espiar(fig, destino, nome, ext, dpi):
        if nome.startswith('GMM08_mapa_risco_'):
            capturas.append({
                'nome': nome,
                'eixos': len(fig.axes),
                'cabecalhos': [t.get_text() for t in fig.texts],
                'escalas': [ax.collections[0].get_clim() for ax in fig.axes[:2]],
            })
        return original(fig, destino, nome, ext, dpi)

    monkeypatch.setattr(m, '_guardar', espiar)
    m._fronteira(c, data, d, tmp_path, 'png', 60, True,
                avaliacao={'y':data['y'], 'p':data['scores'], 'origem':'simulado'})
    assert len(capturas) == 1
    assert capturas[0]['nome'] == 'GMM08_mapa_risco_2d_comparativo'
    assert capturas[0]['eixos'] == 3  # 2 mapas + 1 barra compartilhada.
    assert capturas[0]['escalas'][0] == capturas[0]['escalas'][1]
    textos = ' '.join(capturas[0]['cabecalhos'])
    assert '0,50' in textos and '0.12' in textos
    assert 'MCC da simulação' in textos and 'F10 não utilizado' in textos
    assert m.NUM['mapa_risco_fronteira'] == 8
    assert (tmp_path/'GMM08_mapa_risco_2d_comparativo.png').is_file()
    assert len(list(tmp_path.glob('GMM08_mapa_risco_*.png'))) == 1
    assert len(list(tmp_path.glob('GMM08_malha_*.csv'))) == 1


def test_respeita_pares_geometricos_de_2d_e_3d(sint,tmp_path,monkeypatch):
    df,c,e=sint
    d=copy.deepcopy(e)
    d['configuracao']['reducao']['n_components']=3
    data=m._treinar_vencedor(df,c,d)
    assert m._pares(3,c)==[(0,1),(0,2),(1,2)]
    capturas=[]
    salvar_original=m._guardar

    def espiar_figura(fig,dest,nome,ext,dpi):
        if nome.startswith('GMM08_mapa_risco_3d_'):
            titulos=[ax.get_title() for ax in fig.axes[:6]]
            legenda=[t.get_text() for l in fig.legends for t in l.get_texts()]
            textos=[t.get_text() for t in fig.texts]
            escalas=[ax.collections[0].get_clim() for ax in fig.axes[:6]]
            capturas.append((nome,titulos,legenda,textos,escalas,
                             fig.axes[-1].get_position().x1))
        return salvar_original(fig,dest,nome,ext,dpi)

    monkeypatch.setattr(m,'_guardar',espiar_figura)
    m._fronteira(c,data,d,tmp_path,'png',60,True,
                avaliacao={'y':data['y'],'p':data['scores'],'origem':'simulado'})
    assert len(capturas)==1
    nome,titulos,legenda,textos,escalas,cbar_direita=capturas[0]
    assert nome=='GMM08_mapa_risco_3d_comparativo'
    assert len(titulos)==6
    for i in (0,2,4):
        assert titulos[i]==titulos[i+1]
        assert 'fixada na média do desenvolvimento' in titulos[i]
    assert len(set(escalas))==1
    assert all(any(f'G{k} ' in titulo for titulo in legenda) for k in range(1,5))
    assert sum('MCC da simulação' in t for t in textos)>=2
    assert any('F10 não utilizado' in t for t in textos)
    assert cbar_direita < .95
    assert len(list(tmp_path.glob('GMM08_mapa_risco_*.png')))==1
    assert len(list(tmp_path.glob('GMM08_malha_*.csv')))==3
    m._projecoes(c,data,tmp_path,'png',60,True,'PCA')
    # PDF PCA 12: 3 PCs selecionados => UMA projeção 3D verdadeira.
    # Os três pares continuam obrigatórios para os cortes do GMM 08.
    assert len(list(tmp_path.glob('12_pca_projecao_*.png')))==1
    assert (tmp_path/'12_pca_projecao_PC1_PC2_PC3.png').exists()


def test_covar_full_em_1d(sint):
    df,c,e=sint
    d=m._treinar_vencedor(df,c,e)
    z=m._covariancia(d['gmm'],0)
    assert z.shape==(1,1)
    assert z[0,0]>0


def test_flag_e_exclusao_grafico_nao_aplicavel(cfg):
    assert m._ativo(cfg,'painel_parametros_gaussianos')
    c=copy.deepcopy(cfg)
    c['visualizacao']['graficos']['painel_parametros_gaussianos']['habilitar']=False
    assert not m._ativo(c,'painel_parametros_gaussianos')


def test_nunca_permite_limiar_invalido(cfg):
    with pytest.raises(ValueError,match='invalido'):
        m._limiares(cfg,{'limiar_mcc_oof':1.2})


def test_amostra_plot_indices_validos(cfg):
    y=np.array([0]*950+[1]*50,dtype=np.uint8)
    idx=m._amostra_plot(1000,y,cfg,100)
    assert len(idx)==100 and (np.diff(idx)>0).all()
    assert (y[idx]==1).any()


def test_planejar_nao_reajusta_ou_le_parquet(cfg, sint, tmp_path, monkeypatch):
    _,_,e=sint
    exp=tmp_path/'experimento';exp.mkdir()
    (exp/'manifesto.json').write_text('{"status":"concluido_06_03","tipo":"ensaio"}')
    monkeypatch.setattr(m.vis,'_ler_resultados',lambda _: (cfg,None,e,np.array([0,1]),np.array([.1,.8])))
    monkeypatch.setattr(m,'_treinar_vencedor',lambda *a: pytest.fail('Nao pode treinar no planejamento'))
    r=m.executar(exp,raiz=tmp_path,somente_planejar=True)
    assert r['planejamento'] is True
    assert len(r['graficos'])==8
    assert 'pca_biplot' in r['graficos']
    assert 'tsne_mapa' not in r['graficos']


def test_bloqueia_fold10_nos_indices_originais(cfg, sint, tmp_path, monkeypatch):
    _,_,e=sint
    exp=tmp_path/'experimento';(exp/'oof'/'predicoes').mkdir(parents=True)
    cid=e['id_candidato'];y=np.array([0,1,0,1,0,1,0,1,0],dtype=np.uint8)
    i=np.arange(9,dtype=np.int64)
    np.savez_compressed(exp/'oof'/'predicoes'/f'{cid}.npz',
        linha_original=i,fold=np.arange(1,10,dtype=np.uint8))
    (exp/'manifesto.json').write_text('{"status":"concluido_06_03","tipo":"ensaio",'
        '"sha256_parquet":"hashbase","sha256_folds":"hashfolds",'
        '"registros_desenvolvimento":9}')
    monkeypatch.setattr(m.vis,'_ler_resultados',lambda _: (cfg,None,e,y,np.ones(9)/9))
    folds=np.arange(1,10,dtype=np.uint8);folds[1]=10
    monkeypatch.setattr(m.orq,'_identificar_folds',lambda *a: (folds,tmp_path/'folds.npy','hashbase'))
    monkeypatch.setattr(m.orq,'hash_arquivo',lambda _: 'hashfolds')
    with pytest.raises(ValueError,match='F10 ou folds trocados'):
        m._selecao(exp,tmp_path)
