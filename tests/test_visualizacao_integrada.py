"""Testes 07.03: validacoes sinteticas de integracao e mapa complementar; SEM F10."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import visualizacao_integrada as m
from src import visualizacao_geometrica as geo


@pytest.fixture
def config():
    raiz=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((raiz/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf-8-sig'))
    cfg['visualizacao']['padrao'].update(dpi=55,figsize=[5,3])
    cfg['visualizacao']['graficos']['mapa_risco_fronteira'].update(resolucao_2d=25,n_pontos_scatter=140)
    cfg['visualizacao']['geometria']['pontos_maximos_amostra']=200
    return cfg


@pytest.fixture
def ranking():
    def cand(pca,k):
        return {'metodo':'PCA','reducao':{'n_components':pca,'whiten':False,'svd_solver':'auto','iterated_power':'auto'},
                'gmm':{'n_components':k,'covariance_type':'diag','reg_covar':1e-6,'max_iter':300,
                       'n_init':2,'tol':.001,'init_params':'kmeans'},
                'modo':'NORMAL','razao':None,'suavizacao':0.0}
    grade=pd.DataFrame([
        {'id_candidato':'winner','ap':.012,'mcc':.04,'score':.51,'limiar_mcc':.008,'configuracao':json.dumps(cand(1,4))},
        {'id_candidato':'candidate_2d','ap':.011,'mcc':.035,'score':.50,'limiar_mcc':.009,'configuracao':json.dumps(cand(2,2))},
        {'id_candidato':'candidate_other','ap':.01,'mcc':.03,'score':.49,'limiar_mcc':.01,'configuracao':json.dumps(cand(2,4))},
        {'id_candidato':'candidate_3d','ap':.01,'mcc':.02,'score':.48,'limiar_mcc':.012,'configuracao':json.dumps(cand(3,2))},
    ])
    escolha={'id_candidato':'winner','score_oof':.51,'configuracao':cand(1,4),
             'limiar_mcc_oof':.008,'teste_final_usado':False}
    return grade,escolha


def test_melhor_pca2d_usa_proprio_limiar(ranking):
    grade,escolha=ranking
    c=m._config_candidata_2d(grade,escolha)
    assert c['id_candidato']=='candidate_2d'
    assert c['limiar_mcc_oof']==.009
    assert c['configuracao']['reducao']['n_components']==2
    assert escolha['limiar_mcc_oof']==.008


def test_nao_inventa_segundo_pca(ranking):
    grade,escolha=ranking
    grade=grade.iloc[[0,3]]
    assert m._config_candidata_2d(grade,escolha) is None
    e=deepcopy(escolha)
    e['configuracao']['reducao']['n_components']=2
    assert m._config_candidata_2d(ranking[0],e) is None
    e['configuracao']['metodo']='TSNE'
    assert m._config_candidata_2d(ranking[0],e) is None


def test_npz_candidato_verifica_fold10_e_mesmos_ids(tmp_path,ranking):
    grade,escolha=ranking
    c=m._config_candidata_2d(grade,escolha)
    pasta=tmp_path/'oof'/'predicoes';pasta.mkdir(parents=True)
    ids=np.arange(9,dtype=np.int64); y=np.array([0,1,0,1,0,1,0,1,0],dtype=np.int8)
    folds=np.arange(1,10,dtype=np.uint8)
    destino=pasta/'candidate_2d.npz'
    def salvar(fs,linhas=ids):
        np.savez_compressed(destino,linha_original=linhas,y_true=y,fold=fs,
                            fold_esperado=np.uint8(0),id_candidato=np.array('candidate_2d'))
    salvar(folds)
    m._validar_npz_candidato(tmp_path,c,ids,y,folds)
    errado=folds.copy();errado[2]=10
    salvar(errado)
    with pytest.raises(ValueError,match='Fold F10'):
        m._validar_npz_candidato(tmp_path,c,ids,y,folds)
    salvar(folds,ids[::-1])
    with pytest.raises(ValueError,match='mesmas observacoes'):
        m._validar_npz_candidato(tmp_path,c,ids,y,folds)


def test_copia_reutilizavel_nao_sobrescreve(tmp_path):
    a=tmp_path/'a.png';b=tmp_path/'sub'/'b.png'
    a.write_bytes(b'conteudo1')
    m._arquivo_seguro_copiar(a,b)
    m._arquivo_seguro_copiar(a,b)
    assert b.read_bytes()==a.read_bytes()
    b.write_bytes(b'arquivo outro')
    with pytest.raises(ValueError,match='Conflito'):
        m._arquivo_seguro_copiar(a,b)


def test_html_evidencia_nao_vencedor(ranking):
    _,e=ranking
    texto=m._html_galeria(Path('exp'),[
        (Path('oof/01_a.png'),'Grafico OOF'),
        (Path('complementar_2d/16_b.png'),'Mapa complementar')],
        e,{},'PCA','ensaio')
    assert 'COMPLEMENTAR' not in texto or 'complementar' in texto.lower()
    assert 'NAO representa a fronteira do vencedor' in texto
    assert 'F10' in texto
    assert '01_a.png' in texto and '16_b.png' in texto


def test_planejar_nao_cria_galeria_ou_treina(monkeypatch,tmp_path,config,ranking):
    grade,e=ranking
    root=tmp_path/'ensaio';root.mkdir()
    (root/'manifesto.json').write_text('{"status":"concluido_06_03","tipo":"ensaio"}',encoding='utf-8')
    monkeypatch.setattr(m.oof,'_ler_resultados',lambda _: (config,grade,e,np.array([]),np.array([])))
    monkeypatch.setattr(m.geo,'_treinar_vencedor',lambda *args:pytest.fail('Planejar nunca treina'))
    summary=m.integrar(root,planejar=True)
    assert summary['id_candidato_2d_complementar']=='candidate_2d'
    assert not (root/'visualizacao_consolidada').exists()


def test_integrar_dois_graficos_sem_treinar_complementar(monkeypatch,tmp_path,config,ranking):
    grade,e=ranking
    root=tmp_path/'ensaio';root.mkdir()
    (root/'manifesto.json').write_text('{"status":"concluido_06_03","tipo":"ensaio"}',encoding='utf-8')
    for name in list(m.oof.ARQUIVOS):
        config['visualizacao']['graficos'][name]['habilitar']=(name=='curva_pr')
    for name in list(geo.TIPOS):
        config['visualizacao']['graficos'][name]['habilitar']=(name=='risco_por_gaussiana')
    od=root/'oof'/'graficos_gerados';gd=root/'teste_final'/'graficos_gerados'
    od.mkdir(parents=True);gd.mkdir(parents=True)
    (od/'01_curva_precision_recall.png').write_bytes(b'png1')
    (gd/'07_risco_por_gaussiana.png').write_bytes(b'png2')
    (od/'manifesto_graficos_oof_07_01.json').write_text(json.dumps({'id_vencedor':'winner','usa_f10':False}),encoding='utf-8')
    (gd/'manifesto_geometria_07_02.json').write_text(json.dumps({'id_vencedor':'winner',
        'f10_utilizado':False,'origem':'desenvolvimento_F1_F9'}),encoding='utf-8')
    monkeypatch.setattr(m.oof,'_ler_resultados',lambda _: (config,grade,e,np.array([]),np.array([])))
    monkeypatch.setattr(m.geo,'_treinar_vencedor',lambda *args:pytest.fail('Sem complementar nao treina'))
    r=m.integrar(root,planejar=False,complementar_2d=False)
    assert r['numero_imagens']==2
    assert 'nao_aplicavel'==r['complementar_2d']['situacao']
    assert (root/'visualizacao_consolidada'/'index.html').is_file()
    r2=m.integrar(root,planejar=False,complementar_2d=False)
    assert r2['numero_imagens']==2
    assert (od/'01_curva_precision_recall.png').read_bytes()==b'png1'


def test_desenha_nuvens_e_risco_com_candidato_distinto(config,ranking,tmp_path):
    rng=np.random.default_rng(33);n=300
    x=rng.normal(size=n);z=rng.normal(size=n)
    y=(rng.random(n)<(.12+.05*(x>0))).astype(np.uint8)
    df=pd.DataFrame({'TARGET_OMEGA':y,'NID_ALPHA':np.arange(n),
         'VALOR':x,'V2':z,'RECEIVE_LOC_FEWF':rng.choice(['fraud_x','fraud_y'],n),
         'TRANS_HOUR_SIN':np.sin(x),'TRANS_HOUR_COS':np.cos(x),
         'RECIVE_CATEGORY_OHEWI':rng.choice(['a','b'],n),
         'SEND_GENDER_BE':rng.choice(['F','M'],n)})
    c=m._config_candidata_2d(*ranking)
    data=geo._treinar_vencedor(df,config,c)
    arq=tmp_path/'16_mapa_complementar.png'
    meta=m._desenhar_complementar(config,data,c,arq)
    assert arq.is_file() and arq.stat().st_size>1000
    assert meta['n_pontos_malha']==625
    assert len(data['gmm'].saidas_)==2
    assert c['id_candidato']!='winner'


def test_manifesto_geometrico_invalido_impede_integracao(monkeypatch,tmp_path,config,ranking):
    grade,e=ranking;root=tmp_path/'exp';root.mkdir()
    monkeypatch.setattr(m.oof,'_ler_resultados',lambda _: (config,grade,e,np.array([]),np.array([])))
    gd=root/'teste_final'/'graficos_gerados';gd.mkdir(parents=True)
    (gd/'manifesto_geometria_07_02.json').write_text(json.dumps({'id_vencedor':'winner',
        'f10_utilizado':True,'origem':'desenvolvimento_F1_F9'}),encoding='utf-8')
    (root/'manifesto.json').write_text('{"status":"concluido_06_03","tipo":"ensaio"}',encoding='utf-8')
    with pytest.raises(ValueError,match='Manifesto do 07.02'):
        m.integrar(root,planejar=True)
