"""Novo layout, fluxo mestre e retomada: testes artificiais sem Parquet/F10."""
import json
from pathlib import Path
import pytest
from src import estrutura_saida as layout
from src import orquestracao_oof as orq


def _criar_experimento(tmp_path):
    p=tmp_path/'ensaio_pca_gmm_20261021_999999'
    p.mkdir()
    (p/'manifesto.json').write_text(json.dumps({
        'status':'concluido_06_03', 'tipo':'ensaio',
        'versao_layout':2,'visualizacoes':'pendentes',
        'pendente_em_outros_passos':['modelo_final_serializado','teste_F10'],
    }),encoding='utf-8')
    return p


def test_contrato_pastas_e_f10_isolado(tmp_path):
    cfg={'gmm':{'n_components':6}}
    assert layout.candidato(tmp_path,'abc123',cfg)==tmp_path/'candidatos'/'K06_abc123'
    assert layout.galeria(tmp_path)==tmp_path/'vencedor'/'visualizacao'/'galeria'
    assert layout.imagens_geometria(tmp_path)!=layout.teste_final(tmp_path)
    assert layout.predicoes_vencedor(tmp_path)==tmp_path/'vencedor'/'oof'/'predicoes.npz'


def test_orquestrador_dispara_tres_geradores(monkeypatch,tmp_path):
    from src import visualizacao_oof as oof
    from src import visualizacao_geometrica as geo
    from src import visualizacao_integrada as integ
    p=_criar_experimento(tmp_path)
    chamados=[]
    monkeypatch.setattr(oof,'gerar_graficos_oof',lambda exp:chamados.append(('OOF',exp)))
    monkeypatch.setattr(geo,'executar',lambda exp:chamados.append(('GEO',exp)))
    monkeypatch.setattr(integ,'integrar',lambda exp,**kw:chamados.append(('GAL',exp,kw)))
    orq.executar_visualizacoes(p)
    assert [x[0] for x in chamados]==['OOF','GEO','GAL']
    assert all(x[1]==p for x in chamados)
    assert chamados[-1][-1]=={'complementar_2d':False}
    m=json.loads((p/'manifesto.json').read_text())
    assert m['visualizacoes']=='concluidas'
    assert m['visualizacoes_concluidas']==['07.01','07.02','07.03']
    assert 'teste_F10' in m['pendente_em_outros_passos']
    orq.executar_visualizacoes(p)
    assert len(chamados)==3, 'retomada deve pular etapas anteriores'


def test_falha_visual_preserva_oof_e_retomada(monkeypatch,tmp_path):
    from src import visualizacao_oof as oof
    from src import visualizacao_geometrica as geo
    from src import visualizacao_integrada as integ
    p=_criar_experimento(tmp_path)
    ordem=[]
    monkeypatch.setattr(oof,'gerar_graficos_oof',lambda exp:ordem.append('07.01'))
    def geo_erro(exp):
        ordem.append('07.02')
        raise RuntimeError('falha simulada')
    monkeypatch.setattr(geo,'executar',geo_erro)
    monkeypatch.setattr(integ,'integrar',lambda exp,**kw:ordem.append('07.03'))
    with pytest.raises(RuntimeError,match='falha simulada'):
        orq.executar_visualizacoes(p)
    m=json.loads((p/'manifesto.json').read_text())
    assert m['status']=='concluido_06_03'
    assert m['visualizacoes_concluidas']==['07.01']
    assert m['visualizacoes']=='interrompidas_retomaveis'
    monkeypatch.setattr(geo,'executar',lambda exp:ordem.append('07.02'))
    orq.executar_visualizacoes(p)
    assert ordem==['07.01','07.02','07.02','07.03']
    m=json.loads((p/'manifesto.json').read_text())
    assert m['visualizacoes']=='concluidas'


def test_rejeita_resultado_ou_layout_antigo(tmp_path):
    p=_criar_experimento(tmp_path)
    m=json.loads((p/'manifesto.json').read_text())
    m['versao_layout']=1
    (p/'manifesto.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='novo layout'):
        orq.executar_visualizacoes(p)


def test_tres_candidatos_seleciona_score_e_isola_f10(monkeypatch,tmp_path):
    import numpy as np
    import pandas as pd
    import yaml
    from src.validacao_oof import grade_candidatos_oof
    config=yaml.safe_load((Path(__file__).resolve().parents[1] /
        'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf-8-sig'))
    folds_orig=np.repeat(np.arange(1,11),6).astype(np.uint8)
    orig=np.flatnonzero(folds_orig!=10)
    fdev=folds_orig[orig]
    y=np.tile(np.array([0,0,0,0,1,1],dtype=np.uint8),9)
    df=pd.DataFrame({'TARGET_OMEGA':y})
    grade=list(grade_candidatos_oof(config))
    assert [c['gmm']['n_components'] for c in grade]==[2,4,6]
    monkeypatch.setattr(orq,'grade_candidatos_oof',lambda c:iter(grade))
    def fake_oof(df,folds,config,c,k):
        idx=np.flatnonzero(folds==k)
        kk=int(c['gmm']['n_components'])
        probabilities={2:(.20,.30),4:(.10,.50),6:(.01,.90)}
        p0,p1=probabilities[kk]
        return pd.DataFrame({'linha_original':idx,'fold':folds[idx],
                             'y_true':df['TARGET_OMEGA'].to_numpy()[idx],
                             'prob_fraude':np.where(y[idx]==1,p1,p0)})
    monkeypatch.setattr(orq,'prever_fold_oof',fake_oof)
    pasta,m,_=orq._abrir_execucao(tmp_path,config,'GRADE3-TESTE',ensaio=True)
    escolha=orq.executar_grade(df,fdev,orig,folds_orig,config,pasta,m,ensaio=True)
    assert escolha['configuracao']['gmm']['n_components']==6
    assert len(list((pasta/'candidatos').glob('*/metricas/metricas.json')))==3
    assert len(list((pasta/'candidatos').glob('*/oof/cache/*.npz')))==27
    assert len(list((pasta/'candidatos').glob('*/oof/predicoes.npz')))==3
    assert layout.predicoes_vencedor(pasta).is_file()
    assert (layout.metricas_vencedor(pasta)/'resultados_grade.csv').is_file()
    assert layout.teste_final(pasta).is_dir()
    assert not list(layout.teste_final(pasta).rglob('*.npz'))
    with np.load(layout.predicoes_vencedor(pasta),allow_pickle=False) as a:
        assert not np.any(a['fold']==10)
        assert a['id_candidato'].item()==escolha['id_candidato']
        assert len(a['linha_original'])==54
    ranking=pd.read_csv(layout.metricas_vencedor(pasta)/'resultados_grade.csv')
    assert ranking['score'].is_monotonic_decreasing
    assert ranking.iloc[0]['id_candidato']==escolha['id_candidato']
