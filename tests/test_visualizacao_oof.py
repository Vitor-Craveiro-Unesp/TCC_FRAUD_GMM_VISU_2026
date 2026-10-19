"""Testes isolados do Passo 07.01; nunca usam F10 nem o dataset real."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import average_precision_score, log_loss, matthews_corrcoef

from src.visualizacao_oof import gerar_graficos_oof, _mcc_curva_exata, _numeros_painel_00, _ponto_operacional_pr


@pytest.fixture
def experimento(tmp_path):
    cfg_path = Path(__file__).resolve().parents[1] / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    if cfg_path.is_file():
        config=yaml.safe_load(cfg_path.read_text(encoding='utf-8-sig'))
    else:
        config={
            'saidas':{'pastas':{'oof':'oof','graficos':'graficos_gerados'},'salvar_graficos':True},
            'visualizacao':{'habilitar':True,'gerar_em':['oof','teste_final'],
                'padrao':{'figsize':[9,6],'formato':'png','dpi':100,'salvar_dados_grafico':True},
                'graficos':{name:{'habilitar':True,'gerar_em':['oof']} for name in (
                  'painel_score_composto','curva_pr','curva_mcc','matriz_confusao_05','matriz_confusao_mcc',
                  'distribuicao_probabilidades','comparacao_hiperparametros')},
                'estilo_matrizes':{}}}
    config['visualizacao']['padrao']['dpi']=70
    config['visualizacao']['padrao']['figsize']=[6,4]
    config['visualizacao']['graficos']['curva_mcc']['pontos_visualizacao']=20
    config['visualizacao']['graficos']['distribuicao_probabilidades']['bins']=10
    config['visualizacao']['graficos']['comparacao_hiperparametros']['maximo_configuracoes']=3
    root=tmp_path/'ensaio_pca_gmm_20261017_195252'
    pasta=root/'oof';(pasta/'predicoes').mkdir(parents=True)
    (root/'configuracao.yaml').write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf-8')
    rng=np.random.default_rng(42)
    n=360
    y=rng.choice([0,1],n,p=[.92,.08]).astype(np.uint8)
    p=np.clip(rng.beta(2,18,n) + y*.07,1e-5,1-1e-5)
    limiar=.13
    ap=average_precision_score(y,p)
    mcc=matthews_corrcoef(y,p>=limiar)
    ll=log_loss(y,p,labels=[0,1]);score=(ap+(mcc+1)/2+np.exp(-ll))/3
    cid='abc012'
    cand={'metodo':'PCA','modo':'NORMAL','razao':None,'reducao':{'n_components':2},'gmm':{'n_components':4},'suavizacao':0.0}
    grade=pd.DataFrame([{'id_candidato':cid,'ap':ap,'mcc':mcc,'logloss':ll,'score':score,
        'limiar_mcc':limiar,'n':n,'prevalencia':y.mean(),
        'configuracao':json.dumps(cand)},
        {'id_candidato':'xyz999','ap':ap/2,'mcc':mcc/2,'logloss':ll,'score':score-.1,
         'limiar_mcc':limiar,'n':n,'prevalencia':y.mean(),
         'configuracao':json.dumps({**cand,'gmm':{'n_components':2}})}])
    grade.to_csv(pasta/'resultados_grade.csv',index=False)
    escolha={'id_candidato':cid,'configuracao':cand,'limiar_mcc_oof':limiar,
             'score_oof':score,'status':'selecionado_em_ensaio','teste_final_usado':False}
    (pasta/'melhor_configuracao.json').write_text(json.dumps(escolha),encoding='utf-8')
    np.savez_compressed(pasta/'predicoes'/f'{cid}.npz',
        y_true=y,prob_fraude=p,fold=np.tile(np.arange(1,10,dtype=np.uint8),n//9),
        linha_original=np.arange(n,dtype=np.int64),id_candidato=np.array(cid),
        fold_esperado=np.array(0,dtype=np.uint8))
    return root


def test_gera_sete_tipos_arquivos_e_csv(experimento):
    resumo=gerar_graficos_oof(experimento)
    pasta=experimento/'oof'/'graficos_gerados'
    assert len(resumo['tipos_gerados'])==7
    assert len(list(pasta.glob('*.png')))==7
    assert len(list(pasta.glob('*.csv')))==9  # PR e MCC incluem CSVs adicionais de pontos operacionais
    assert resumo['usa_f10'] is False and resumo['reajustou_modelo'] is False
    assert resumo['id_vencedor']=='abc012'
    assert (pasta/'manifesto_graficos_oof_07_01.json').is_file()


def test_idempotente_e_preserva_fontes(experimento):
    fontes=[experimento/'configuracao.yaml',experimento/'oof'/'resultados_grade.csv',
            experimento/'oof'/'melhor_configuracao.json',
            experimento/'oof'/'predicoes'/'abc012.npz']
    antes={str(f):f.read_bytes() for f in fontes}
    gerar_graficos_oof(experimento)
    png=experimento/'oof'/'graficos_gerados'/'01_curva_precision_recall.png'
    bytes_png=png.read_bytes()
    gerar_graficos_oof(experimento)
    assert png.read_bytes()==bytes_png
    assert all(f.read_bytes()==antes[str(f)] for f in fontes)


def test_respeita_flag_yaml_desativada(experimento):
    caminho=experimento/'configuracao.yaml'
    c=yaml.safe_load(caminho.read_text(encoding='utf-8'))
    c['visualizacao']['graficos']['curva_mcc']['habilitar']=False
    caminho.write_text(yaml.safe_dump(c),encoding='utf-8')
    resumo=gerar_graficos_oof(experimento)
    assert len(resumo['tipos_gerados'])==6
    assert 'curva_mcc' in resumo['tipos_omitidos_por_yaml']
    assert not (experimento/'oof'/'graficos_gerados'/'02_mcc_por_limiar.png').exists()


def test_bloqueia_f10(experimento):
    arq=experimento/'oof'/'predicoes'/'abc012.npz'
    with np.load(arq) as z:
        data={key:z[key] for key in z.files}
    data['fold'][0]=10
    np.savez_compressed(arq,**data)
    with pytest.raises(ValueError,match='F1..F9'):
        gerar_graficos_oof(experimento)


def test_bloqueia_vencedor_inconsistente(experimento):
    arquivo=experimento/'oof'/'melhor_configuracao.json'
    escolha=json.loads(arquivo.read_text())
    escolha['id_candidato']='xyz999'
    arquivo.write_text(json.dumps(escolha))
    with pytest.raises(ValueError,match='Vencedor divergente'):
        gerar_graficos_oof(experimento)


def test_mcc_vetorizado_inclui_empates_sem_quebrar():
    y=np.array([1,0,0,1,1,0,0,0],dtype=np.uint8)
    p=np.array([.8,.8,.4,.4,.4,.1,.1,.1])
    thresholds, valores=_mcc_curva_exata(y,p)
    assert np.array_equal(thresholds,np.array([.8,.4,.1]))
    assert np.allclose(valores,[matthews_corrcoef(y,p>=t) for t in thresholds])


def test_painel_00_valores_corretos_e_fontes_preservadas(experimento):
    import numpy as np
    pasta=experimento/'oof'/'predicoes'/'abc012.npz'
    with np.load(pasta,allow_pickle=False) as arq:
        y=arq['y_true'];p=arq['prob_fraude']
    grade=pd.read_csv(experimento/'oof'/'resultados_grade.csv')
    vencedor=grade.iloc[0]
    escolha=json.loads((experimento/'oof'/'melhor_configuracao.json').read_text())
    numeros=_numeros_painel_00(y,p,vencedor,escolha)
    assert np.isclose(numeros['mcc_01'],(numeros['mcc_bruto']+1)/2)
    assert np.isclose(numeros['logloss_01'],np.exp(-numeros['logloss_bruto']))
    assert np.isclose(numeros['score_composto'],
        (numeros['ap']+numeros['mcc_01']+numeros['logloss_01'])/3)
    gerar_graficos_oof(experimento)
    saida=experimento/'oof'/'graficos_gerados'
    assert (saida/'00_painel_score_composto.png').stat().st_size>1000
    csv=pd.read_csv(saida/'00_painel_score_composto.csv')
    assert list(csv['indicador'])==['AP','MCC','LogLoss','Score composto']
    assert np.allclose(csv['valor_0_a_1'],[
        numeros['ap'], numeros['mcc_01'], numeros['logloss_01'], numeros['score_composto']])
    manifesto=json.loads((saida/'manifesto_painel_00.json').read_text())
    assert manifesto['f10_utilizado'] is False


def test_yaml_painel_desativado_nao_gera_painel(experimento):
    arq=experimento/'configuracao.yaml'
    cfg=yaml.safe_load(arq.read_text())
    cfg['visualizacao']['graficos']['painel_score_composto']['habilitar']=False
    arq.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    gerar_graficos_oof(experimento)
    assert not (experimento/'oof'/'graficos_gerados'/'00_painel_score_composto.png').exists()


def test_yaml_antigo_sem_chave_painel_suportado(experimento):
    arq=experimento/'configuracao.yaml'
    cfg=yaml.safe_load(arq.read_text())
    del cfg['visualizacao']['graficos']['painel_score_composto']
    arq.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    gerar_graficos_oof(experimento)
    assert (experimento/'oof'/'graficos_gerados'/'00_painel_score_composto.png').exists()


def test_painel_00_refino_exibe_titulo_nota_e_fontes_configuradas(experimento, monkeypatch):
    from matplotlib.figure import Figure

    cfg_path = experimento/'configuracao.yaml'
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    painel = cfg['visualizacao']['graficos']['painel_score_composto']
    painel.update({'figsize':[10.2,15.0], 'fonte_titulo':22,
                   'fonte_secundaria':10.8,'fonte_rodape':9.5})
    cfg_path.write_text(yaml.safe_dump(cfg),encoding='utf-8')

    rotulos=[]
    original = Figure.text
    def registrar(self, x, y, s, *args, **kwargs):
        rotulos.append((s,kwargs.get('fontsize')))
        return original(self,x,y,s,*args,**kwargs)
    monkeypatch.setattr(Figure,'text',registrar)
    gerar_graficos_oof(experimento)

    assert ('Indicadores de desempenho e score composto',22) in rotulos
    assert any('As cores não definem qualidade universal' in s for s,_ in rotulos)
    assert any('e LogLoss com uma previsão de referência' in s for s,_ in rotulos)
    assert any(s.startswith('Dados: OOF F1') and f>=9 for s,f in rotulos)
    assert (experimento/'oof'/'graficos_gerados'/'00_painel_score_composto.png').is_file()


def test_pr_pontos_operacionais_calculados_sem_novo_limiar():
    y=np.array([0,1,0,1,0,1],dtype=np.uint8)
    p=np.array([.10,.20,.50,.70,.80,.90])
    a=_ponto_operacional_pr(y,p,.60)
    b=_ponto_operacional_pr(y,p,.50)
    assert (a['verdadeiros_positivos'],a['positivos_previstos'])==(2,3)
    assert np.isclose(a['precision'],2/3) and np.isclose(a['recall'],2/3)
    assert (b['verdadeiros_positivos'],b['positivos_previstos'])==(2,4)
    assert np.isclose(b['precision'],.5) and np.isclose(b['recall'],2/3)
    c=_ponto_operacional_pr(y,p,1.0)
    assert c['positivos_previstos']==0 and c['precision_definida'] is False


def test_pr_csv_pontos_coincidem_com_predicoes_oof(experimento):
    gerar_graficos_oof(experimento)
    arquivo=experimento/'oof'/'graficos_gerados'/'01_curva_precision_recall_pontos.csv'
    dados=pd.read_csv(arquivo)
    assert list(dados['corte'])==['MCC OOF','Fixo 0,50']
    with np.load(experimento/'oof'/'predicoes'/'abc012.npz',allow_pickle=False) as a:
        y=a['y_true'];p=a['prob_fraude']
    for row in dados.itertuples():
        pt=_ponto_operacional_pr(y,p,row.limiar)
        assert np.isclose(row.precision,pt['precision'])
        assert np.isclose(row.recall,pt['recall'])
        assert row.positivos_previstos==pt['positivos_previstos']
        assert row.origem=='OOF_F1_F9'
    assert np.isclose(dados['ap_oof'].iloc[0],average_precision_score(y,p))


def test_pr_yaml_desativa_pontos_sem_alterar_ap(experimento):
    arq=experimento/'configuracao.yaml'
    cfg=yaml.safe_load(arq.read_text())
    cfg['visualizacao']['graficos']['curva_pr']['mostrar_ponto_05']=False
    cfg['visualizacao']['graficos']['curva_pr']['mostrar_ponto_mcc_oof']=False
    arq.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    gerar_graficos_oof(experimento)
    assert (experimento/'oof'/'graficos_gerados'/'01_curva_precision_recall.png').exists()
    assert (experimento/'oof'/'graficos_gerados'/'01_curva_precision_recall_pontos.csv').exists()


def test_pr_zoom_auto_mantem_ponto_mcc_e_limites_uteis():
    from src.visualizacao_oof import _limites_zoom_pr
    cfg={'visualizacao':{'graficos':{'curva_pr':{
        'modo_eixos':'zoom_automatico',
        'zoom_recall_alvo':0.35,
        'zoom_precision_alvo':0.08,
    }}}}
    x,y,ampliado=_limites_zoom_pr(cfg,{'recall':0.3023,'precision':0.0153},0.005218)
    assert ampliado is True
    assert x==(0.0,0.35) and y==(0.0,0.08)
    x,y,ampliado=_limites_zoom_pr(cfg,{'recall':0.8,'precision':0.25},0.005)
    assert x[1]>=.8 and y[1]>=.25 and ampliado


def test_pr_limites_manuais_e_validacao():
    from src.visualizacao_oof import _limites_zoom_pr
    cfg={'visualizacao':{'graficos':{'curva_pr':{
        'modo_eixos':'manual','limites_recall':[0,0.42],
        'limites_precision':[0,.10]}}}}
    assert _limites_zoom_pr(cfg,{'recall':0.8,'precision':.5},.005)==((0,.42),(0,.1),False)
    cfg['visualizacao']['graficos']['curva_pr']['limites_recall']=[.8,.1]
    with pytest.raises(ValueError,match='Limites manuais'):
        _limites_zoom_pr(cfg,{'recall':.3,'precision':.01},.005)


def test_pr_visualizacao_zoom_sem_inset_e_mantem_pontos_csv(experimento, monkeypatch):
    from src import visualizacao_oof as visu
    import matplotlib.pyplot as plt

    vistos={}
    original=visu._salvar
    def capturar(fig, destino, *, dpi):
        if destino.name=='01_curva_precision_recall.png':
            principal=fig.axes[0]
            xlim=principal.get_xlim(); ylim=principal.get_ylim()
            vistos['limites']=(xlim,ylim)
            vistos['detalhes']=[ax.get_xlim() for ax in principal.child_axes]
            vistos['textos']=[a.get_text() for a in fig.texts]
        return original(fig,destino,dpi=dpi)
    monkeypatch.setattr(visu,'_salvar',capturar)
    gerar_graficos_oof(experimento)
    assert vistos['limites'][0][1] <= 1
    assert vistos['limites'][1][1] <= 1
    assert vistos['detalhes'] == []
    assert any('Zoom de leitura:' in s for s in vistos['textos'])
    assert not any('visão geral' in s.lower() for s in vistos['textos'])
    assert (experimento/'oof'/'graficos_gerados'/'01_curva_precision_recall_pontos.csv').exists()


def test_pr_omitir_ponto_degenerado_so_no_desenho(experimento, monkeypatch):
    from src import visualizacao_oof as visu
    from matplotlib.axes import Axes
    arq=experimento/'oof'/'predicoes'/'abc012.npz'
    with np.load(arq, allow_pickle=False) as npz:
        valores={k:npz[k] for k in npz.files}
    # Scores menores que 0.5, sem alterar identidade do fold.
    valores['prob_fraude'] *= .4
    np.savez_compressed(arq, **valores)
    csv=experimento/'oof'/'resultados_grade.csv'
    grade=pd.read_csv(csv)
    y=valores['y_true'];p=valores['prob_fraude']; limiar=.13
    ap=average_precision_score(y,p)
    mcc=matthews_corrcoef(y,p>=limiar)
    ll=log_loss(y,p,labels=[0,1]);score=(ap+(mcc+1)/2+np.exp(-ll))/3
    for nome,val in [('ap',ap),('mcc',mcc),('logloss',ll),('score',score)]:
        grade.loc[grade.id_candidato=='abc012',nome]=val
    grade.to_csv(csv,index=False)
    escolha_path=experimento/'oof'/'melhor_configuracao.json'
    escolha=json.loads(escolha_path.read_text())
    escolha['score_oof']=score
    escolha_path.write_text(json.dumps(escolha),encoding='utf-8')
    vistos=[]
    real=Axes.scatter
    def spy(self,x,y,*args,**kwargs):
        vistos.append(kwargs.get('label',''))
        return real(self,x,y,*args,**kwargs)
    monkeypatch.setattr(Axes,'scatter',spy)
    gerar_graficos_oof(experimento)
    # MCC e outros graficos usam scatter; verificar texto de ponto 0.50 na PR.
    assert not any(str(x).startswith('τ=0,50 ·') for x in vistos)
    data=pd.read_csv(experimento/'oof'/'graficos_gerados'/'01_curva_precision_recall_pontos.csv')
    assert data.loc[data.corte=='Fixo 0,50','positivos_previstos'].iloc[0]==0


def test_mcc_02_cortes_oof_e_05_exatos_e_mesma_amostra(experimento):
    from src.visualizacao_oof import _pontos_operacionais_mcc
    with np.load(experimento/'oof'/'predicoes'/'abc012.npz',allow_pickle=False) as a:
        y=a['y_true'];p=a['prob_fraude']
    escolha=json.loads((experimento/'oof'/'melhor_configuracao.json').read_text())
    pontos=_pontos_operacionais_mcc(y,p,escolha['limiar_mcc_oof'])
    assert len(pontos)==2
    assert [x['limiar'] for x in pontos]==[.13,.5]
    for pt in pontos:
        assert np.isclose(pt['mcc_bruto'],matthews_corrcoef(y,p>=pt['limiar']))
        assert pt['positivos_previstos']==np.count_nonzero(p>=pt['limiar'])
        assert pt['origem']=='OOF_F1_F9'
    gerar_graficos_oof(experimento)
    saida=experimento/'oof'/'graficos_gerados'
    csv=pd.read_csv(saida/'02_mcc_pontos_operacionais.csv')
    assert list(csv['corte'])==['MCC OOF','Fixo 0,50']
    assert np.allclose(csv['mcc_bruto'],[x['mcc_bruto'] for x in pontos])
    assert (saida/'02_mcc_por_limiar.png').stat().st_size>1000


def test_mcc_02_fixo_sem_positivos_previstos_gera_mcc_zero():
    from src.visualizacao_oof import _pontos_operacionais_mcc
    y=np.array([0,1,0,1,0,0],dtype=np.uint8)
    p=np.array([.01,.10,.08,.19,.03,.05])
    melhor,fijo=_pontos_operacionais_mcc(y,p,.10)
    assert fijo['limiar']==.5
    assert fijo['positivos_previstos']==0
    assert fijo['mcc_bruto']==0
    assert np.isclose(melhor['mcc_bruto'],matthews_corrcoef(y,p>=.1))


def test_mcc_02_y_eixo_exibe_mcc_bruto_sem_transformacao(experimento,monkeypatch):
    from src import visualizacao_oof as mod
    original=mod._salvar
    capturado={}
    def spy(fig,destino,*,dpi):
        if destino.name=='02_mcc_por_limiar.png':
            capturado['ylabel']=fig.axes[0].get_ylabel()
            capturado['textos']=[v.get_text() for v in fig.texts]
            capturado['xlim']=fig.axes[0].get_xlim()
        return original(fig,destino,dpi=dpi)
    monkeypatch.setattr(mod,'_salvar',spy)
    gerar_graficos_oof(experimento)
    assert capturado['ylabel']=='MCC bruto (−1 a +1)'
    assert capturado['xlim'][0] == 0.0
    assert capturado['xlim'][0] <= .13 <= capturado['xlim'][1]
    assert 0 < capturado['xlim'][1] <= 1
    assert any('MCC(0,50)' in x for x in capturado['textos'])
    assert any('F10 não utilizado' in x for x in capturado['textos'])


def test_mcc_02_zoom_oculta_marcador_05_sem_mudar_metricas(experimento, monkeypatch):
    from src import visualizacao_oof as mod
    from matplotlib.axes import Axes
    original=mod._salvar
    vistos={}
    scatter_original=Axes.scatter
    scatter_x=[]
    def capturar_scatter(self,x,y,*args,**kwargs):
        if kwargs.get('marker') in ('o','X'):
            scatter_x.extend(np.atleast_1d(x).tolist())
        return scatter_original(self,x,y,*args,**kwargs)
    def capturar(fig,destino,*,dpi):
        if destino.name=='02_mcc_por_limiar.png':
            vistos['xlim']=fig.axes[0].get_xlim()
            vistos['textos']=[t.get_text() for t in fig.texts]
        return original(fig,destino,dpi=dpi)
    monkeypatch.setattr(Axes,'scatter',capturar_scatter)
    monkeypatch.setattr(mod,'_salvar',capturar)
    gerar_graficos_oof(experimento)
    assert vistos['xlim'][0] <= .13 <= vistos['xlim'][1] < .5
    assert any('(fora do zoom)' in txt for txt in vistos['textos'])
    # Ponto otimo pertence ao zoom, referencia 0,50 fica so no texto.
    assert any(np.isclose(x,.13) for x in scatter_x)
    assert not any(np.isclose(x,.5) for x in scatter_x)
    csv=pd.read_csv(experimento/'oof'/'graficos_gerados'/'02_mcc_pontos_operacionais.csv')
    assert np.isclose(csv.loc[csv['corte']=='Fixo 0,50','limiar'].iloc[0],.5)


def test_mcc_02_modo_manual_exibe_limiar_05(experimento,monkeypatch):
    from src import visualizacao_oof as mod
    cfg=experimento/'configuracao.yaml'
    c=yaml.safe_load(cfg.read_text())
    c['visualizacao']['graficos']['curva_mcc']['modo_eixos']='manual'
    c['visualizacao']['graficos']['curva_mcc']['limites_limiar']=[0.,.6]
    cfg.write_text(yaml.safe_dump(c),encoding='utf-8')
    salvos={}
    original=mod._salvar
    def capturar(fig,destino,*,dpi):
        if destino.name=='02_mcc_por_limiar.png':
            salvos['xlim']=fig.axes[0].get_xlim()
        return original(fig,destino,dpi=dpi)
    monkeypatch.setattr(mod,'_salvar',capturar)
    gerar_graficos_oof(experimento)
    assert salvos['xlim']==(0.,.6)


def test_mcc_02_zoom_invalido_bloqueado(experimento):
    from src.visualizacao_oof import _limites_zoom_mcc
    cfg=yaml.safe_load((experimento/'configuracao.yaml').read_text())
    cfg['visualizacao']['graficos']['curva_mcc']['zoom_fracao_mcc_relevante']=1.0
    with pytest.raises(ValueError,match='Parametros do zoom horizontal MCC invalidos'):
        _limites_zoom_mcc(cfg,.01,np.array([.01,.005]))


def test_mcc_02_zoom_horizontal_depende_dos_dados_nao_de_tres_vezes_tau():
    from src.visualizacao_oof import _limites_zoom_mcc
    cfg = {'visualizacao':{'graficos':{'curva_mcc':{'modo_eixos':'zoom_automatico'}}}}
    threshold = np.linspace(.001, .0114, 1200)
    mcc = np.maximum(0, .04 * (threshold-.001)/(.0104-.001))
    mcc[threshold>.011] = 0
    limites, zoom = _limites_zoom_mcc(cfg,.010396,threshold,mcc)
    assert zoom and limites[0] <= .010396 <= limites[1]
    assert limites[1] < .015   # nao ampliar arbitrariamente ate 3*tau
    assert limites[0] >= 0


def test_mcc_02_zoom_resiste_outlier_de_score_alto_com_mcc_zero():
    from src.visualizacao_oof import _limites_zoom_mcc
    cfg = {'visualizacao':{'graficos':{'curva_mcc':{'modo_eixos':'zoom_automatico'}}}}
    threshold = np.r_[np.linspace(.002,.012,800),.97]
    valores = np.r_[np.linspace(.01,.08,800),0]
    limites,_ = _limites_zoom_mcc(cfg,.0102,threshold,valores)
    assert limites[1] < .02


def test_mcc_02_cabecalho_e_rodape_sem_sobreposicao(experimento,monkeypatch):
    from src import visualizacao_oof as mod
    from matplotlib.transforms import Bbox
    salvar_real=mod._salvar
    capturas=[]
    def conferir(fig,destino,*,dpi):
        if destino.name=='02_mcc_por_limiar.png':
            fig.canvas.draw()
            rend=fig.canvas.get_renderer()
            xlabel=fig.axes[0].xaxis.label.get_window_extent(rend)
            titulo=fig._suptitle.get_window_extent(rend)
            rleg=fig.legends[0].get_window_extent(rend)
            textos=[t for t in fig.texts if t is not fig._suptitle]
            subtitulo=next(t for t in textos if 'Zoom automatico:' in t.get_text())
            capturas.append((xlabel, titulo, rleg, subtitulo.get_window_extent(rend),
                             [t.get_window_extent(rend) for t in textos if 'MCC otimo OOF:' in t.get_text()]))
            assert not xlabel.overlaps(capturas[-1][-1][0])
            assert not titulo.overlaps(rleg)
            assert not subtitulo.get_window_extent(rend).overlaps(rleg)
            # Legenda fisicamente acima da malha de dados.
            assert rleg.y0 > fig.axes[0].get_window_extent(rend).y1
        return salvar_real(fig,destino,dpi=dpi)
    monkeypatch.setattr(mod,'_salvar',conferir)
    gerar_graficos_oof(experimento)
    assert len(capturas)==1


def test_confusao_quadrantes_e_faixas_pdf_sem_aproximacoes():
    from src.visualizacao_oof import _dados_quadrantes
    y=np.r_[np.zeros(100,dtype=np.uint8),np.ones(100,dtype=np.uint8)]
    p=np.r_[np.full(98,.1),np.full(2,.8),np.full(20,.1),np.full(80,.8)]
    cfg={'visualizacao':{'estilo_matrizes':{'faixas_qualidade':{
        'tn':[85,97],'fp':[1,5],'fn':[10,30],'tp':[40,70]}}}}
    cm, linhas=_dados_quadrantes(y,p,.5,cfg)
    assert cm.tolist()==[[98,2],[20,80]]
    r={v['quadrante']:v for v in linhas}
    assert {c:r[c]['pct_da_classe_real'] for c in r}=={'TN':98.,'FP':2.,'FN':20.,'TP':80.}
    assert {c:r[c]['faixa_qualitativa'] for c in r}=={'TN':'Bom','FP':'Médio','FN':'Médio','TP':'Bom'}
    assert sum(r[c]['pct_da_classe_real'] for c in ('TN','FP'))==100
    assert sum(r[c]['pct_da_classe_real'] for c in ('FN','TP'))==100
    assert all(v['denominador_classe_real']==100 for v in linhas)
    assert r['FP']['ideal']==0 and r['TN']['ideal']==100


def test_confusao_limiar_fixo_e_otimo_mesmos_dados(experimento):
    from src.visualizacao_oof import _matriz, _ler_resultados
    cfg, grade, escolha, y,p=_ler_resultados(experimento)
    destino=experimento/'oof'/'graficos_gerados'
    _matriz(cfg,y,p,.5,'matriz_confusao_05',destino,'png',75,True)
    _matriz(cfg,y,p,escolha['limiar_mcc_oof'],'matriz_confusao_mcc',destino,'png',75,True)
    assert (destino/'03_matriz_confusao_05.png').stat().st_size>10000
    assert (destino/'04_matriz_confusao_mcc.png').stat().st_size>10000
    for arq,lim in (('03_matriz_confusao_05.csv',.5),
                     ('04_matriz_confusao_mcc.csv',escolha['limiar_mcc_oof'])):
        df=pd.read_csv(destino/arq)
        from sklearn.metrics import confusion_matrix
        m=confusion_matrix(y,p>=lim,labels=[0,1])
        assert df['observado'].tolist()==[m[0,0],m[0,1],m[1,0],m[1,1]]
        assert df['denominador_classe_real'].tolist()==[
            int(m[0].sum()),int(m[0].sum()),int(m[1].sum()),int(m[1].sum())]


def test_confusao_sem_uma_classe_marca_nao_aplicavel():
    from src.visualizacao_oof import _dados_quadrantes
    y=np.array([0,0,0],dtype=np.uint8)
    p=np.array([.9,.2,.1])
    _,linhas=_dados_quadrantes(y,p,.5,{'visualizacao':{}})
    r={v['quadrante']:v for v in linhas}
    assert r['FN']['pct_da_classe_real'] is None
    assert r['TP']['faixa_qualitativa']=='N/A'
    assert r['FP']['pct_da_classe_real']==pytest.approx(100/3)


def test_confusao_faixas_configuracao_invalida_bloqueada():
    from src.visualizacao_oof import _dados_quadrantes
    with pytest.raises(ValueError,match='Faixas de qualidade invalidas'):
        _dados_quadrantes(np.array([0,1]),np.array([0.1,0.8]),.5,
            {'visualizacao':{'estilo_matrizes':{'faixas_qualidade':{'tn':[97,85]}}}})


def test_distribuicao_05_frequencia_normalizada_por_classe(experimento):
    from src.visualizacao_oof import _dados_distribuicao, _limites_x_distribuicao
    cfg=yaml.safe_load((experimento/'configuracao.yaml').read_text(encoding='utf-8'))
    y=np.array([0,0,0,0,1,1,1],dtype=np.uint8)
    p=np.array([.001,.002,.003,.40,.004,.008,.90])
    op=cfg['visualizacao']['graficos']['distribuicao_probabilidades']
    lo, zoom=_limites_x_distribuicao(op,y,p,.01)
    assert zoom is True
    assert lo[0] == 0 and lo[1] >= .01
    tabela,_=_dados_distribuicao(y,p,lo,12)
    assert tabela['contagem_legitima'].sum()+tabela['fora_do_zoom_legitima'].iloc[0]==4
    assert tabela['contagem_fraude'].sum()+tabela['fora_do_zoom_fraude'].iloc[0]==3
    assert np.isclose(tabela['frequencia_relativa_legitima'].sum()
           +tabela['fora_do_zoom_legitima'].iloc[0]/4,1)
    assert np.isclose(tabela['frequencia_relativa_fraude'].sum()
           +tabela['fora_do_zoom_fraude'].iloc[0]/3,1)


def test_distribuicao_05_escala_completa_mostra_050():
    from src.visualizacao_oof import _limites_x_distribuicao
    y=np.array([0,0,1,1]);p=np.array([.001,.3,.006,.9])
    lim,zoom=_limites_x_distribuicao({'modo_eixo_x':'completo'},y,p,.01)
    assert lim==(0.,1.) and not zoom


def test_distribuicao_05_yml_zoom_invalido():
    from src.visualizacao_oof import _limites_x_distribuicao
    y=np.array([0,1]);p=np.array([.001,.2])
    with pytest.raises(ValueError,match='Parametros do zoom'):
        _limites_x_distribuicao({'quantil_zoom':.2},y,p,.01)


def test_distribuicao_05_gera_csv_com_frequencias_reais(experimento):
    from src.visualizacao_oof import _ler_resultados, _distribuicao
    cfg,grade,escolha,y,p=_ler_resultados(experimento)
    d=experimento/'oof'/'graficos_gerados'
    _distribuicao(cfg,y,p,escolha,d,'png',70,True)
    tab=pd.read_csv(d/'05_distribuicao_probabilidades.csv')
    for classe in ('legitima','fraude'):
        counts=tab[f'contagem_{classe}'].sum()
        fora=int(tab[f'fora_do_zoom_{classe}'].iloc[0])
        n=int(tab[f'total_{classe}'].iloc[0])
        assert counts+fora==n
        assert np.isclose(counts/n,tab[f'frequencia_relativa_{classe}'].sum())
    assert (d/'05_distribuicao_probabilidades.png').is_file()
