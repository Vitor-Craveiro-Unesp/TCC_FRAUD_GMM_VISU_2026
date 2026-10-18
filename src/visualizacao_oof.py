"""Passo 07.01 — seis tipos de gráficos de validação OOF, somente vencedor.

Executar, na raiz de TCC_EXPERIMENTAL_V3:
    python -m src.visualizacao_oof --experimento outputs/ensaio_pca_gmm_...

Nao carrega dataset_raw.parquet, folds.npy ou F10, nao ajusta modelos e
nao altera metricas, predições, YAML, manifestos da execucao ou selecao.
Os graficos sao gravados em <experimento>/oof/graficos_gerados/.
O gráfico 06 usa todos os candidatos; os demais usam apenas o vencedor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import yaml
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    average_precision_score, confusion_matrix, log_loss, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, f1_score,
)


ARQUIVOS = {
    'curva_pr': '01_curva_precision_recall',
    'curva_mcc': '02_mcc_por_limiar',
    'matriz_confusao_05': '03_matriz_confusao_05',
    'matriz_confusao_mcc': '04_matriz_confusao_mcc',
    'distribuicao_probabilidades': '05_distribuicao_probabilidades',
    'comparacao_hiperparametros': '06_comparacao_hiperparametros',
}


def _json_abrir(caminho: Path) -> dict:
    with caminho.open(encoding='utf-8-sig') as arq:
        return json.load(arq)


def _exigir(condicao: bool, mensagem: str):
    if not condicao:
        raise ValueError(mensagem)


def _numero_proximo(a, b, nome, *, atol=1e-9):
    _exigir(np.isclose(float(a), float(b), atol=atol, rtol=1e-7),
            f'Valor inconsistente em {nome}: {a} vs {b}.')


def _ler_resultados(experimento: Path):
    """Carrega exclusivamente os dados OOF já produzidos no Passo 06.03."""
    experimento = experimento.resolve()
    arquivo_config = experimento / 'configuracao.yaml'
    arquivo_grade = experimento / 'oof' / 'resultados_grade.csv'
    arquivo_escolha = experimento / 'oof' / 'melhor_configuracao.json'
    for arquivo in (arquivo_config, arquivo_grade, arquivo_escolha):
        if not arquivo.is_file():
            raise FileNotFoundError(f'Arquivo obrigatorio ausente: {arquivo}')
    config = yaml.safe_load(arquivo_config.read_text(encoding='utf-8-sig'))
    _exigir(isinstance(config, dict), 'YAML da execucao invalido.')
    _exigir(config['saidas']['pastas']['oof'] == 'oof', 'Pasta OOF inesperada.')
    _exigir(config['saidas']['pastas']['graficos'] == 'graficos_gerados', 'Pasta graficos inesperada.')
    grade = pd.read_csv(arquivo_grade, dtype={'id_candidato': str})
    exigidas = {'id_candidato','ap','mcc','logloss','score','limiar_mcc','n','prevalencia','configuracao'}
    _exigir(exigidas.issubset(grade.columns), f'CSV sem colunas {exigidas - set(grade.columns)}.')
    _exigir(len(grade) > 0 and grade['id_candidato'].is_unique, 'Grade vazia ou com IDs duplicados.')
    for nome in ('ap','mcc','logloss','score','limiar_mcc','n','prevalencia'):
        _exigir(np.isfinite(pd.to_numeric(grade[nome], errors='coerce')).all(), f'Coluna {nome} invalida.')
    # Desempate metodologico é o mesmo do 06.03.
    grade = grade.sort_values(['score','ap','id_candidato'], ascending=[False,False,True], kind='stable').reset_index(drop=True)
    escolha = _json_abrir(arquivo_escolha)
    cid = str(escolha['id_candidato'])
    _exigir(cid == str(grade.iloc[0]['id_candidato']), 'Vencedor divergente do ranking OOF.')
    vencedor = grade.iloc[0]
    _exigir(escolha.get('teste_final_usado') is False, 'F10 nao pode ter sido usado na selecao.')
    _numero_proximo(escolha['limiar_mcc_oof'], vencedor['limiar_mcc'], 'limiar escolhido')
    _numero_proximo(escolha['score_oof'], vencedor['score'], 'score vencedor')
    _exigir(escolha['configuracao'] == json.loads(vencedor['configuracao']),
            'Configuracao escolhida diverge da grade.')
    caminho_pred = experimento / 'oof' / 'predicoes' / f'{cid}.npz'
    if not caminho_pred.is_file():
        raise FileNotFoundError(f'Predicoes OOF do vencedor ausentes: {caminho_pred}')
    with np.load(caminho_pred, allow_pickle=False) as arq:
        chaves = set(arq.files)
        necessarias = {'y_true','prob_fraude','fold','linha_original','id_candidato','fold_esperado'}
        _exigir(necessarias.issubset(chaves), f'NPZ sem campos: {necessarias - chaves}.')
        y = np.asarray(arq['y_true']).astype(np.int8)
        p = np.asarray(arq['prob_fraude'], dtype=np.float64)
        folds = np.asarray(arq['fold'])
        ids = np.asarray(arq['linha_original'])
        _exigir(str(arq['id_candidato'].item()) == cid and int(arq['fold_esperado'].item()) == 0,
                'NPZ nao corresponde ao OOF consolidado vencedor.')
    _exigir(y.ndim == p.ndim == folds.ndim == ids.ndim == 1, 'Predicoes OOF devem ser vetores 1D.')
    _exigir(len(y) == int(vencedor['n']) == len(p) == len(folds) == len(ids), 'Quantidade OOF inconsistente.')
    _exigir(set(np.unique(folds)) == set(range(1,10)), 'OOF deve incluir somente F1..F9.')
    _exigir(len(np.unique(ids)) == len(ids), 'Indices OOF duplicados.')
    _exigir(np.isin(y,[0,1]).all() and 0 < y.sum() < len(y), 'Rotulos invalidos ou classe ausente.')
    _exigir(np.isfinite(p).all() and np.all((p>=0)&(p<=1)), 'Probabilidades invalidas.')
    # Confere metricas, mas NUNCA recalcula o limiar ótimo para escolher outro.
    limiar = float(escolha['limiar_mcc_oof'])
    ap = float(average_precision_score(y,p))
    mcc = float(matthews_corrcoef(y, p>=limiar))
    ll = float(log_loss(y,p,labels=[0,1]))
    score = (ap+(mcc+1)/2+np.exp(-ll))/3
    for nome, valor in [('ap',ap),('mcc',mcc),('logloss',ll),('score',score),('prevalencia',y.mean())]:
        _numero_proximo(valor, vencedor[nome], nome, atol=1e-7)
    return config, grade, escolha, y, p


def _ativado(config: dict, nome: str, fase='oof') -> bool:
    visu = config.get('visualizacao', {})
    g = visu.get('graficos', {}).get(nome, {})
    return (visu.get('habilitar', False) is True
            and config.get('saidas', {}).get('salvar_graficos', True) is True
            and g.get('habilitar') is True
            and fase in visu.get('gerar_em', [])
            and fase in g.get('gerar_em', []))


def _op(config, nome):
    return config['visualizacao']['graficos'][nome]


def _figura(config, *, escala=1.0):
    pad = config['visualizacao']['padrao']
    tam = np.asarray(pad.get('figsize',[9,6]), dtype=float)
    _exigir(tam.shape == (2,) and np.isfinite(tam).all() and (tam>0).all(), 'figsize invalido.')
    return plt.subplots(figsize=tuple(tam*escala))


def _salvar(fig, destino, *, dpi):
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        plt.close(fig)
        print(f'[PRESERVADO] {destino.name}')
        return 'existente'
    try:
        # Salva figura no mesmo formato pedido, sem mudar arquivos anteriores.
        fig.savefig(destino, dpi=dpi, bbox_inches='tight')
        print(f'[GERADO] {destino.name}')
        return 'criado'
    finally:
        plt.close(fig)


def _csv_protegido(df, destino):
    if destino.exists():
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False, encoding='utf-8-sig')


def _caminho_png(destino, nome, extensao):
    return destino / (ARQUIVOS[nome]+'.'+extensao)


def _mcc_curva_exata(y, p):
    """MCC em cada score distinto; os empates no score não são quebrados."""
    ordem = np.argsort(p)[::-1]
    s = p[ordem]
    ys = y[ordem].astype(np.int64)
    fim = np.flatnonzero(np.r_[s[:-1] != s[1:], True])
    tp = np.cumsum(ys)[fim].astype(np.float64)
    n = len(y)
    npos = float(np.sum(y))
    nneg = float(n-npos)
    pred_pos = (fim+1).astype(np.float64)
    denom = np.sqrt(pred_pos*npos*nneg*(n-pred_pos))
    valores = np.divide(n*tp-npos*pred_pos, denom,
                        out=np.zeros_like(tp), where=denom>0)
    return s[fim], valores


def _plot_pr(config, y, p, vencedor, destino, ext, dpi, dados):
    op = _op(config,'curva_pr')
    prec, rec, _ = precision_recall_curve(y,p)
    fig, ax = _figura(config)
    rotulo = f'Vencedor — AP={float(vencedor["ap"]):.5f}' if op.get('mostrar_ap',True) else 'Vencedor OOF'
    ax.plot(rec,prec, lw=1.5, label=rotulo)
    if op.get('mostrar_baseline_prevalencia',True):
        ax.axhline(float(np.mean(y)),ls='--',lw=1,color='gray',label=f'Prevalência = {np.mean(y):.3%}')
    ax.set(xlabel='Revocação (Recall)', ylabel='Precisão (Precision)',
           xlim=op.get('limites_recall',[0,1]), ylim=op.get('limites_precision',[0,1]))
    if config['visualizacao']['padrao'].get('titulo',True):
        ax.set_title('01 — Curva Precision–Recall | Vencedor OOF')
    ax.grid(config['visualizacao']['padrao'].get('grade',True), alpha=.25)
    if config['visualizacao']['padrao'].get('legenda',True): ax.legend()
    _salvar(fig,_caminho_png(destino,'curva_pr',ext),dpi=dpi)
    if dados:
        _csv_protegido(pd.DataFrame({'recall':rec,'precision':prec}),destino/'01_curva_precision_recall.csv')


def _plot_mcc(config, y, p, vencedor, escolha, destino, ext, dpi, dados):
    op = _op(config,'curva_mcc')
    threshold, mcc = _mcc_curva_exata(y,p)
    pontos = int(op.get('pontos_visualizacao',2000))
    _exigir(pontos>0,'curva_mcc.pontos_visualizacao deve ser positivo.')
    vis = np.unique(np.linspace(0,len(threshold)-1,min(pontos,len(threshold)),dtype=np.int64))
    limiar = float(escolha['limiar_mcc_oof'])
    fig,ax = _figura(config)
    ax.plot(threshold[vis],mcc[vis],lw=1.2,label='MCC dos limiares OOF')
    if op.get('mostrar_limiar_oof',True):
        ax.axvline(limiar,ls='--',color='black',lw=1,label=f'Limiar OOF={limiar:.6g}')
    if op.get('mostrar_mcc_no_limiar_oof',True):
        ax.scatter([limiar],[float(vencedor['mcc'])],color='darkred',zorder=4,
                   label=f'MCC no limiar = {float(vencedor["mcc"]):.5f}')
    if op.get('mostrar_limiar_05',True):
        ax.axvline(.5,ls=':',color='gray',lw=1,label='Referência 0,5')
    ax.set(xlabel='Limiar de probabilidade',ylabel='MCC')
    if config['visualizacao']['padrao'].get('titulo',True):
        ax.set_title('02 — MCC por limiar | Vencedor OOF')
    ax.grid(config['visualizacao']['padrao'].get('grade',True),alpha=.25)
    if config['visualizacao']['padrao'].get('legenda',True): ax.legend()
    _salvar(fig,_caminho_png(destino,'curva_mcc',ext),dpi=dpi)
    if dados:
        _csv_protegido(pd.DataFrame({'limiar':threshold,'mcc':mcc}),destino/'02_mcc_por_limiar_completo.csv')


def _matriz(config, y, p, limiar, nome, destino, ext, dpi, dados):
    cm = confusion_matrix(y, p>=limiar, labels=[0,1]); tn,fp,fn,tp = cm.ravel()
    op = config['visualizacao'].get('estilo_matrizes',{})
    vis = op.get('mostrar',{})
    linhas = np.sum(cm,axis=1)
    ideal = np.array([[linhas[0],0],[0,linhas[1]]],dtype=np.int64)
    proximidade = np.array([[tn/linhas[0],1-fp/linhas[0]],
                            [1-fn/linhas[1],tp/linhas[1]]],dtype=np.float64)*100
    proxima = op.get('escala_proximidade', {})
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('proximidade',
        [proxima.get('cor_longe','#f7fbff'),proxima.get('cor_meio','#6baed6'),proxima.get('cor_perto','#08306b')])
    fig,ax = _figura(config,escala=1.12)
    ax.imshow(proximidade,cmap=cmap,vmin=0,vmax=100)
    quadrantes = [['TN','FP'],['FN','TP']]
    descricoes = [['Legítima correta','Alarme falso'],['Fraude não detectada','Fraude detectada']]
    cores = op.get('bordas',{})
    for i in range(2):
        for j in range(2):
            linha_n=linhas[i]
            pct_classe=cm[i,j]/linha_n*100 if linha_n else float('nan')
            linhas_txt = []
            if vis.get('rotulo_tn_fp_fn_tp',True): linhas_txt.append(quadrantes[i][j])
            if vis.get('descricao_quadrante',True): linhas_txt.append(descricoes[i][j])
            if vis.get('contagem_observada',True): linhas_txt.append(f'Observado: {cm[i,j]:,}')
            if vis.get('contagem_ideal',True): linhas_txt.append(f'Ideal: {ideal[i,j]:,}')
            if vis.get('diferenca_absoluta_do_ideal',True): linhas_txt.append(f'Δ ideal: {abs(cm[i,j]-ideal[i,j]):,}')
            if vis.get('percentual_da_classe',True): linhas_txt.append(f'{pct_classe:.2f}% da classe real')
            if vis.get('percentual_proximidade',True): linhas_txt.append(f'Proximidade: {proximidade[i,j]:.2f}%')
            ax.text(j,i,'\n'.join(linhas_txt),ha='center',va='center',fontsize=8.3,
                    color=('white' if proximidade[i,j] >= 65 else 'black')
                       if proxima.get('adaptar_cor_texto',True) else 'black')
            ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,
                                   edgecolor=cores.get(quadrantes[i][j].lower(),'#555555'),linewidth=3.0))
    titulo=f'{ARQUIVOS[nome][:2]} — Matriz de confusão | OOF | Limiar={limiar:.6g}'
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['Legítima','Fraude'],
           yticklabels=['Legítima','Fraude'],xlabel='Classe predita',ylabel='Classe real')
    if config['visualizacao']['padrao'].get('titulo',True):ax.set_title(titulo)
    if vis.get('totais_linhas_colunas',False):
        ax.text(1.04,.76,f'Total legítimas: {linhas[0]:,}',transform=ax.transAxes,fontsize=8)
        ax.text(1.04,.20,f'Total fraudes: {linhas[1]:,}',transform=ax.transAxes,fontsize=8)
        ax.text(.5,-.15,f'Total predito legítima: {tn+fn:,} | fraude: {tp+fp:,}',
                transform=ax.transAxes,ha='center',fontsize=8)
    # Métricas do rodapé calculadas no limiar representado, sem otimizá-lo.
    foot=op.get('rodape',{})
    pred=(p>=limiar)
    pr=precision_score(y,pred,zero_division=0)
    re=recall_score(y,pred,zero_division=0)
    f1=f1_score(y,pred,zero_division=0)
    spec=tn/(tn+fp) if tn+fp else float('nan')
    mcc=matthews_corrcoef(y,pred)
    total=len(y)
    kpi = {
        'accuracy':f'Accuracy={(tn+tp)/total:.3f}',
        'precision':f'Precisão={pr:.3f}',
        'recall_sensibilidade':f'Recall={re:.3f}',
        'especificidade':f'Especificidade={spec:.3f}',
        'f1':f'F1={f1:.3f}',
        'fpr':f'FPR={fp/(fp+tn) if fp+tn else float("nan"):.3f}',
        'fnr':f'FNR={fn/(fn+tp) if fn+tp else float("nan"):.3f}',
        'mcc':f'MCC={mcc:.4f}',
        'prevalencia':f'Prevalência={y.mean():.3%}',
    }
    texto=' | '.join(valor for chave,valor in kpi.items() if foot.get(chave,False))
    if texto:
        # Quebrar em 2 ou 3 linhas para imagens legíveis.
        selecionados=[valor for chave,valor in kpi.items() if foot.get(chave,False)]
        grupos=[selecionados[i:i+3] for i in range(0,len(selecionados),3)]
        fig.text(.5,-.02,'\n'.join(' | '.join(g) for g in grupos),ha='center',va='top',fontsize=9)
    _salvar(fig,_caminho_png(destino,nome,ext),dpi=dpi)
    if dados:
        registros=[]
        for i in range(2):
            for j in range(2):
                registros.append({'quadrante':quadrantes[i][j], 'real':i, 'predito':j,
                     'observado':int(cm[i,j]),'ideal':int(ideal[i,j]),
                     'proximidade_pct':float(proximidade[i,j]),
                     'pct_da_classe_real':float(cm[i,j]/linhas[i]*100)})
        _csv_protegido(pd.DataFrame(registros),destino/(ARQUIVOS[nome]+'.csv'))


def _distribuicao(config,y,p,escolha,destino,ext,dpi,dados):
    op = _op(config,'distribuicao_probabilidades')
    bins = int(op.get('bins',50))
    _exigir(bins>0,'distribuicao_probabilidades.bins deve ser positivo.')
    fig,ax=_figura(config)
    separar = op.get('separar_classes',True)
    dens = op.get('normalizar',True)
    # Histograma e densidade usam escala de probabilidade real, sem fabricar dados.
    tipo=op.get('tipo','histograma')
    _exigir(tipo in ('histograma','densidade'), 'distribuicao_probabilidades.tipo invalido.')
    bins_ = np.linspace(float(p.min()),float(p.max()) if p.max()>p.min() else float(p.min())+1e-9,bins+1)
    conjuntos=[(p[y==0],'Legítimas','#1f77b4'),(p[y==1],'Fraudes','#d62728')] if separar else [(p,'Todas as transações','#1f77b4')]
    for vals,titulo,cor in conjuntos:
        if tipo == 'histograma':
            ax.hist(vals,bins=bins_,density=dens,histtype='step',lw=1.6,label=titulo,color=cor)
        else:
            _exigir(len(vals)>2 and np.std(vals)>0,
                    f'KDE requer variacao e mais de duas observacoes: {titulo}')
            xs=np.linspace(float(min(vals.min(),p.min())),float(max(vals.max(),p.max())),400)
            kde=gaussian_kde(vals)
            ys=kde(xs)
            if not dens:ys=ys*len(vals)*(bins_[1]-bins_[0])
            ax.plot(xs,ys,label=titulo,color=cor,lw=1.6)
    if op.get('marcar_limiar_oof',True):
        ax.axvline(float(escolha['limiar_mcc_oof']),ls='--',color='black',lw=1,
                   label=f'Limiar MCC OOF={float(escolha["limiar_mcc_oof"]):.5f}')
    if op.get('marcar_limiar_05',True):
        ax.axvline(.5,ls=':',color='gray',lw=1,label='Limiar 0,5')
    escala=op.get('escala_y','linear')
    _exigir(escala in ('linear','log'),'escala_y deve ser linear ou log.')
    ax.set_yscale(escala)
    ax.set(xlabel='Probabilidade estimada de fraude',ylabel='Densidade' if dens else 'Contagem aproximada por bin')
    if config['visualizacao']['padrao'].get('titulo',True):
        ax.set_title('05 — Distribuição dos scores | Vencedor OOF')
    ax.grid(config['visualizacao']['padrao'].get('grade',True),alpha=.2)
    if config['visualizacao']['padrao'].get('legenda',True): ax.legend()
    _salvar(fig,_caminho_png(destino,'distribuicao_probabilidades',ext),dpi=dpi)
    if dados:
        df=pd.DataFrame({'inicio_bin':bins_[:-1], 'fim_bin':bins_[1:],
           'contagem_legitima':np.histogram(p[y==0],bins=bins_)[0],
           'contagem_fraude':np.histogram(p[y==1],bins=bins_)[0]})
        _csv_protegido(df,destino/'05_distribuicao_probabilidades.csv')


def _rotulo_config(linha):
    c=json.loads(linha['configuracao'])
    red=c['reducao']
    if c['metodo'] == 'PCA':
        nome=f'PCA {red["n_components"]}'
    else:
        nome=f't-SNE {red["n_components"]}D p={red["perplexity"]}'
    rot=f'{nome} · K={c["gmm"]["n_components"]} · {c["modo"]}'
    if c['modo']=='RUS':rot+=f' {c["razao"]}'
    return rot


def _comparacao(config,grade,escolha,destino,ext,dpi,dados):
    op=_op(config,'comparacao_hiperparametros')
    chave=op.get('ordenar_por','score')
    validas={'ap','mcc','logloss','score'}
    _exigir(chave in validas,'comparacao_hiperparametros.ordenar_por invalido.')
    ordem_asc = chave == 'logloss'
    grade=grade.sort_values([chave,'id_candidato'],ascending=[ordem_asc,True],kind='stable')
    maximo=int(op.get('maximo_configuracoes',25))
    _exigir(maximo>=1,'maximo_configuracoes deve ser positivo.')
    limitado=grade.head(maximo).copy()
    metricas=op.get('metricas',['ap','mcc','logloss','score'])
    _exigir(bool(metricas) and all(m in validas for m in metricas), 'Métricas de comparação desconhecidas.')
    # Uma imagem da categoria 06, com painéis separados para escalas distintas.
    fig,axes=plt.subplots(len(metricas),1,figsize=(max(10, len(limitado)*.8+4),max(4,3.3*len(metricas))),
                          squeeze=False)
    labels=[_rotulo_config(r) for _,r in limitado.iterrows()]
    x=np.arange(len(limitado))
    for ax,m in zip(axes.flat,metricas):
        vals=limitado[m].to_numpy(float)
        barras=ax.bar(x,vals)
        for barra,(_,r) in zip(barras,limitado.iterrows()):
            if op.get('destacar_vencedor',True) and r['id_candidato']==escolha['id_candidato']:
                barra.set_edgecolor('black');barra.set_linewidth(2.5);barra.set_hatch('///')
        ax.set_ylabel(m.upper());ax.set_xticks(x,labels,rotation=30,ha='right',fontsize=8)
        # baseline = 0 evita eixos manipulados visualmente
        minimo=min(0,float(np.min(vals)))
        maximo_v=max(0,float(np.max(vals)))
        margem=max(.03*max(abs(maximo_v),abs(minimo)),.003)
        ax.set_ylim(minimo-margem if minimo<0 else 0,maximo_v+margem)
        ax.grid(axis='y',alpha=.2)
        for barra,v in zip(barras,vals):
            ax.annotate(f'{v:.5f}',xy=(barra.get_x()+barra.get_width()/2,barra.get_height()),
                        xytext=(0,3),textcoords='offset points',ha='center',va='bottom',fontsize=8)
    if config['visualizacao']['padrao'].get('titulo',True):
        fig.suptitle('06 — Comparação OOF de hiperparâmetros (vencedor hachurado)',fontsize=12)
    fig.tight_layout()
    _salvar(fig,_caminho_png(destino,'comparacao_hiperparametros',ext),dpi=dpi)
    if dados:
        # Salva quadro completo, não somente o top N visualizado.
        _csv_protegido(grade,destino/'06_comparacao_hiperparametros_todos.csv')


def gerar_graficos_oof(experimento: Path):
    """Gera os seis tipos configurados e retorna resumo (sem tocar no F10)."""
    experimento=Path(experimento).resolve()
    config, grade, escolha, y, p = _ler_resultados(experimento)
    pad=config['visualizacao']['padrao']
    ext=str(pad.get('formato','png')).lower()
    _exigir(ext in {'png','svg','pdf'}, 'visualizacao.padrao.formato deve ser png, svg ou pdf.')
    dpi=int(pad.get('dpi',180))
    _exigir(50<=dpi<=1200,'visualizacao.padrao.dpi fora da faixa 50..1200.')
    dados=pad.get('salvar_dados_grafico',True) is True
    destino=experimento/'oof'/'graficos_gerados'
    destino.mkdir(parents=True,exist_ok=True)
    vencedor=grade.iloc[0]
    gerados=[];omitidos=[]
    funcoes={
        'curva_pr':lambda:_plot_pr(config,y,p,vencedor,destino,ext,dpi,dados),
        'curva_mcc':lambda:_plot_mcc(config,y,p,vencedor,escolha,destino,ext,dpi,dados),
        'matriz_confusao_05':lambda:_matriz(config,y,p,.5,'matriz_confusao_05',destino,ext,dpi,dados),
        'matriz_confusao_mcc':lambda:_matriz(config,y,p,float(escolha['limiar_mcc_oof']), 'matriz_confusao_mcc',destino,ext,dpi,dados),
        'distribuicao_probabilidades':lambda:_distribuicao(config,y,p,escolha,destino,ext,dpi,dados),
        'comparacao_hiperparametros':lambda:_comparacao(config,grade,escolha,destino,ext,dpi,dados),
    }
    for nome,func in funcoes.items():
        if _ativado(config,nome):
            func();gerados.append(nome)
        else:
            print(f'[OMITIDO NO YAML] {nome}')
            omitidos.append(nome)
    resumo={'etapa':'07.01', 'tipo':'graficos_preditivos_oof',
            'id_vencedor':escolha['id_candidato'], 'numero_candidatos_comparados':len(grade),
            'status_escolha':escolha['status'],'usa_f10':False,
            'reajustou_modelo':False,'limiar_mcc_oof':float(escolha['limiar_mcc_oof']),
            'tipos_gerados':gerados,'tipos_omitidos_por_yaml':omitidos,
            'tipos_geometricos_pendentes_passo_07_02':list(config['visualizacao']['graficos'])[6:],
            'diretorio':str(destino)}
    arq=destino/'manifesto_graficos_oof_07_01.json'
    content=json.dumps(resumo,ensure_ascii=False,indent=2)+'\n'
    if arq.exists() and arq.read_text(encoding='utf-8')!=content:
        warnings.warn('Manifesto anterior diferente; preservado sem sobrescrever.',stacklevel=2)
    elif not arq.exists():
        arq.write_text(content,encoding='utf-8')
    print(f'\nPasso 07.01: {len(gerados)} tipos de graficos OOF em {destino}')
    print(f'Vencedor: {escolha["id_candidato"]}; F10 nao lido; GMM nao reajustado.')
    return resumo


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experimento', type=Path, required=True,
                        help='Pasta de experimento/ensaio concluido pelo Passo 06.03')
    args=parser.parse_args(argv)
    gerar_graficos_oof(args.experimento)


if __name__ == '__main__':
    main()
