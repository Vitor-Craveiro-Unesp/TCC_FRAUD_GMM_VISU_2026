"""Passo 07.01 — oito tipos de gráficos de validação OOF, somente vencedor.

Executar, na raiz de TCC_EXPERIMENTAL_V3:
    python -m src.visualizacao_oof --experimento outputs/ensaio_pca_gmm_...

Nao carrega dataset_raw.parquet, folds.npy ou F10, nao ajusta modelos e
nao altera metricas, predições, YAML, manifestos da execucao ou selecao.
Os graficos sao gravados em <experimento>/vencedor/visualizacao/oof/.
O gráfico 06 usa todos os candidatos; os demais usam apenas o vencedor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from src import estrutura_saida as layout
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import yaml
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    average_precision_score, confusion_matrix, log_loss, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, f1_score,
)


ARQUIVOS = {
    'painel_score_composto': '00_painel_score_composto',
    'curva_pr': '01_curva_precision_recall',
    'curva_mcc': '02_mcc_por_limiar',
    'matriz_confusao_05': '03_matriz_confusao_05',
    'matriz_confusao_mcc': '04_matriz_confusao_mcc',
    'distribuicao_probabilidades': '05_distribuicao_probabilidades',
    'comparacao_hiperparametros': '06_comparacao_hiperparametros',
    'metricas_por_fold': '07_metricas_por_fold',
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
    arquivo_grade = layout.metricas_vencedor(experimento) / 'resultados_grade.csv'
    arquivo_escolha = layout.metricas_vencedor(experimento) / 'melhor_configuracao.json'
    for arquivo in (arquivo_config, arquivo_grade, arquivo_escolha):
        if not arquivo.is_file():
            raise FileNotFoundError(f'Arquivo obrigatorio ausente: {arquivo}')
    config = yaml.safe_load(arquivo_config.read_text(encoding='utf-8-sig'))
    _exigir(isinstance(config, dict), 'YAML da execucao invalido.')
    _exigir(config['saidas']['pastas'].get('oof') == 'oof', 'Pasta OOF inesperada.')
    _exigir(config['saidas']['pastas'].get('vencedor') == 'vencedor', 'Pasta do vencedor inesperada.')
    _exigir(config['saidas']['pastas'].get('visualizacao') == 'visualizacao', 'Pasta de visualizacao inesperada.')
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
    caminho_pred = layout.predicoes_vencedor(experimento)
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
    graficos = visu.get('graficos', {})
    g = graficos.get(nome, {})
    # Compatibilidade: YAML de ensaios anteriores a criação do Painel 00.
    # Esses YAMLs continuam imutáveis; não habilitar se a chave existe e está False.
    if nome == 'painel_score_composto' and nome not in graficos:
        return (visu.get('habilitar') is True
                and config.get('saidas', {}).get('salvar_graficos', True) is True
                and fase == 'oof' and fase in visu.get('gerar_em', []))
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


def _numeros_painel_00(y, p, vencedor, escolha):
    """Quatro indicadores do catálogo: todos calculados do MESMO OOF."""
    ap = float(average_precision_score(y, p))
    mcc = float(matthews_corrcoef(y, p >= float(escolha['limiar_mcc_oof'])))
    loss = float(log_loss(y, p, labels=[0, 1]))
    mcc_01 = (mcc + 1.0) / 2.0
    loss_01 = float(np.exp(-loss))
    composto = (ap + mcc_01 + loss_01) / 3.0
    for nome, valor in [('ap', ap), ('mcc', mcc), ('logloss', loss), ('score', composto)]:
        _numero_proximo(valor, vencedor[nome], f'Painel 00: {nome}', atol=1e-7)
    return {
        'ap': ap, 'prevalencia_oof': float(np.mean(y)),
        'mcc_bruto': mcc, 'mcc_01': mcc_01,
        'logloss_bruto': loss, 'logloss_01': loss_01,
        'score_composto': composto,
        'limiar_mcc_oof': float(escolha['limiar_mcc_oof']),
        'n_oof': int(len(y)),
    }


def _plot_painel_00(config, y, p, vencedor, escolha, destino, ext, dpi, dados):
    """Painel escuro e quatro barras 0–1 seguindo catálogo em PDF.

    Não rotula valores como bom/médio/ruim, não usa números simulados.
    """
    v = _numeros_painel_00(y, p, vencedor, escolha)
    tema = config['visualizacao']['graficos'].get('painel_score_composto', {})
    cor_fundo = tema.get('cor_fundo', '#07101f')
    cor_cartao = tema.get('cor_cartao', '#111d33')
    cor_texto = '#edf4ff'
    cor_suave = '#b7c9e2'
    cor_borda = '#345273'
    cmap = LinearSegmentedColormap.from_list(
        'escala_descritiva_00', ['#d6333a', '#f26b29', '#f0d94c', '#87c646', '#1b9e62'])
    # Apenas layout; nunca altera as probabilidades ou métricas OOF.
    tamanho = np.asarray(tema.get('figsize', [10.2, 15.0]), dtype=float)
    _exigir(tamanho.shape == (2,) and np.isfinite(tamanho).all()
            and (tamanho > 0).all(), 'painel_score_composto.figsize invalido.')
    fonte_titulo = float(tema.get('fonte_titulo', 21))
    fonte_secundaria = float(tema.get('fonte_secundaria', 10.5))
    fonte_rodape = float(tema.get('fonte_rodape', 9.6))
    _exigir(all(np.isfinite(f) and 6 <= f <= 36 for f in
                (fonte_titulo, fonte_secundaria, fonte_rodape)),
            'Fontes do Painel 00 fora de 6..36.')
    fig = plt.figure(figsize=tuple(tamanho), facecolor=cor_fundo)


    fig.text(.075, .967, 'PAINEL INICIAL',
             color='#7ac8fd', fontsize=12, weight='bold')



    fig.text(.075, .932, 'Indicadores de desempenho e score composto',
             color=cor_texto, fontsize=fonte_titulo, weight='bold')







    cards = [
    ('AP (AVERAGE PRECISION)',
     'Resumo da curva Precision-Recall ao longo dos limiares.',
     'AP', v['ap'], f"AP bruto: {v['ap']:.7f}",
     'AP calculada sobre as predições OOF.'),
    ('MCC (MATTHEWS CORRELATION COEFFICIENT)',
     'Qualidade da classificação, considerando VP, VN, FP e FN.',
     '(MCC + 1) / 2', v['mcc_01'], f"MCC bruto: {v['mcc_bruto']:.7f}",
     'MCC calculado no limiar ótimo das predições OOF.'),
    ('LOGLOSS',
     'Qualidade das probabilidades previstas; menor LogLoss é melhor.',
     'exp(-LogLoss)', v['logloss_01'], f"LogLoss bruto: {v['logloss_bruto']:.7f}",
     'LogLoss calculado sobre as predições OOF.'),
    ('SCORE COMPOSTO',
     'Síntese dos três critérios com pesos iguais.',
     'S = [AP + (MCC(τ*)+1)/2 + exp(-LogLoss)] / 3', v['score_composto'],
     'Três parcelas com peso 1/3',
     'Critério de seleção interno do TCC, não métrica padronizada.'),
]



    carta_h = .181
    for idx, (titulo, descricao, formula, valor, nota, detalhe) in enumerate(cards):
        fundo_y = .720 - idx * .197

        fig.add_artist(FancyBboxPatch((.065, fundo_y), .87, carta_h,
            boxstyle='round,pad=0.006,rounding_size=0.021',
            transform=fig.transFigure, linewidth=1, edgecolor=cor_borda,
            facecolor=cor_cartao))
        fig.text(.09, fundo_y+.155, titulo, color=cor_texto, fontsize=12, weight='bold')
        fig.text(.09, fundo_y+.137, descricao, color=cor_suave, fontsize=fonte_secundaria)
        fig.add_artist(FancyBboxPatch((.09, fundo_y+.101), .82, .023,
            boxstyle='round,pad=0.004,rounding_size=0.006',
            transform=fig.transFigure, linewidth=.8,
            edgecolor=cor_borda, facecolor='#0d1a2e'))
        fig.text(.103, fundo_y+.107, formula, color=cor_texto, fontsize=fonte_secundaria)
        fig.text(.09, fundo_y+.068, f'{valor:.7f}',
                 color=cmap(np.clip(valor,0,1)), fontsize=24, weight='bold')
        fig.text(.91, fundo_y+.073, 'escala 0–1', color=cor_suave,
                 fontsize=fonte_secundaria-1, ha='right')
        bar = fig.add_axes([.09, fundo_y+.048, .82, .010])
        bar.set_zorder(10)
        bar.imshow(np.linspace(0,1,600)[None,:], origin='lower',
                   cmap=cmap, extent=(0,1,0,1), aspect='auto', interpolation='bicubic')
        bar.axvline(np.clip(valor,0,1), color='#eaf3ff', lw=2.7)
        bar.set(xlim=(0,1), ylim=(0,1)); bar.axis('off')
        fig.text(.09, fundo_y+.029, nota,
                 color=cor_texto, fontsize=fonte_secundaria, weight='bold')
        fig.text(.09, fundo_y+.010, detalhe, color=cor_suave, fontsize=fonte_secundaria-0.3)


    # Caixa de informacoes uteis

    info_y = .032
    info_h = .075

    fig.add_artist(
        FancyBboxPatch(
            (.065, info_y),
            .87,
            info_h,
            boxstyle='round,pad=0.004,rounding_size=0.008',
            transform=fig.transFigure,
            linewidth=1,
            edgecolor=cor_borda,
            facecolor='#0d1a2e'
        )
    )

    fig.text(
        .082,
        info_y + .047,
        'INFORMAÇÕES ÚTEIS',
        color='#7ac8fd',
        fontsize=fonte_rodape,
        weight='bold'
    )

    fig.text(
        .082,
        info_y + .020,
        (
            f"OOF F1–F9  |  "
            f"n={v['n_oof']}  |  "
            f"Fraudes={int(np.count_nonzero(np.asarray(y) == 1))}  |  "
            f"τ*={v['limiar_mcc_oof']:.7f}  |  "
            "F10 não utilizado"
        ),
        color=cor_suave,
        fontsize=fonte_rodape-0.3
    )


    _salvar(fig, _caminho_png(destino, 'painel_score_composto', ext), dpi=dpi)
    if dados:
        tabela = pd.DataFrame([
            {'indicador':'AP','valor_bruto':v['ap'],'transformacao':'identidade',
             'valor_0_a_1':v['ap'],'baseline_prevalencia':v['prevalencia_oof']},
            {'indicador':'MCC','valor_bruto':v['mcc_bruto'],'transformacao':'(MCC+1)/2',
             'valor_0_a_1':v['mcc_01'],'baseline_prevalencia':np.nan},
            {'indicador':'LogLoss','valor_bruto':v['logloss_bruto'],'transformacao':'exp(-LogLoss)',
             'valor_0_a_1':v['logloss_01'],'baseline_prevalencia':np.nan},
            {'indicador':'Score composto','valor_bruto':v['score_composto'],
             'transformacao':'media das tres parcelas',
             'valor_0_a_1':v['score_composto'],'baseline_prevalencia':np.nan},
        ])
        tabela['limiar_mcc_oof'] = v['limiar_mcc_oof']
        tabela['n_oof'] = v['n_oof']
        _csv_protegido(tabela, destino/'00_painel_score_composto.csv')
    return v


def _ponto_operacional_pr(y, p, limiar):
    """Precisao e recall obtidos das classificacoes OOF no corte informado.

    Uma classificacao sem positivos previstos possui precisao indefinida;
    usamos 0 somente para representar numericamente o ponto operacional.
    """
    _exigir(np.isfinite(limiar) and 0 <= limiar <= 1,
            'Limiar PR fora do intervalo [0,1].')
    pred = p >= limiar
    total_prev = int(np.count_nonzero(pred))
    tp = int(np.count_nonzero(pred & (y == 1)))
    return {'limiar':float(limiar), 'precision':float(tp/total_prev) if total_prev else 0.0,
            'recall':float(tp/np.count_nonzero(y == 1)),
            'positivos_previstos':total_prev, 'verdadeiros_positivos':tp,
            'precision_definida':bool(total_prev)}


def _limites_zoom_pr(config, ponto_mcc, prevalencia):
    """Define enquadramento visual, sem descartar ou recalcular dados OOF.

    O inicio da PR pode ter valores acima do teto do zoom. Esta limitacao
    do enquadramento deve ser informada no rodape, sem miniatura interna.
    """
    op = _op(config, 'curva_pr')
    modo = op.get('modo_eixos', 'zoom_automatico')
    _exigir(modo in ('zoom_automatico', 'manual'),
            'curva_pr.modo_eixos deve ser zoom_automatico ou manual.')
    if modo == 'manual':
        limites_x = np.asarray(op.get('limites_recall', [0, 1]), dtype=float)
        limites_y = np.asarray(op.get('limites_precision', [0, 1]), dtype=float)
        _exigir(limites_x.shape == limites_y.shape == (2,) and
                np.isfinite(limites_x).all() and np.isfinite(limites_y).all() and
                0 <= limites_x[0] < limites_x[1] <= 1 and
                0 <= limites_y[0] < limites_y[1] <= 1,
                'Limites manuais PR invalidos.')
        return tuple(limites_x), tuple(limites_y), False

    alvo_x = float(op.get('zoom_recall_alvo', 0.35))
    alvo_y = float(op.get('zoom_precision_alvo', 0.08))
    _exigir(np.isfinite([alvo_x, alvo_y]).all() and
            0 < alvo_x <= 1 and 0 < alvo_y <= 1,
            'Zoom PR deve estar no intervalo (0,1].')
    # Um minimo de zoom orientado pelo ponto MCC; jamais esconde esse ponto.
    max_x = min(1.0, max(alvo_x, ponto_mcc['recall'] * 1.12))
    max_y = min(1.0, max(alvo_y, ponto_mcc['precision'] * 1.35,
                          prevalencia * 2.0))
    return (0.0, max_x), (0.0, max_y), True


















def _plot_pr(config, y, p, vencedor, escolha, destino, ext, dpi, dados):
    """Curva Precision-Recall OOF. O zoom modifica apenas a visualizacao."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    op = _op(config, 'curva_pr')
    precisao, recall, _ = precision_recall_curve(y, p)
    ap = float(average_precision_score(y, p))
    prevalencia = float(np.mean(y))
    _numero_proximo(ap, vencedor['ap'], 'AP na curva PR', atol=1e-7)

    # O limiar selecionado em OOF e utilizado sem nova otimizacao.
    tau = float(escolha['limiar_mcc_oof'])
    ponto_otimo = _ponto_operacional_pr(y, p, tau)
    ponto_fixo = _ponto_operacional_pr(y, p, 0.5)
    xlim, ylim, ampliado = _limites_zoom_pr(config, ponto_otimo, prevalencia)


    fig, ax = plt.subplots(figsize=(10.5, 7.0), facecolor='white')
    fig.subplots_adjust(left=.115, right=.97, bottom=.280, top=.805)

    # Dados originais: nao suavizar nem interpolar a curva.
    ax.plot(
        recall, precisao, color='#21558c', linewidth=1.8,
        label='Curva Precision–Recall'
    )
    if op.get('mostrar_baseline_prevalencia', True):
        ax.axhline(
            prevalencia, color='#8a736b', linestyle='--', linewidth=1.35,
            label='Prevalência de fraude'
        )
    if op.get('mostrar_ponto_mcc_oof', True):
        ax.scatter(
            [ponto_otimo['recall']], [ponto_otimo['precision']],
            color='#178f66', marker='o', s=82, zorder=7,
            edgecolors='white', linewidths=.9,
            label='Limiar ótimo do MCC (τ*)'
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


    ax.set_xlabel(
        'Recall — proporção de fraudes identificadas',
        fontsize=10.5,
        labelpad=8
    )

    ax.set_ylabel(
        'Precisão — proporção de alertas corretos',
        fontsize=10.5,
        labelpad=8
    )


    # Sete casas em todos os rotulos numericos, sem zero negativo.
    def formato_eixo(valor, _pos):
        return f'{0.0 if abs(valor) < 0.00000005 else valor:.7f}'

    formato = FuncFormatter(formato_eixo)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(formato)
    ax.yaxis.set_major_formatter(formato)
    ax.tick_params(axis='both', labelsize=8.2, length=4)

    # Estilo discreto, semelhante ao de figuras de artigos cientificos.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if config['visualizacao']['padrao'].get('grade', True):
        ax.grid(axis='y', color='#d7dfe8', linewidth=.75, alpha=.65)
        ax.set_axisbelow(True)

        # Miniatura da curva Precision-Recall completa
    if ampliado:
        mini = ax.inset_axes(
            [.74, .53, .23, .39],
            zorder=10
        )

        mini.set_facecolor('white')

        # Curva original, sem suavização
        mini.plot(
            recall,
            precisao,
            color='#21558c',
            linewidth=1.15
        )

        # Prevalência da classe fraude
        mini.axhline(
            prevalencia,
            color='#8a736b',
            linestyle='--',
            linewidth=.9
        )

        # Ponto do limiar ótimo
        mini.scatter(
            [ponto_otimo['recall']],
            [ponto_otimo['precision']],
            color='#178f66',
            s=15,
            zorder=5
        )

        # Retângulo indicando a janela do gráfico principal
        janela = Rectangle(
            (xlim[0], ylim[0]),
            xlim[1] - xlim[0],
            ylim[1] - ylim[0],
            fill=False,
            edgecolor='#ba633d',
            linewidth=1.3,
            linestyle='--'
        )

        mini.add_patch(janela)

        # Limites completos da curva PR
        mini.set_xlim(0.0, 1.0)
        mini.set_ylim(0.0, 1.0)

        mini.set_title(
            'Curva completa',
            fontsize=8.8,
            pad=4
        )

        # Apenas os extremos, mantendo sete casas decimais
        mini.set_xticks([0.0, 1.0])
        mini.set_yticks([0.0, 1.0])

        mini.xaxis.set_major_formatter(formato)
        mini.yaxis.set_major_formatter(formato)

        mini.tick_params(
            axis='both',
            labelsize=6.4,
            length=2,
            pad=1
        )

        # Bordas discretas
        for lado in mini.spines.values():
            lado.set_edgecolor('#8493a5')
            lado.set_linewidth(.85)

        # Evita que números ultrapassem as bordas laterais
        for lbl, align in zip(
            mini.get_xticklabels(),
            ['left', 'right']
        ):
            lbl.set_ha(align)


    if config['visualizacao']['padrao'].get('titulo', True):
        fig.suptitle(
            'Curva Precision–Recall do modelo selecionado em OOF',
            fontsize=14.5, y=.972, color='#202c3a'
        )


    if config['visualizacao']['padrao'].get('legenda', True):
        fig.legend(
            *ax.get_legend_handles_labels(),
            loc='upper center', bbox_to_anchor=(.53, .871),
            ncol=3,
            frameon=False, fontsize=9.0,
            columnspacing=2.2, handletextpad=.65
        )

    # Resultados concentrados em um unico local, sem repetir a legenda.
    if op.get('mostrar_resumo_operacional', True):
        fig.text(
            .115, .178, 'Resultados da validação OOF',
            fontsize=10.5, weight='bold', color='#26374d'
        )
        fig.text(
            .115, .139,
            f'AP = {ap:.7f}   |   Prevalência = {prevalencia:.7f}   |   τ* = {tau:.7f}',
            fontsize=9.5, color='#26374d'
        )
        fig.text(
            .115, .101,
            f'Em τ*: Precisão = {ponto_otimo["precision"]:.7f}   |   '
            f'Recall = {ponto_otimo["recall"]:.7f}',
            fontsize=9.5, color='#26374d'
        )

    _salvar(fig, _caminho_png(destino, 'curva_pr', ext), dpi=dpi)
    if dados:
        _csv_protegido(
            pd.DataFrame({'recall': recall, 'precision': precisao}),
            destino / '01_curva_precision_recall.csv'
        )
        _csv_protegido(
            pd.DataFrame([
                {'corte': nome, **pt, 'ap_oof': ap,
                 'prevalencia_oof': prevalencia, 'origem': 'OOF_F1_F9'}
                for nome, pt in [('MCC OOF', ponto_otimo), ('Fixo 0,50', ponto_fixo)]
            ]),
            destino / '01_curva_precision_recall_pontos.csv'
        )



def _pontos_operacionais_mcc(y, p, limiar_oof):
    """MCC bruto do mesmo OOF em tau* congelado e no corte fixo 0,50.

    Nao procura novo limiar. O valor de 0,50 deve aparecer mesmo que
    nao exista nenhum score nesse valor ou nenhuma previsao positiva.
    """
    _exigir(np.isfinite(limiar_oof) and 0 <= limiar_oof <= 1,
            'Limiar MCC OOF invalido.')
    return [
        {'corte':'MCC OOF', 'limiar':float(limiar_oof),
         'mcc_bruto':float(matthews_corrcoef(y, p >= limiar_oof)),
         'positivos_previstos':int(np.count_nonzero(p >= limiar_oof)),
         'origem':'OOF_F1_F9'},
        {'corte':'Fixo 0,50', 'limiar':0.5,
         'mcc_bruto':float(matthews_corrcoef(y, p >= 0.5)),
         'positivos_previstos':int(np.count_nonzero(p >= 0.5)),
         'origem':'OOF_F1_F9'},
    ]


def _limites_zoom_mcc(config, limiar_oof, threshold, mcc=None):
    """Enquadra a parte informativa da curva MCC sem selecionar novo limiar.

    Usa a distribuicao *dos pontos da propria curva* e nao um multiplo de
    tau*. Alguns valores extremos de score podem ficar fora da janela;
    eles continuam no CSV completo. O limiar oficial e sempre exibido.
    """
    op = _op(config, 'curva_mcc')
    modo = op.get('modo_eixos', 'zoom_automatico')
    _exigir(modo in ('zoom_automatico', 'manual'),
            'curva_mcc.modo_eixos deve ser zoom_automatico ou manual.')
    if modo == 'manual':
        limites = np.asarray(op.get('limites_limiar', [0, 1]), dtype=float)
        _exigir(limites.shape == (2,) and np.isfinite(limites).all() and
                0 <= limites[0] < limites[1] <= 1 and
                limites[0] <= limiar_oof <= limites[1],
                'curva_mcc.limites_limiar invalidos ou excluem tau*.')
        return (float(limites[0]), float(limites[1])), False

    threshold = np.asarray(threshold, dtype=float)
    _exigir(threshold.ndim == 1 and threshold.size > 0 and
            np.isfinite(threshold).all() and np.all((threshold >= 0) & (threshold <= 1)) and
            np.isfinite(limiar_oof) and 0 <= limiar_oof <= 1,
            'Limiares invalidos na curva MCC.')
    fracao = float(op.get('zoom_fracao_mcc_relevante', 0.025))
    extremo = float(op.get('zoom_quantile_extremo', 0.005))
    folga = float(op.get('zoom_margem_horizontal', 0.07))
    _exigir(0 <= fracao < 1 and 0 <= extremo < .5 and 0 < folga <= .5 and
            np.isfinite([fracao, extremo, folga]).all(),
            'Parametros do zoom horizontal MCC invalidos.')

    candidatos = threshold
    if mcc is not None:
        valores = np.asarray(mcc, dtype=float)
        _exigir(valores.shape == threshold.shape and np.isfinite(valores).all(),
                'Curva MCC e limiares nao coincidem.')
        amplitude = float(np.max(np.abs(valores)))
        if amplitude > 1e-12:
            significativos = np.abs(valores) >= fracao * amplitude
            if np.any(significativos):
                candidatos = threshold[significativos]

    # Cortar so extremos visuais raros, nunca o tau* congelado.
    if candidatos.size >= 20:
        baixo, alto = np.quantile(candidatos, [extremo, 1-extremo])
    else:
        baixo, alto = float(np.min(candidatos)), float(np.max(candidatos))
    baixo = min(float(baixo), float(limiar_oof))
    alto = max(float(alto), float(limiar_oof))
    largura = max(alto - baixo, alto*.04, 1e-12)
    esquerdo = max(0., baixo - folga*largura)
    direito = min(1., alto + folga*largura)
    if esquerdo <= .18 * max(direito, 1e-12):
        esquerdo = 0.
    if direito <= esquerdo:
        # Caso degenerado (scores todos iguais): janela pequena mas valida.
        esquerdo = max(0., float(limiar_oof) - .025)
        direito = min(1., float(limiar_oof) + .025)
        if direito <= esquerdo:
            esquerdo, direito = 0., 1.
    return (float(esquerdo), float(direito)), bool(esquerdo > 0 or direito < 1)


def _plot_mcc(config, y, p, vencedor, escolha, destino, ext, dpi, dados):
    """MCC bruto OOF com zoom de leitura e comparacao exata em tau* e 0,50."""
    op = _op(config, 'curva_mcc')
    threshold, mcc = _mcc_curva_exata(y, p)
    pontos = int(op.get('pontos_visualizacao', 2000))
    _exigir(pontos > 0, 'curva_mcc.pontos_visualizacao deve ser positivo.')
    tau = float(escolha['limiar_mcc_oof'])
    operacoes = _pontos_operacionais_mcc(y, p, tau)
    otimo, fixo = operacoes
    _numero_proximo(otimo['mcc_bruto'], vencedor['mcc'],
                    'MCC do corte congelado no grafico 02', atol=1e-7)

    xlim, ampliado = _limites_zoom_mcc(config, tau, threshold, mcc)
    no_zoom = np.flatnonzero((threshold >= xlim[0]) & (threshold <= xlim[1]))
    # Amostrar para o desenho SOMENTE dentro da janela. Manter todos os
    # valores e limiares originais no CSV e nao usar a grade visual na selecao.
    if len(no_zoom) > pontos:
        pos = np.unique(np.linspace(0, len(no_zoom)-1, pontos, dtype=np.int64))
        vis = no_zoom[pos]
        # Preservar os extremos do MCC do intervalo ao subamostrar.
        vis = np.unique(np.r_[vis, no_zoom[np.argmax(mcc[no_zoom])],
                              no_zoom[np.argmin(mcc[no_zoom])]])
    else:
        vis = no_zoom
    if not len(vis):
        # Valores de tau* entre scores distintos (caso raro): nao fabricar
        # pontos de curva, mas continuar mostrando tau* e sua metrica OOF.
        vis = np.array([], dtype=np.int64)

    # MCC e sempre bruto em [-1,+1], nunca a transformacao do Painel 00.
    fig, ax = plt.subplots(figsize=(11.0, 7.3))
    # Reservas exclusivas: cabecalho, legenda, malha, rotulos e rodape.
    # Nao posicionar textos manuais sobre a area do xlabel.
    fig.subplots_adjust(left=.105, right=.965, bottom=.265, top=.775)


    ax.plot(threshold[vis], mcc[vis], lw=1.65, color='#385aaf',
            label='MCC por limiar')

    if op.get('mostrar_limiar_oof', True):
        ax.axvline(tau, ls='--', color='#168e66', lw=1.25,
                   label='Limiar ótimo (τ*)')    

    if op.get('mostrar_mcc_no_limiar_oof', True):
        ax.scatter([tau], [otimo['mcc_bruto']], s=78, marker='o',
                   color='#168e66', edgecolors='white', linewidths=.9, zorder=6,
                   label='MCC no limiar ótimo')

    # Em bases desbalanceadas, 0,50 costuma ficar fora do zoom: manter a
    # comparacao numerica no rodape, sem achatar a curva ou criar ponto falso.
    fixo_visivel = xlim[0] <= .5 <= xlim[1]
    if fixo_visivel and op.get('mostrar_limiar_05', True):
        ax.axvline(.5, ls='--', color='#ce584b', lw=1.25,
                   label='Referência fixa: τ=0,50')
        if op.get('mostrar_mcc_no_limiar_05', True):
            ax.scatter([.5], [fixo['mcc_bruto']], s=70, marker='X',
                       color='#ce584b', zorder=7,
                       label=f'MCC(0,50)={fixo["mcc_bruto"]:.5f}')

    ax.set(xlabel='Limiar de classificação (τ)', ylabel='MCC bruto (−1 a +1)',
           xlim=xlim)
    # Usar somente a faixa visivel na escala Y para evitar compressao.
    valores_y = [0.0, otimo['mcc_bruto']]
    if len(no_zoom):
        valores_y.extend((float(np.min(mcc[no_zoom])), float(np.max(mcc[no_zoom]))))
    if fixo_visivel:
        valores_y.append(fixo['mcc_bruto'])
    minimo, maximo = min(valores_y), max(valores_y)
    margem = max(.005, (maximo-minimo)*.12)
    ax.set_ylim(max(-1., minimo-margem), min(1., maximo+margem))

    from matplotlib.ticker import FuncFormatter, MaxNLocator

    def formato_eixo(valor, _pos):
        return f'{0.0 if abs(valor) < 0.00000005 else valor:.7f}'

    formato = FuncFormatter(formato_eixo)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.xaxis.set_major_formatter(formato)
    ax.yaxis.set_major_formatter(formato)

    ax.tick_params(axis='both', labelsize=8.5)


    if config['visualizacao']['padrao'].get('titulo', True):
        fig.suptitle('MCC por limiar', fontsize=15, y=.972)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if config['visualizacao']['padrao'].get('grade', True):
        ax.grid(axis='y', color='#d7dfe8', linewidth=.75, alpha=.65)
        ax.set_axisbelow(True)

    if config['visualizacao']['padrao'].get('legenda', True):
        fig.legend(*ax.get_legend_handles_labels(), loc='upper center',
                   bbox_to_anchor=(.535, .864), ncol=3, fontsize=9.0,
                   frameon=False, columnspacing=1.8, handlelength=2.1)

    if op.get('mostrar_resumo_operacional', True):
        fig.text(
            .105, .150,
            'Resultados da validação OOF',
            fontsize=10.5,
            weight='bold',
            color='#26374d',
            va='top'
        )

        fig.text(
            .105, .108,
            f'Limiar ótimo τ* = {tau:.7f}   |   '
            f'MCC(τ*) = {otimo["mcc_bruto"]:.7f}',
            fontsize=9.7,
            color='#26374d',
            va='top'
        )

        fig.text(
            .105, .068,
            f'Em τ = 0.5000000: MCC = {fixo["mcc_bruto"]:.7f}'
            + (' (fora da região exibida)' if not fixo_visivel else ''),
            fontsize=9.3,
            color='#52647a',
            va='top'
        )


    _salvar(fig, _caminho_png(destino, 'curva_mcc', ext), dpi=dpi)
    if dados:
        _csv_protegido(pd.DataFrame({'limiar':threshold, 'mcc':mcc}),
                       destino/'02_mcc_por_limiar_completo.csv')
        _csv_protegido(pd.DataFrame(operacoes),
                       destino/'02_mcc_pontos_operacionais.csv')


# Faixas visuais exatamente como no catalogo de referencia do usuario.
# Sao referencias CONTEXTUAIS, nao limites universais de desempenho.
_FAIXAS_PDF = {
    'TN': (85., 97.), # <85 ruim | 85..97 medio | >=97 bom
    'FP': (1., 5.),   # <=1 bom | 1..5 medio | >5 ruim
    'FN': (10., 30.), # <=10 bom | 10..30 medio | >30 ruim
    'TP': (40., 70.), # <40 ruim | 40..70 medio | >=70 bom
}

_DESCRICOES_PDF = {
    'TN': ('VN', 'Verdadeiro negativo',
           'Legítimas corretamente classificadas como legítimas.',
           ('Muitas legítimas tratadas como suspeitas',
            'Custo operacional ainda relevante',
            'Maioria das legítimas liberada corretamente')),
    'FP': ('FP', 'Falso positivo',
           'Legítimas classificadas incorretamente como fraude.',
           ('Baixo atrito para clientes legítimos',
            'Tolerância depende da operação',
            'Muitas legítimas bloqueadas por engano')),
    'FN': ('FN', 'Falso negativo',
           'Fraudes reais que passaram sem serem identificadas.',
           ('Poucas fraudes escapam',
            'Detecção intermediária',
            'Muitas fraudes não identificadas')),
    'TP': ('VP', 'Verdadeiro positivo',
           'Fraudes reais corretamente detectadas.',
           ('Poucas fraudes identificadas',
            'Detecção com espaço para melhora',
            'Boa parcela das fraudes identificada')),
}


def _dados_quadrantes(y, p, limiar, config):
    """Números exatos do OOF; percentuais condicionados à CLASSE REAL.

    TN+FP = número de legítimas, FN+TP = número de fraudes.
    Não classifica qualidade quando alguma classe real está ausente.
    """
    _exigir(np.isfinite(limiar) and 0 <= limiar <= 1,
            'Limiar da matriz deve estar em [0,1].')
    cm = confusion_matrix(y, p >= limiar, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())
    n0, n1 = tn + fp, fn + tp
    op = config.get('visualizacao', {}).get('estilo_matrizes', {})
    limites_cfg = op.get('faixas_qualidade', {})
    valores = {'TN':(tn,n0,n0), 'FP':(fp,n0,0),
               'FN':(fn,n1,0), 'TP':(tp,n1,n1)}
    linhas=[]
    for chave, (observado, denominador, ideal) in valores.items():
        lims = np.asarray(limites_cfg.get(chave.lower(), _FAIXAS_PDF[chave]), dtype=float)
        _exigir(lims.shape == (2,) and np.isfinite(lims).all()
                and 0 <= lims[0] < lims[1] <= 100,
                f'Faixas de qualidade invalidas para {chave}.')
        pct = (100.*observado/denominador) if denominador else None
        proximidade = (100.*(1-abs(observado-ideal)/denominador)) if denominador else None
        if pct is None:
            faixa = 'N/A'
        elif chave in ('TN','TP'):
            faixa = 'Ruim' if pct < lims[0] else 'Médio' if pct < lims[1] else 'Bom'
        else:
            faixa = 'Bom' if pct <= lims[0] else 'Médio' if pct <= lims[1] else 'Ruim'
        linhas.append({
            'quadrante': chave, 'real':0 if chave in ('TN','FP') else 1,
            'predito':0 if chave in ('TN','FN') else 1,
            'observado': observado, 'ideal': ideal,
            'delta_ideal': abs(observado-ideal),
            'denominador_classe_real': denominador,
            'pct_da_classe_real': pct,
            'proximidade_pct': proximidade,
            'faixa_qualitativa': faixa,
            'faixa_limite_1_pct':float(lims[0]),
            'faixa_limite_2_pct':float(lims[1]),
        })
    return cm, linhas


def _qualidade_cor(lims, invertido):
    """Gradiente local: as transições seguem os cortes do PDF, não 1/3 fixo."""
    verde, amarelo, vermelho='#25b576','#ecc84d','#de3d55'
    a,b=float(lims[0])/100,float(lims[1])/100
    # Sem pontos repetidos em 0/1 (as faixas atuais estão no interior).
    _exigir(0<a<b<1, 'Faixas de qualidade devem estar dentro de 0..100 para gradiente.')
    if invertido:
        pontos=[(0,verde),(a,verde),((a+b)/2,amarelo),(b,vermelho),(1,vermelho)]
    else:
        pontos=[(0,vermelho),(a,vermelho),((a+b)/2,amarelo),(b,verde),(1,verde)]
    return LinearSegmentedColormap.from_list('qualidade_'+('invertida' if invertido else 'crescente'),pontos)


def _matriz(config, y, p, limiar, nome, destino, ext, dpi, dados):
    """Matriz OOF por quadrante, com barras de proximidade ao ideal.

    As cores indicam somente a distancia numerica ao ideal. Nao ha
    classificacao em bom, medio ou ruim. Os dados e o CSV sao preservados.
    """
    cm, linhas = _dados_quadrantes(y, p, limiar, config)
    tn, fp, fn, tp = map(int, cm.ravel())
    n0, n1 = tn + fp, fn + tp

    op = config['visualizacao'].get('estilo_matrizes', {})
    _exigir(
        op.get('layout', 'quadrantes_pdf') == 'quadrantes_pdf',
        'Layout de matriz desconhecido.'
    )
    pal = op.get('cores_painel', {})
    bg = pal.get('fundo', '#091425')
    cartao = pal.get('cartao', '#12213b')
    borda = pal.get('borda', '#294766')
    texto = pal.get('texto', '#f6f8fc')
    secundario = pal.get('secundario', '#b2bfd3')

    def fracao(pct):
        return 'N/A' if pct is None else f'{float(pct) / 100.0:.7f}'

    def inteiro(valor):
        return str(int(valor))

    def caixa(x, y0, largura, altura):
        fig.add_artist(FancyBboxPatch(
            (x, y0), largura, altura,
            boxstyle='round,pad=0.006,rounding_size=0.010',
            transform=fig.transFigure, facecolor=cartao,
            edgecolor=borda, linewidth=.9, zorder=0
        ))

    def escrever(x, y0, mensagem, **kwargs):
        fig.text(
            x, y0, mensagem, transform=fig.transFigure,
            color=kwargs.pop('color', texto), va='top', **kwargs
        )

    por_chave = {linha['quadrante']: linha for linha in linhas}
    is_mcc = nome == 'matriz_confusao_mcc'
    titulo_corte = (
        f'LIMIAR ÓTIMO DO MCC: τ* = {limiar:.7f}'
        if is_mcc else f'CORTE FIXO: τ = {limiar:.7f}'
    )
    escala = LinearSegmentedColormap.from_list(
        'proximidade_ideal_matriz', ['#d6333a', '#f0d94c', '#1b9e62']
    )

    fig = plt.figure(figsize=(11.6, 12.6), facecolor=bg)
    escrever(.065, .967, 'Matriz de confusão por quadrante',
             fontsize=20, weight='bold')
    escrever(.065, .916, titulo_corte,
             fontsize=11, weight='bold', color='#a9c8eb')

    resumos = [
        ('CLASSE REAL: LEGÍTIMA', n0, 'TN', 'FP', 'VN', 'FP'),
        ('CLASSE REAL: FRAUDE', n1, 'FN', 'TP', 'FN', 'VP'),
    ]
    for indice, (titulo, total, a, b, sigla_a, sigla_b) in enumerate(resumos):
        x = .065 + indice * .44
        caixa(x, .800, .420, .080)
        escrever(x + .016, .864, titulo, fontsize=10.6, weight='bold')
        escrever(
            x + .016, .828,
            f'{sigla_a}: {fracao(por_chave[a]["pct_da_classe_real"])}  |  '
            f'{sigla_b}: {fracao(por_chave[b]["pct_da_classe_real"])}  |  '
            f'n={inteiro(total)}',
            fontsize=9.1, color=secundario
        )

    topo_primeiro = .782
    altura = .164
    intervalo = .180
    for indice, chave in enumerate(('TN', 'FP', 'FN', 'TP')):
        linha = por_chave[chave]
        y0 = topo_primeiro - altura - indice * intervalo
        caixa(.065, y0, .868, altura)

        codigo, titulo, descricao, _notas = _DESCRICOES_PDF[chave]
        observado = inteiro(linha['observado'])
        ideal = inteiro(linha['ideal'])
        delta = inteiro(linha['delta_ideal'])
        valor_observado = fracao(linha['pct_da_classe_real'])
        proximidade_pct = linha['proximidade_pct']
        valor_proximidade = fracao(proximidade_pct)

        escrever(.089, y0 + .140, f'{codigo} — {titulo}',
                 fontsize=14, weight='bold')
        escrever(.909, y0 + .141, valor_observado,
                 fontsize=18.5, weight='bold', ha='right')
        escrever(.089, y0 + .112, descricao,
                 fontsize=10.0, color=secundario)
        escrever(.089, y0 + .081,
                 f'Observado: {observado}   |   Ideal: {ideal}   |   Δ ideal: {delta}',
                 fontsize=10.0, color=texto)
        escrever(.089, y0 + .058,
                 f'Proximidade ao ideal: {valor_proximidade}',
                 fontsize=9.7, color=secundario)

        # A barra usa sempre a escala de proximidade ao ideal, de 0 a 1.
        barra = fig.add_axes([.09, y0 + .025, .816, .014])
        barra.imshow(
            np.linspace(0, 1, 600)[None, :],
            origin='lower', cmap=escala, extent=(0, 1, 0, 1),
            aspect='auto', interpolation='bicubic'
        )
        if proximidade_pct is not None:
            proximidade = float(proximidade_pct) / 100.0
            barra.axvline(proximidade, color='#f6f8fc', lw=2.7,
                          clip_on=False)
        barra.set(xlim=(0, 1), ylim=(0, 1))
        barra.axis('off')

    _salvar(fig, _caminho_png(destino, nome, ext), dpi=dpi)
    if dados:
        # Mesmo conteudo de auditoria usado antes da alteracao visual.
        _csv_protegido(
            pd.DataFrame(linhas), destino / (ARQUIVOS[nome] + '.csv')
        )


def _limites_x_distribuicao(op, y, p, tau):
    """Janela de leitura baseada nos scores OOF de AMBAS as classes.

    Usar quantis por classe evita que a predominância de negativas
    oculte as caudas das fraudes. Não recalcula nem altera scores.
    """
    modo = op.get('modo_eixo_x', 'zoom_automatico')
    _exigir(modo in ('zoom_automatico', 'completo'),
            'distribuicao_probabilidades.modo_eixo_x invalido.')
    _exigir(0 <= tau <= 1 and np.isfinite(tau), 'Limiar OOF invalido.')
    if modo == 'completo':
        return (0., 1.), False
    q = float(op.get('quantil_zoom', .995))
    margem = float(op.get('margem_zoom', .10))
    _exigir(.5 <= q <= 1. and 0. <= margem <= .5,
            'Parametros do zoom de distribuicao invalidos.')
    candidatos = [float(np.quantile(p[y == classe], q))
                  for classe in (0, 1) if np.any(y == classe)]
    maximo = max([float(tau), *candidatos, float(np.min(p)), 1e-10])
    teto = min(1., maximo * (1. + margem))
    # Um minimo numerico garante ao menos dois limites diferentes.
    if teto <= 1e-10:
        teto = min(1., max(1e-6, float(np.max(p))*1.1))
    return (0., teto), teto < (1. - 1e-10)


def _dados_distribuicao(y, p, limites, bins):
    """Contagens reais e frequências normalizadas por classe real.

    As frequências dos bins são divididas por todas as observações
    daquela classe, mesmo quando o eixo usa zoom e oculta a cauda.
    """
    _exigir(bins > 0, 'Numero de bins deve ser positivo.')
    edges=np.linspace(limites[0],limites[1],bins+1)
    counts0=np.histogram(p[y==0],bins=edges)[0]
    counts1=np.histogram(p[y==1],bins=edges)[0]
    n0=int(np.sum(y==0)); n1=int(np.sum(y==1))
    fora0=n0-int(np.sum(counts0));fora1=n1-int(np.sum(counts1))
    df=pd.DataFrame({
        'inicio_bin':edges[:-1], 'fim_bin':edges[1:],
        'contagem_legitima':counts0, 'contagem_fraude':counts1,
        'frequencia_relativa_legitima':counts0/n0 if n0 else np.zeros(bins),
        'frequencia_relativa_fraude':counts1/n1 if n1 else np.zeros(bins),
        'total_legitima':n0,'total_fraude':n1,
        'fora_do_zoom_legitima':fora0,'fora_do_zoom_fraude':fora1,
    })
    return df, edges


def _distribuicao(config,y,p,escolha,destino,ext,dpi,dados):
    op=_op(config, 'distribuicao_probabilidades')
    bins=int(op.get('bins',50))
    _exigir(5<=bins<=2000,'distribuicao_probabilidades.bins fora da faixa 5..2000.')
    separar=op.get('separar_classes',True)
    normalizar=op.get('normalizar',True)
    tipo=op.get('tipo','histograma')
    _exigir(tipo in ('histograma','densidade'), 'distribuicao_probabilidades.tipo invalido.')
    escala=str(op.get('escala_y','linear'))
    _exigir(escala in ('linear','log'),'escala_y deve ser linear ou log.')
    tau=float(escolha['limiar_mcc_oof'])
    limites, zoom=_limites_x_distribuicao(op,y,p,tau)
    tab, edges=_dados_distribuicao(y,p,limites,bins)
    n0=int(np.sum(y==0)); n1=int(np.sum(y==1))

    # Figura principal
    fig = plt.figure(figsize=(12.0, 7.4), facecolor='white')
    ax = fig.add_axes([.105, .205, .855, .625])

    fig.suptitle(
        'Distribuição das probabilidades de fraude',
        x=.5,
        y=.967,
        fontsize=15,
        color='#202c3a'
    )

    # Estilos do exemplo do catalogo: legitima azul, fraude laranja.
    # Na sobreposicao, nenhuma classe e removida ou reponderada.
    conjuntos=([(p[y==0],'Legítima','#5b9bd5',n0),
                (p[y==1],'Fraude','#f29a43',n1)] if separar
               else [(p,'Todas as transações','#5b9bd5',len(p))])
    handles=[]; labels=[]
    for vals, nome, cor, n in conjuntos:
        if tipo=='histograma':
            pesos=(np.full(len(vals),1./n,dtype=float) if normalizar
                    else np.ones(len(vals),dtype=float))
            _,_,art=ax.hist(vals,bins=edges,weights=pesos,alpha=.55,
                  color=cor,edgecolor='white',linewidth=.4,label=f'{nome} (n={n})')
            handles.append(art[0]);labels.append(f'{nome} (n={n})')
        else:
            _exigir(len(vals)>2 and np.std(vals)>0,
                    f'KDE requer variacao e mais de duas observacoes: {nome}')
            xs=np.linspace(*limites,500)
            dens=gaussian_kde(vals)(xs)
            if normalizar:
                # Densidade KDE e diferente de frequencia relativa por bin.
                # Multiplicacao por largura do bin faz a escala comparavel
                # a frequencia por bin (aproximadamente, nao soma 1 no zoom).
                dens=dens*(edges[1]-edges[0])
            else:
                dens=dens*n*(edges[1]-edges[0])
            linha,=ax.plot(xs,dens,color=cor,lw=2,label=f'{nome} (n={n})')
            handles.append(linha);labels.append(f'{nome} (n={n})')

    if op.get('marcar_limiar_oof', True) and limites[0] <= tau <= limites[1]:
        l = ax.axvline(
            tau,
            color='#cb2935',
            lw=2,
            ls='-',
            label='Limiar ótimo do MCC (τ*)'
        )
        handles.append(l)
        labels.append('Limiar ótimo do MCC (τ*)')




    if op.get('marcar_limiar_05', True) and limites[0] <= .5 <= limites[1]:
        l = ax.axvline(
            .5,
            color='#242e41',
            lw=1.6,
            ls='--',
            label='Corte fixo (τ = 0.5000000)'
        )
        handles.append(l)
        labels.append('Corte fixo (τ = 0.5000000)')





    ax.set_xlim(*limites)
    ax.set_yscale(escala)
    ax.set_xlabel('Score / probabilidade estimada de fraude',fontsize=11,labelpad=8)
    ylabel=('Frequência relativa dentro de cada classe' if normalizar
            else 'Número de transações por bin')
    if tipo=='densidade':
        ylabel=('Frequência relativa aproximada por bin (KDE)' if normalizar
                else 'Contagem aproximada por bin (KDE)')

    ax.set_ylabel(ylabel, fontsize=10.8, labelpad=9)

    from matplotlib.ticker import FuncFormatter

    def formato_eixo(valor, _pos):
        valor = 0.0 if abs(valor) < 0.00000005 else valor
        return f'{valor:.7f}'

    ax.xaxis.set_major_formatter(FuncFormatter(formato_eixo))
    ax.yaxis.set_major_formatter(FuncFormatter(formato_eixo))

    ax.tick_params(axis='both', labelsize=8.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(
        config['visualizacao']['padrao'].get('grade', True),
        axis='y',
        color='#d7dfe8',
        linewidth=.75,
        alpha=.65
    )

    ax.set_axisbelow(True)
    if config['visualizacao']['padrao'].get('legenda',True):
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.892),
                   ncol=min(4,len(labels)),frameon=False,fontsize=9.4)

    # Resultados principais da validacao OOF
    ap = float(average_precision_score(y, p))
    prevalencia = float(np.mean(y))

    fig.text(
        .105, .105,
        'Resultados da validação OOF',
        fontsize=10.5,
        weight='bold',
        color='#26374d'
    )

    fig.text(
        .105, .065,
        f'AP = {ap:.7f}   |   '
        f'Prevalência = {prevalencia:.7f}   |   '
        f'τ* MCC = {tau:.7f}',
        fontsize=9.7,
        color='#26374d'
    )

    _salvar(fig,_caminho_png(destino,'distribuicao_probabilidades',ext),dpi=dpi)
    if dados:
        _csv_protegido(tab,destino/'05_distribuicao_probabilidades.csv')


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


def _comparacao(config, grade, escolha, destino, ext, dpi, dados):
    """Compara as configuracoes OOF sem alterar metricas ou selecionar outro modelo.

    Candidatos ordenados pelo score composto e distribuidos em paginas.
    Escalas horizontais iniciam em zero (salvo valores negativos) e sao
    constantes entre as paginas de uma mesma metrica.
    """
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from matplotlib.patches import Patch

    op = _op(config, 'comparacao_hiperparametros')
    validas = {'ap', 'mcc', 'logloss', 'score'}
    metricas_config = list(op.get('metricas', ['ap', 'mcc', 'logloss', 'score']))
    _exigir(
        bool(metricas_config)
        and len(set(metricas_config)) == len(metricas_config)
        and set(metricas_config).issubset(validas),
        'Metricas de comparacao desconhecidas ou repetidas.'
    )
    # O score aparece primeiro quando habilitado, pois foi o criterio de selecao.
    metricas = (['score'] if 'score' in metricas_config else []) + [
        m for m in metricas_config if m != 'score'
    ]
    _exigir(len(grade) > 0, 'Grade de candidatos vazia.')
    _exigir(grade['id_candidato'].is_unique, 'IDs de candidatos repetidos.')

    # Mesma regra de classificacao OOF: score, AP e ID em caso de empate.
    ordenada = grade.sort_values(
        ['score', 'ap', 'id_candidato'],
        ascending=[False, False, True],
        kind='stable'
    ).reset_index(drop=True)
    vencedor_id = str(escolha['id_candidato'])
    _exigir(
        str(ordenada.iloc[0]['id_candidato']) == vencedor_id,
        'Candidato selecionado diverge da ordenacao pelo score composto.'
    )

    # O parametro historico funciona como teto por pagina: nenhum candidato
    # e excluido da comparacao. O padrao e de ate oito por imagem.
    maximo = int(op.get('maximo_configuracoes', 25))
    _exigir(maximo >= 1, 'maximo_configuracoes deve ser positivo.')
    por_pagina = int(op.get('configuracoes_por_pagina', min(maximo, 8)))
    _exigir(1 <= por_pagina <= maximo, 'configuracoes_por_pagina invalido.')


    titulos = {
        'score': 'SCORE COMPOSTO',
        'ap': 'AP (AVERAGE PRECISION)',
        'mcc': 'MCC (MATTHEWS CORRELATION COEFFICIENT)',
        'logloss': 'LOGLOSS',
    }



    # Evitar zero negativo e preservar sete casas em todos os eixos.
    formato = FuncFormatter(
        lambda valor, _pos: f'{0.0 if abs(valor) < 0.00000005 else valor:.7f}'
    )
    cor_selecionado = '#245b89'
    cor_demais = '#a5c1d9'

    # Os limites sao calculados com a grade COMPLETA para permitir comparacao
    # entre paginas sem mudanca artificial de escala.
    escalas = {}
    for metrica in metricas:
        numeros = ordenada[metrica].to_numpy(dtype=float)
        _exigir(np.isfinite(numeros).all(), f'Valores invalidos para {metrica}.')
        menor = min(0.0, float(np.min(numeros)))
        maior = max(0.0, float(np.max(numeros)))
        amplitude = maior - menor
        margem = max(0.07 * amplitude, 0.0000001)
        escalas[metrica] = (
            menor - margem if menor < 0.0 else 0.0,
            maior + margem
        )

    total_paginas = (len(ordenada) + por_pagina - 1) // por_pagina
    for pagina in range(total_paginas):
        inicio = pagina * por_pagina
        parte = ordenada.iloc[inicio:inicio + por_pagina]
        posicoes = np.arange(len(parte))
        rotulos = [_rotulo_config(linha) for _, linha in parte.iterrows()]
        selecionados = (
            parte['id_candidato'].astype(str).to_numpy() == vencedor_id
        )
        cores = [cor_selecionado if s else cor_demais for s in selecionados]

        # Quatro paineis com barras horizontais: nomes nunca ficam inclinados.
        altura = max(9.7, len(metricas) * (1.5 + 0.34 * len(parte)))
        fig, eixos = plt.subplots(
            len(metricas), 1,
            figsize=(12.5, altura), squeeze=False, facecolor='white'
        )
        fig.subplots_adjust(
            left=.30, right=.81, top=.84, bottom=.065, hspace=.57
        )

        if config['visualizacao']['padrao'].get('titulo', True):
            titulo = 'Comparação de hiperparâmetros'
            if total_paginas > 1:
                titulo += f' — parte {pagina + 1}/{total_paginas}'
            fig.suptitle(titulo, fontsize=16, color='#26374d', y=.981)

        if config['visualizacao']['padrao'].get('legenda', True):
            fig.legend(
                handles=[
                    Patch(facecolor=cor_selecionado,
                          label='Modelo selecionado pelo score composto'),
                    Patch(facecolor=cor_demais, label='Demais configurações'),
                ],
                loc='upper center', bbox_to_anchor=(.55, .931),
                frameon=False, ncol=2, fontsize=9.5
            )

        for ax, metrica in zip(eixos.flat, metricas):
            valores = parte[metrica].to_numpy(dtype=float)
            ax.barh(posicoes, valores, height=.63, color=cores, zorder=3)
            ax.set_yticks(posicoes, rotulos)
            ax.invert_yaxis()  # candidato de maior score no topo
            ax.set_xlim(*escalas[metrica])
            ax.set_title(titulos[metrica], loc='left', fontsize=11.2,
                         weight='semibold', color='#26374d', pad=12)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.set_major_formatter(formato)
            ax.tick_params(axis='x', labelsize=8.4)
            ax.tick_params(axis='y', labelsize=9.0, length=0, pad=9)
            ax.grid(axis='x', color='#d7dfe8', alpha=.7,
                    linewidth=.75, zorder=0)
            ax.set_axisbelow(True)
            ax.axvline(0.0, color='#8291a3', linewidth=.8, zorder=2)
            for lateral in ('top', 'right', 'left'):
                ax.spines[lateral].set_visible(False)
            ax.spines['bottom'].set_color('#a7b4c2')

            # Coluna de valores alinhada; as barras nao encobrem os numeros.
            for indice, (valor, selecionado) in enumerate(zip(valores, selecionados)):
                ax.text(
                    1.022, indice,
                    f'{0.0 if abs(valor) < 0.00000005 else valor:.7f}',
                    transform=ax.get_yaxis_transform(),
                    ha='left', va='center', fontsize=9.0,
                    weight='bold' if selecionado else 'normal',
                    color='#245b89' if selecionado else '#35465a',
                    clip_on=False
                )
            for rotulo, selecionado in zip(ax.get_yticklabels(), selecionados):
                if selecionado:
                    rotulo.set_weight('bold')
                    rotulo.set_color('#245b89')

        if pagina == 0:
            arquivo = _caminho_png(destino, 'comparacao_hiperparametros', ext)
        else:
            arquivo = destino / (
                f'{ARQUIVOS["comparacao_hiperparametros"]}'
                f'_parte_{pagina + 1:02d}.{ext}'
            )
        _salvar(fig, arquivo, dpi=dpi)

    if dados:
        # Todos os candidatos, sem arredondamento e sem omissao por paginacao.
        _csv_protegido(
            ordenada,
            destino / '06_comparacao_hiperparametros_todos.csv'
        )



def _plot_metricas_por_fold(experimento, config, y, p, escolha, destino, ext, dpi, dados):
    """AP, MCC bruto e exp(-LogLoss) por F1..F9; sem otimizar limiares."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    # A origem unica e o NPZ consolidado do vencedor, ja validado no 06.03.
    caminho = layout.predicoes_vencedor(experimento)
    with np.load(caminho, allow_pickle=False) as arq:
        _exigir('fold' in arq.files, 'NPZ vencedor sem identificacao dos folds.')
        _exigir(str(arq['id_candidato'].item()) == str(escolha['id_candidato'])
                and int(arq['fold_esperado'].item()) == 0,
                'NPZ de folds nao corresponde ao OOF do vencedor.')
        folds = np.asarray(arq['fold'], dtype=np.int64)
        _exigir(np.array_equal(np.asarray(arq['y_true'], dtype=np.int8), y)
                and np.array_equal(np.asarray(arq['prob_fraude'], dtype=np.float64), p),
                'Folds, rotulos e probabilidades OOF desalinhados.')

    _exigir(folds.shape == y.shape
            and set(np.unique(folds)) == set(range(1, 10)),
            'Esperados exclusivamente os nove folds F1..F9.')
    tau = float(escolha['limiar_mcc_oof'])
    _exigir(np.isfinite(tau) and 0.0 <= tau <= 1.0,
            'Limiar MCC global OOF invalido.')

    linhas = []
    for k in range(1, 10):
        mascara = folds == k
        yk, pk = y[mascara], p[mascara]
        _exigir(0 < int(yk.sum()) < len(yk),
                f'Fold F{k} sem as duas classes reais.')
        perda = float(log_loss(yk, pk, labels=[0, 1]))
        linhas.append({
            'fold': k,
            'n_observacoes': int(len(yk)),
            'n_fraudes': int(yk.sum()),
            'prevalencia_fraude': float(yk.mean()),
            'ap': float(average_precision_score(yk, pk)),
            'mcc_corte_050': float(matthews_corrcoef(yk, pk >= 0.5)),
            'mcc_corte_otimo_oof': float(matthews_corrcoef(yk, pk >= tau)),
            'logloss': perda,
            'exp_menos_logloss': float(np.exp(-perda)),
            'limiar_otimo_oof_global': tau,
        })
    tabela = pd.DataFrame(linhas)
    _exigir(int(tabela['n_observacoes'].sum()) == len(y),
            'Contagem dos folds difere do OOF total.')

    x = np.arange(1, 10)
    formato = FuncFormatter(
        lambda v, _: f'{0.0 if abs(v) < 0.00000005 else v:.7f}'
    )
    fig, eixos = plt.subplots(3, 1, figsize=(12.5, 11.5), facecolor='white')
    fig.subplots_adjust(left=.13, right=.965, top=.88, bottom=.075, hspace=.49)

    fig.suptitle('ESTABILIDADE DAS MÉTRICAS POR FOLD OOF',
                 fontsize=16, fontweight='bold', y=.980, color='#20324a')










    ax = eixos[0]
    ax.plot(x, tabela['ap'], '-o', color='#285da2', lw=2,
            ms=5, label='AP por fold')
    ax.plot(x, tabela['prevalencia_fraude'], '--s', color='#ad8573',
            lw=1.3, ms=3.5, label='Prevalência por fold')
    ax.set_title('A — Average Precision (AP)', loc='left',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('AP / prevalência')
    ax.legend(frameon=False, fontsize=9, loc='upper right')

    ax = eixos[1]
    ax.plot(x, tabela['mcc_corte_050'], '-o', color='#d08040',
            lw=1.8, ms=5, label='MCC | τ = 0,5000000')
    ax.plot(x, tabela['mcc_corte_otimo_oof'], '-o', color='#218765',
            lw=1.8, ms=5, label=f'MCC | τ* = {tau:.7f}')
    ax.axhline(0, color='#8b96a6', lw=.85, ls=':')
    ax.set_title('B — Matthews Correlation Coefficient (MCC)',
                 loc='left', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('MCC bruto')
    ax.legend(frameon=False, fontsize=9, loc='upper right')

    ax = eixos[2]
    ax.plot(x, tabela['exp_menos_logloss'], '-o', color='#7656a5',
            lw=2, ms=5, label='exp(-LogLoss) por fold')
    ax.set_title('C — exp(-LogLoss)', loc='left',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('exp(-LogLoss)')
    ax.legend(frameon=False, fontsize=9, loc='upper right')

    for ax in eixos:
        ax.set_xlim(.7, 9.3)
        ax.set_xticks(x, [f'F{k}' for k in x])
        ax.set_xlabel('Fold de validação OOF')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(formato)
        ax.tick_params(axis='y', labelsize=8.7)
        ax.grid(axis='y', color='#dbe3ed', alpha=.7, lw=.8)
        ax.set_axisbelow(True)
        for lado in ('top', 'right'):
            ax.spines[lado].set_visible(False)

    _salvar(fig, _caminho_png(destino, 'metricas_por_fold', ext), dpi=dpi)
    if dados:
        _csv_protegido(tabela, destino / '07_metricas_por_fold.csv')


def gerar_graficos_oof(experimento: Path):
    """Gera os graficos preditivos e o Painel 00 configurados e retorna resumo (sem tocar no F10)."""
    experimento=Path(experimento).resolve()
    config, grade, escolha, y, p = _ler_resultados(experimento)
    pad=config['visualizacao']['padrao']
    ext=str(pad.get('formato','png')).lower()
    _exigir(ext in {'png','svg','pdf'}, 'visualizacao.padrao.formato deve ser png, svg ou pdf.')
    dpi=int(pad.get('dpi',180))
    _exigir(50<=dpi<=1200,'visualizacao.padrao.dpi fora da faixa 50..1200.')
    dados=pad.get('salvar_dados_grafico',True) is True
    destino=layout.imagens_oof(experimento)
    destino.mkdir(parents=True,exist_ok=True)
    vencedor=grade.iloc[0]
    gerados=[];omitidos=[]
    funcoes={
        'painel_score_composto':lambda:_plot_painel_00(config,y,p,vencedor,escolha,destino,ext,dpi,dados),
        'curva_pr':lambda:_plot_pr(config,y,p,vencedor,escolha,destino,ext,dpi,dados),
        'curva_mcc':lambda:_plot_mcc(config,y,p,vencedor,escolha,destino,ext,dpi,dados),
        'matriz_confusao_05':lambda:_matriz(config,y,p,.5,'matriz_confusao_05',destino,ext,dpi,dados),
        'matriz_confusao_mcc':lambda:_matriz(config,y,p,float(escolha['limiar_mcc_oof']), 'matriz_confusao_mcc',destino,ext,dpi,dados),
        'distribuicao_probabilidades':lambda:_distribuicao(config,y,p,escolha,destino,ext,dpi,dados),
        'comparacao_hiperparametros':lambda:_comparacao(config,grade,escolha,destino,ext,dpi,dados),
        'metricas_por_fold':lambda:_plot_metricas_por_fold(experimento,config,y,p,escolha,destino,ext,dpi,dados),
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
            'tipos_geometricos_pendentes_passo_07_02':[
                nome for nome in config['visualizacao']['graficos'] if nome not in ARQUIVOS
            ],
            'diretorio':str(destino)}
    if 'painel_score_composto' in gerados:
        painel_meta = {'tipo':'painel_00_composto', 'origem':'OOF_F1_F9',
                       'id_vencedor':escolha['id_candidato'], 'f10_utilizado':False,
                       'indicadores':_numeros_painel_00(y,p,vencedor,escolha)}
        painel_path = destino/'manifesto_painel_00.json'
        conteudo = json.dumps(painel_meta,ensure_ascii=False,indent=2)+'\n'
        if not painel_path.exists():
            painel_path.write_text(conteudo,encoding='utf-8')
        elif painel_path.read_text(encoding='utf-8') != conteudo:
            warnings.warn('Manifesto Painel 00 anterior divergente: preservado.',stacklevel=2)
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
