# -*- coding: utf-8 -*-
"""ETAPA 9/9 — galeria F10, SOMENTE dos artefatos da Etapa 8.

    python scripts/execucao/ETAPA9_GERAR_GALERIA_F10_PCA.py --verificar
    python scripts/execucao/ETAPA9_GERAR_GALERIA_F10_PCA.py --gerar

Este programa NAO importa os modulos de treinamento/inferencia, NAO consulta o
Parquet, NAO carrega modelos .joblib, NAO recalcula o limiar do MCC e NAO
reexecuta avaliacao no F10. Lê apenas CSVs/JSONs JA SALVOS da Etapa 8.
Os resultados visuais sao DESCRITIVOS, amostrais e exploratorios.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
ORIGEM_PADRAO = RAIZ / 'outputs' / 'avaliacao_final_f10_pca_10pct_20261029T232726363532Z'
METRICAS_7 = ('ap', 'mcc_limiar_oof', 'logloss', 'score_composto_limiar_oof')


def exigir(condicao, mensagem):
    if not condicao:
        raise RuntimeError(mensagem)


def sha256(arquivo: Path) -> str:
    digest = hashlib.sha256()
    with arquivo.open('rb') as file:
        for trecho in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(trecho)
    return digest.hexdigest()


def json_ler(arquivo: Path) -> dict:
    with arquivo.open('r', encoding='utf-8') as f:
        return json.load(f)


def conferir_fonte(origem: Path) -> tuple[dict, dict]:
    """Le SOMENTE os resultados produzidos na Etapa 8; NENHUMA escrita."""
    exigir(origem.is_dir(), f'Pasta da Etapa 8 nao encontrada: {origem}')
    arquivo_m = origem / 'manifesto_avaliacao_f10.json'
    exigir(arquivo_m.is_file(), 'Manifesto da avaliacao F10 ausente')
    manifesto = json_ler(arquivo_m)
    exigir(manifesto.get('versao') == 'PCA_F10_10PCT_V1', 'Versao de F10 inesperada')
    exigir(manifesto.get('carater') == 'AVALIACAO_EXPLORATORIA_AMOSTRAL', 'Carater nao exploratorio')
    exigir(manifesto.get('f10_avaliado') is True, 'F10 nao foi concluido')
    exigir(manifesto.get('otimizacao_limiar_em_f10') is False, 'Otimizacao no F10 proibida')
    exigir(manifesto.get('selecionar_vencedor_em_f10') is False, 'Selecao por F10 proibida')
    protocolo_relativo = manifesto.get('referencia_protocolo')
    exigir(isinstance(protocolo_relativo, str), 'Manifesto sem referencia pre-F10')
    arquivo_protocolo = (RAIZ / protocolo_relativo).resolve()
    exigir(arquivo_protocolo.is_relative_to((RAIZ / 'outputs').resolve()),
           'Protocolo precisa estar na pasta outputs')
    exigir(arquivo_protocolo.is_file(), 'Manifesto pre-F10 nao encontrado')
    exigir(sha256(arquivo_protocolo) == manifesto.get('sha256_protocolo_pre_f10'),
           'Manifesto PRE-F10 nao corresponde ao SHA-256 congelado')
    pre = json_ler(arquivo_protocolo)
    exigir(pre.get('situacao') == 'PREFLIGHT_CONGELADO_SEM_AVALIACAO',
           'Protocolo PRE-F10 nao congelado')
    exigir(pre.get('protocolo_teste_pre_registrado', {}).get('fracao_teste') == .1,
           'Fracao do protocolo congelado divergente')
    modelos_pre = {x['pca_dimensao']: x for x in pre.get('modelos', [])}
    exigir(set(modelos_pre) == {1, 2, 3}, 'Protocolo pre-F10 sem os tres modelos')
    amostra = manifesto.get('amostra_f10', {})
    exigir(amostra.get('tipo') == 'AMOSTRA_F10_10PCT_ESTRATIFICADA_PRE_REGISTRADA',
           'Amostra nao corresponde ao protocolo pre-registrado')
    exigir(amostra.get('fracao') == 0.1 and amostra.get('semente') == 42,
           'Fracao ou semente diferem do protocolo')

    entradas = manifesto.get('resultados_por_dimensao', [])
    exigir(len(entradas) == 3 and {e.get('pca_dimensao') for e in entradas} == {1, 2, 3},
           'Manifesto precisa conter exatamente PCA 1D/2D/3D')
    resultados = {}
    referencia_ids = referencia_y = None
    for entrada in entradas:
        dim = entrada['pca_dimensao']
        pre_modelo = modelos_pre[dim]
        exigir(entrada['limiar_mcc_oof_congelado'] == pre_modelo['limiar_mcc_oof_congelado'],
               f'Limiar PCA {dim}D diferente do PRE-F10 congelado')
        exigir(entrada['candidato_oof'] == pre_modelo['candidato_oof'],
               f'Candidato PCA {dim}D diferente do PRE-F10 congelado')
        pasta = origem / f'PCA_{dim}D'
        csv_path = pasta / 'predicoes_f10.csv'
        metricas_path = pasta / 'metricas_f10.json'
        exigir(csv_path.is_file() and metricas_path.is_file(), f'Arquivos faltantes em PCA {dim}D')
        exigir(sha256(csv_path) == entrada['sha256_predicoes_f10'], f'CSV PCA {dim}D alterado')
        exigir(sha256(metricas_path) == entrada['sha256_metricas_f10'], f'Metricas PCA {dim}D alteradas')
        exigir(entrada['predicoes_f10'].replace('\\', '/').endswith(f'/PCA_{dim}D/predicoes_f10.csv'),
               f'Localizacao CSV declarada difere PCA {dim}D')
        exigir(entrada['metricas_f10'].replace('\\', '/').endswith(f'/PCA_{dim}D/metricas_f10.json'),
               f'Localizacao JSON declarada difere PCA {dim}D')
        tabela = pd.read_csv(csv_path)
        met = json_ler(metricas_path)
        colunas = ['linha_original', 'fold', 'y_true', 'prob_fraude',
                   'predicao_tau_0_50', 'predicao_tau_mcc_oof']
        exigir(tabela.columns.tolist() == colunas, f'Colunas inesperadas PCA {dim}D')
        exigir(len(tabela) == amostra['n_linhas'] == met['n'], f'Tamanho F10 PCA {dim}D diferente')
        ids = tabela['linha_original'].to_numpy()
        labels = tabela['y_true'].to_numpy()
        probabilidades = tabela['prob_fraude'].to_numpy(dtype=float)
        exigir(np.issubdtype(ids.dtype, np.integer), f'IDs nao inteiros PCA {dim}D')
        exigir(np.all(ids >= 0) and np.all(np.diff(ids) > 0), f'IDs F10 nao crescentes PCA {dim}D')
        exigir(np.all(tabela['fold'].to_numpy() == 10), f'Folds != 10 PCA {dim}D')
        exigir(np.isin(labels, [0, 1]).all(), f'Rotulos invalidos PCA {dim}D')
        exigir(np.isfinite(probabilidades).all() and np.all((probabilidades >= 0) & (probabilidades <= 1)),
               f'Scores invalidos PCA {dim}D')
        exigir(int(np.count_nonzero(labels == 0)) == amostra['n_legitimas'] == met['n_legitimas'],
               f'Legitimas divergentes PCA {dim}D')
        exigir(int(np.count_nonzero(labels == 1)) == amostra['n_fraudes'] == met['n_fraudes'],
               f'Fraudes divergentes PCA {dim}D')
        if referencia_ids is None:
            referencia_ids = ids.copy()
            referencia_y = labels.copy()
        else:
            exigir(np.array_equal(ids, referencia_ids) and np.array_equal(labels, referencia_y),
                   f'Diferenca na amostra ou nos rotulos PCA {dim}D')
        tau = met.get('limiar_mcc_oof_congelado')
        exigir(isinstance(tau, (int, float)) and np.isfinite(tau) and 0 <= tau <= np.nextafter(1.0, np.inf),
               f'Tau invalido PCA {dim}D')
        exigir(tau == entrada['limiar_mcc_oof_congelado'], f'Tau manifesto divergente PCA {dim}D')
        exigir(met.get('limiar_otimizado_no_f10') is False, 'O limiar foi otimizado no F10')
        exigir(met.get('dimensao_pca') == dim, f'Dimensao nas metricas divergente PCA {dim}D')
        exigir(met.get('id_candidato_oof') == entrada['candidato_oof'], f'Candidato divergente PCA {dim}D')
        exigir(np.array_equal(tabela['predicao_tau_0_50'].to_numpy(), (probabilidades >= 0.5).astype(int)),
               f'Predicao fixa 0.5 nao confere PCA {dim}D')
        exigir(np.array_equal(tabela['predicao_tau_mcc_oof'].to_numpy(), (probabilidades >= tau).astype(int)),
               f'Predicao τ* nao confere PCA {dim}D')
        for nome, chave in [('tau_0_50', 'predicao_tau_0_50'),
                            ('tau_mcc_oof', 'predicao_tau_mcc_oof')]:
            corte = met['comparacao_limiares'][nome]
            pred = tabela[chave].to_numpy()
            tn = int(np.count_nonzero((labels == 0) & (pred == 0)))
            fp = int(np.count_nonzero((labels == 0) & (pred == 1)))
            fn = int(np.count_nonzero((labels == 1) & (pred == 0)))
            tp = int(np.count_nonzero((labels == 1) & (pred == 1)))
            exigir((tn, fp, fn, tp) == tuple(corte['matriz_confusao'][k] for k in ('tn', 'fp', 'fn', 'tp')),
                   f'Matriz divergente PCA {dim}D {nome}')
            exigir(corte['positivos_preditos'] == tp + fp, 'Total previsto positivo divergente')
            exigir(abs(corte['limiar'] - (0.5 if nome == 'tau_0_50' else tau)) < 1e-14,
                   'Corte das metricas divergente')
        for k in METRICAS_7:
            exigir(isinstance(met[k], (int, float)) and math.isfinite(met[k]), f'Metrica invalida: {k}')
        exigir(all(abs(float(met[k]) - float(entrada[out])) < 1e-12 for k, out in [
            ('ap','ap'), ('mcc_limiar_oof','mcc_limiar_oof'),
            ('logloss','logloss'), ('score_composto_limiar_oof','score_composto_limiar_oof')]),
            f'Metricas do manifesto divergentes PCA {dim}D')
        resultados[dim] = {'tabela': tabela, 'metricas': met, 'arquivos': {
            'csv_sha256': entrada['sha256_predicoes_f10'],
            'metricas_sha256': entrada['sha256_metricas_f10'],
        }}
        print(f'[APROVADO] PCA {dim}D | n={len(tabela)} | fraudes={int(labels.sum())} | '
              f'tau={tau:.17g} | CSV/JSON conferidos', flush=True)
    return manifesto, resultados


def preparar_plotagem():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'font.weight': 'normal',
        'axes.titlesize': 13, 'axes.titleweight': 'bold',
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.facecolor': '#ffffff', 'figure.facecolor': '#ffffff',
        'axes.labelcolor': '#20304b', 'text.color': '#132239',
        'axes.edgecolor': '#b8c2d1', 'xtick.color': '#50627b', 'ytick.color': '#50627b',
    })
    return plt


def rodape(fig, nota):
    fig.text(0.02, 0.01, nota, fontsize=8.2, color='#54647d')


def salvar(fig, pasta: Path, nome: str, plt):
    fig.savefig(pasta / nome, dpi=160, bbox_inches='tight', pad_inches=0.22)
    plt.close(fig)
    return nome


def painel_resumo(plt, dim, met, pasta, nfra, n):
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(13.6, 6.3))
    fig.text(.05, .89, f'PAINEL FINAL F10  |  GMM + PCA {dim}D', fontsize=22, weight='bold')
    fig.text(.05, .83, 'Avaliação exploratória • 10% estratificado • limiar congelado no OOF', fontsize=11, color='#516079')
    rotulos = [('AP', met['ap']), ('MCC (τ* OOF)', met['mcc_limiar_oof']),
               ('LOGLOSS', met['logloss']), ('SCORE COMPOSTO', met['score_composto_limiar_oof'])]
    for i, (titulo, val) in enumerate(rotulos):
        x = .05 + i * .233
        rect = FancyBboxPatch((x, .48), .213, .27, transform=fig.transFigure,
                              boxstyle='round,pad=.01,rounding_size=.015',
                              linewidth=.9, edgecolor='#dee5ee', facecolor='#f5f8fc')
        fig.add_artist(rect)
        fig.text(x+.017, .665, titulo, fontsize=10, weight='bold', color='#506079')
        fig.text(x+.017, .545, f'{val:.7f}', fontsize=19.5, weight='bold', color='#15233b')
    fig.text(.05, .373, 'INFORMAÇÕES ÚTEIS', weight='bold', fontsize=13)
    linhas = [f'Amostra F10: {n:,} transações  •  fraudes: {nfra:,}  •  legítimas: {n-nfra:,}',
              f'Prevalência: {met["prevalencia"]:.7f}  •  τ*= {met["limiar_mcc_oof_congelado"]:.7f}',
              f'MCC τ=0,50: {met["comparacao_limiares"]["tau_0_50"]["mcc"]:.7f}  •  MCC τ*: {met["mcc_limiar_oof"]:.7f}',
              'τ* definido exclusivamente em F1–F9 (OOF). Nenhum ajuste foi feito usando F10.']
    for i, linha in enumerate(linhas):
        fig.text(.05, .315-i*.053, linha, fontsize=11, color='#35465d')
    rodape(fig, 'Fonte: predições e métricas F10 da Etapa 8 • Mesma amostra para PCA 1D / 2D / 3D')
    return salvar(fig, pasta, '01_painel_final.png', plt)


def curva_pr(plt, dim, tabela, met, pasta):
    from sklearn.metrics import precision_recall_curve
    y = tabela['y_true'].to_numpy()
    s = tabela['prob_fraude'].to_numpy()
    precision, recall, _ = precision_recall_curve(y, s)
    corte = met['comparacao_limiares']['tau_mcc_oof']
    px, rx = corte['precision'], corte['recall_sensibilidade']
    # É uma curva DESCRITIVA do mesmo F10 já avaliado, não uma busca de tau.
    fig, axs = plt.subplots(1, 2, figsize=(13.8, 5.7), gridspec_kw={'width_ratios': [1, 1.8]})
    fig.suptitle(f'Curva Precision–Recall no F10  |  PCA {dim}D', fontsize=18, weight='bold', y=.985)
    base = met['prevalencia']
    for ax in axs:
        ax.step(recall, precision, where='post', color='#1769aa', linewidth=1.7, label='Precision–Recall (F10)')
        ax.axhline(base, ls=':', color='#a55c26', lw=1.4, label=f'Prevalência = {base:.7f}')
        ax.scatter([rx], [px], s=54, marker='o', color='#ba3843', zorder=5, label=f'τ* OOF = {met["limiar_mcc_oof_congelado"]:.7f}')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Recall (sensibilidade)')
        ax.set_ylabel('Precision')
        ax.grid(alpha=.13)
    axs[0].set_ylim(0, 1.01)
    axs[0].set_title('Visão completa — Precision 0 a 1')
    interior = precision[(recall > 0.01) & (recall < 0.99)]
    q = float(np.quantile(interior, .90)) if len(interior) else base
    alto = min(1.0, max(.03, q * 2.0, px * 1.18, base * 5))
    axs[1].set_ylim(0, alto)
    axs[1].set_title(f'Detalhe da faixa inferior (até {alto:.3f})')
    axs[1].legend(loc='upper right', fontsize=8.5)
    fig.subplots_adjust(bottom=.16, top=.83, wspace=.26)
    rodape(fig, f'AP={met["ap"]:.7f} • τ* congelado em OOF, apenas assinalado no F10; sem otimização de limiar')
    return salvar(fig, pasta, '02_curva_precision_recall.png', plt)


def distribuicao_scores(plt, dim, tabela, met, pasta):
    from matplotlib.ticker import FuncFormatter
    labels = tabela['y_true'].to_numpy()
    scores = tabela['prob_fraude'].to_numpy(dtype=float)
    # log10(score + eps): não oculta probabilidades muito pequenas.
    xs = np.log10(scores + 1e-10)
    minimo = float(min(-10, np.min(xs)))
    bins = np.linspace(minimo, np.log10(1 + 1e-10), 75)
    fig, axs = plt.subplots(1, 2, figsize=(13.8, 5.7), sharex=True, sharey=True)
    fig.suptitle(f'Distribuição dos scores F10  |  PCA {dim}D', fontsize=18, weight='bold')
    for ax, (valor, nome) in zip(axs, [(0.5,'τ = 0,50'), (met['limiar_mcc_oof_congelado'],'τ* OOF')]):
        for classe, cor, rotulo in [(0, '#2378ac', 'Legítimas'), (1, '#c7464e', 'Fraudes')]:
            sample = xs[labels == classe]
            ax.hist(sample, bins=bins, weights=np.full(len(sample), 100 / len(sample)),
                    histtype='step', linewidth=1.7, color=cor, label=rotulo)
        ax.axvline(np.log10(valor + 1e-10), color='#262e39', ls='--', lw=1.7,
                   label=f'Corte {valor:.7f}')
        ax.set_title(nome)
        ax.set_xlabel('Score de fraude (escala logarítmica)')
        ax.grid(alpha=.14, axis='y')
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'$10^{{{int(x)}}}$' if abs(x-round(x))<.1 else ''))
    axs[0].set_ylabel('Frequência por classe (%) — normalizada separadamente')
    fig.subplots_adjust(bottom=.17, top=.86, wspace=.16)
    rodape(fig, 'Distribuição descritiva; normalização por classe evita ocultar as 96 fraudes. Dois cortes fixos; não se procura corte ótimo no F10.')
    return salvar(fig, pasta, '03_distribuicao_scores_2_limiares.png', plt)


def matrizes_confusao(plt, dim, met, pasta):
    from matplotlib.patches import Rectangle
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.8))
    fig.suptitle(f'Matrizes de confusão do F10  |  PCA {dim}D', fontsize=18, weight='bold')
    for ax, nome, limiar in zip(axs, ('tau_0_50','tau_mcc_oof'), (0.5,met['limiar_mcc_oof_congelado'])):
        d = met['comparacao_limiares'][nome]
        mat = d['matriz_confusao']['contagens']
        percent = d['matriz_confusao']['percentuais_por_classe_real']
        ax.set_xlim(0, 2)
        ax.set_ylim(2, 0)
        for i in range(2):
            for j in range(2):
                cor = '#e4f5e9' if i == j else '#fce9e8'
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=cor, edgecolor='white', linewidth=4))
                ax.text(j+.5, i+.42, f'{int(mat[i][j]):,}', ha='center', va='center', fontsize=21, weight='bold')
                ax.text(j+.5, i+.70, f'{percent[i][j]:.2f}% da classe', ha='center', fontsize=10, color='#42536b')
        ax.set_xticks([.5, 1.5], ['Pred. legítima', 'Pred. fraude'])
        ax.set_yticks([.5, 1.5], ['Real legítima', 'Real fraude'])
        ax.tick_params(length=0)
        ax.set_aspect('equal')
        ax.set_title(f'τ = {limiar:.7f}   |   MCC = {d["mcc"]:.7f}', fontsize=12)
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.subplots_adjust(bottom=.18, top=.83, wspace=.3)
    rodape(fig, 'Verde = acerto (TN / TP); vermelho = erro (FP / FN). Percentuais por classe real; valores absolutos incluídos.')
    return salvar(fig, pasta, '04_matrizes_confusao_050_vs_tau_oof.png', plt)


def comparacao_geral(plt, resultados, pasta):
    fig, axs = plt.subplots(2, 2, figsize=(13.8, 9.1))
    valores = [('ap','Average Precision (AP)'), ('mcc_limiar_oof','MCC em τ* OOF'),
               ('logloss','LogLoss (menor = melhor)'), ('score_composto_limiar_oof','Score composto')]
    dims = [1, 2, 3]
    labels = [f'PCA {d}D' for d in dims]
    cores = ['#2f658e','#618cad','#94b1c8']
    fig.suptitle('Comparativo descritivo • três modelos GMM + PCA no mesmo F10', fontsize=18, weight='bold')
    for ax, (key, titulo) in zip(axs.ravel(), valores):
        y = [resultados[d]['metricas'][key] for d in dims]
        bars = ax.bar(labels, y, color=cores, width=.55)
        ax.set_title(titulo)
        ax.set_ylim(0, max(y)*1.25 if max(y) > 0 else 1)
        ax.grid(axis='y', alpha=.13)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, y):
            ax.annotate(f'{val:.7f}', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9.5)
    fig.subplots_adjust(top=.90, bottom=.09, hspace=.38, wspace=.26)
    rodape(fig, 'F10 10% • somente descrição pós-teste • proibição de selecionar modelo ou alterar configurações com base neste comparativo.')
    return salvar(fig, pasta, '00_comparativo_metricas_f10.png', plt)


def html_galeria(manifesto: dict, resultados: dict, caminhos: dict, pasta: Path):
    cards = []
    for dim in (1, 2, 3):
        met = resultados[dim]['metricas']
        imagens = caminhos[dim]
        itens = '\n'.join(f'<a href="PCA_{dim}D/{html.escape(img)}" target="_blank"><img src="PCA_{dim}D/{html.escape(img)}" loading="lazy" alt="{html.escape(img)}"></a>' for img in imagens)
        cards.append(f'''<section id="pca{dim}"><h2>PCA {dim}D</h2>
        <p class="info">AP {met['ap']:.7f} · MCC τ* {met['mcc_limiar_oof']:.7f} · LogLoss {met['logloss']:.7f} · Score {met['score_composto_limiar_oof']:.7f} · τ* {met['limiar_mcc_oof_congelado']:.7f}</p>
        <div class="figuras">{itens}</div></section>''')
    quant = manifesto['amostra_f10']
    corpo = '''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Galeria F10 | GMM + PCA 1D, 2D, 3D</title>
    <style>:root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#15233a;background:#f0f3f8}
    body{margin:0}.wrap{max-width:1250px;margin:auto;padding:28px}header{background:#10263c;color:white;padding:38px 25px}
    header .wrap{padding:0 25px}h1{font-size:30px;margin:0 0 12px}h2{font-size:24px;margin:8px 0 6px}
    .sub{color:#d0dce9;line-height:1.55}.aviso{border-left:4px solid #b76d22;background:#fff6e8;padding:16px 20px;margin:25px 0;border-radius:5px;line-height:1.55}
    section{background:white;border-radius:14px;margin:26px 0;padding:22px 26px;box-shadow:0 2px 9px #19283a14}
    .figuras{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:18px;margin-top:20px}
    .figuras a{background:#fafbfd;border:1px solid #e3e9f1;border-radius:10px;padding:10px;display:flex;align-items:center}
    .figuras img{width:100%;height:auto;display:block}.info{color:#53657c;line-height:1.7}
    nav{display:flex;gap:10px;flex-wrap:wrap}nav a{color:white;border:1px solid #8495a5;border-radius:20px;padding:7px 13px;text-decoration:none}
    .mono{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}footer{padding:20px 0 50px;color:#53657c}</style></head>
    <body><header><div class="wrap"><h1>Resultados F10 — GMM + PCA</h1>
    <p class="sub">Avaliação exploratória com amostra estratificada pré-registrada de 10% do F10. Apenas dados já exportados na Etapa 8; nenhuma nova inferência.</p>
    <nav><a href="#global">Comparação geral</a><a href="#pca1">PCA 1D</a><a href="#pca2">PCA 2D</a><a href="#pca3">PCA 3D</a></nav></div></header>
    <main class="wrap"><div class="aviso"><b>Interpretação científica:</b> os limiares τ* vieram exclusivamente do OOF F1–F9. O F10 foi consultado anteriormente na Etapa 8; os resultados são exploratórios e não podem ser usados para reajustar limiares, refazer o ranking de configurações ou declarar um vencedor para o treino integral.</div>
    <section id="global"><h2>Visão conjunta</h2>
    <p class="info">''' + f'{quant["n_linhas"]:,} transações · {quant["n_fraudes"]:,} fraudes · {quant["n_legitimas"]:,} legítimas · semente {quant["semente"]}' + '''</p>
    <div class="figuras"><a href="00_comparativo_metricas_f10.png" target="_blank"><img src="00_comparativo_metricas_f10.png" alt="Comparativo das quatro métricas F10"></a></div></section>
    ''' + '\n'.join(cards) + '''<footer>Fonte: artefatos imutáveis da Etapa 8, verificados por SHA-256. Galeria gerada sem ler Parquet, modelos, folds ou F10 original.</footer></main></body></html>'''
    (pasta / 'index.html').write_text(corpo, encoding='utf-8')


def gerar(origem: Path, manifesto: dict, resultados: dict) -> Path:
    plt = preparar_plotagem()
    agora = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destino = RAIZ / 'outputs' / ('galeria_final_f10_pca_10pct_' + agora)
    exigir(destino != origem and not destino.exists(), 'Destino de figuras ja existe')
    destino.mkdir(parents=False, exist_ok=False)
    caminhos = {}
    print(f'\n[DESTINO EXCLUSIVO] {destino}', flush=True)
    arquivo_global = comparacao_geral(plt, resultados, destino)
    for dim in (1, 2, 3):
        print(f'[GRAFICOS] PCA {dim}D: 4 arquivos', flush=True)
        pasta = destino / f'PCA_{dim}D'
        pasta.mkdir(exist_ok=False)
        tabela = resultados[dim]['tabela']
        met = resultados[dim]['metricas']
        caminhos[dim] = [
            painel_resumo(plt, dim, met, pasta, int((tabela['y_true']==1).sum()), len(tabela)),
            curva_pr(plt, dim, tabela, met, pasta),
            distribuicao_scores(plt, dim, tabela, met, pasta),
            matrizes_confusao(plt, dim, met, pasta),
        ]
    html_galeria(manifesto, resultados, caminhos, destino)
    gerados = [arquivo_global, 'index.html'] + [f'PCA_{d}D/{nome}' for d in (1,2,3) for nome in caminhos[d]]
    auditoria = {
        'versao': 'GALERIA_F10_10PCT_V1',
        'origem_f10_ja_avaliado': str(origem.relative_to(RAIZ)).replace('\\','/'),
        'sha256_manifesto_f10': sha256(origem / 'manifesto_avaliacao_f10.json'),
        'amostra': manifesto['amostra_f10'],
        'tipo': 'VISUALIZACAO_DESCRITIVA_EXPLORATORIA_SEM_REAVALIAR_F10',
        'treinamento': False, 'nova_inferencia': False, 'nova_selecao': False,
        'nova_otimizacao_limiar': False,
        'arquivos_gerados': [{'arquivo': a, 'sha256': sha256(destino / a)} for a in gerados],
    }
    (destino/'manifesto_galeria_f10.json').write_text(json.dumps(auditoria, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(f'[CONCLUIDO] 13 graficos PNG + galeria HTML em: {destino / "index.html"}', flush=True)
    return destino


def main():
    p = argparse.ArgumentParser(description=__doc__)
    op = p.add_mutually_exclusive_group(required=True)
    op.add_argument('--verificar', action='store_true')
    op.add_argument('--gerar', action='store_true')
    p.add_argument('--origem', default=str(ORIGEM_PADRAO), help='Pasta da avaliacao F10 JA CONCLUIDA')
    a = p.parse_args()
    origem = Path(a.origem).resolve()
    exigir(origem.is_relative_to((RAIZ / 'outputs').resolve()), 'Origem precisa estar em outputs/')
    print('=== ETAPA 9/9: GALERIA DOS RESULTADOS F10 JÁ SALVOS ===', flush=True)
    print('Nao acessa Parquet, folds, modelos ou rótulos F10 originais; NAO treina.', flush=True)
    manifesto, resultados = conferir_fonte(origem)
    if a.verificar:
        print('\n[APROVADO] 3 resultados conferidos: hashes, contagens, alinhamento e cortes fixos.', flush=True)
        print('Nenhum arquivo escrito. Use --gerar para produzir as novas figuras.', flush=True)
    else:
        gerar(origem, manifesto, resultados)


if __name__ == '__main__':
    main()
