# -*- coding: utf-8 -*-
"""V7: galeria HTML offline e auditoria F10 t-SNE+GMM 1D.

Modos:
    --planejar  : apenas leitura/validacao de manifestos e hashes V1-V6.
    --executar  : gera UMA NOVA galeria, sem modificar V1-V6, OOF ou PCA.
    --validar   : valida galeria ja gerada e toda sua proveniencia.

Nao le Parquet nem arquivos de folds mestre, nao desserializa joblib,
nao executa fit/transform/predict, nao calcula novo limiar, nao le rede.
Os 12 PNGs e o PRE01 HTML sao COPIADOS (e verificados), nunca recriados.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
DIR_NAME = 'galeria'
LOCK_NAME = '.galeria_tsne_f10_v7_em_andamento.lock'

GRAPH_SPECS = (
    ('v4', '00_painel_score_composto.png', '00 — Painel inicial', 'Indicadores AP, MCC, LogLoss e score composto no F10.', 'desempenho'),
    ('v4', '01_curva_precision_recall.png', '01 — Curva Precision–Recall', 'Ordenação de fraudes, prevalência e limiar τ* OOF fixo.', 'desempenho'),
    ('v4', '02_mcc_por_limiar.png', '02 — MCC por limiar', 'Curva diagnóstica F10 com marcador no limiar escolhido em OOF; não reotimizar.', 'desempenho'),
    ('v4', '03_matriz_confusao_05.png', '03 — Matriz de confusão τ=0,50', 'Contagens e proporções no corte de referência.', 'desempenho'),
    ('v4', '04_matriz_confusao_mcc.png', '04 — Matriz de confusão τ* OOF', 'Classificação F10 usando o limiar congelado no desenvolvimento.', 'desempenho'),
    ('v4', '05_distribuicao_probabilidades.png', '05 — Distribuição dos scores', 'Scores das duas classes e limiares, sem alterar probabilidades.', 'desempenho'),
    ('v5', 'GMM01_densidade_marginal_1d_TSNE1.png', 'GMM01 — Densidade marginal', 'Mistura e componentes gaussianos aprendidos em F1–F9; densidade não é risco.', 'geometria'),
    ('v5', 'GMM04_dominancia_1d.png', 'GMM04 — Dominância gaussiana', 'Componente predominante por posição no eixo TSNE1.', 'geometria'),
    ('v5', 'GMM07_sensibilidade_1d_comparativo.png', 'GMM07 — Sensibilidade da fronteira', 'Mesma função de risco; comparação entre τ=0,50 e τ* OOF com MCC F10.', 'geometria'),
    ('v5', 'GMM08_mapa_risco_1d_comparativo.png', 'GMM08 — Risco e geometria 95%', 'Mapa contínuo com gaussianas 95% e dois limiares, sem refit.', 'geometria'),
    ('v6', '13_tsne_mapa_TSNE1.png', 'TSNE13 — Histograma TSNE1', 'Coordenadas fora da amostra (OOS), classes reais sobrepostas para descrição.', 'diagnosticos'),
    ('v6', '14_tsne_diagnostico.png', 'TSNE14 — KL e tempo', 'KL do ajuste F1–F9; tempo OOS da V3 não registrado.', 'diagnosticos'),
)
PRE01_REL = 'PRE01_F10/PRE01_F10_diagnostico_padronizacao_completo.html'
PRE01_COPY = 'PRE01_F10/PRE01_F10_diagnostico_padronizacao_completo.html'


def exigir(cond, mensagem):
    if not cond:
        raise RuntimeError(mensagem)


def sha_file(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def ler_json(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        obj = json.load(f)
    exigir(isinstance(obj, dict), 'JSON nao e objeto: ' + str(path))
    return obj


def posix_rel(destino, base):
    return Path(os.path.relpath(destino, base)).as_posix()


def png_valido(p):
    with Path(p).open('rb') as fd:
        return fd.read(8) == b'\x89PNG\r\n\x1a\n'


def preflight():
    # As verificacoes V6->V5->V4->V3->V2 e V1 sao SOMENTE LEITURA.
    from scripts.execucao import v6_diagnosticos_tsne1d_f10 as v6
    from scripts.execucao import v5_geometria_f10_tsne1d as v5
    from scripts.execucao import v4_graficos_preditivos_tsne1d_f10 as v4
    from scripts.execucao import v3_avaliacao_f10_tsne1d as v3

    v6_info = v6.preflight()
    m6 = v6.conferir_saida(v6_info)
    v5_info = v6_info['st']
    v4_info = v5_info['q']
    v3_info = v5_info['info']
    m5 = v5.verificar_saida(v5_info)
    m4 = v4.saida_valida(v4_info)
    m3 = v3.verify_existing(v3_info)

    exp = v3_info['info']['pasta']
    escolha = v5_info['escolha']
    met_path = v5_info['v3'] / 'metricas_f10.json'
    exigir(met_path.is_file(), 'Metricas V3 ausentes.')
    exigir(escolha['id_candidato'] == '757edec8b54d68', 'Candidato t-SNE mudou.')
    tau = float(escolha['limiar_mcc_oof'])
    exigir(tau == 0.011926665574414961, 'Tau MCC OOF foi alterado.')
    exigir(all(m.get('status') == 'APROVADA' for m in (m3, m4, m5, m6)),
           'Alguma etapa V3-V6 nao esta aprovada.')
    exigir(all(m.get('n_f10') == 18523 for m in (m3, m4, m5, m6)),
           'Quantidade F10 divergente entre as etapas.')
    exigir(all(m.get('id_candidato_oof') == escolha['id_candidato'] for m in (m3,m4,m5,m6)),
           'Identificador OOF divergente entre V3-V6.')
    exigir(all(float(m.get(campo)) == tau for m, campo in (
        (m3, 'limiar_tau_mcc_oof_congelado'),
        (m4, 'limiar_mcc_oof_congelado'),
        (m5, 'limiar_mcc_oof_congelado'),
        (m6, 'limiar_mcc_oof_congelado'))), 'Limiar divergente entre V3-V6.')
    exigir(m6['tempo_inferencia_oos_v3_segundos'] is None,
           'Tempo V3 foi indevidamente preenchido.')

    origins = {'v4': v4_info['out'], 'v5': v5_info['out'], 'v6': v6_info['out']}
    for origin, name, *_ in GRAPH_SPECS:
        p = origins[origin] / name
        exigir(p.is_file() and p.stat().st_size > 100 and png_valido(p),
               'PNG ausente/invalido: ' + str(p))
    pre01 = origins['v6'] / PRE01_REL
    exigir(pre01.is_file() and pre01.stat().st_size > 100000,
           'PRE01 F10 interativo V6 ausente/incompleto.')

    pasta_final = exp / 'vencedor' / 'teste_final'
    destino = pasta_final / DIR_NAME
    staging = pasta_final / ('.' + DIR_NAME + '_incompleta_v7')
    lock = ROOT / 'outputs' / LOCK_NAME
    pca = v3_info['pca'] / 'vencedor' / 'teste_final' / 'galeria' / 'index.html'
    oof = exp / 'vencedor' / 'visualizacao' / 'galeria' / 'index.html'
    exigir(v3_info['modelo_hash'] == m4['sha256_modelo_v2'] == m5['sha256_modelo_v2'] == m6['sha256_modelo_v2'],
           'Modelo SHA-256 mudou de V2 para V6.')
    return {
        'exp': exp, 'v6_info': v6_info, 'v5_info': v5_info, 'v4_info': v4_info,
        'v3_info': v3_info, 'met_path': met_path, 'origins': origins,
        'm3': m3, 'm4': m4, 'm5': m5, 'm6': m6, 'pre01': pre01,
        'tau': tau, 'model_hash': v3_info['modelo_hash'], 'candidate': escolha['id_candidato'],
        'dest': destino, 'staging': staging, 'lock': lock, 'pca_html': pca, 'oof_html': oof,
    }


def fontes_protegidas(info):
    # Arquivos realmente relevantes e checados pelos manifestos anteriores.
    st = info['v5_info']
    v3 = st['v3']
    src = {
        st['info']['modelo_dir'] / 'modelo_final_f1_f9.joblib',
        st['info']['modelo_dir'] / 'manifesto_modelo_final.json',
        *[v3 / n for n in ('manifesto_v3.json', 'predicoes_f10.csv',
             'metricas_f10.json', 'coordenadas_TSNE1_f10.csv', 'indices_f10_amostra.npy')],
        st['v4'] / 'manifesto_v4.json',
        st['out'] / 'manifesto_v5.json',
        info['v6_info']['out'] / 'manifesto_v6.json',
        info['pre01'],
    }
    for origin, name, *_ in GRAPH_SPECS:
        src.add(info['origins'][origin] / name)
    for folder, manifest in ((st['v4'], info['m4']), (st['out'], info['m5']),
                             (info['v6_info']['out'], info['m6'])):
        for name in manifest['sha256_arquivos']:
            src.add(folder / name)
    for path in src:
        exigir(path.is_file(), 'Fonte V2-V6 desapareceu: ' + str(path))
    return {posix_rel(p, ROOT): sha_file(p) for p in sorted(src)}


def auditar_conteudo(info):
    """Ler apenas CSVs/JSONs que V3 ja congelou; nunca refazer scores."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import matthews_corrcoef
    from src.avaliacao_final_tsne import avaliar_scores_com_limiar_oof_congelado

    v3 = info['v5_info']['v3']
    pred = pd.read_csv(v3 / 'predicoes_f10.csv')
    coord = pd.read_csv(v3 / 'coordenadas_TSNE1_f10.csv')
    met = ler_json(v3 / 'metricas_f10.json')
    ids = info['v3_info']['ids']
    y = pred['y_true'].to_numpy(dtype=np.uint8)
    p = pred['prob_fraude'].to_numpy(dtype=np.float64)
    z = coord['TSNE1'].to_numpy(dtype=np.float64)
    tau = info['tau']
    exigir(len(y) == len(z) == len(ids) == 18523 and
           np.array_equal(pred['linha_original'].to_numpy(dtype=np.int64), ids) and
           np.array_equal(coord['linha_original'].to_numpy(dtype=np.int64), ids) and
           np.isfinite(p).all() and np.isfinite(z).all() and
           ((0 <= p) & (p <= 1)).all() and
           (pred['fold'] == 10).all(), 'V3 predicoes/embedding e indices nao alinhados.')
    exigir(np.array_equal((p >= tau).astype(np.uint8), pred['predicao_tau_mcc_oof'].to_numpy(dtype=np.uint8)) and
           np.array_equal((p >= .5).astype(np.uint8), pred['predicao_tau_0_50'].to_numpy(dtype=np.uint8)),
           'Classes da V3 nao refletem os limiares declarados.')
    esperado = avaliar_scores_com_limiar_oof_congelado(y, p, limiar_mcc_oof=tau)
    for nome in ('ap','mcc_limiar_oof','logloss','score_composto_limiar_oof'):
        exigir(np.isclose(float(met[nome]), esperado[nome], atol=1e-11, rtol=1e-11),
               'Metricas F10 divergentes no arquivo V3: '+nome)
    exigir(met.get('limiar_otimizado_no_f10') is False and
           met.get('f10_reutilizado') is True, 'F10 deixou de ser exploratorio/congelado.')
    cm = esperado['comparacao_limiares']['tau_mcc_oof']['matriz_confusao']
    m4 = info['m4']; m5 = info['m5']; m6 = info['m6']
    exigir(all(cm[k] == m4['matriz_confusao_tau_oof'][k] for k in ('tn','fp','fn','tp')),
           'Matriz de confusao V4 nao corresponde aos scores V3.')
    positivos = int(np.count_nonzero(p >= tau))
    positivos05 = int(np.count_nonzero(p >= .5))
    exigir(positivos == m4['n_preditos_positivos_tau_oof'] == m5['n_preditos_positivos_tau_oof']
           and positivos05 == m4['n_preditos_positivos_0_50'] == m5['n_preditos_positivos_050'],
           'V4 e V5 divergem em positivos previstos.')
    exigir(np.isclose(float(m5['mcc_f10_tau_oof']), esperado['mcc_limiar_oof'], atol=1e-12) and
           np.isclose(float(m5['mcc_f10_050']), esperado['comparacao_limiares']['tau_0_50']['mcc'], atol=1e-12),
           'MCC geométrico V5 nao corresponde aos dados F10.')
    exigir(m6['n_variaveis_pre01'] == 39 and m6['n_padronizadas_pre01'] == 12,
           'Quantidade de variaveis PRE01 V6 mudou.')
    exigir(np.isclose(float(m4['score_min']), float(p.min()), atol=1e-12) and
           np.isclose(float(m4['score_max']), float(p.max()), atol=1e-12),
           'Faixa de scores V4 nao corresponde V3.')
    exigir(y.sum() == m4['n_fraudes'] == m5['n_fraudes_f10'] == m6['n_fraudes_f10'],
           'Total de fraudes divergente entre V4-V6.')
    return {
        'n_f10': len(y), 'n_fraudes': int(y.sum()), 'n_legitimas': int(len(y)-y.sum()),
        'n_alertas_tau': positivos, 'n_alertas_05': positivos05,
        'tau': tau, 'ap': float(met['ap']), 'mcc':float(met['mcc_limiar_oof']),
        'mcc_05': float(esperado['comparacao_limiares']['tau_0_50']['mcc']),
        'logloss': float(met['logloss']), 'score_composto': float(met['score_composto_limiar_oof']),
        'prevalencia': float(met['prevalencia']), 'score_min':float(p.min()),
        'score_max': float(p.max()), 'cm': {k:int(cm[k]) for k in ('tn','fp','fn','tp')},
    }


def imprimir_plano(info):
    print('=== V7 - GALERIA FINAL F10 TSNE+GMM 1D ===', flush=True)
    print('EXPERIMENTO_TSNE=' + str(info['exp']), flush=True)
    print('CANDIDATO_OOF=' + info['candidate'], flush=True)
    print('TAU_OOF_CONGELADO=' + format(info['tau'], '.17g'), flush=True)
    print('MODELO_V2_SHA256=' + info['model_hash'], flush=True)
    print('F10_PAREADO_PCA_N=18523', flush=True)
    print('V4_GRAFICOS=6/6', flush=True)
    print('V5_GRAFICOS=4/4', flush=True)
    print('V6_GRAFICOS=2/2', flush=True)
    print('PRE01_INTERATIVO=SIM', flush=True)
    print('DESTINO_V7=' + str(info['dest']), flush=True)
    print('OOF_REEXECUTADO=NAO', flush=True)
    print('MODELO_REAJUSTADO=NAO', flush=True)
    print('PREVISOES_RECALCULADAS=NAO', flush=True)
    print('LIMIAR_F10_REOTIMIZADO=NAO', flush=True)


def render_html(info, result, csv_links):
    e = escape
    def num(x):return f'{float(x):.7f}'.replace('.', ',')
    tau = result['tau']
    cm = result['cm']
    alerts = result['n_alertas_tau']
    # Nomes e caminhos sao somente os previamente conferidos em preflight.
    cards = []
    for group, filename, title, desc, category in GRAPH_SPECS:
        path = 'figuras/' + filename
        cards.append(f'''<article class="graph" data-category="{category}">
  <button class="zoom" type="button" data-full="{e(path, quote=True)}" data-alt="{e(title, quote=True)}" title="Ampliar grafico {e(title, quote=True)}">
    <img src="{e(path, quote=True)}" loading="lazy" alt="{e(title, quote=True)}"></button>
  <div class="graph-copy"><div class="graph-head"><span class="tag">{e(group.upper())}</span><h3>{e(title)}</h3></div>
    <p>{e(desc)}</p><a href="{e(path, quote=True)}" target="_blank" rel="noopener">Abrir imagem em tamanho completo</a></div></article>''')
    links = ''.join(f'<li><a href="{e(url,quote=True)}" download>{e(label)}</a></li>' for label,url in csv_links)
    extras = []
    if info['pca_html'].is_file():
        extras.append(f'<a href="{e(posix_rel(info["pca_html"], info["dest"]),quote=True)}">Galeria F10 PCA 1D</a>')
    if info['oof_html'].is_file():
        extras.append(f'<a href="{e(posix_rel(info["oof_html"], info["dest"]),quote=True)}">Galeria OOF t-SNE 1D</a>')
    links_extra = ' · '.join(extras)
    rank_note = ('Nenhuma transação da amostra F10 alcançou τ* ou 0,50. O MCC zero decorre de todas as predições serem legítimas.'
                 if alerts == 0 and result['n_alertas_05'] == 0 else
                 'Os indicadores de classificação foram calculados com os limiares previamente fixados; o F10 não escolheu novo corte.')
    comparison = ('O maior score da amostra F10 é inferior a τ* OOF. Isto explica os zero alertas neste lote.'
                  if result['score_max'] < tau else
                  'Os scores cobrem a faixa descrita nos gráficos, sem mudança de limiar.')
    return f'''<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>F10 | t-SNE + GMM 1D — Galeria final</title>
<style>
:root{{--ink:#132944;--muted:#61738a;--bg:#f5f7fb;--line:#dce5ef;--navy:#101e35;--blue:#235c9f;--white:#fff;--teal:#067c70}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);font:15px/1.58 'Segoe UI',system-ui,sans-serif;color:var(--ink)}}
a{{color:#195b9d;text-decoration:none}}a:hover{{text-decoration:underline}}button{{font:inherit;cursor:pointer}}
header{{background:linear-gradient(115deg,#0b1b32,#18375a 68%,#1a5965);color:white;padding:45px max(22px,calc((100vw - 1220px)/2)) 36px}}
header .eyebrow{{text-transform:uppercase;letter-spacing:.19em;color:#97c6e9;font-size:12px;font-weight:800}}h1{{font-size:clamp(29px,4.6vw,47px);letter-spacing:-.035em;line-height:1.12;margin:10px 0 13px}}header p{{max-width:870px;color:#d1e1f4;margin:0}}.meta{{display:flex;flex-wrap:wrap;gap:9px;margin-top:24px}}
.meta span{{border:1px solid #57718d;background:#ffffff11;border-radius:7px;padding:5px 10px;font-size:12px}}
nav{{background:#fff;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10}}nav .inner{{max-width:1220px;margin:auto;display:flex;flex-wrap:wrap;gap:7px;padding:11px 22px}}nav a{{font-size:13px;font-weight:700;padding:7px 12px;border-radius:7px}}nav a:hover{{background:#eaf2fb}}
main{{max-width:1220px;margin:0 auto;padding:28px 22px 65px}}section{{scroll-margin-top:95px;margin-top:32px}}h2{{font-size:25px;letter-spacing:-.025em;margin:0 0 8px}}.lead{{color:var(--muted);margin:0 0 18px}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px}}
.metric{{background:#fff;border:1px solid var(--line);padding:18px;border-radius:12px;box-shadow:0 3px 16px #193c6507}}.metric .label{{font-size:12px;color:var(--muted);font-weight:800;text-transform:uppercase;letter-spacing:.08em}}.metric .value{{font-size:clamp(23px,3vw,31px);font-weight:800;letter-spacing:-.035em;margin-top:7px;font-variant-numeric:tabular-nums}}
.metric small{{color:var(--muted)}}.note{{border-left:4px solid #407da9;background:#edf4fa;border-radius:5px;padding:14px 18px;margin-top:18px}}.note strong{{display:block;margin-bottom:5px}}.note p{{margin:0}}.audit{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.panel{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px}}.panel h3{{margin:0 0 10px}}.panel p{{margin:9px 0}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:9px 12px;text-align:right;border-bottom:1px solid #e2e9f1;font-variant-numeric:tabular-nums}}th:first-child,td:first-child{{text-align:left}}th{{font-size:12px;color:#506680;background:#f0f4f9}}tr:last-child td{{border-bottom:0}}
.filters{{display:flex;flex-wrap:wrap;gap:7px;margin:17px 0 18px}}.filters button{{border:1px solid #d6e2ee;border-radius:9px;padding:8px 15px;background:#fff;color:var(--ink);font-weight:650}}.filters button.active{{background:#183b68;color:#fff;border-color:#183b68}}
.graphs{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:17px}}.graph{{background:#fff;border:1px solid var(--line);border-radius:13px;overflow:hidden;box-shadow:0 3px 16px #1b426008}}.zoom{{display:block;width:100%;height:315px;overflow:hidden;border:0;border-bottom:1px solid var(--line);padding:14px;background:#f8fafc}}.zoom img{{width:100%;height:100%;object-fit:contain;display:block;transition:transform .2s ease}}.zoom:hover img{{transform:scale(1.035)}}.graph-copy{{padding:15px 18px 19px}}.graph-head{{display:flex;gap:9px;align-items:center;flex-wrap:wrap}}h3{{font-size:16px;letter-spacing:-.015em;margin:0}}.tag{{font-size:10px;letter-spacing:.08em;font-weight:800;color:#42698b;background:#e9f3fb;border-radius:5px;padding:3px 6px}}.graph p{{font-size:13px;color:var(--muted);margin:9px 0 12px}}.graph a{{font-size:12px;font-weight:750}}.source-list{{columns:2;column-gap:26px;padding-left:22px}}.source-list li{{margin:7px 0;break-inside:avoid;font-size:13px}}
.open-pre01{{display:inline-flex;align-items:center;gap:8px;padding:12px 17px;border-radius:8px;background:#175c89;color:#fff;font-weight:750;margin:9px 0 8px}}.open-pre01:hover{{background:#123f62;text-decoration:none}}footer{{margin:30px 0 0;color:#61738a;font-size:12px;border-top:1px solid #dce5ef;padding-top:17px}}
#lightbox{{position:fixed;inset:0;background:#07111de8;z-index:30;display:none;align-items:center;justify-content:center;flex-direction:column;padding:30px 16px}}#lightbox.open{{display:flex}}#lightbox img{{max-width:min(95vw,1450px);max-height:84vh;object-fit:contain;background:#fff;border-radius:5px}}#lightbox button{{position:absolute;right:25px;top:15px;background:#ffffff24;color:#fff;border:1px solid #ffffff70;border-radius:7px;padding:8px 13px}}
@media(max-width:900px){{.cards{{grid-template-columns:repeat(2,minmax(0,1fr))}}.graphs{{grid-template-columns:1fr}}.audit{{grid-template-columns:1fr}}.source-list{{columns:1}}}}@media(max-width:520px){{.cards{{grid-template-columns:1fr 1fr;gap:8px}}.metric{{padding:12px}}.metric .value{{font-size:20px}}.zoom{{height:255px}}header{{padding-top:34px}}}}
</style></head><body>
<header><div class="eyebrow">TCC experimental · F10 · t-SNE + GMM</div><h1>Galeria final<br>t-SNE + GMM 1D</h1><p>Resultados de inferência fora da amostra sobre as mesmas 18.523 transações usadas na avaliação PCA 1D. Modelo e limiar congelados no desenvolvimento F1–F9.</p>
<div class="meta"><span>F10 exploratório reutilizado</span><span>18.523 observações</span><span>Candidato {e(info['candidate'])}</span><span>τ* OOF = {num(tau)}</span><span>12 gráficos + PRE01</span></div></header>
<nav aria-label="Navegação"><div class="inner"><a href="#resumo">Resultados</a><a href="#classificacao">Classificação</a><a href="#catalogo">Gráficos</a><a href="#pre01">PRE01</a><a href="#fontes">Dados e auditoria</a></div></nav>
<main><section id="resumo"><h2>Indicadores no F10</h2><p class="lead">Métricas da V3, apresentadas com sete casas decimais. O MCC usa exclusivamente o limiar τ* definido no OOF.</p>
<div class="cards"><div class="metric"><div class="label">Average Precision</div><div class="value">{num(result['ap'])}</div><small>Ordenação das fraudes</small></div>
<div class="metric"><div class="label">MCC em τ* OOF</div><div class="value">{num(result['mcc'])}</div><small>Classificação no F10</small></div>
<div class="metric"><div class="label">LogLoss</div><div class="value">{num(result['logloss'])}</div><small>Probabilidades previstas</small></div>
<div class="metric"><div class="label">Score composto</div><div class="value">{num(result['score_composto'])}</div><small>AP, MCC transformado e exp(−LogLoss)</small></div></div>
<div class="note"><strong>Avaliação exploratória reutilizada, não holdout inédito</strong><p>O F10 já foi acessado previamente no projeto. Não houve novo ajuste de modelo, escolha de hiperparâmetros ou otimização de limiar com F10 nas etapas V3–V7.</p></div></section>
<section id="classificacao"><h2>Classificação e distribuição dos scores</h2><p class="lead">{e(rank_note)}</p><div class="audit">
<div class="panel"><h3>Matriz de confusão — τ* OOF</h3><table><thead><tr><th>Classe real</th><th>Predita legítima</th><th>Predita fraude</th></tr></thead><tbody><tr><td>Legítima</td><td>{cm['tn']:,}</td><td>{cm['fp']:,}</td></tr><tr><td>Fraude</td><td>{cm['fn']:,}</td><td>{cm['tp']:,}</td></tr></tbody></table><p><b>Alertas em τ*:</b> {result['n_alertas_tau']:,} &nbsp; | &nbsp; <b>Alertas em 0,50:</b> {result['n_alertas_05']:,}</p></div>
<div class="panel"><h3>Pontos de operação congelados</h3><p><b>Limiar fixo:</b> 0,5000000 · MCC F10 = {num(result['mcc_05'])}</p><p><b>Limiar τ* OOF:</b> {num(tau)} · MCC F10 = {num(result['mcc'])}</p><p><b>Scores F10:</b> mínimo {num(result['score_min'])}, máximo {num(result['score_max'])}.</p><p>{e(comparison)}</p></div></div></section>
<section id="catalogo"><h2>Catálogo de 12 gráficos</h2><p class="lead">Imagens copiadas byte a byte das etapas aprovadas V4–V6. Clique para ampliar. O GMM foi ajustado exclusivamente com F1–F9.</p>
<div class="filters" role="group" aria-label="Filtrar gráficos"><button class="active" type="button" data-filter="todos" aria-pressed="true">Todos (12)</button><button type="button" data-filter="desempenho" aria-pressed="false">Desempenho (6)</button><button type="button" data-filter="geometria" aria-pressed="false">Geometria (4)</button><button type="button" data-filter="diagnosticos" aria-pressed="false">Diagnósticos (2)</button></div>
<div class="graphs">{''.join(cards)}</div></section>
<section id="pre01"><h2>PRE01 — Diagnóstico interativo</h2><div class="panel"><p>Análise das 39 variáveis transformadas no F10, das quais 12 foram padronizadas com parâmetros aprendidos somente em F1–F9. A página tem gráficos interativos e busca por variável, e funciona offline.</p>
<a class="open-pre01" href="{e(PRE01_COPY,quote=True)}" target="_blank" rel="noopener">Abrir PRE01 do F10</a><p class="lead">Não foram recalculadas previsões nem realizada padronização por <i>fit</i> em F10.</p></div></section>
<section id="fontes"><h2>Dados, proveniência e auditoria</h2><p class="lead">CSVs originais continuam nas pastas das respectivas etapas. Os hashes SHA-256 e as verificações estão em <a href="manifesto_v7.json">manifesto_v7.json</a>.</p>
<div class="panel"><h3>Arquivos CSV disponíveis</h3><ul class="source-list">{links}</ul><p class="lead">O tempo real da inferência OOS da V3 não foi registrado. O KL exibido em TSNE14 é do ajuste F1–F9.</p>{links_extra}</div></section>
<footer>t-SNE+GMM FULL K=4 · F1–F9: treino e seleção OOF · F10: amostra estratificada exploratória pareada com PCA · limiar OOF inalterado · V7 sem novos ajustes.</footer></main>
<div id="lightbox" role="dialog" aria-modal="true" aria-label="Ampliar gráfico"><button id="closeLightbox" type="button">Fechar ×</button><img alt=""></div>
<script>
const filters=document.querySelectorAll('[data-filter]');
filters.forEach(b=>b.addEventListener('click',()=>{{const selected=b.dataset.filter;filters.forEach(x=>{{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));}});document.querySelectorAll('.graph').forEach(card=>{{card.hidden=(selected!=='todos'&&card.dataset.category!==selected);}});}}));
const box=document.getElementById('lightbox'),photo=box.querySelector('img');
document.querySelectorAll('.zoom').forEach(b=>b.addEventListener('click',()=>{{photo.src=b.dataset.full;photo.alt=b.dataset.alt;box.classList.add('open');document.getElementById('closeLightbox').focus();}}));
function closeBox(){{box.classList.remove('open');photo.removeAttribute('src');}}
document.getElementById('closeLightbox').addEventListener('click',closeBox);
box.addEventListener('click',e=>{{if(e.target===box)closeBox();}});
document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeBox();}});
</script></body></html>'''


def copy_checked(source, destination):
    exigir(source.is_file() and not destination.exists(), 'Fonte/destino invalido: '+str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    exigir(sha_file(source) == sha_file(destination), 'Copia nao preservou SHA-256: '+str(destination))


def csv_catalog(info):
    links = []
    for group in ('v4','v5','v6'):
        folder = info['origins'][group]
        for p in sorted(folder.glob('*.csv')):
            links.append((group.upper() + ' / ' + p.name, posix_rel(p, info['dest'])))
    for p in sorted((info['origins']['v6'] / 'PRE01_F10').glob('*.csv')):
        links.append(('PRE01 / ' + p.name, posix_rel(p, info['dest'])))
    return links


def verificar_saida(info):
    out = info['dest']; manifest = out / 'manifesto_v7.json'; html_path = out / 'index.html'
    exigir(out.is_dir() and manifest.is_file() and html_path.is_file(),
           'Galeria V7 ausente ou parcial. Preservar e informar.')
    meta = ler_json(manifest)
    exigir(meta.get('status') == 'APROVADA' and meta.get('versao') == 'V7_GALERIA_F10_TSNE1D_V1',
           'Manifesto V7 nao aprovado.')
    exigir(meta.get('id_candidato_oof') == info['candidate'] and
           meta.get('limiar_mcc_oof_congelado') == info['tau'] and
           meta.get('modelo_v2_sha256') == info['model_hash'] and
           meta.get('n_f10') == 18523 and
           meta.get('avaliacao_f10_exploratoria_reutilizada') is True and
           meta.get('novo_treinamento') is False and
           meta.get('novas_predicoes') is False and
           meta.get('limiar_reotimizado_f10') is False,
           'Manifesto V7 divergiu do experimento congelado.')
    imgs = {'figuras/' + x[1] for x in GRAPH_SPECS}
    copied = imgs | {PRE01_COPY, 'index.html'}
    hashes = meta.get('sha256_galeria', {})
    exigir(isinstance(hashes, dict) and set(hashes) == copied,
           'Arquivos V7 divergentes dos 12 PNGs + PRE01 + index.')
    for name, expected in hashes.items():
        exigir((out/name).is_file() and sha_file(out/name) == expected,
               'Conteudo V7 alterado: ' + name)
    exigir({x.name for x in (out/'figuras').glob('*.png')} == {x[1] for x in GRAPH_SPECS},
           'Conjunto de PNGs V7 deveria ter exatamente 12 imagens.')
    exigir(meta.get('sha256_fontes') == fontes_protegidas(info),
           'Algum arquivo V2-V6 foi alterado desde a geracao V7.')
    for origin, name, *_ in GRAPH_SPECS:
        exigir(sha_file(info['origins'][origin] / name) == hashes['figuras/' + name],
               'Copia V7 PNG nao corresponde ao original: ' + name)
    exigir(sha_file(info['pre01']) == hashes[PRE01_COPY],
           'Copia V7 PRE01 nao corresponde ao original.')
    return meta


def executar(info):
    imprimir_plano(info)
    if info['dest'].exists():
        meta = verificar_saida(info)
        print('V7_STATUS=JA_APROVADA_SAIDA_VALIDADA', flush=True)
        print('V7_GRAFICOS_VERIFICADOS=12/12', flush=True)
        print('V7_HTML=' + str(info['dest'] / 'index.html'), flush=True)
        return 0
    exigir(not info['staging'].exists(), 'Pasta V7 staging parcial: preservar e investigar.')
    exigir(not info['lock'].exists(), 'Lock V7 existe: verificar processo antes de modificar.')
    with info['lock'].open('x', encoding='utf-8') as f:
        f.write('V7 galeria F10 t-SNE+GMM 1D; sem ajuste e sem novas previsoes.\n')
    try:
        before = fontes_protegidas(info)
        data = auditar_conteudo(info)
        stage = info['staging']
        stage.mkdir(parents=False, exist_ok=False)
        print('[V7] Copiando 12 PNGs ja aprovados (sem renderizar novamente).', flush=True)
        for group, name, *_ in GRAPH_SPECS:
            copy_checked(info['origins'][group] / name, stage / 'figuras' / name)
            print('[V7 IMAGEM] ' + name, flush=True)
        print('[V7] Copiando PRE01 interativo original da V6.', flush=True)
        copy_checked(info['pre01'], stage / PRE01_COPY)
        html = render_html(info, data, csv_catalog(info))
        exigir('F10 exploratória' not in html or 'τ*' in html, 'Rotulos da galeria invalidos.')
        (stage / 'index.html').write_text(html, encoding='utf-8')
        exigir(len(html) > 10000 and html.count('<article class="graph"') == 12,
               'HTML incompleto ou numero de cards != 12.')
        exigir(before == fontes_protegidas(info), 'Fontes V2-V6 mudaram durante a criacao V7.')
        paths = ['figuras/' + x[1] for x in GRAPH_SPECS] + [PRE01_COPY, 'index.html']
        hashes_copias = {name: sha_file(stage/name) for name in paths}
        manifest = {
            'versao':'V7_GALERIA_F10_TSNE1D_V1', 'status':'APROVADA',
            'data_utc':datetime.now(timezone.utc).isoformat(),
            'metodo':'TSNE+GMM FULL K=4, 1D',
            'experimento_oof':posix_rel(info['exp'], ROOT),
            'id_candidato_oof':info['candidate'],
            'limiar_mcc_oof_congelado':info['tau'],
            'modelo_v2_sha256':info['model_hash'],
            'n_f10':data['n_f10'], 'n_fraudes':data['n_fraudes'],
            'n_legitimas':data['n_legitimas'],
            'n_previstas_fraude_tau_oof':data['n_alertas_tau'],
            'matriz_confusao_tau_oof':data['cm'],
            'metricas_f10':{n:data[n] for n in ('ap','mcc','logloss','score_composto','mcc_05')},
            'graficos_v4':6, 'graficos_v5':4, 'graficos_v6':2,
            'total_graficos':12, 'pre01_interativo':True,
            'avaliacao_f10_exploratoria_reutilizada':True,
            'tempo_inferencia_oos_v3_segundos':None,
            'novo_treinamento':False, 'novas_predicoes':False,
            'limiar_reotimizado_f10':False, 'oof_reexecutado':False,
            'sha256_fontes':before, 'sha256_galeria':hashes_copias,
        }
        (stage / 'manifesto_v7.json').write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+'\n',encoding='utf-8')
        exigir(not info['dest'].exists(), 'Galeria destino apareceu durante a execucao.')
        os.replace(stage, info['dest'])
        verificar_saida(info)
        print('V7_GRAFICOS_VERIFICADOS=12/12', flush=True)
        print('V7_PRE01_INTERATIVO=APROVADO', flush=True)
        print('V7_F10_N=' + str(data['n_f10']), flush=True)
        print('V7_ALERTAS_TAU_OOF=' + str(data['n_alertas_tau']), flush=True)
        print('V7_MCC_TAU_OOF=' + format(data['mcc'], '.7f'), flush=True)
        print('V7_STATUS=APROVADA', flush=True)
        print('V7_HTML=' + str(info['dest'] / 'index.html'), flush=True)
        print('OOF_REEXECUTADO=NAO', flush=True)
        print('PREDICOES_V3_MODIFICADAS=NAO', flush=True)
        return 0
    finally:
        info['lock'].unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument('--planejar', action='store_true')
    choices.add_argument('--executar', action='store_true')
    choices.add_argument('--validar', action='store_true')
    args = parser.parse_args(argv)
    info = preflight()
    if args.planejar:
        imprimir_plano(info)
        print('V7_STATUS=PLANEJAMENTO_APROVADO_SEM_LEITURA_DOS_SCORES_F10', flush=True)
        return 0
    if args.validar:
        verificar_saida(info)
        print('V7_STATUS=APROVADA_VALIDADA', flush=True)
        print('V7_GRAFICOS_VERIFICADOS=12/12', flush=True)
        return 0
    return executar(info)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print('V7_STATUS=BLOQUEADA', file=sys.stderr, flush=True)
        print('V7_MOTIVO=' + str(exc), file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)
