# -*- coding: utf-8 -*-
"""Fase 3A: seis graficos preditivos do F10 t-SNE 1D JA avaliado.

Entrada: predicoes e metricas congeladas da Fase 2. Nenhum ajuste, acesso ao
Parquet, previsao, transformacao t-SNE ou otimizacao de limiar ocorre aqui.
Entrega deliberadamente parcial: a galeria completa GMM/TSNE/PRE01 requer
coordenadas F10 associadas ao mesmo passe OOS da inferencia (Fase 3B).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from src import visualizacao_oof as vo
from src.avaliacao_final_tsne import avaliar_scores_com_limiar_oof_congelado
from src.folds import hash_arquivo
from src.amostragem_f10 import fracao_amostra_f10

PNG_FILES = (
    '00_painel_score_composto.png',
    '01_curva_precision_recall.png',
    '02_mcc_por_limiar.png',
    '03_matriz_confusao_05.png',
    '04_matriz_confusao_mcc.png',
    '05_distribuicao_probabilidades.png',
)


def exigir(c, mensagem):
    if not c:
        raise RuntimeError(mensagem)


def ler_json(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        return json.load(f)


@contextmanager
def _rotulos_f10():
    """Adapta SOMENTE rótulos das figuras geradas; restaura funções originais."""
    from matplotlib.text import Text
    originals = (vo._salvar, vo._csv_protegido, vo._pontos_operacionais_mcc)
    pares = (
        ('Curva Precision–Recall do modelo selecionado em OOF',
         'Curva Precision–Recall do modelo avaliado em F10'),
        ('Resultados da validação OOF', 'Resultados do F10 exploratório'),
        ('AP calculada sobre as predições OOF.', 'AP calculada sobre as predições F10.'),
        ('MCC calculado no limiar ótimo das predições OOF.',
         'MCC F10 no limiar τ* congelado em OOF.'),
        ('LogLoss calculado sobre as predições OOF.',
         'LogLoss calculado sobre as predições F10.'),
        ('OOF F1–F9', 'F10 exploratório'),
        ('Limiar ótimo do MCC (τ*)', 'Limiar τ* congelado em OOF'),
        ('Limiar ótimo (τ*)', 'Limiar τ* congelado em OOF'),
        ('MCC no limiar ótimo', 'MCC F10 em τ* OOF'),
        ('LIMIAR ÓTIMO DO MCC:', 'LIMIAR τ* CONGELADO EM OOF:'),
    )

    def salvar(fig, dest, *, dpi):
        for t in fig.findobj(match=Text):
            atual = t.get_text()
            novo = atual
            for old, new in pares:
                novo = novo.replace(old, new)
            if novo != atual:
                t.set_text(novo)
        return originals[0](fig, dest, dpi=dpi)

    def salvar_csv(tab, dest):
        tab = tab.copy().rename(columns={
            'n_oof': 'n_f10', 'prevalencia_oof': 'prevalencia_f10'})
        if 'origem' in tab:
            tab['origem'] = 'F10_EXPLORATORIO_TAU_OOF_CONGELADO'
        if 'corte' in tab:
            tab['corte'] = tab['corte'].replace({'MCC OOF': 'Tau MCC OOF congelado'})
        return originals[1](tab, dest)

    def pontos_mcc(y, p, tau):
        ops = originals[2](y, p, tau)
        for item in ops:
            item['origem'] = 'F10_EXPLORATORIO_TAU_OOF_CONGELADO'
            if item['corte'] == 'MCC OOF':
                item['corte'] = 'Tau MCC OOF congelado'
        return ops

    vo._salvar, vo._csv_protegido, vo._pontos_operacionais_mcc = salvar, salvar_csv, pontos_mcc
    try:
        yield
    finally:
        vo._salvar, vo._csv_protegido, vo._pontos_operacionais_mcc = originals


def _entradas(pasta):
    import yaml
    pasta = Path(pasta).resolve()
    raiz = pasta / 'vencedor' / 'teste_final'
    m = ler_json(raiz / 'manifesto_finalizacao_tsne.json')
    exigir(m.get('etapa') == 'F10_AVALIADO_GALERIA_PENDENTE' and
           m.get('metodo') == 'TSNE' and m.get('dimensao') == 1,
           'Exige avaliacao t-SNE 1D Fase 2 concluida; galeria completa ainda pendente.')
    config = yaml.safe_load((pasta/'configuracao.yaml').read_text(encoding='utf-8-sig'))
    exigir(str(config['reducao']['metodo']).upper() == 'TSNE' and
           config['tsne']['n_components'] in (1, [1]), 'Configuracao nao corresponde a TSNE 1D.')
    origem = {
        'predicoes_f10.csv': 'sha256_predicoes',
        'metricas_f10.json': 'sha256_metricas',
        'indices_f10_amostra.npy': 'sha256_indices_arquivo',
        'modelo_final_f1_f9/modelo_final_f1_f9.joblib': 'sha256_modelo',
    }
    for nome, chave in origem.items():
        p = raiz / nome
        exigir(p.is_file() and hash_arquivo(p) == m.get(chave),
               'Fonte F10 ou modelo ausente/alterado: '+nome)
    scores = pd.read_csv(raiz/'predicoes_f10.csv')
    met = ler_json(raiz/'metricas_f10.json')
    ids = np.load(raiz/'indices_f10_amostra.npy', allow_pickle=False)
    required = {'linha_original', 'fold', 'y_true', 'prob_fraude',
                'predicao_tau_0_50', 'predicao_tau_mcc_oof'}
    exigir(required.issubset(scores.columns) and len(scores)==len(ids)>0,
           'Tabela de predicoes F10 incompleta ou n diferente.')
    obs = scores['linha_original'].to_numpy(dtype=np.int64)
    y = scores['y_true'].to_numpy(dtype=np.uint8)
    p = scores['prob_fraude'].to_numpy(dtype=np.float64)
    tau = float(met['limiar_mcc_oof_congelado'])
    exigir(np.array_equal(obs, ids) and np.all(scores['fold']==10) and
           np.isin(y, [0,1]).all() and set(np.unique(y).tolist())=={0,1} and
           np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and
           0 <= tau <= 1 and
           np.array_equal(scores['predicao_tau_0_50'].to_numpy(dtype=np.uint8), (p>=.5).astype(np.uint8)) and
           np.array_equal(scores['predicao_tau_mcc_oof'].to_numpy(dtype=np.uint8), (p>=tau).astype(np.uint8)),
           'Origem, fold, scores ou classes F10 inconsistentes.')
    recalculado = avaliar_scores_com_limiar_oof_congelado(y,p,limiar_mcc_oof=tau)
    exigir(met.get('limiar_otimizado_no_f10') is False and
           m.get('id_candidato_oof') == met.get('id_candidato_oof') and
           np.isclose(float(m['fracao_f10_dentro_do_fold']), fracao_amostra_f10(config)) and
           np.isclose(float(met['fracao_f10_dentro_do_fold']), fracao_amostra_f10(config)),
           'Limiar ou origem F10 divergentes do manifesto.')
    for campo in ('ap','mcc_limiar_oof','logloss','score_composto_limiar_oof'):
        exigir(np.isclose(float(met[campo]), float(recalculado[campo]), rtol=1e-9, atol=1e-9),
               'Metrica congelada diverge dos scores: '+campo)
    return raiz, config, met, m, y, p, tau


def _verificar_existente(dest, m, fonte):
    man = ler_json(dest / 'manifesto_visualizacao_preditiva.json')
    exigir(man.get('estado')=='PREDITIVOS_6_DE_12_GALERIA_COMPLETA_PENDENTE'
           and man.get('sha256_predicoes_f10')==m['sha256_predicoes']
           and man.get('sha256_metricas_f10')==m['sha256_metricas'],
           'Previa F10 anterior incompatível; nao sobrescrever.')
    for nome, digest in man['sha256_arquivos'].items():
        arq = dest/nome
        exigir(arq.is_file() and hash_arquivo(arq)==digest,
               'Arquivo de previa ausente/alterado: '+nome)
    exigir(set(PNG_FILES).issubset(man['sha256_arquivos']), 'Faltam PNGs previstos.')
    return dest


def gerar(pasta):
    """Renderiza seis figuras de F10 já avaliado; arquivo novo e imutável."""
    raiz, cfg, met, marker, y, p, tau = _entradas(pasta)
    dest = raiz/'visualizacao_preditiva'
    if dest.exists():
        return _verificar_existente(dest, marker, raiz)
    stage = raiz/'.visualizacao_preditiva_incompleta'
    exigir(not stage.exists(), 'Staging de visualizacao F10 encontrado: preservar para auditoria.')
    with (raiz/'.visualizacao_preditiva.lock').open('x',encoding='utf-8'):
        pass
    lock = raiz/'.visualizacao_preditiva.lock'
    try:
        stage.mkdir(exist_ok=False)
        vencedor = pd.Series({
            'ap':float(met['ap']), 'mcc':float(met['mcc_limiar_oof']),
            'logloss':float(met['logloss']),
            'score':float(met['score_composto_limiar_oof']),
            'prevalencia':float(met['prevalencia']),
        })
        escolha = {'limiar_mcc_oof':tau}
        dpi = int(cfg['visualizacao']['padrao'].get('dpi',180))
        with _rotulos_f10():
            vo._plot_painel_00(cfg,y,p,vencedor,escolha,stage,'png',dpi,True)
            vo._plot_pr(cfg,y,p,vencedor,escolha,stage,'png',dpi,True)
            vo._plot_mcc(cfg,y,p,vencedor,escolha,stage,'png',dpi,True)
            vo._matriz(cfg,y,p,.5,'matriz_confusao_05',stage,'png',dpi,True)
            vo._matriz(cfg,y,p,tau,'matriz_confusao_mcc',stage,'png',dpi,True)
            vo._distribuicao(cfg,y,p,escolha,stage,'png',dpi,True)
        for nome in PNG_FILES:
            f=stage/nome
            exigir(f.is_file() and f.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n',
                   'Grafico F10 invalido ou ausente: '+nome)
        # HTML explicitamente parcial. Nunca registrar como galeria final completa.
        cards = ''.join('<a class="card" href="'+escape(nome)+'"><img src="'+escape(nome)+'" alt="'+escape(nome)+'"><span>'+escape(nome)+'</span></a>'
                        for nome in PNG_FILES)
        html = ('<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>F10 t-SNE 1D - prévia preditiva</title>'
                '<style>body{font:15px system-ui;background:#f4f7fb;color:#1b2b45;max-width:1400px;margin:auto;padding:22px}'
                '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:16px}'
                '.card{display:block;background:white;padding:12px;border:1px solid #ccd6e1;border-radius:10px;color:#134e87;text-decoration:none}'
                'img{width:100%;height:auto}span{display:block;padding:6px}</style></head><body>'
                '<h1>t-SNE 1D + GMM — F10 exploratório</h1>'
                '<p><strong>PRÉVIA PARCIAL: 6 gráficos preditivos.</strong> Gráficos geométricos, TSNE13/14 e PRE01 ainda não integrados. '
                'Não constitui a galeria completa da simulação do professor.</p>'
                f'<p>F10: n={len(y)}; desenvolvimento={float(marker["fracao_desenvolvimento"]):.2%}; '
                f'F10={float(marker["fracao_f10_dentro_do_fold"]):.2%} do fold; '
                f'τ* OOF congelado={tau:.7f}; AP={float(met["ap"]):.7f}; MCC={float(met["mcc_limiar_oof"]):.7f}.</p>'
                '<div class="grid">'+cards+'</div></body></html>')
        (stage/'index.html').write_text(html,encoding='utf-8')
        files={f.name:hash_arquivo(f) for f in stage.iterdir() if f.is_file()}
        info={
            'estado':'PREDITIVOS_6_DE_12_GALERIA_COMPLETA_PENDENTE',
            'n_f10':int(len(y)),'mcc_f10_tau_oof':float(matthews_corrcoef(y,p>=tau)),
            'limiar_mcc_oof_congelado':tau,
            'sha256_predicoes_f10':marker['sha256_predicoes'],
            'sha256_metricas_f10':marker['sha256_metricas'],
            'sha256_arquivos':files,
            'sem_treinamento':True,'sem_novas_predicoes':True,
            'sem_otimizacao_limiar_f10':True,
            'galeria_f10_completa':False,
        }
        (stage/'manifesto_visualizacao_preditiva.json').write_text(
            json.dumps(info,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        exigir(not dest.exists(),'Destino apareceu durante geracao; nao sobrescrever.')
        os.replace(stage,dest)
        return _verificar_existente(dest,marker,raiz)
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--experimento',type=Path,required=True)
    x=a.parse_args(argv)
    p=gerar(x.experimento)
    print('[TSNE1D F10] Previa preditiva gerada/verificada: '+str(p/'index.html'))
    print('[TSNE1D F10] Galeria completa ainda PENDENTE (geometria, TSNE e PRE01).')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
