# -*- coding: utf-8 -*-
"""Inclui HTML TSNE13 3D na galeria OOF existente a partir do CSV F1-F9.

Nao recalcula t-SNE, GMM, probabilidades ou tau. Consome o scatter e CSV
ja gerados pelo passo 07.02 a partir do desenvolvimento. Nunca consulta F10.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import os
import shutil

import numpy as np
import pandas as pd
import yaml

from src.folds import hash_arquivo
from src.ponte_oof_final_tsne import verificar_oof_concluido_tsne
from src.visualizacao_tsne3d_interativa import gerar_tsne13_3d, HTML

MARCA = '<!-- TSNE13_3D_HTML_INTERATIVO_OOF_V74 -->'
LINK = 'interativos/' + HTML


def integrar_tsne13_oof(pasta_experimento: str | Path, *, raiz: str | Path | None = None):
    base = Path(raiz).resolve() if raiz else Path(__file__).resolve().parents[1]
    pasta = Path(pasta_experimento).resolve()
    if not pasta.is_relative_to((base / 'outputs').resolve()):
        raise ValueError('Galeria OOF 3D so aceita outputs/.')
    cfg = yaml.safe_load((pasta / 'configuracao.yaml').read_text(encoding='utf-8-sig'))
    if cfg['reducao']['metodo'].upper() != 'TSNE' or cfg['tsne']['n_components'] not in (3,[3]):
        raise ValueError('Apenas t-SNE 3D pode usar esta integracao.')
    pre = verificar_oof_concluido_tsne(pasta)
    raiz_vis = pasta / 'vencedor' / 'visualizacao'
    origem = raiz_vis / 'geometria_desenvolvimento' / '13_tsne_mapa_TSNE1_TSNE2_TSNE3.csv'
    galeria = raiz_vis / 'galeria'
    indice = galeria / 'index.html'
    destino = galeria / LINK
    if not indice.is_file():
        raise FileNotFoundError('Galeria OOF nao esta concluida.')
    antigo = indice.read_text(encoding='utf-8')
    if MARCA in antigo or destino.exists():
        if MARCA not in antigo or not destino.is_file() or LINK not in antigo:
            raise RuntimeError('TSNE13 OOF parcialmente publicado: preservar para auditoria.')
        return indice
    if not origem.is_file():
        raise FileNotFoundError('CSV TSNE13 F1-F9 nao encontrado; habilite salvar_dados_grafico no YAML.')
    dados = pd.read_csv(origem)
    colunas = ['TSNE1', 'TSNE2', 'TSNE3']
    obrigatorias = {'classe_real', 'risco_estimado_treino', *colunas}
    if not obrigatorias.issubset(dados.columns):
        raise ValueError('CSV TSNE13 OOF nao contem coordenadas e scores esperados.')
    if len(dados) < 4 or dados[colunas].isna().any().any():
        raise ValueError('CSV TSNE13 OOF vazio ou com coordenadas invalidas.')
    coord = dados[colunas].copy()
    y = dados['classe_real'].to_numpy(dtype=np.uint8)
    scores = dados['risco_estimado_treino'].to_numpy(dtype=np.float64)
    if not np.isin(y,[0,1]).all() or len(np.unique(y)) != 2 or not np.isfinite(scores).all() or not np.all((scores>=0)&(scores<=1)):
        raise ValueError('CSV OOF com classes/scores invalidos.')
    tau = float(pre.escolha_oof['limiar_mcc_oof'])
    gamma = None
    if 'gaussiana_dominante' in dados:
        dom = dados['gaussiana_dominante'].to_numpy(dtype=int)
        if np.isin(dom, [1, 2, 3, 4]).all():
            gamma = np.eye(4)[dom-1]
    vis_tmp = galeria / '.tsne13_html_oof_v74_em_andamento'
    if vis_tmp.exists():
        raise RuntimeError('Pasta temporaria TSNE13 OOF ja existe. Preserve para auditoria.')
    vis_tmp.mkdir()
    try:
        gerar_tsne13_3d(coord, y, vis_tmp, scores=scores,
                        limiar_oof=tau, responsabilidades=gamma,
                        origem='desenvolvimento',
                        max_pontos=int(cfg['visualizacao']['geometria'].get('pontos_maximos_amostra', 20000)),
                        semente=int(cfg['experimento']['semente']),
                        dpi=int(cfg['visualizacao']['padrao'].get('dpi', 180)))
        novo = antigo.replace('</body>',
            MARCA + '<p><a href="' + LINK + '">TSNE13 — Projecao tridimensional interativa (modelo reajustado em F1-F9)</a></p></body>')
        if novo == antigo or MARCA not in novo:
            raise RuntimeError('Galeria OOF nao contem fechamento body para novo link.')
        (galeria / 'interativos').mkdir(exist_ok=True)
        if destino.exists():
            raise RuntimeError('HTML TSNE13 publicado por outro processo. Nao sobrescrever.')
        shutil.copyfile(vis_tmp / HTML, destino)
        if hash_arquivo(destino) != hash_arquivo(vis_tmp / HTML):
            raise RuntimeError('HTML TSNE13 copiado com hash divergente.')
        tmp_index = indice.with_name('index.html.tsne13_v74_tmp')
        if tmp_index.exists():
            raise RuntimeError('Indice temporario ja existe: nao sobrescrever.')
        with tmp_index.open('x',encoding='utf-8') as f:
            f.write(novo)
        os.replace(tmp_index, indice)
        print('[OOF TSNE13 3D] HTML offline integrado a galeria.', flush=True)
        return indice
    finally:
        shutil.rmtree(vis_tmp)
