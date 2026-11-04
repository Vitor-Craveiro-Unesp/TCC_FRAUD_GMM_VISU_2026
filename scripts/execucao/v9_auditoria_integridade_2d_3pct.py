# -*- coding: utf-8 -*-
"""V9: auditoria final PCA+GMM 2D e t-SNE+GMM 2D, 3% reais.

Comandos:
  python -B -m scripts.execucao.v9_auditoria_integridade_2d_3pct --auditar
  python -B -m scripts.execucao.v9_auditoria_integridade_2d_3pct --relatorio
  python -B -m scripts.execucao.v9_auditoria_integridade_2d_3pct --abrir

NENHUM comando executa OOF, ajuste de modelo, inferencia F10 ou renderizacao.
Auditoria sem comparador/ranqueamento automatico de metodos.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from html.parser import HTMLParser
import json
import logging
import os
from pathlib import Path
import sys
import traceback
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd

STATUS = 'INTEGRIDADE_PCA_TSNE_2D_3PCT_APROVADA'
VERSAO = 'v9-integridade-2d-3pct-20261103-v1'
DIRETORIO_RELATORIO = 'v9_auditoria_integridade_pca_tsne_2d_3pct'


def exigir(condicao, mensagem):
    if not condicao:
        raise RuntimeError(mensagem)


class _RecursosHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.imagens = []
    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'img':
            src = dict(attrs).get('src')
            if src:
                self.imagens.append(src)


def _conferir_html_local(pagina: Path):
    exigir(pagina.is_file() and pagina.stat().st_size > 250,
           'Pagina HTML inexistente ou vazia: ' + str(pagina))
    conteudo = pagina.read_text(encoding='utf-8-sig')
    html = _RecursosHTML()
    html.feed(conteudo)
    exigir(len(html.imagens) > 0, 'Galeria HTML sem imagens: ' + str(pagina))
    # Somente as fontes locais podem ser checadas no disco. Imagens embutidas
    # ou remotas nao sao tratadas como caminho relativo local.
    locais = 0
    for src in html.imagens:
        u = urlsplit(src)
        if u.scheme or u.netloc or src.startswith('//'):
            continue
        caminho = unquote(u.path).replace('/', os.sep)
        exigir(bool(caminho), 'Caminho vazio de imagem em ' + str(pagina))
        destino = (pagina.parent / caminho).resolve()
        exigir(destino.is_file(), 'Imagem referenciada mas ausente: ' + str(destino))
        locais += 1
    return {'sha256': _hash(pagina), 'imagens_html': len(html.imagens),
            'imagens_locais_conferidas': locais, 'caminho': str(pagina)}


def _hash(path: Path):
    h = sha256()
    with path.open('rb') as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def _num7(x):
    return f'{float(x):.7f}'


def _dados_metodo(drv, pasta, info, metodo, manifest_modelo):
    base = pasta/'vencedor'
    f10 = base/'teste_final/avaliacao_f10_3pct'
    graf = base/'teste_final/graficos_f10_3pct'
    gal_f10 = base/'teste_final/galeria_3pct/index.html'
    gal_oof = base/'visualizacao/galeria/index.html'
    met = drv.ler_json(f10/'metricas_f10.json')
    mf = drv.ler_json(f10/'manifesto_f10_3pct.json')
    p = pd.read_csv(f10/'predicoes_f10.csv', usecols=['linha_original','fold','y_true','prob_fraude'])
    idx = np.load(f10/'indices_f10_amostra.npy', allow_pickle=False)
    exigir(len(idx) == len(p) == 5557 and
           np.array_equal(idx, p['linha_original'].to_numpy(dtype=np.int64)) and
           (p['fold'] == 10).all() and int(p['y_true'].sum()) == 29 and
           np.isfinite(p['prob_fraude'].to_numpy()).all(),
           'F10 de ' + metodo + ': contagens, indices ou scores invalidos.')
    exigir(met['limiar_otimizado_no_f10'] is False and
           mf['sem_fit_no_f10'] is True and
           mf['sem_otimizacao_de_tau_no_f10'] is True and
           mf['avaliacao_exploratoria'] is True,
           metodo + ': indicador de avaliacao congelada ou exploratoria invalido.')
    m = base/'teste_final/modelo_final_f1_f9/modelo_final_f1_f9.joblib'
    exigir(_hash(m) == manifest_modelo['sha256_modelo_final'],
           metodo + ': modelo nao corresponde ao manifesto.')
    exigir(int(manifest_modelo['n_treino_original']) == 50013,
           metodo + ': numero de linhas F1-F9 invalido.')
    pngs = sorted(graf.glob('*.png'))
    exigir(len(pngs) == 14,
           metodo + ': esperado exatamente 14 PNGs na pasta F10.')
    gh = {p.name:_hash(p) for p in pngs}
    oof_gal = _conferir_html_local(gal_oof)
    f10_gal = _conferir_html_local(gal_f10)
    exigir(f10_gal['imagens_html'] == 14,
           metodo + ': HTML F10 nao apresenta 14 imagens.')
    escolha = info['escolha']
    return {
        'metodo': metodo.upper(),
        'experimento': str(pasta),
        'n_oof_f1_f9': len(info['ids']),
        'n_f10': int(met['n']),
        'fraudes_f10': int(met['n_fraudes']),
        'legitimas_f10': int(met['n_legitimas']),
        'modelo': {
            'caminho': str(m),
            'sha256': manifest_modelo['sha256_modelo_final'],
            'candidato_oof': escolha['id_candidato'],
            'limiar_mcc_oof_congelado': float(escolha['limiar_mcc_oof']),
            'treino_somente_f1_f9': True,
        },
        'oof': {
            'score_composto': float(escolha['score_oof']),
            'sha256_predicoes_npz': _hash(base/'oof/predicoes.npz'),
            'galeria': oof_gal,
        },
        'f10': {
            'ap': float(met['ap']),
            'mcc_limiar_oof': float(met['mcc_limiar_oof']),
            'logloss': float(met['logloss']),
            'score_composto': float(met['score_composto_limiar_oof']),
            'limiar_reotimizado': False,
            'avaliacao_exploratoria': True,
            'sha256_predicoes_csv': _hash(f10/'predicoes_f10.csv'),
            'sha256_metricas_json': _hash(f10/'metricas_f10.json'),
            'sha256_indices_npy': _hash(f10/'indices_f10_amostra.npy'),
            'graficos_f10_14_sha256': gh,
            'galeria': f10_gal,
        },
    }


def auditar():
    """Confere os dois resultados sem executar etapas de producao."""
    from scripts.execucao import v8_pca_gmm_2d_3pct as v8
    from scripts.execucao import v7_graficos_galeria_tsne2d_3pct as v7
    from scripts.execucao import experimentos_reais_3pct_2d as drv

    print('=== V9 | AUDITORIA FINAL SEM TREINAMENTO ===', flush=True)
    # V8 verificou IDs, folds, classes, hash parquet e folds, recalcou metricas
    # PCA, conferiu 14 PNGs e os links HTML. Nenhuma chamada de execucao.
    base = v8.conferir_base()
    v8.etapa_validar(base)
    _, pca_cfg, tsne_cfg, tsne_pasta, tinfo, pca_pasta, _ = base
    pinfo = v8.conferir_oof_pca(drv, pca_cfg, pca_pasta, tinfo)
    p_mod_dir, p_mod_man = v8.conferir_modelo(drv, pca_cfg, pca_pasta, pinfo)
    # V7 verifica V4/V5/V6, recalcula metricas TSNE e audita todos os PNGs.
    d, t_cfg, t_pasta, t_info, t_mod_dir, t_f10, t_pngs, t_gal, t_met = v7.conferir_integridade_base()
    exigir(d is drv and t_pasta.resolve() == tsne_pasta.resolve(),
           'V7 e V8 discordam sobre ensaio t-SNE.')
    v7.conferir_saida(d, t_pasta, t_f10, t_pngs, t_gal, t_met)
    t_mod_man = drv.validar_modelo(t_mod_dir, t_cfg, t_info['escolha'], 'tsne', 50013)
    exigir(np.array_equal(tinfo['ids'], pinfo['ids']) and
           np.array_equal(tinfo['folds'], pinfo['folds']) and
           np.array_equal(tinfo['y'], pinfo['y']),
           'Critico: OOF PCA e TSNE nao sao as mesmas observacoes.')
    tsne_f10_ids = np.load(t_f10/'indices_f10_amostra.npy', allow_pickle=False)
    pca_f10_ids = np.load(pca_pasta/'vencedor/teste_final/avaliacao_f10_3pct/indices_f10_amostra.npy', allow_pickle=False)
    exigir(np.array_equal(tsne_f10_ids, pca_f10_ids),
           'Critico: F10 PCA/TSNE usa observacoes distintas.')
    exigir(len(tsne_f10_ids) == 5557 and len(tinfo['ids']) == 50013,
           'Tamanhos OOF/F10 divergentes do experimento aprovado.')
    exigir(set(map(int, tsne_f10_ids)).isdisjoint(map(int, tinfo['ids'])),
           'Vazamento: indices F10 presentes no OOF.')
    # SHA logico/canonico reproduz a V8, inclusive se Windows regravar LF como CRLF.
    yaml_digests = {
        'pca': sha256(drv.yaml_texto(pca_cfg).encode('utf-8')).hexdigest(),
        'tsne': sha256(drv.yaml_texto(tsne_cfg).encode('utf-8')).hexdigest(),
    }
    exigir(yaml_digests['pca'] == v8.SHA_PCA_YAML and
           yaml_digests['tsne'] == v8.SHA_TSNE_YAML,
           'Arquivos YAML congelados mudaram desde a V2.')
    resultados = {
        'versao': VERSAO,
        'status': STATUS,
        'tipo': 'AUDITORIA_TECNICA_DE_ARTEFATOS_EXISTENTES',
        'escopo': 'PCA_GMM_2D_E_TSNE_GMM_2D_3PCT_REAL',
        'sem_treinamento': True,
        'sem_inferencia_f10': True,
        'sem_renderizar': True,
        'sem_comparador_automatico': True,
        'f10_exploratorio': True,
        'mesmos_ids_oof': True,
        'mesmos_folds_e_rotulos_oof': True,
        'mesmos_ids_f10': True,
        'sem_intersecao_oof_f10': True,
        'n_f10': int(len(tsne_f10_ids)),
        'fraudes_f10': 29,
        'parquet_sha256': pinfo['manifesto']['sha256_parquet'],
        'folds_sha256': pinfo['manifesto']['sha256_folds'],
        'yaml_sha256': yaml_digests,
        'metodos': {
            'PCA': _dados_metodo(drv, pca_pasta, pinfo, 'pca', p_mod_man),
            'TSNE': _dados_metodo(drv, t_pasta, tinfo, 'tsne', t_mod_man),
        },
        'observacao_metodologica': (
            'F10 reutilizado em experimentos anteriores: resultados exploratorios, '
            'nao constituem validacao independente inedita nem inferencia de superioridade estatistica.'
        ),
    }
    for metodo, rel in resultados['metodos'].items():
        exigir(rel['n_oof_f1_f9'] == 50013 and rel['n_f10'] == 5557 and
               rel['fraudes_f10'] == 29 and
               len(rel['f10']['graficos_f10_14_sha256']) == 14,
               'Contagens de artefatos invalidas para ' + metodo)
    return resultados, drv.ROOT


def _markdown(relatorio):
    linhas = [
        '# Auditoria final — PCA + GMM 2D e t-SNE + GMM 2D (3% real)',
        '',
        '**Estado:** ' + relatorio['status'],
        '',
        'Auditoria técnica somente de leitura dos experimentos. Não treina modelos, '
        'não infere no F10, não reotimiza limiares e não gera gráficos.',
        '',
        '## Integridade e comparabilidade das amostras',
        '',
        '- Mesmas 50.013 observações OOF F1–F9, com folds e rótulos idênticos: **SIM**.',
        '- Mesmas 5.557 observações F10, incluindo 29 fraudes e 5.528 legítimas: **SIM**.',
        '- Interseção OOF × F10: **VAZIA**.',
        '- YAMLs congelados, hashes de dados/folds, modelos e artefatos: **CONFERIDOS**.',
        '- Total de imagens F10: **28 PNGs, 14 por método**.',
        '- Total de galerias HTML: **4**.',
        '',
        '## Resultados auditados — sem classificação automática de métodos',
    ]
    for nome in ('PCA','TSNE'):
        x = relatorio['metodos'][nome]
        f = x['f10']
        linhas += [
            '', '### ' + nome,
            '',
            '- Candidato OOF: `' + x['modelo']['candidato_oof'] + '`',
            '- Limiar ótimo congelado do OOF: **' + _num7(x['modelo']['limiar_mcc_oof_congelado']) + '**',
            '- Score composto OOF: **' + _num7(x['oof']['score_composto']) + '**',
            '- AP F10: **' + _num7(f['ap']) + '**',
            '- MCC F10 no τ* OOF: **' + _num7(f['mcc_limiar_oof']) + '**',
            '- LogLoss F10: **' + _num7(f['logloss']) + '**',
            '- Score composto F10: **' + _num7(f['score_composto']) + '**',
            '- SHA-256 do modelo final: `' + x['modelo']['sha256'] + '`',
            '- Galeria OOF: `' + x['oof']['galeria']['caminho'] + '`',
            '- Galeria F10: `' + f['galeria']['caminho'] + '`',
        ]
    linhas += [
        '', '## Restrição metodológica', '',
        relatorio['observacao_metodologica'], '',
        '**Condição visual:** a auditoria verifica arquivos e links, mas não substitui '
        'a inspeção humana do conteúdo e do layout dos gráficos.',
        '', '## Reprodutibilidade', '',
        '- SHA-256 Parquet: `' + relatorio['parquet_sha256'] + '`',
        '- SHA-256 folds: `' + relatorio['folds_sha256'] + '`',
        '- SHA-256 YAML PCA: `' + relatorio['yaml_sha256']['pca'] + '`',
        '- SHA-256 YAML t-SNE: `' + relatorio['yaml_sha256']['tsne'] + '`',
        '',
        'O arquivo `auditoria_integridade_v9.json` contém os hashes individuais '
        'dos 28 PNGs e dos principais dados de predição, métricas, modelos e galerias.',
        '',
    ]
    return '\n'.join(linhas)


def _salvar_imutavel(caminho, dados):
    if caminho.exists():
        exigir(caminho.is_file() and caminho.read_bytes() == dados,
               'Relatorio existente difere da auditoria atual; nao sobrescrever: ' + str(caminho))
        return 'REUTILIZADO_INTEGRO'
    with caminho.open('xb') as f:
        f.write(dados)
    return 'CRIADO'


def gerar_relatorio(relatorio, raiz):
    destino = raiz/'outputs'/'validacao'/DIRETORIO_RELATORIO
    destino.mkdir(parents=True, exist_ok=True)
    j = (json.dumps(relatorio, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    m = _markdown(relatorio).encode('utf-8')
    json_path = destino/'auditoria_integridade_v9.json'
    md_path = destino/'RELATORIO_AUDITORIA_V9.md'
    print('V9_JSON_STATUS=' + _salvar_imutavel(json_path, j), flush=True)
    print('V9_MD_STATUS=' + _salvar_imutavel(md_path, m), flush=True)
    print('V9_JSON=' + str(json_path), flush=True)
    print('V9_MARKDOWN=' + str(md_path), flush=True)
    print('V9_SHA256_JSON=' + _hash(json_path), flush=True)
    print('V9_SHA256_MARKDOWN=' + _hash(md_path), flush=True)


def abrir_htmls(relatorio):
    import webbrowser
    for metodo in ('PCA','TSNE'):
        info = relatorio['metodos'][metodo]
        for gal in (info['oof']['galeria'], info['f10']['galeria']):
            p = Path(gal['caminho'])
            exigir(p.is_file(), 'Galeria ausente: ' + str(p))
            if hasattr(os, 'startfile'):
                os.startfile(str(p))
            else:
                webbrowser.open(p.as_uri())
            print('V9_HTML_ABERTO=' + str(p), flush=True)
    print('V9_HTML_ABERTOS=4', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    opt = p.add_mutually_exclusive_group(required=True)
    for mode in ('auditar','relatorio','abrir'):
        opt.add_argument('--'+mode, action='store_true')
    a = p.parse_args(argv)
    try:
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        resultado, raiz = auditar()
        if a.relatorio:
            gerar_relatorio(resultado, raiz)
        elif a.abrir:
            abrir_htmls(resultado)
        print('V9_STATUS=' + STATUS, flush=True)
        print('V9_IDS_OOF_IGUAIS=SIM', flush=True)
        print('V9_IDS_F10_IGUAIS=SIM', flush=True)
        print('V9_INTERSECAO_OOF_F10=0', flush=True)
        print('V9_GRAFICOS_F10=28/28', flush=True)
        print('V9_GALERIAS_HTML=4/4', flush=True)
        print('V9_TREINAMENTO_REEXECUTADO=NAO', flush=True)
        print('V9_INFERENCIA_F10_REPETIDA=NAO', flush=True)
        print('V9_F10_EXPLORATORIO=SIM', flush=True)
        print('EXIT_CODE_V9=0', flush=True)
        return 0
    except Exception as e:
        print('V9_STATUS=BLOQUEADO_SEM_ALTERAR_EXPERIMENTOS', file=sys.stderr, flush=True)
        print('V9_MOTIVO=' + str(e), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
