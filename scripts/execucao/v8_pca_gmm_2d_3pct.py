# -*- coding: utf-8 -*-
"""V8: PCA+GMM 2D em 3% reais; fases independentes e comparabilidade por IDs.

USO: python -B -m scripts.execucao.v8_pca_gmm_2d_3pct --planejar|--oof|--modelo|--f10|--galeria|--validar

NUNCA chama o executor completo --pca. Cada fase e opcional e idempotente;
usa somente o escopo apropriado. Sem comparador de metricas entre metodos.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
from html.parser import HTMLParser
import logging
from pathlib import Path
import struct
import sys
import traceback

import numpy as np
import pandas as pd

METODO = 'pca'
N_OOF_REFERENCIA = 50013
N_F10_REFERENCIA = 5557
N_FRAUDES_F10_REFERENCIA = 29
TSNE_PASTA_REFERENCIA = 'ensaio_tsne_gmm_20261103_135207'
TSNE_SHA_MODELO = 'd9fb29d0c3f900a856f87724315a48e223968cfa0ea61d88bf3ee53546eff588'
TSNE_CANDIDATO = 'd517359b0f6645'
TSNE_TAU_7D = 0.0124164
SHA_PCA_YAML = '29b88f4d7f1de6c5ad86bbff0421e20f14ba635599aba559464dcb12a24a7604'
SHA_TSNE_YAML = '64f105eb35e6928cc0843a27be187ded9951e6cb84191a875ebeb835974440c4'
LOCK = '.ensaio_real_3pct_2d_pca.lock'
PERMITIDOS = frozenset({'modelo_final_f1_f9', 'avaliacao_f10_3pct', 'graficos_f10_3pct', 'galeria_3pct'})


def exigir(ok, motivo):
    if not ok:
        raise RuntimeError(motivo)


class _HTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.images = []
        self.links = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'img' and a.get('src'):
            self.images.append(a['src'])
        if tag == 'a' and a.get('href'):
            self.links.append(a['href'])


def conferir_base():
    # Reaproveita validacoes V3/V4 sem executar treino ou F10.
    from scripts.execucao import experimentos_reais_3pct_2d as drv
    from scripts.execucao import v4_oof_real_tsne2d_3pct as v4
    d2, tsne_cfg = v4.conferir_homologacao()
    exigir(d2 is drv, 'Executor V2 nao corresponde ao modulo homologado.')
    pca_cfg, grade = drv.construir_config(METODO)
    exigir(tsne_cfg == drv.construir_config('tsne')[0], 'YAML TSNE mudou.')
    exigir(len(grade) == 1 and grade[0]['metodo'] == 'PCA' and
           grade[0]['reducao']['n_components'] == 2,
           'PCA 2D deve possuir exatamente um candidato no OOF.')
    for metodo, digest in [('pca', SHA_PCA_YAML), ('tsne', SHA_TSNE_YAML)]:
        c, _ = drv.construir_config(metodo)
        exigir(sha256(drv.yaml_texto(c).encode('utf-8')).hexdigest() == digest,
               f'YAML {metodo.upper()} difere do aprovado na V2.')
    exigir(not drv.LIBERACAO_V3.exists(),
           'O executor completo 2D foi liberado; abortar para nao misturar fluxos.')
    from src.garantia_layout_tsne import verificar_layout_tsne
    verificar_layout_tsne(tsne_cfg, fase='executor_3pct_2d')

    # V8 usa a amostra de OOF real do TSNE como identidade de referencia.
    tsne_pasta, parcial = drv.localizar_experimento(tsne_cfg, 'tsne')
    exigir(tsne_pasta is not None and parcial is None and
           tsne_pasta.name == TSNE_PASTA_REFERENCIA,
           'Ensaio TSNE 2D de referencia nao e o aprovado V4-V7.')
    info_tsne = drv.validar_oof_f1f9(tsne_pasta, tsne_cfg, 'tsne')
    escolha_t = info_tsne['escolha']
    exigir(len(info_tsne['ids']) == N_OOF_REFERENCIA and
           escolha_t['id_candidato'] == TSNE_CANDIDATO and
           abs(float(escolha_t['limiar_mcc_oof']) - TSNE_TAU_7D) <= 5.1e-8 and
           escolha_t['teste_final_usado'] is False,
           'OOF do t-SNE 2D nao corresponde a V4 aprovada.')
    pca_pasta, pca_parcial = drv.localizar_experimento(pca_cfg, METODO)
    exigir(not (pca_pasta and pca_parcial), 'PCA 2D completo e parcial coexistem.')
    return drv, pca_cfg, tsne_cfg, tsne_pasta, info_tsne, pca_pasta, pca_parcial


def conferir_escopo(teste, etapa):
    """Pasta vazia criada pelo OOF e permitida; nunca apagar conteudo."""
    if not teste.exists():
        return
    exigir(teste.is_dir(), 'teste_final existe mas nao e pasta: ' + str(teste))
    nomes = {p.name for p in teste.iterdir()}
    exigir(not (nomes - PERMITIDOS),
           'Conteudo inesperado em teste_final: ' + str(sorted(nomes-PERMITIDOS)))
    limite = {'oof': set(), 'modelo': {'modelo_final_f1_f9'},
              'f10': {'modelo_final_f1_f9', 'avaliacao_f10_3pct'},
              'galeria': set(PERMITIDOS), 'validar': set(PERMITIDOS)}[etapa]
    exigir(nomes <= limite,
           f'Etapa {etapa}: pasta teste_final contem fases posteriores. '
           'Use validar/abrir, nao execute fases anteriores novamente.')
    for nome in nomes:
        exigir((teste/nome).is_dir(), 'Item inesperado dentro teste_final: ' + nome)


def conferir_oof_pca(drv, cfg, pasta, tsne_info):
    exigir(pasta is not None and pasta.is_dir(), 'OOF PCA 2D ainda nao existe.')
    man = drv.ler_json(pasta/'manifesto.json')
    exigir(man.get('status') == 'concluido_06_03' and
           man.get('visualizacoes') == 'concluidas' and
           man.get('tipo') == 'ensaio', 'OOF PCA nao esta concluido ou visualizado.')
    exigir((pasta/'vencedor/visualizacao/galeria/index.html').is_file(),
           'Galeria OOF do PCA ausente.')
    info = drv.validar_oof_f1f9(pasta, cfg, METODO)
    exigir(len(info['ids']) == N_OOF_REFERENCIA and
           np.array_equal(info['ids'], tsne_info['ids']) and
           np.array_equal(info['folds'], tsne_info['folds']) and
           np.array_equal(info['y'], tsne_info['y']),
           'CRITICO: PCA e TSNE usam registros/folds/classes OOF diferentes.')
    exigir(man['sha256_parquet'] == tsne_info['manifesto']['sha256_parquet'] and
           man['sha256_folds'] == tsne_info['manifesto']['sha256_folds'],
           'Parquet/folds PCA diferentes da referencia TSNE.')
    esc = info['escolha']
    exigir(esc['teste_final_usado'] is False and
           esc['configuracao']['metodo'] == 'PCA' and
           esc['configuracao']['reducao']['n_components'] == 2,
           'Vencedor PCA nao e o candidato PCA 2D do OOF.')
    return info


def conferir_modelo(drv, cfg, pasta, info):
    modelo = pasta/'vencedor/teste_final/modelo_final_f1_f9'
    exigir(modelo.is_dir(), 'PCA sem modelo final: execute -Modo modelo.')
    man = drv.validar_modelo(modelo, cfg, info['escolha'], METODO, N_OOF_REFERENCIA)
    exigir(man.get('tipo') == 'TREINAMENTO_FINAL_F1_F9' and
           int(man.get('dimensao_pca', -1)) == 2 and
           man.get('teste_f10_executado') is False and
           int(man['n_treino_original']) == N_OOF_REFERENCIA and
           man['id_candidato_oof'] == info['escolha']['id_candidato'],
           'Manifesto do modelo PCA nao corresponde a V8.')
    for campo in ('sha256_parquet','sha256_folds','sha256_yaml'):
        exigir(man['proveniencia'].get(campo) == info['manifesto'].get(campo),
               'Proveniencia de modelo PCA divergente: ' + campo)
    exigir(Path(man['proveniencia']['experimento_oof']).resolve() == pasta.resolve(),
           'Modelo PCA nao pertence ao ensaio OOF.')
    return modelo, man


def conferir_tsne_f10(drv, tsne_pasta, tsne_cfg, tsne_info):
    # Apenas depois do reajuste PCA final. F10 NAO e usado para escolher candidato PCA.
    base = tsne_pasta/'vencedor/teste_final'
    modelo = base/'modelo_final_f1_f9'
    manifesto_mod = drv.validar_modelo(modelo, tsne_cfg, tsne_info['escolha'],
                                       'tsne', N_OOF_REFERENCIA)
    exigir(manifesto_mod['sha256_modelo_final'] == TSNE_SHA_MODELO,
           'Modelo t-SNE V5 mudou desde a aprovacao.')
    f10, manifesto = drv.validar_f10(tsne_pasta, tsne_cfg, tsne_info,
                                    'tsne', modelo)
    exigir(int(manifesto['n_f10']) == N_F10_REFERENCIA and
           int(manifesto['n_fraudes_f10']) == N_FRAUDES_F10_REFERENCIA and
           manifesto.get('sem_fit_no_f10') is True and
           manifesto.get('sem_otimizacao_de_tau_no_f10') is True,
           'F10 do t-SNE nao corresponde ao aprovado V6.')
    ids = np.load(f10/'indices_f10_amostra.npy', allow_pickle=False)
    pred = pd.read_csv(f10/'predicoes_f10.csv')
    exigir(len(ids) == N_F10_REFERENCIA and
           np.array_equal(ids, pred['linha_original'].to_numpy(dtype=np.int64)) and
           (pred['fold'] == 10).all() and
           int(pred['y_true'].sum()) == N_FRAUDES_F10_REFERENCIA and
           np.all(tsne_info['folds_master'][ids] == 10),
           'IDs do F10 TSNE sao inconsistentes.')
    return ids, pred['y_true'].to_numpy(dtype=np.uint8)


def conferir_f10_pca(drv, cfg, pasta, info, modelo, ids_tsne, y_tsne):
    f10, manifesto = drv.validar_f10(pasta, cfg, info, METODO, modelo)
    exigir(manifesto['metodo'] == 'PCA' and
           int(manifesto['n_f10']) == N_F10_REFERENCIA and
           int(manifesto['n_fraudes_f10']) == N_FRAUDES_F10_REFERENCIA and
           manifesto.get('sem_fit_no_f10') is True and
           manifesto.get('sem_otimizacao_de_tau_no_f10') is True and
           manifesto.get('avaliacao_exploratoria') is True,
           'F10 PCA nao respeita o contrato V8.')
    ids = np.load(f10/'indices_f10_amostra.npy', allow_pickle=False)
    pred = pd.read_csv(f10/'predicoes_f10.csv')
    coords = pd.read_csv(f10/'coordenadas_PC1_PC2_f10.csv')
    met = drv.ler_json(f10/'metricas_f10.json')
    exigir(len(ids) == N_F10_REFERENCIA and
           np.array_equal(ids, ids_tsne) and
           np.array_equal(ids, pred['linha_original'].to_numpy(dtype=np.int64)) and
           np.array_equal(ids, coords['linha_original'].to_numpy(dtype=np.int64)) and
           np.array_equal(pred['y_true'].to_numpy(dtype=np.uint8), y_tsne) and
           (pred['fold'] == 10).all() and
           np.isfinite(coords[['PC1','PC2']].to_numpy()).all(),
           'CRITICO: amostra F10 PCA/TSNE nao coincide ou coordenadas divergentes.')
    y = pred['y_true'].to_numpy(dtype=np.uint8)
    p = pred['prob_fraude'].to_numpy(dtype=np.float64)
    tau = float(info['escolha']['limiar_mcc_oof'])
    exigir(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)) and
           np.array_equal(pred['predicao_tau_0_50'], (p >= .5).astype(int)) and
           np.array_equal(pred['predicao_tau_mcc_oof'], (p >= tau).astype(int)) and
           met['limiar_otimizado_no_f10'] is False,
           'Scores, limiar ou predicoes F10 PCA invalidos.')
    from src.avaliacao_final_pca import avaliar_scores_com_limiar_oof_congelado
    reaval = avaliar_scores_com_limiar_oof_congelado(y, p, limiar_mcc_oof=tau)
    for campo in ('ap','mcc_limiar_oof','logloss','score_composto_limiar_oof',
                  'prevalencia','exp_menos_logloss'):
        exigir(abs(float(reaval[campo]) - float(met[campo])) < 1e-10,
               'Metrica PCA F10 inconsistente: ' + campo)
    for corte in ('tau_0_50','tau_mcc_oof'):
        for k in ('tn','fp','fn','tp'):
            exigir(int(reaval['comparacao_limiares'][corte]['matriz_confusao'][k]) ==
                   int(met['comparacao_limiares'][corte]['matriz_confusao'][k]),
                   'Matriz F10 PCA inconsistente: '+corte+'/'+k)
    return f10, met


def png_valido(arq):
    with arq.open('rb') as f:
        b = f.read(24)
    exigir(b[:8] == b'\x89PNG\r\n\x1a\n' and len(b) == 24 and b[12:16] == b'IHDR',
           'PNG invalido: ' + str(arq))
    w, h = struct.unpack('>II', b[16:24])
    exigir(200 <= w <= 16000 and 200 <= h <= 16000,
           'Dimensoes PNG inesperadas: ' + str(arq))


def conferir_galeria_pca(drv, pasta, f10):
    graficos = pasta/'vencedor/teste_final/graficos_f10_3pct'
    galeria = pasta/'vencedor/teste_final/galeria_3pct'
    exigir(graficos.is_dir() and galeria.is_dir(),
           'Graficos e galeria PCA ainda nao completos.')
    _g, mg = drv.validar_graficos(pasta, METODO, f10)
    nomes = set(drv.IMAGENS_PREDITIVAS) | set(drv.imagens_geometricas(METODO)) | {drv.imagem_projecao(METODO)}
    exigir(len(nomes) == 14 and mg.get('n_imagens') == 14 and
           set(mg.get('imagens', [])) == nomes and
           {p.name for p in graficos.glob('*.png')} == nomes and
           mg.get('sem_fit') is True and
           mg.get('limiar_f10_reotimizado') is False,
           '14 graficos PCA nao correspondem ao contrato.')
    for nome in nomes:
        png_valido(graficos/nome)
        exigir(mg['sha256_arquivos'][nome] == drv.hash_arquivo(graficos/nome),
               'Hash grafico PCA divergente: ' + nome)
    m = drv.ler_json(galeria/'manifesto_galeria_3pct.json')
    exigir(m['status'] == 'APROVADA' and m['metodo'] == 'PCA' and
           m['sha256_html'] == drv.hash_arquivo(galeria/'index.html') and
           m['sha256_predicoes_f10'] == drv.hash_arquivo(f10/'predicoes_f10.csv') and
           set(m['hashes_png']) == nomes and
           {p.name for p in galeria.glob('*.png')} == nomes,
           'Manifesto/figuras da galeria PCA divergente.')
    for nome in nomes:
        png_valido(galeria/nome)
        exigir(m['hashes_png'][nome] == drv.hash_arquivo(galeria/nome) ==
               drv.hash_arquivo(graficos/nome),
               'Galeria PCA possui arquivo divergente: ' + nome)
    extr = _HTML()
    extr.feed((galeria/'index.html').read_text(encoding='utf-8-sig'))
    exigir(len(extr.images) == 14 and set(extr.images) == nomes,
           'HTML PCA nao possui exatamente 14 referencias de imagem.')
    links = {'../../visualizacao/galeria/index.html',
             '../avaliacao_f10_3pct/metricas_f10.json',
             '../avaliacao_f10_3pct/predicoes_f10.csv'}
    exigir(links <= set(extr.links) and
           all((galeria/x).resolve().is_file() for x in links),
           'HTML PCA contem links quebrados.')
    return graficos, galeria


@contextmanager
def lock_pca(drv):
    raiz = drv.ROOT/'outputs'
    exigir(raiz.is_dir(), 'Diretorio outputs ausente.')
    lock = raiz/LOCK
    exigir(not lock.exists(), 'Lock PCA V8 presente; audite antes de repetir: ' + str(lock))
    with lock.open('x', encoding='utf-8') as arq:
        arq.write('V8 PCA GMM 2D 3%: comando isolado; F10 nao treina.\n')
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def etapa_oof(base):
    drv, cfg, _t_cfg, _t_pasta, t_info, pca_pasta, parcial = base
    exigir(pca_pasta is None or parcial is None, 'Pasta PCA ambigua.')
    if pca_pasta is not None:
        info = conferir_oof_pca(drv, cfg, pca_pasta, t_info)
        pasta = pca_pasta
        print('V8_OOF_REUTILIZADO_SEM_TREINAR=SIM', flush=True)
    else:
        with lock_pca(drv):
            pasta = drv.executar_oof(METODO, cfg)
            info = conferir_oof_pca(drv, cfg, pasta, t_info)
            conferir_escopo(pasta/'vencedor/teste_final', 'oof')
    print('V8_OOF_STATUS=APROVADO', flush=True)
    print('V8_OOF_N=' + str(len(info['ids'])), flush=True)
    print('V8_OOF_VENCEDOR=' + info['escolha']['id_candidato'], flush=True)
    print('V8_OOF_TAU=' + f'{float(info["escolha"]["limiar_mcc_oof"]):.7f}', flush=True)
    print('V8_OOF_SCORE=' + f'{float(info["escolha"]["score_oof"]):.7f}', flush=True)
    print('V8_REGISTROS_OOF_IGUAIS_AO_TSNE=SIM', flush=True)
    print('V8_F10_AVALIADO=NAO', flush=True)
    print('V8_OOF_GALERIA=' + str(pasta/'vencedor/visualizacao/galeria/index.html'), flush=True)


def etapa_modelo(base):
    drv, cfg, _t_cfg, _t_pasta, t_info, pasta, _parcial = base
    info = conferir_oof_pca(drv, cfg, pasta, t_info)
    teste = pasta/'vencedor/teste_final'
    conferir_escopo(teste, 'modelo')
    m = teste/'modelo_final_f1_f9'
    if m.is_dir():
        conferir_modelo(drv, cfg, pasta, info)
        print('V8_MODELO_REUTILIZADO_SEM_FIT=SIM', flush=True)
    else:
        exigir(not m.exists(), 'Destino modelo existe como arquivo.')
        with lock_pca(drv):
            drv.executar_modelo(pasta, cfg, info, METODO)
            conferir_modelo(drv, cfg, pasta, info)
    print('V8_MODELO_STATUS=APROVADO', flush=True)
    print('V8_F10_AVALIADO=NAO', flush=True)


def etapa_f10(base):
    drv, cfg, tsne_cfg, tsne_pasta, t_info, pasta, _parcial = base
    info = conferir_oof_pca(drv, cfg, pasta, t_info)
    conferir_escopo(pasta/'vencedor/teste_final', 'f10')
    modelo, m = conferir_modelo(drv, cfg, pasta, info)
    ids, ys = conferir_tsne_f10(drv, tsne_pasta, tsne_cfg, t_info)
    esperados, yprev = drv.escolher_indices_f10(info['folds_master'],
                              info['arquivo_parquet'], cfg['dados']['coluna_alvo'])
    exigir(np.array_equal(esperados, ids) and np.array_equal(yprev, ys),
           'PCA selecionaria F10 distinto do TSNE. BLOQUEADO ANTES DA INFERENCIA.')
    f10_dir = pasta/'vencedor/teste_final/avaliacao_f10_3pct'
    if not f10_dir.is_dir():
        exigir(not f10_dir.exists(), 'Destino F10 ja existe como arquivo.')
        with lock_pca(drv):
            drv.executar_f10(pasta, cfg, info, METODO, modelo)
    f10, met = conferir_f10_pca(drv, cfg, pasta, info, modelo, ids, ys)
    print('V8_F10_STATUS=APROVADO', flush=True)
    print('V8_F10_N=' + str(met['n']), flush=True)
    print('V8_F10_FRAUDES=' + str(met['n_fraudes']), flush=True)
    for campo, rotulo in [('ap','AP'), ('mcc_limiar_oof','MCC_TAU_OOF'),
                          ('logloss','LOGLOSS'), ('score_composto_limiar_oof','SCORE_COMPOSTO')]:
        print('V8_F10_' + rotulo + '=' + f'{float(met[campo]):.7f}', flush=True)
    print('V8_F10_IDS_IGUAIS_AO_TSNE=SIM', flush=True)
    print('V8_F10_LIMIAR_REOTIMIZADO=NAO', flush=True)


def etapa_galeria(base):
    drv, cfg, tsne_cfg, tsne_pasta, t_info, pasta, _parcial = base
    info = conferir_oof_pca(drv, cfg, pasta, t_info)
    conferir_escopo(pasta/'vencedor/teste_final', 'galeria')
    modelo, _m = conferir_modelo(drv, cfg, pasta, info)
    ids, y = conferir_tsne_f10(drv, tsne_pasta, tsne_cfg, t_info)
    f10, met = conferir_f10_pca(drv, cfg, pasta, info, modelo, ids, y)
    graficos = pasta/'vencedor/teste_final/graficos_f10_3pct'
    galeria = pasta/'vencedor/teste_final/galeria_3pct'
    if galeria.is_dir():
        conferir_galeria_pca(drv, pasta, f10)
        print('V8_GALERIA_REUTILIZADA=SIM', flush=True)
    else:
        exigir(not galeria.exists(), 'Destino galeria ja existe como arquivo.')
        with lock_pca(drv):
            if not graficos.is_dir():
                exigir(not graficos.exists(), 'Destino graficos ja existe como arquivo.')
                from src import visualizacao_geometrica as geo
                import inspect
                exigir('f10_real_3pct' in inspect.getsource(geo._fronteira),
                       'GMM08 nao possui hotfix F10 real. Instale hotfix V7 primeiro.')
            gd = drv.executar_graficos(pasta, cfg, info, METODO, modelo, f10)
            drv.executar_galeria(pasta, cfg, METODO, f10, gd)
            conferir_galeria_pca(drv, pasta, f10)
    print('V8_GALERIA_STATUS=APROVADA', flush=True)
    print('V8_GRAFICOS_F10=14/14', flush=True)
    print('V8_GALERIA_F10=' + str(galeria/'index.html'), flush=True)
    print('V8_MODELO_REAJUSTADO_NA_GALERIA=NAO', flush=True)
    print('V8_F10_INFERENCIA_REPETIDA_NA_GALERIA=NAO', flush=True)


def etapa_validar(base):
    drv, cfg, tsne_cfg, tsne_pasta, t_info, pasta, _parcial = base
    info = conferir_oof_pca(drv, cfg, pasta, t_info)
    conferir_escopo(pasta/'vencedor/teste_final', 'validar')
    modelo, m = conferir_modelo(drv, cfg, pasta, info)
    ids, y = conferir_tsne_f10(drv, tsne_pasta, tsne_cfg, t_info)
    f10, met = conferir_f10_pca(drv, cfg, pasta, info, modelo, ids, y)
    _, galeria = conferir_galeria_pca(drv, pasta, f10)
    tsne_oof = tsne_pasta/'vencedor/visualizacao/galeria/index.html'
    tsne_f10 = tsne_pasta/'vencedor/teste_final/galeria_3pct/index.html'
    exigir(tsne_oof.is_file() and tsne_f10.is_file(), 'HTMLs TSNE V7 ausentes.')
    print('=== V8 | PCA + GMM 2D REAL 3% ===', flush=True)
    print('V8_STATUS=PCA_2D_OOF_F10_GALERIAS_APROVADOS', flush=True)
    print('V8_PCA_OOF=' + str(pasta/'vencedor/visualizacao/galeria/index.html'), flush=True)
    print('V8_PCA_F10=' + str(galeria/'index.html'), flush=True)
    print('V8_TSNE_OOF=' + str(tsne_oof), flush=True)
    print('V8_TSNE_F10=' + str(tsne_f10), flush=True)
    print('V8_REGISTROS_OOF_IGUAIS=SIM', flush=True)
    print('V8_REGISTROS_F10_IGUAIS=SIM', flush=True)
    print('V8_F10_N=' + str(met['n']), flush=True)
    print('V8_F10_FRAUDES=' + str(met['n_fraudes']), flush=True)
    print('V8_F10_AP=' + f'{float(met["ap"]):.7f}', flush=True)
    print('V8_F10_MCC=' + f'{float(met["mcc_limiar_oof"]):.7f}', flush=True)
    print('V8_F10_LOGLOSS=' + f'{float(met["logloss"]):.7f}', flush=True)
    print('V8_F10_SCORE=' + f'{float(met["score_composto_limiar_oof"]):.7f}', flush=True)
    print('V8_OOF_REEXECUTADO_NA_VALIDACAO=NAO', flush=True)
    print('V8_TREINO_REEXECUTADO_NA_VALIDACAO=NAO', flush=True)
    print('V8_F10_INFERENCIA_REPETIDA_NA_VALIDACAO=NAO', flush=True)
    print('V8_F10_EXPLORATORIO=SIM', flush=True)


def etapa_abrir(base):
    import os
    import webbrowser
    # Sempre validar os 4 resultados antes de abrir arquivos.
    etapa_validar(base)
    drv, _c, _t, tsne_pasta, _ti, pasta, _p = base
    paginas = [
        pasta/'vencedor/visualizacao/galeria/index.html',
        pasta/'vencedor/teste_final/galeria_3pct/index.html',
        tsne_pasta/'vencedor/visualizacao/galeria/index.html',
        tsne_pasta/'vencedor/teste_final/galeria_3pct/index.html',
    ]
    for p in paginas:
        exigir(p.is_file(), 'HTML nao encontrado: ' + str(p))
        if hasattr(os, 'startfile'):
            os.startfile(str(p))
        else:
            webbrowser.open(p.as_uri())
        print('V8_HTML_ABERTO=' + str(p), flush=True)
    print('V8_HTML_ABERTOS=4', flush=True)


def etapa_planejar(base):
    drv, cfg, tsne_cfg, tsne_pasta, t_info, pasta, parcial = base
    print('=== V8 | PLANEJAMENTO PCA + GMM 2D REAL 3% ===', flush=True)
    print('V8_YAML=' + str(drv.YAMLS_CONGELADOS/'pca_gmm_2d_3pct.yaml'), flush=True)
    print('V8_AMOSTRAGEM=3%_OOF_E_F10_ESTRATIFICADOS', flush=True)
    print('V8_OOF_REGISTROS_TSNE=' + str(len(t_info['ids'])), flush=True)
    print('V8_OOF_CANDIDATOS_PCA=1', flush=True)
    print('V8_OOF_AJUSTES_GMM=9', flush=True)
    print('V8_MODELO_FINAL=SO_F1_F9', flush=True)
    print('V8_F10=APENAS_APOS_MODELO_PERSISTIDO', flush=True)
    print('V8_PCA_OOF_EXISTENTE=' + str(pasta or ''), flush=True)
    print('V8_PCA_OOF_PARCIAL=' + str(parcial or ''), flush=True)
    print('V8_TSNE_REFERENCIA=' + str(tsne_pasta), flush=True)
    print('V8_F10_PREVISTO=' + str(N_F10_REFERENCIA), flush=True)
    print('V8_REGISTROS_COMPARADOS_POR_ID=SIM', flush=True)
    print('V8_PLANEJAMENTO=APROVADO_SEM_TREINAR_OU_LER_F10', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    for opc in ('planejar','oof','modelo','f10','galeria','validar','abrir'):
        g.add_argument('--'+opc, action='store_true')
    a = parser.parse_args(argv)
    try:
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        base = conferir_base()
        modo = next(v for v in ('planejar','oof','modelo','f10','galeria','validar','abrir') if getattr(a,v))
        if modo == 'planejar':
            etapa_planejar(base)
        elif modo == 'oof':
            etapa_oof(base)
        elif modo == 'modelo':
            etapa_modelo(base)
        elif modo == 'f10':
            etapa_f10(base)
        elif modo == 'galeria':
            etapa_galeria(base)
        elif modo == 'validar':
            etapa_validar(base)
        else:
            etapa_abrir(base)
        print('EXIT_CODE_V8=0', flush=True)
        return 0
    except Exception as exc:
        print('V8_STATUS=BLOQUEADO_SEM_SOBRESCRITA', file=sys.stderr, flush=True)
        print('V8_MOTIVO=' + str(exc), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
