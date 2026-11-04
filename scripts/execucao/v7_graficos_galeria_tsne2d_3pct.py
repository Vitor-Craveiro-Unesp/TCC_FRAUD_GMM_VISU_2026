# -*- coding: utf-8 -*-
"""V7 — gera SOMENTE graficos e galeria F10 t-SNE+GMM 2D, 3% real.

Reutiliza modelo V5, CSVs/metrica F10 V6 e renderer V2 ja instalados.
Nao invoca OOF, fit, transformacao de F10 nem reavaliacao do F10.

python -B -m scripts.execucao.v7_graficos_galeria_tsne2d_3pct --planejar
python -B -m scripts.execucao.v7_graficos_galeria_tsne2d_3pct --executar
python -B -m scripts.execucao.v7_graficos_galeria_tsne2d_3pct --validar
"""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import logging
from pathlib import Path
import struct
import sys
import traceback

METODO = 'tsne'
PASTA_APROVADA = 'ensaio_tsne_gmm_20261103_135207'
OOF_N = 50013
F10_N = 5557
F10_FRAUDES = 29
CANDIDATO = 'd517359b0f6645'
MODELO_SHA256 = 'd9fb29d0c3f900a856f87724315a48e223968cfa0ea61d88bf3ee53546eff588'
TAU_APROX = .0124164
RESULTADOS_F10_ESPERADOS_7D = {
    'ap': .0780832,
    'mcc_limiar_oof': .1011681,
    'logloss': .0306042,
    'score_composto_limiar_oof': .5328422,
}
LOCK = '.ensaio_real_3pct_2d_tsne.lock'


def exigir(cond, mensagem):
    if not cond:
        raise RuntimeError(mensagem)


class LinksHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.images = []
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == 'img' and values.get('src'):
            self.images.append(values['src'])
        elif tag == 'a' and values.get('href'):
            self.hrefs.append(values['href'])


def nomes_esperados(drv):
    nomes = set(drv.IMAGENS_PREDITIVAS) | set(drv.imagens_geometricas(METODO)) | {drv.imagem_projecao(METODO)}
    exigir(len(nomes) == 14 and all(n.endswith('.png') for n in nomes),
           'Contrato V2 nao contem exatamente 14 nomes PNG distintos.')
    return nomes


def conferir_integridade_base():
    """Nao chama v6.conferir_base() (V6 nao conhecia as pastas novas da V7).

    Chama conferidor V5 de OOF (sem treino) e validador F10 V6 diretamente,
    que nao alteram arquivos nem reotimizam limiares.
    """
    from scripts.execucao import v5_modelo_final_tsne2d_3pct as v5
    from scripts.execucao import v6_f10_real_tsne2d_3pct as v6
    from scripts.execucao import experimentos_reais_3pct_2d as drv
    from src.garantia_layout_tsne import verificar_layout_tsne

    base_drv, cfg, pasta, info = v5.conferir_base()
    exigir(base_drv is drv and pasta.name == PASTA_APROVADA,
           'Modulo V2 ou experimento OOF difere do aprovado na V5.')
    verificar_layout_tsne(cfg, fase='executor_3pct_2d')
    exigir(len(info['ids']) == OOF_N and info['escolha']['id_candidato'] == CANDIDATO
           and info['escolha']['teste_final_usado'] is False
           and int(info['escolha']['configuracao']['reducao']['n_components']) == 2,
           'OOF F1..F9 diverge da V4 aprovada.')

    teste_final = pasta / 'vencedor' / 'teste_final'
    exigir(teste_final.is_dir(), 'vencedor/teste_final nao existe.')
    permitidos = {drv.SAIDA_MODELO, drv.SAIDA_F10, drv.SAIDA_GRAFICOS, drv.SAIDA_GALERIA}
    presentes = {p.name for p in teste_final.iterdir()}
    exigir(not (presentes - permitidos),
           'Arquivos nao previstos no teste_final: ' + str(sorted(presentes-permitidos)) + '. Nao apagar.')
    exigir({drv.SAIDA_MODELO, drv.SAIDA_F10} <= presentes,
           'V5 e V6 devem existir antes da V7; nao executar F10 novamente.')

    modelo = teste_final / drv.SAIDA_MODELO
    f10 = teste_final / drv.SAIDA_F10
    graficos = teste_final / drv.SAIDA_GRAFICOS
    galeria = teste_final / drv.SAIDA_GALERIA
    modelo_arq = modelo / 'modelo_final_f1_f9.joblib'
    model_manifest = drv.validar_modelo(modelo, cfg, info['escolha'], METODO, OOF_N)
    exigir(model_manifest.get('sha256_modelo_final') == MODELO_SHA256 and
           drv.hash_arquivo(modelo_arq) == MODELO_SHA256,
           'SHA-256 do modelo final V5 nao confere.')
    exigir(abs(float(info['escolha']['limiar_mcc_oof']) - TAU_APROX) <= 5e-8,
           'Limiar OOF nao corresponde ao aprovado.')
    destino, mf = drv.validar_f10(pasta, cfg, info, METODO, modelo)
    exigir(destino == f10 and mf.get('n_f10') == F10_N and
           mf.get('n_fraudes_f10') == F10_FRAUDES and
           mf.get('sem_fit_no_f10') is True and
           mf.get('sem_otimizacao_de_tau_no_f10') is True and
           mf.get('avaliacao_exploratoria') is True and
           mf.get('sha256_modelo') == MODELO_SHA256,
           'Avaliacao F10 V6 diverge dos resultados aprovados.')
    # Verifica matriz de confusao, indices F10 e todas as metricas sem inferencia.
    met = v6.conferir_resultados(drv, cfg, pasta, info, modelo, f10)
    exigir(int(met['n']) == F10_N and int(met['n_fraudes']) == F10_FRAUDES,
           'Metricas/contagens F10 divergentes.')
    for nome, esperado in RESULTADOS_F10_ESPERADOS_7D.items():
        exigir(abs(float(met[nome]) - esperado) < .000000051,
               'Metrica F10 mudou desde V6: ' + nome)
    return drv, cfg, pasta, info, modelo, f10, graficos, galeria, met


def status_diretorio(p):
    if p.is_dir():
        return 'PRESENTE_VALIDAR_OU_REUTILIZAR'
    exigir(not p.exists(), 'Caminho existe e nao e pasta: ' + str(p))
    return 'NAO_GERADO'


def imprimir_plano(drv, pasta, f10, graficos, galeria, met):
    print('=== V7 | PLANEJAR GRAFICOS E GALERIA F10 TSNE 2D ===', flush=True)
    print('V7_EXPERIMENTO=' + str(pasta), flush=True)
    print('V7_PREDICOES_F10_EXISTENTES=' + str(f10/'predicoes_f10.csv'), flush=True)
    print('V7_GRAFICOS_DESTINO=' + str(graficos), flush=True)
    print('V7_GRAFICOS_ESTADO=' + status_diretorio(graficos), flush=True)
    print('V7_GALERIA_DESTINO=' + str(galeria/'index.html'), flush=True)
    print('V7_GALERIA_ESTADO=' + status_diretorio(galeria), flush=True)
    print('V7_N_PNG_CONTRATO=' + str(len(nomes_esperados(drv))), flush=True)
    print('V7_N_F10=' + str(met['n']), flush=True)
    print('V7_N_FRAUDES=' + str(met['n_fraudes']), flush=True)
    print('V7_OOF_REEXECUTADO=NAO', flush=True)
    print('V7_MODELO_REAJUSTADO=NAO', flush=True)
    print('V7_F10_INFERENCIA_REPETIDA=NAO', flush=True)
    print('V7_PLANO=APROVADO_SEM_MODIFICACOES', flush=True)


def validar_png(arquivo):
    with arquivo.open('rb') as f:
        cab = f.read(24)
    exigir(len(cab) == 24 and cab[:8] == b'\x89PNG\r\n\x1a\n' and cab[12:16] == b'IHDR',
           'PNG invalido ou incompleto: ' + str(arquivo))
    largura, altura = struct.unpack('>II', cab[16:24])
    exigir(200 <= largura <= 16000 and 200 <= altura <= 16000,
           'Dimensoes PNG inesperadas: ' + str(arquivo))


def conferir_saida(drv, pasta, f10, graficos, galeria, met):
    esperados = nomes_esperados(drv)
    exigir(graficos.is_dir() and galeria.is_dir(),
           'Imagens e galeria ainda nao geradas; execute V7 --executar.')
    gdir, m_g = drv.validar_graficos(pasta, METODO, f10)
    exigir(gdir == graficos and m_g.get('status') == 'APROVADA' and
           m_g.get('n_imagens') == 14 and set(m_g.get('imagens', [])) == esperados and
           m_g.get('sem_fit') is True and m_g.get('limiar_f10_reotimizado') is False,
           'Manifesto de figuras F10 nao corresponde ao contrato aprovado.')
    pngs = {p.name for p in graficos.glob('*.png')}
    exigir(pngs == esperados,
           'Os 14 PNGs originais estao incompletos ou incluem extras: ' + str(sorted(pngs ^ esperados)))
    for nome in sorted(esperados):
        png = graficos/nome
        validar_png(png)
        exigir(m_g['sha256_arquivos'].get(nome) == drv.hash_arquivo(png),
               'Hash PNG original divergente: ' + nome)
    manifest_gal = drv.ler_json(galeria/'manifesto_galeria_3pct.json')
    exigir(manifest_gal.get('status') == 'APROVADA' and
           manifest_gal.get('metodo') == 'TSNE' and
           manifest_gal.get('sha256_predicoes_f10') == drv.hash_arquivo(f10/'predicoes_f10.csv') and
           manifest_gal.get('sha256_html') == drv.hash_arquivo(galeria/'index.html') and
           set(manifest_gal.get('hashes_png', {})) == esperados,
           'Galeria HTML ou seu manifesto nao correspondem a V6.')
    png_galeria = {p.name for p in galeria.glob('*.png')}
    exigir(png_galeria == esperados,
           'Galeria contem PNGs extras ou faltantes: ' + str(sorted(png_galeria ^ esperados)))
    for nome in sorted(esperados):
        png = galeria/nome
        validar_png(png)
        digest = drv.hash_arquivo(png)
        exigir(digest == drv.hash_arquivo(graficos/nome) and
               digest == manifest_gal['hashes_png'][nome],
               'PNG do HTML nao corresponde ao original: ' + nome)
    html = (galeria/'index.html').read_text(encoding='utf-8-sig')
    extrator = LinksHTML()
    extrator.feed(html)
    exigir(set(extrator.images) == esperados and len(extrator.images) == 14,
           'HTML nao referencia exatamente as 14 figuras esperadas.')
    # Navegacao funcional entre o HTML F10 e galeria OOF da mesma execucao.
    links_esperados = {
        '../../visualizacao/galeria/index.html',
        '../avaliacao_f10_3pct/metricas_f10.json',
        '../avaliacao_f10_3pct/predicoes_f10.csv',
    }
    exigir(links_esperados <= set(extrator.hrefs),
           'HTML nao contem os tres links de referencia esperados.')
    for href in links_esperados:
        destino = (galeria/href).resolve()
        exigir(destino.is_file(), 'Link quebrado no HTML: ' + href)
    exigir('exploratorio' in html.lower() or 'exploratório' in html.lower(),
           'Aviso de teste exploratorio nao consta no HTML.')
    exigir((pasta/'vencedor/visualizacao/galeria/index.html').is_file(),
           'Galeria OOF real ausente.')
    print('=== V7 | GRAFICOS + GALERIA REAL F10 TSNE GMM 2D ===', flush=True)
    print('V7_STATUS=GRAFICOS_GALERIA_TSNE_2D_3PCT_APROVADOS', flush=True)
    print('V7_EXPERIMENTO=' + str(pasta), flush=True)
    print('V7_FIGURAS_F10=14/14', flush=True)
    print('V7_HTML=' + str(galeria/'index.html'), flush=True)
    print('V7_HTML_OOF=' + str(pasta/'vencedor/visualizacao/galeria/index.html'), flush=True)
    print('V7_F10_N=' + str(met['n']), flush=True)
    print('V7_F10_FRAUDES=' + str(met['n_fraudes']), flush=True)
    print('V7_AP_F10=' + f'{float(met["ap"]):.7f}', flush=True)
    print('V7_MCC_F10=' + f'{float(met["mcc_limiar_oof"]):.7f}', flush=True)
    print('V7_LOGLOSS_F10=' + f'{float(met["logloss"]):.7f}', flush=True)
    print('V7_SCORE_F10=' + f'{float(met["score_composto_limiar_oof"]):.7f}', flush=True)
    print('V7_OOF_REEXECUTADO=NAO', flush=True)
    print('V7_MODELO_REAJUSTADO=NAO', flush=True)
    print('V7_F10_INFERENCIA_REPETIDA=NAO', flush=True)
    print('V7_PCA_EXECUTADO=NAO', flush=True)
    return True


def executar(drv, cfg, pasta, info, modelo, f10, graficos, galeria, met):
    lock = drv.ROOT / 'outputs' / LOCK
    exigir(not lock.exists(), 'Outro processamento TSNE 2D pode estar ativo: ' + str(lock))
    # Manifestos de resultados sao validados caso pastas ja existam.
    if graficos.is_dir():
        drv.validar_graficos(pasta, METODO, f10)
    if galeria.is_dir():
        exigir(graficos.is_dir(), 'Galeria ja existe, mas graficos originais ausentes.')
        conferir_saida(drv, pasta, f10, graficos, galeria, met)
        print('V7_REUTILIZACAO=VALIDADA_SEM_RENDERIZAR', flush=True)
        return
    with lock.open('x', encoding='utf-8') as arq:
        arq.write('V7: somente renderizacao em F10 ja avaliado na V6.\n')
    try:
        exigir(not galeria.exists(), 'Galeria surgiu durante a aquisicao do lock; nao sobrescrever.')
        print('[V7] Reutilizando modelo V5 e resultados F10 V6; sem treino nem inferencia.', flush=True)
        gdir = drv.executar_graficos(pasta, cfg, info, METODO, modelo, f10)
        exigir(gdir == graficos and graficos.is_dir(),
               'Diretorio de graficos retornado incorreto.')
        gal_dir = drv.executar_galeria(pasta, cfg, METODO, f10, graficos)
        exigir(gal_dir == galeria and galeria.is_dir(),
               'Diretorio de galeria retornado incorreto.')
        conferir_saida(drv, pasta, f10, graficos, galeria, met)
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    opc = parser.add_mutually_exclusive_group(required=True)
    opc.add_argument('--planejar', action='store_true')
    opc.add_argument('--executar', action='store_true')
    opc.add_argument('--validar', action='store_true')
    args = parser.parse_args(argv)
    try:
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        drv, cfg, pasta, info, modelo, f10, graficos, galeria, met = conferir_integridade_base()
        if args.planejar:
            imprimir_plano(drv, pasta, f10, graficos, galeria, met)
        elif args.executar:
            executar(drv, cfg, pasta, info, modelo, f10, graficos, galeria, met)
        else:
            conferir_saida(drv, pasta, f10, graficos, galeria, met)
        print('EXIT_CODE_V7=0', flush=True)
        return 0
    except Exception as exc:
        print('V7_STATUS=BLOQUEADO_SEM_TREINAR', file=sys.stderr, flush=True)
        print('V7_MOTIVO=' + str(exc), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
