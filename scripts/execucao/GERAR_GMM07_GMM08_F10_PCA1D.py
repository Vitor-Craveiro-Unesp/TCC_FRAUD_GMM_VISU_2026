# -*- coding: utf-8 -*-
"""PCA 1D — GMM07/GMM08 F10 com as funcoes de desenho OOF aprovadas.

--verificar: confere arquivos ja salvos; sem joblib, Parquet, ajuste ou escrita.
--gerar: usa PC1 F10 ja salvo, GMM/RH congelados e MCC F10 ja registrado.
          Apenas calcula campos geometricos para desenho, nunca novos scores de F10.
          Todos os arquivos novos em vencedor/teste_final; OOF e 00..05 intocados.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

ENSAIO = RAIZ / 'outputs/ensaio_pca_gmm_20261029_173123'
MODELO = RAIZ / 'outputs/treinamento_final_pca_10pct_20261029_174241/PCA_1D'
TESTE = ENSAIO / 'vencedor/teste_final'
GEOMETRIA = TESTE / 'dados_geometria'
FIGURAS = TESTE / 'graficos_gerados'
GALERIA = TESTE / 'galeria/complemento_gmm07_gmm08_pca1d_f10.html'
MANIFESTO = TESTE / 'manifesto_complemento_gmm07_gmm08_pca1d_f10.json'

PNG07 = 'GMM07_sensibilidade_1d_comparativo.png'
PNG08 = 'GMM08_mapa_risco_1d_comparativo.png'
ARQUIVOS_GERADOS = (
    PNG07, 'GMM07_malha_PC1.csv', 'GMM07_metricas_comparativas.csv',
    PNG08, 'GMM08_malha_PC1.csv',
)


def exigir(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise RuntimeError(mensagem)


def ler_json(arquivo: Path) -> dict:
    exigir(arquivo.is_file(), f'JSON ausente: {arquivo}')
    resultado = json.loads(arquivo.read_text(encoding='utf-8-sig'))
    exigir(isinstance(resultado, dict), f'JSON incorreto: {arquivo}')
    return resultado


def hash_arquivo(arquivo: Path) -> str:
    exigir(arquivo.is_file(), f'Arquivo ausente: {arquivo}')
    h = sha256()
    with arquivo.open('rb') as f:
        for bloco in iter(lambda: f.read(1 << 20), b''):
            h.update(bloco)
    return h.hexdigest()


def comparar_ids(coords: pd.DataFrame, preds: pd.DataFrame, n: int) -> None:
    exigir(list(coords.columns) == ['linha_original', 'PC1'], 'CSV PC1 com colunas inesperadas.')
    exigir(list(preds.columns) == ['linha_original', 'fold', 'y_true', 'prob_fraude',
                                  'predicao_tau_0_50', 'predicao_tau_mcc_oof'],
           'Predicoes F10 com colunas inesperadas.')
    exigir(len(coords) == len(preds) == n, 'Quantidade de linhas PC1/F10 inesperada.')
    ids = coords['linha_original'].to_numpy()
    exigir(np.issubdtype(ids.dtype, np.integer) and np.all(np.diff(ids) > 0),
           'Indices F10 ausentes, repetidos ou desordenados.')
    exigir(np.array_equal(ids, preds['linha_original'].to_numpy()),
           'Indices F10 divergentes entre geometria e predicoes.')
    exigir((preds['fold'] == 10).all(), 'Predicoes fora do fold F10.')
    exigir(preds['y_true'].isin([0, 1]).all(), 'Alvos F10 invalidos.')
    pc1 = coords['PC1'].to_numpy(dtype=float)
    prob = preds['prob_fraude'].to_numpy(dtype=float)
    exigir(np.isfinite(pc1).all() and np.isfinite(prob).all() and
           ((prob >= 0) & (prob <= 1)).all(), 'Coordenadas/scores invalidos.')


def cortes_fixos(preds: pd.DataFrame, metricas: dict, tau: float) -> tuple[float, float]:
    exigir(np.isfinite(tau) and 0 <= tau <= 1, 'Limiar OOF invalido.')
    exigir(metricas['limiar_otimizado_no_f10'] is False, 'Limiar F10 nao pode ser otimizado.')
    exigir(float(metricas['limiar_mcc_oof_congelado']) == tau, 'Limiar F10/OOF divergente.')
    p = preds['prob_fraude'].to_numpy(dtype=float)
    exigir(preds['predicao_tau_0_50'].isin([0, 1]).all() and
           preds['predicao_tau_mcc_oof'].isin([0, 1]).all(), 'Classes preditas invalidas.')
    exigir(np.array_equal((p >= .5).astype(int), preds['predicao_tau_0_50'].to_numpy()),
           'Corte 0,50 diverge do CSV F10 salvo.')
    exigir(np.array_equal((p >= tau).astype(int), preds['predicao_tau_mcc_oof'].to_numpy()),
           'Corte tau* diverge do CSV F10 salvo.')
    c = metricas['comparacao_limiares']
    exigir(float(c['tau_0_50']['limiar']) == .5 and
           float(c['tau_mcc_oof']['limiar']) == tau, 'Metricas F10 com cortes incorretos.')
    mc0 = float(c['tau_0_50']['mcc'])
    mc1 = float(c['tau_mcc_oof']['mcc'])
    exigir(np.isfinite(mc0) and np.isfinite(mc1) and
           abs(mc0) <= 1 and abs(mc1) <= 1, 'MCCs F10 invalidos.')
    exigir(mc1 == float(metricas['mcc_limiar_oof']), 'MCC tau* F10 inconsistente.')
    return mc0, mc1


def destino_livre() -> None:
    exigir(not GALERIA.exists() and not MANIFESTO.exists(),
           'Complemento GMM07/08 ja iniciado/concluido; preservar arquivos.')
    for nome in ARQUIVOS_GERADOS:
        exigir(not (FIGURAS / nome).exists(),
               f'Arquivo existente, nao sobrescrever: {FIGURAS / nome}')


def verificar() -> dict:
    print('=== PCA 1D — PREPARACAO GMM07 E GMM08 F10 ===', flush=True)
    geom = ler_json(GEOMETRIA / 'manifesto_geometria_f10.json')
    modelo = ler_json(MODELO / 'manifesto_modelo_final.json')
    escolha = ler_json(ENSAIO / 'vencedor/metricas/melhor_configuracao.json')
    esp = ler_json(TESTE / 'manifesto_espelho_graficos_f10.json')
    met = ler_json(TESTE / 'metricas_f10.json')
    p1 = ler_json(TESTE / 'manifesto_complemento_geometria_pca1d_f10.json')
    exigir(geom['dimensao'] == modelo['dimensao_pca'] == 1, 'Modelo nao e PCA 1D.')
    exigir(geom['n'] == esp['n'] == met['n'] == 18523, 'Amostra F10 divergente.')
    exigir(modelo['n_treino_original'] == 166716, 'Treinamento F1–F9 divergente.')
    exigir(geom['tipo'] == 'EXTRACAO_DESCRITIVA_F10_JA_AVALIADO' and
           geom['parquet_target_lido'] is False and
           geom['nova_avaliacao_f10'] is False and geom['novo_treinamento'] is False,
           'Coordenadas F10 nao foram extraidas no protocolo esperado.')
    tau = float(escolha['limiar_mcc_oof'])
    exigir(tau == float(geom['limiar_mcc_oof_congelado']) ==
           float(modelo['limiar_mcc_oof_congelado']) ==
           float(esp['limiar_mcc_congelado_oof']) == float(p1['tau_star_oof']),
           'Tau* OOF nao corresponde ao pacote final.')
    exigir(str(escolha['id_candidato']) == modelo['id_candidato_oof'], 'Candidato OOF mudou.')
    exigir(hash_arquivo(MODELO/'modelo_final_f1_f9.joblib') ==
           modelo['sha256_modelo_final'] == geom['sha256_modelo_final'] == p1['sha256_modelo'],
           'Hash do modelo congelado divergente.')
    exigir(hash_arquivo(TESTE/'predicoes_f10.csv') == esp['predicoes_sha256'] ==
           p1['sha256_predicoes_f10'], 'Predicoes F10 alteradas.')
    exigir(hash_arquivo(TESTE/'metricas_f10.json') == esp['metricas_sha256'],
           'Metricas F10 alteradas.')
    for nome in ('coordenadas_PC1_f10.csv', 'resumo_padronizacao_f10.csv'):
        exigir(hash_arquivo(GEOMETRIA/nome) == geom['arquivos'][nome],
               f'Arquivo de geometria F10 alterado: {nome}')
    for nome, valor in p1['fontes_codigos'].items():
        exigir(hash_arquivo(RAIZ/'src'/nome) == valor,
               f'Codigo original alterado desde Parte 1: {nome}')
    for nome in p1['graficos']:
        exigir(hash_arquivo(FIGURAS/nome) == p1['graficos'][nome],
               f'Figura/CSV Parte 1 alterado: {nome}')
    config = yaml.safe_load((ENSAIO/'configuracao.yaml').read_text(encoding='utf-8-sig'))
    exigir(config['reducao']['metodo'].upper() == 'PCA' and config['pca']['n_components'] == [1],
           'Configuracao PCA 1D incorreta.')
    for nome in ('sensibilidade_fronteira', 'mapa_risco_fronteira'):
        exigir(config['visualizacao']['graficos'][nome]['habilitar'] is True,
               f'Grafico {nome} desativado.')
    exigir(config['visualizacao']['padrao']['formato'].lower() == 'png',
           'Este complemento requer PNG.')
    coords = pd.read_csv(GEOMETRIA/'coordenadas_PC1_f10.csv')
    preds = pd.read_csv(TESTE/'predicoes_f10.csv')
    comparar_ids(coords, preds, geom['n'])
    mc0, mc1 = cortes_fixos(preds, met, tau)
    destino_livre()
    print(f'[APROVADO] PCA 1D, {len(coords)} linhas F10, hash/modelo/indices/tau OK.', flush=True)
    print(f'[APROVADO] MCC F10 ja salvo: corte 0,50 = {mc0:.7f} | tau* OOF = {mc1:.7f}.', flush=True)
    print('[APROVADO] Sem leitura de Parquet, joblib ou novos ajustes.', flush=True)
    return dict(geom=geom, modelo=modelo, escolha=escolha, config=config,
                coords=coords, preds=preds, tau=tau, mcc_fixo=mc0, mcc_tau=mc1)


def rotulos_para_f10(fig) -> None:
    """Altera apenas os textos em memoria antes de salvar; nada em src/ muda."""
    from matplotlib.text import Text
    substituicoes = (
        ('B — Corte ótimo MCC OOF', 'B — Limiar τ* congelado OOF'),
        ('LIMIAR ÓTIMO DO MCC', 'LIMIAR τ* CONGELADO (OOF)'),
        ('MCC OOF', 'MCC F10'),
    )
    for texto in fig.findobj(match=Text):
        antigo = texto.get_text()
        novo = antigo
        for de, para in substituicoes:
            novo = novo.replace(de, para)
        if novo != antigo:
            texto.set_text(novo)
    fig.text(.985, .010, 'F10 • AMOSTRAL • τ* FIXO EM OOF', ha='right',
             fontsize=7.5, color='#8091a6')


@contextmanager
def rotinas_de_desenho_f10(geo, gmm07, tau: float, mcc_fixo: float, mcc_tau: float):
    """Substituicoes temporarias SOMENTE do processo de desenho: restauradas no finally."""
    original_mcc = geo._avaliar_cortes_oof
    original_guardar = geo._guardar
    original_rotulo = gmm07._rotulo_mcc

    def mcc_registrado(avaliacao, limiar):
        exigir(avaliacao.get('origem') == 'f10_amostral_congelado' and
               limiar == tau, 'Origem do MCC ou limiar incorreto.')
        return mcc_fixo, mcc_tau

    def rotulo(avaliacao):
        exigir(avaliacao.get('origem') == 'f10_amostral_congelado',
               'Origem do MCC F10 incorreta.')
        return 'MCC F10', 'MCC da avaliação F10 já salva; τ* congelado em OOF.'

    def guardar(fig, pasta, base, formato, dpi):
        exigir(base in {Path(PNG07).stem, Path(PNG08).stem},
               'Tentativa inesperada de salvar grafico.')
        rotulos_para_f10(fig)
        return original_guardar(fig, pasta, base, formato, dpi)

    geo._avaliar_cortes_oof = mcc_registrado
    geo._guardar = guardar
    gmm07._rotulo_mcc = rotulo
    try:
        yield
    finally:
        geo._avaliar_cortes_oof = original_mcc
        geo._guardar = original_guardar
        gmm07._rotulo_mcc = original_rotulo


def pagina_html(estado: dict) -> str:
    def cartao(nome, titulo):
        return (f'<article><h2>{html.escape(titulo)}</h2><a href="../graficos_gerados/{nome}" target="_blank">'
                f'<img src="../graficos_gerados/{nome}" alt="{html.escape(titulo)}"></a></article>')
    return ('<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>PCA 1D F10 — GMM07 e GMM08</title><style>'
            'body{font:15px system-ui,sans-serif;background:#f5f7fa;color:#172236;max-width:1300px;margin:auto;padding:24px}'
            'article{background:white;border:1px solid #d7dee6;border-radius:10px;padding:14px;margin:20px 0}'
            'img{width:100%;max-height:720px;object-fit:contain}a{color:#245386}'
            '.aviso{background:#e8edf4;border-left:4px solid #355877;padding:13px}'
            '</style></head><body><h1>PCA 1D — GMM07 e GMM08 no F10</h1>'
            f'<p class="aviso">Amostra F10 já avaliada: n={len(estado["preds"]):,}. '
            f'τ*={estado["tau"]:.17g} foi congelado em OOF, sem otimização em F10. '
            f'MCC F10 em τ=0,50: {estado["mcc_fixo"]:.7f}; MCC F10 em τ*: {estado["mcc_tau"]:.7f}. '
            'Mesma rotina visual do OOF, com título/rodapé de origem corrigidos. '
            'A malha é calculada no GMM congelado em F1–F9; as observações sobrepostas são as mesmas do F10. '
            'Nenhuma predição F10, treinamento ou cálculo de nova métrica neste complemento.</p>'
            + cartao(PNG07, 'GMM07 — Sensibilidade da fronteira (F10)')
            + cartao(PNG08, 'GMM08 — Mapa de risco e gaussianas 95% (F10)')
            + '<p>Arquivos CSV das malhas e métricas já registradas em ../graficos_gerados/.</p></body></html>')


def gerar(estado: dict) -> None:
    from src.treinamento_final_pca import carregar_pacote_confiavel
    from src import visualizacao_geometrica as geo
    from src import visualizacao_gmm07 as gmm07

    pacote = carregar_pacote_confiavel(MODELO)
    exigir(pacote.n_treino_original == 166716 and pacote.limiar_oof == estado['tau'],
           'Modelo carregado nao corresponde ao F10 congelado.')
    x = estado['coords']['PC1'].to_numpy(dtype=float)
    coords = pd.DataFrame({'PC1': x})
    data = {'coord': coords, 'y': estado['preds']['y_true'].to_numpy(dtype=np.uint8),
            'scores': estado['preds']['prob_fraude'].to_numpy(dtype=float),
            'gmm': pacote.gmm, 'rh': pacote.risco, 'redutor': pacote.pca,
            'limiar_oof': estado['tau']}
    escolha = {'limiar_mcc_oof': estado['tau']}
    avaliacao = {'origem': 'f10_amostral_congelado'}
    cfg = estado['config']
    dpi = int(cfg['visualizacao']['padrao'].get('dpi', 180))

    TESTE.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.parcial_gmm07_gmm08_f10_', dir=TESTE))
    try:
        with rotinas_de_desenho_f10(geo, gmm07, estado['tau'], estado['mcc_fixo'], estado['mcc_tau']):
            print('[GRAFICO] GMM07 — mesmo padrao OOF; rotulo MCC F10', flush=True)
            info07 = gmm07.gerar(cfg, data, escolha, stage, 'png', dpi, True, avaliacao=avaliacao)
            exigir(info07['arquivo'] == PNG07 and
                   info07['origem_mcc'] == 'f10_amostral_congelado',
                   'GMM07 incorreto.')
            print('[GRAFICO] GMM08 — mesmo padrao OOF; rotulo MCC F10', flush=True)
            geo._fronteira(cfg, data, escolha, stage, 'png', dpi, True, avaliacao=avaliacao)
        caminho = [stage / nome for nome in ARQUIVOS_GERADOS]
        exigir(all(f.is_file() and f.stat().st_size > 0 for f in caminho),
               'Saidas GMM07/GMM08 incompletas; nada publicado.')
        metricas_csv = pd.read_csv(stage/'GMM07_metricas_comparativas.csv')
        exigir(len(metricas_csv) == 2 and
               set(metricas_csv['origem_mcc']) == {'f10_amostral_congelado'},
               'CSV de metricas GMM07 nao identifica corretamente o F10.')
        exigir(np.allclose(metricas_csv['mcc'].to_numpy(dtype=float),
                           [estado['mcc_fixo'], estado['mcc_tau']], rtol=0, atol=1e-15),
               'CSV MCC recalculado ou divergente.')
        hashes = {f.name: hash_arquivo(f) for f in caminho}
        info = {
            'versao': 'GMM07_GMM08_F10_PCA1D_V1',
            'fase': 'teste_final', 'tipo': 'GEOMETRIA_GMM_F10_AMOSTRAL',
            'n_f10': len(x), 'tau_star_oof_congelado': estado['tau'],
            'mcc_f10_0_50_ja_salvo': estado['mcc_fixo'],
            'mcc_f10_tau_oof_ja_salvo': estado['mcc_tau'],
            'sem_treinamento': True, 'sem_nova_avaliacao_f10': True,
            'sem_nova_inferencia_f10': True, 'sem_leitura_parquet': True,
            'sha256_modelo': estado['modelo']['sha256_modelo_final'],
            'sha256_coordenadas_f10': estado['geom']['arquivos']['coordenadas_PC1_f10.csv'],
            'sha256_metricas_f10': hash_arquivo(TESTE/'metricas_f10.json'),
            'sha256_predicoes_f10': hash_arquivo(TESTE/'predicoes_f10.csv'),
            'fontes_codigo_sha256': {nome: hash_arquivo(RAIZ/'src'/nome) for nome in
                                    ('visualizacao_geometrica.py','visualizacao_gmm07.py')},
            'arquivos': hashes,
            'nota': 'MCC lido do JSON F10 ja avaliado. Fronteiras calculadas sobre o GMM congelado em F1–F9.',
            'pendente': ['PRE01_F10'],
        }
        (stage/MANIFESTO.name).write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
        (stage/GALERIA.name).write_text(pagina_html(estado), encoding='utf-8')
        destino_livre()
        FIGURAS.mkdir(parents=True, exist_ok=True)
        GALERIA.parent.mkdir(parents=True, exist_ok=True)
        for f in caminho:
            os.rename(f, FIGURAS / f.name)
        os.rename(stage/GALERIA.name, GALERIA)
        os.rename(stage/MANIFESTO.name, MANIFESTO)
    finally:
        if stage.exists():
            if any(stage.iterdir()):
                print('[AVISO] Arquivos parciais preservados em:', stage, flush=True)
            else:
                stage.rmdir()
    print('[CONCLUIDO] GMM07 + GMM08; 2 PNG + 3 CSV, sem alterar 00..05/OOF.', flush=True)
    print('[GALERIA]', GALERIA, flush=True)
    print('[MANIFESTO]', MANIFESTO, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    modo = ap.add_mutually_exclusive_group(required=True)
    modo.add_argument('--verificar', action='store_true')
    modo.add_argument('--gerar', action='store_true')
    args = ap.parse_args()
    estado = verificar()
    if args.gerar:
        gerar(estado)


if __name__ == '__main__':
    main()
