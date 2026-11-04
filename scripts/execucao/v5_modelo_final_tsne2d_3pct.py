# -*- coding: utf-8 -*-
"""V5 — Reajuste final TSNE+GMM 2D (3%) EXCLUSIVAMENTE F1..F9.

Reutiliza o vencedor OOF aprovado da V4 e a funcao de persistencia V2.
Nunca invoca treinamento OOF, inferencia F10, geracao de graficos F10
nem executor completo. F10 fica reservado para a V6.

Uso:
  python -B -m scripts.execucao.v5_modelo_final_tsne2d_3pct --planejar
  python -B -m scripts.execucao.v5_modelo_final_tsne2d_3pct --executar
  python -B -m scripts.execucao.v5_modelo_final_tsne2d_3pct --validar
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import logging
from pathlib import Path
import sys
import traceback

METODO = 'tsne'
PASTA_OOF_APROVADA = 'ensaio_tsne_gmm_20261103_135207'
SAIDA_MODELO = 'modelo_final_f1_f9'
LOCK_RELATIVO = 'outputs/.ensaio_real_3pct_2d_tsne.lock'


def exigir(condicao, motivo):
    if not condicao:
        raise RuntimeError(motivo)


def sha_arquivo(caminho):
    h = sha256()
    with Path(caminho).open('rb') as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def conferir_base():
    """Guarda V3 + selecao OOF; nenhuma leitura de amostras do F10."""
    from scripts.execucao import v4_oof_real_tsne2d_3pct as v4
    from scripts.execucao import experimentos_reais_3pct_2d as drv

    drv_v4, cfg = v4.conferir_homologacao()
    exigir(drv_v4 is drv, 'O executor homologado V4 e o V2 divergem.')
    pasta, parcial = drv.localizar_experimento(cfg, METODO)
    exigir(parcial is None and pasta is not None,
           'OOF incompleto ou nao localizado. Nao reexecutar: audite a V4.')
    exigir(pasta.name == PASTA_OOF_APROVADA,
           'Pasta OOF diferente da aprovada na V4; V5 bloqueada: ' + str(pasta))
    exigir(not drv.LIBERACAO_V3.exists(),
           'Executor completo foi liberado; V5 requer F10 bloqueado.')

    man = drv.ler_json(pasta / 'manifesto.json')
    exigir(man.get('status') == 'concluido_06_03' and
           man.get('visualizacoes') == 'concluidas' and
           man.get('tipo') == 'ensaio',
           'Manifesto OOF V4 nao concluido ou nao e ensaio.')
    exigir((pasta / 'vencedor' / 'visualizacao' / 'galeria' / 'index.html').is_file(),
           'Galeria OOF da V4 ausente.')

    info = drv.validar_oof_f1f9(pasta, cfg, METODO)
    escolha = info['escolha']
    exigir(len(info['ids']) == 50013,
           'Contagem OOF diferente das 50.013 previsoes aprovadas na V4.')
    exigir(escolha.get('id_candidato') == 'd517359b0f6645',
           'Candidato OOF mudou desde a V4.')
    exigir(float(escolha.get('limiar_mcc_oof')) > 0 and
           escolha.get('teste_final_usado') is False,
           'Limiar OOF invalido ou F10 declarado na selecao.')
    exigir(int(escolha['configuracao']['reducao']['n_components']) == 2 and
           escolha['configuracao']['metodo'] == 'TSNE',
           'Vencedor nao corresponde a TSNE 2D.')
    return drv, cfg, pasta, info


def conferir_conteudo_teste_final(pasta):
    """Orquestrador OOF cria teste_final/ vazia. V5 permite apenas modelo final."""
    base = pasta / 'vencedor' / 'teste_final'
    if not base.exists():
        return base, base / SAIDA_MODELO, 'AUSENTE'
    exigir(base.is_dir(), 'teste_final existe mas nao e diretorio: ' + str(base))
    nomes = sorted(p.name for p in base.iterdir())
    inesperados = [nome for nome in nomes if nome != SAIDA_MODELO]
    exigir(not inesperados,
           'teste_final contem arquivos/pastas inesperados: ' + repr(inesperados) +
           '. Nao apagar nem alterar; envie os nomes para analise.')
    modelo = base / SAIDA_MODELO
    if modelo.exists():
        exigir(modelo.is_dir(), 'Caminho do modelo final nao e diretorio.')
        return base, modelo, 'MODELO_PRESENTE'
    return base, modelo, 'DIRETORIO_VAZIO'


def validar_modelo(drv, cfg, pasta, info):
    _base, modelo, estado = conferir_conteudo_teste_final(pasta)
    exigir(estado == 'MODELO_PRESENTE', 'Modelo final ainda nao foi produzido.')
    man = drv.validar_modelo(modelo, cfg, info['escolha'], METODO,
                             len(info['ids']))
    exigir(man.get('versao') is not None and
           man.get('tipo') == 'TREINAMENTO_FINAL_TSNE_F1_F9' and
           man.get('metodo') == 'TSNE' and
           int(man.get('dimensao_tsne')) == 2 and
           man.get('teste_f10_executado') is False and
           int(man['n_treino_original']) == len(info['ids']) and
           int(man['n_ajuste_gmm']) > 0,
           'Manifesto do modelo incompatível com F1..F9 / TSNE 2D.')

    origem = man.get('proveniencia')
    exigir(isinstance(origem, dict), 'Proveniencia do modelo ausente.')
    for chave in ('sha256_parquet', 'sha256_folds', 'sha256_yaml'):
        exigir(origem.get(chave) == info['manifesto'].get(chave),
               'Hash de proveniencia do modelo diverge: ' + chave)
    exigir(Path(origem.get('experimento_oof', '')).resolve() == pasta.resolve(),
           'Proveniencia aponta para outro experimento OOF.')
    exigir(float(man['limiar_mcc_oof_congelado']) ==
           float(info['escolha']['limiar_mcc_oof']),
           'Limiar do modelo nao foi congelado a partir do OOF.')

    arquivo = modelo / 'modelo_final_f1_f9.joblib'
    digest = sha_arquivo(arquivo)
    exigir(digest == man['sha256_modelo_final'],
           'Hash fisico do pacote final diverge do manifesto.')
    print('=== V5 | MODELO FINAL TSNE + GMM 2D | F1..F9 ===', flush=True)
    print('V5_STATUS=MODELO_FINAL_TSNE_2D_3PCT_APROVADO', flush=True)
    print('V5_EXPERIMENTO=' + str(pasta), flush=True)
    print('V5_MODELO=' + str(arquivo), flush=True)
    print('V5_MANIFESTO=' + str(modelo / 'manifesto_modelo_final.json'), flush=True)
    print('V5_SHA256_MODELO=' + digest, flush=True)
    print('V5_CANDIDATO_OOF=' + str(info['escolha']['id_candidato']), flush=True)
    print('V5_N_TREINO_F1_F9=' + str(man['n_treino_original']), flush=True)
    print('V5_N_GMM_RUS=' + str(man['n_ajuste_gmm']), flush=True)
    print('V5_TAU_MCC_OOF=' + f'{float(man["limiar_mcc_oof_congelado"]):.7f}', flush=True)
    print('V5_F10_AVALIADO=NAO', flush=True)
    print('V5_OOF_REEXECUTADO=NAO', flush=True)
    print('V5_PCA_2D_EXECUTADO=NAO', flush=True)
    print('V5_GALERIA_OOF=' + str(pasta/'vencedor/visualizacao/galeria/index.html'), flush=True)
    return man


def planejar(drv, cfg, pasta, info):
    _base, destino, estado = conferir_conteudo_teste_final(pasta)
    print('=== V5 | PLANEJAR MODELO FINAL TSNE 2D ===', flush=True)
    print('V5_OOF_EXPERIMENTO=' + str(pasta), flush=True)
    print('V5_OOF_VALIDADO=SIM', flush=True)
    print('V5_PREDICOES_OOF=' + str(len(info['ids'])), flush=True)
    print('V5_VENCEDOR=' + str(info['escolha']['id_candidato']), flush=True)
    print('V5_TAU_CONGELADO=' + f'{float(info["escolha"]["limiar_mcc_oof"]):.7f}', flush=True)
    print('V5_DESTINO_MODELO=' + str(destino), flush=True)
    print('V5_ESTADO_MODELO=' + estado, flush=True)
    print('V5_TREINAMENTO_EXECUTADO=NAO', flush=True)
    print('V5_F10_RESERVADO=SIM', flush=True)
    print('V5_PLANEJAMENTO=APROVADO', flush=True)
    if estado == 'MODELO_PRESENTE':
        validar_modelo(drv, cfg, pasta, info)


def executar(drv, cfg, pasta, info):
    _base, destino, estado = conferir_conteudo_teste_final(pasta)
    if estado == 'MODELO_PRESENTE':
        print('[V5] Modelo existente; somente validar, sem repetir ajuste.', flush=True)
        validar_modelo(drv, cfg, pasta, info)
        return

    lock = drv.ROOT / LOCK_RELATIVO
    exigir(not lock.exists(),
           'Lock de outra execucao TSNE 2D encontrado: ' + str(lock))
    # Bloqueia simultaneamente o executor 2D completo e novas V4/V5.
    with lock.open('x', encoding='utf-8') as arq:
        arq.write('V5: ajuste final F1..F9 do vencedor OOF. F10 nao avaliado.\n')
    try:
        _base, _destino, estado_atual = conferir_conteudo_teste_final(pasta)
        exigir(estado_atual != 'MODELO_PRESENTE',
               'Modelo apareceu durante aquisicao do lock. Nao sobrescrever.')
        print('[V5] Reajuste final somente F1..F9; sem executar OOF/F10.', flush=True)
        resultado = drv.executar_modelo(pasta, cfg, info, METODO)
        exigir(resultado == destino and resultado.is_dir(),
               'Destino inesperado na persistencia do modelo final.')
        validar_modelo(drv, cfg, pasta, info)
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    op = parser.add_mutually_exclusive_group(required=True)
    op.add_argument('--planejar', action='store_true')
    op.add_argument('--executar', action='store_true')
    op.add_argument('--validar', action='store_true')
    args = parser.parse_args(argv)
    try:
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        drv, cfg, pasta, info = conferir_base()
        if args.planejar:
            planejar(drv, cfg, pasta, info)
        elif args.executar:
            executar(drv, cfg, pasta, info)
        else:
            validar_modelo(drv, cfg, pasta, info)
        print('EXIT_CODE_V5=0', flush=True)
        return 0
    except Exception as exc:
        print('V5_STATUS=BLOQUEADO_SEM_F10', file=sys.stderr, flush=True)
        print('V5_MOTIVO=' + str(exc), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
