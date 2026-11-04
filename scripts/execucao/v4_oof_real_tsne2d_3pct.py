# -*- coding: utf-8 -*-
"""V4: somente OOF real TSNE+GMM 2D, 3%, F1..F9.

Usa o YAML congelado e as fontes homologadas V2/V3.
NUNCA chama ajustar_final_f1_f9(), executar_modelo(), executar_f10(),
executar_graficos() ou executar_galeria() do executor 2D.
Nao cria HOMOLOGACAO_V3_APROVADA.json (o executor completo permanece bloqueado).

Uso: python -B -m scripts.execucao.v4_oof_real_tsne2d_3pct --planejar
     python -B -m scripts.execucao.v4_oof_real_tsne2d_3pct --executar
     python -B -m scripts.execucao.v4_oof_real_tsne2d_3pct --validar
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import traceback

METODO = 'tsne'


def exigir(condicao, motivo):
    if not condicao:
        raise RuntimeError(motivo)


def conferir_homologacao():
    # Ambas sao rotinas de LEITURA, sem fit, sem Parquet e sem F10.
    from scripts.validacao import homologar_v3_2d as v3
    from scripts.execucao import experimentos_reais_3pct_2d as drv

    configuracoes, _ = v3.inspecionar_v2()
    v3.validar_previa(configuracoes)
    cfg, grade = drv.construir_config(METODO)
    exigir(cfg == configuracoes[METODO],
           'YAML TSNE congelado diverge da configuracao homologada V3.')
    exigir(len(grade) == 2 and all(c['metodo'] == 'TSNE' and
           int(c['reducao']['n_components']) == 2 for c in grade),
           'Grade OOF TSNE 2D diferente da homologada (2 candidatos).')
    exigir(not drv.LIBERACAO_V3.exists(),
           'Marcador de liberacao completa encontrado. V4 exige bloqueio do F10.')
    from src.garantia_layout_tsne import verificar_layout_tsne
    verificar_layout_tsne(cfg, fase='executor_3pct_2d')
    print('V3_TECNICA=INTEGRA_17_PREVIAS', flush=True)
    print('V3_VISUAL=APROVADA_PELO_USUARIO', flush=True)
    print('EXECUTOR_COMPLETO_OOF_MODELO_F10=BLOQUEADO', flush=True)
    return drv, cfg


def conferir_sem_teste_final(drv, cfg):
    final, parcial = drv.localizar_experimento(cfg, METODO)
    for pasta in (final, parcial):
        if pasta is None:
            continue
        pasta_f10 = Path(pasta) / 'vencedor' / 'teste_final'
        # O orquestrador OOF cria antecipadamente teste_final/ VAZIA.
        # A existencia da pasta NAO significa que F10 tenha sido utilizado.
        # Recusar qualquer arquivo/subpasta: V4 so pode validar F1..F9.
        if pasta_f10.exists():
            exigir(pasta_f10.is_dir(),
                   'Caminho teste_final nao e diretorio: ' + str(pasta_f10))
            presentes = sorted(p.name for p in pasta_f10.iterdir())
            exigir(not presentes,
                   'Pasta teste_final NAO VAZIA: V4 bloqueada. '
                   'Inspecione antes de continuar. Caminho=' + str(pasta_f10) +
                   ' | entradas=' + repr(presentes))
    return final, parcial


def validar_oof_existente(drv, cfg):
    pasta, parcial = conferir_sem_teste_final(drv, cfg)
    exigir(pasta is not None and parcial is None,
           'OOF TSNE 2D nao esta concluido. Examine o ensaio em andamento antes de validar.')
    info = drv.validar_oof_f1f9(pasta, cfg, METODO)
    manifesto = drv.ler_json(pasta / 'manifesto.json')
    exigir(manifesto.get('status') == 'concluido_06_03' and
           manifesto.get('visualizacoes') == 'concluidas',
           'OOF ou visualizacao ainda nao foi finalizado.')
    html = pasta / 'vencedor' / 'visualizacao' / 'galeria' / 'index.html'
    exigir(html.is_file(), 'Galeria OOF ausente: '+str(html))
    esc = info['escolha']
    exigir(esc['teste_final_usado'] is False and
           esc['configuracao']['metodo'] == 'TSNE' and
           int(esc['configuracao']['reducao']['n_components']) == 2,
           'Vencedor OOF nao corresponde ao t-SNE 2D; F10 deve permanecer isolado.')
    print('=== V4 | RESULTADOS OOF REAL TSNE + GMM 2D ===', flush=True)
    print('V4_STATUS=OOF_TSNE_2D_3PCT_APROVADO', flush=True)
    print('V4_OOF_FOLDS=F1_F9', flush=True)
    print('V4_PREDICOES_OOF=' + str(len(info['ids'])), flush=True)
    print('V4_CANDIDATO_VENCEDOR=' + str(esc['id_candidato']), flush=True)
    print('V4_TAU_MCC_OOF=' + f'{float(esc["limiar_mcc_oof"]):.7f}', flush=True)
    if 'score_oof' in esc:
        print('V4_SCORE_OOF=' + f'{float(esc["score_oof"]):.7f}', flush=True)
    print('V4_EXPERIMENTO=' + str(pasta), flush=True)
    print('V4_GALERIA_OOF=' + str(html), flush=True)
    print('V4_F10_AVALIADO=NAO', flush=True)
    print('V4_MODELO_FINAL_REAJUSTADO=NAO', flush=True)
    print('V4_PCA_2D_EXECUTADO=NAO', flush=True)
    return pasta


def modo_planejar(drv, cfg):
    print('=== V4 | PLANEJAMENTO SOMENTE OOF TSNE 2D ===', flush=True)
    drv.imprimir_plano(METODO)
    final, parcial = conferir_sem_teste_final(drv, cfg)
    print('V4_OOF_EXISTENTE=' + str(final or ''), flush=True)
    print('V4_OOF_PARCIAL=' + str(parcial or ''), flush=True)
    print('V4_PROXIMA_OPERACAO=OOF_F1_F9', flush=True)
    print('V4_F10_PERMANECE_BLOQUEADO=SIM', flush=True)
    print('V4_PLANEJAMENTO=APROVADO', flush=True)


def modo_executar(drv, cfg):
    """Unica chamada que treina: o orquestrador OOF, sem F10 ou modelo final."""
    pasta_lock = drv.ROOT / 'outputs'
    pasta_lock.mkdir(exist_ok=True)
    lock = pasta_lock / '.ensaio_real_3pct_2d_tsne.lock'
    exigir(not lock.exists(),
           'Lock de t-SNE 2D presente. Verifique se outra execucao esta ativa: '+str(lock))
    # Exclusao atomica: se houver outra sessao, open('x') falha sem treinar.
    with lock.open('x', encoding='utf-8') as f:
        f.write('V4: somente OOF F1..F9 TSNE GMM 2D 3%. Sem F10.\n')
    try:
        final, parcial = conferir_sem_teste_final(drv, cfg)
        if final is None:
            print('V4_INICIANDO=OOF_REAL_TSNE_2D_3PCT', flush=True)
            print('V4_ESCOPO=9_FOLDS_2_CANDIDATOS_SEM_F10', flush=True)
        else:
            print('V4_REUTILIZANDO_OOF_EXISTENTE=' + str(final), flush=True)
        # O metodo do executor V2 chama unicamente src.orquestracao_oof.
        # Nunca chamar drv.executar('tsne') ou drv._executar_sem_lock('tsne').
        pasta = drv.executar_oof(METODO, cfg)
        exigir(pasta.is_dir(), 'Orquestrador nao devolveu pasta OOF concluida.')
        return validar_oof_existente(drv, cfg)
    finally:
        # Remove somente lock criado nesta execucao; nao remove locks alheios.
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selecao = parser.add_mutually_exclusive_group(required=True)
    selecao.add_argument('--planejar', action='store_true')
    selecao.add_argument('--executar', action='store_true')
    selecao.add_argument('--validar', action='store_true')
    args = parser.parse_args(argv)
    try:
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        drv, cfg = conferir_homologacao()
        if args.planejar:
            modo_planejar(drv, cfg)
        elif args.executar:
            modo_executar(drv, cfg)
        else:
            validar_oof_existente(drv, cfg)
        return 0
    except Exception as exc:
        print('V4_STATUS=BLOQUEADO_SEM_LIBERAR_F10', file=sys.stderr, flush=True)
        print('V4_MOTIVO=' + str(exc), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
