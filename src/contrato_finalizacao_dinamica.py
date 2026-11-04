"""Contrato cientifico e operacional para a finalizacao PCA/t-SNE dinamica.

SEM IO, SEM FIT, SEM F10: validacao compartilhada por seis finalizadores.
Uma modalidade (dimensao) por execucao; K, n_init e razao RUS podem formar grade.
"""
from __future__ import annotations

import math
from hashlib import sha256
import json

from src.amostragem_f10 import fracao_amostra_f10


def _unica_dimensao(valor, nome: str) -> int:
    if isinstance(valor, list):
        if len(valor) != 1:
            raise ValueError(f'{nome}: finalizacao exige uma unica dimensao por YAML.')
        valor = valor[0]
    if type(valor) is not int or valor not in (1, 2, 3):
        raise ValueError(f'{nome} deve ser um inteiro em 1, 2 ou 3.')
    return valor


def _erro_dimensao(metodo: str, esperada: int, detalhe: str) -> ValueError:
    if metodo == 'TSNE' and esperada == 1:
        return ValueError(f'Finalizador TSNE aceita somente dimensao 1; 2D/3D proibido: {detalhe}.')
    if metodo == 'TSNE' and esperada == 2:
        return ValueError(f'Finalizador TSNE aceita somente dimensao 2: {detalhe}.')
    return ValueError(f'Finalizador {metodo} exige dimensao {esperada}: {detalhe}.')


def validar_configuracao_finalizacao(config: dict, *, metodo: str | None = None,
                                     dimensao: int | None = None):
    """Rejeita combinacoes cientificamente invalidas ANTES do primeiro fit.

    Nao exige K4, RUS16, n_init5 ou valores historicos de fracao/seed.
    A viabilidade do tamanho da RUS depende dos dados e e checada pelo
    orquestrador apos selecionar F1..F9, antes de abrir o experimento.
    """
    if not isinstance(config, dict):
        raise ValueError('Configuracao YAML deve ser um objeto.')
    real = str(config.get('reducao', {}).get('metodo', '')).upper()
    if real not in ('PCA', 'TSNE') or (metodo is not None and real != metodo):
        raise ValueError(f'Metodo de finalizacao incompatível: {real!r}.')
    try:
        dim = _unica_dimensao(config.get('pca' if real == 'PCA' else 'tsne', {}).get('n_components'),
                              f'{real.lower()}.n_components')
    except ValueError as exc:
        if dimensao is not None:
            raise _erro_dimensao(real, dimensao, str(exc)) from exc
        raise
    if dimensao is not None and dim != dimensao:
        raise _erro_dimensao(real, dimensao, f'recebeu {dim}D')
    frac = config.get('dados', {}).get('fracao_base')
    if type(frac) not in (float, int) or not math.isfinite(frac) or not 0 < frac <= 1:
        raise ValueError('dados.fracao_base precisa pertencer ao intervalo (0,1].')
    fracao_amostra_f10(config)
    divisao = config.get('divisao', {})
    cv = config.get('validacao_cruzada', {})
    if (divisao.get('total_folds') != 10 or divisao.get('qual_fold_teste') != 10
        or divisao.get('quantidade_folds_teste') != 1 or divisao.get('estratificada') is not True
        or cv.get('estrategia') != 'rotacao_folds_restantes'
        or cv.get('evitar_vazamento') is not True):
        raise ValueError('Finalizacao exige F1-F9 e F10 reservado, estratificado e sem vazamento.')
    seed = config.get('experimento', {}).get('semente')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('experimento.semente deve ser inteiro em [0, 2**32-1].')
    if config.get('amostragem', {}).get('aplicar_somente_treino') is not True:
        raise ValueError('RUS deve ocorrer apenas no treinamento.')
    if config.get('execucao', {}).get('proteger_resultados') is not True:
        raise ValueError('execucao.proteger_resultados=true e obrigatorio na finalizacao.')

    from src.validacao_oof import grade_candidatos_oof
    from src.gmm import GMMNaoSupervisionado
    from src.reducao import RedutorPCA
    from src.reducao_tsne import RedutorTSNE
    grade = list(grade_candidatos_oof(config))
    if not grade:
        raise ValueError('A grade OOF esta vazia.')
    for candidato in grade:
        param = candidato['gmm']
        if param['covariance_type'] != 'full':
            raise ValueError('Metodologia exige gmm.covariance_type=full em TODA a grade.')
        GMMNaoSupervisionado(config, param)  # valida K, n_init, max_iter etc sem fit
        if candidato['modo'] not in ('RUS', 'NORMAL'):
            raise ValueError('Amostragem precisa ser RUS ou NORMAL.')
        red = candidato['reducao']
        if red['n_components'] != dim:
            raise ValueError('Dimensao do candidato diverge da finalizacao.')
        if real == 'PCA':
            RedutorPCA(config, red)
        else:
            RedutorTSNE(config, red)
    return grade


def validar_vencedor(config: dict, escolha: dict, *, metodo: str,
                     dimensao: int | None = None) -> dict:
    grade = validar_configuracao_finalizacao(config, metodo=metodo, dimensao=dimensao)
    if not isinstance(escolha, dict) or escolha.get('teste_final_usado') is not False:
        raise ValueError('Vencedor OOF invalido: F10 nao pode selecionar candidato.')
    if escolha.get('status') not in ('selecionado_em_ensaio', 'selecionado_em_oof_integral'):
        raise ValueError('Status OOF do vencedor invalido.')
    fracao = config['dados']['fracao_base']
    if fracao < 1 and escolha.get('status') != 'selecionado_em_ensaio':
        raise ValueError('OOF amostral precisa de status selecionado_em_ensaio.')
    escolhido = escolha.get('configuracao')
    if escolhido not in grade:
        raise ValueError('Vencedor nao pertence a grade OOF congelada.')
    esperado = sha256(json.dumps(escolhido, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()[:14]
    if escolha.get('id_candidato') != esperado:
        raise ValueError('Identificador do vencedor OOF divergente da configuracao.')
    tau = escolha.get('limiar_mcc_oof')
    if type(tau) not in (int, float) or not math.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError('Limiar MCC OOF congelado invalido.')
    return escolhido


def validar_viabilidade_oof(folds_dev, y_dev, config, *, grade=None):
    """Verifica minimos de amostra ANTES do primeiro fit (so F1..F9).

    Nao modifica RUS nem escolhe um candidato. Confronta todas as combinacoes
    com o numero de registros que elas efetivamente receberiam no treinamento.
    """
    import numpy as np
    f, y = np.asarray(folds_dev), np.asarray(y_dev)
    if f.shape != y.shape or f.ndim != 1 or set(np.unique(f)) != set(range(1, 10)):
        raise ValueError('Preflight cientifico exige somente folds de F1 a F9.')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Preflight exige alvo binario, sem F10.')
    candidatos = grade if grade is not None else validar_configuracao_finalizacao(config)
    for fold in range(1, 10):
        mask = f != fold
        npos = int(np.count_nonzero(y[mask] == 1))
        nneg = int(np.count_nonzero(y[mask] == 0))
        if not npos or not nneg:
            raise ValueError(f'F{fold}: treinamento sem ambas as classes.')
        ntrain = npos + nneg
        for c in candidatos:
            if c['modo'] == 'RUS':
                n_gmm = npos + min(nneg, int(c['razao']) * npos)
            else:
                n_gmm = ntrain
            k = int(c['gmm']['n_components'])
            if n_gmm < max(2, k):
                raise ValueError(f'F{fold}: GMM K={k} exige pelo menos {max(2,k)} linhas '
                                 f'apos {c["modo"]}; disponiveis {n_gmm}.')
            if c['metodo'] == 'TSNE' and c['reducao']['perplexity'] >= ntrain:
                raise ValueError(f'F{fold}: t-SNE perplexity={c["reducao"]["perplexity"]} '
                                 f'exige menos que {ntrain} registros de treino.')
    return True
