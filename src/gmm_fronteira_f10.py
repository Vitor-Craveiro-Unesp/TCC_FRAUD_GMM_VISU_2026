# -*- coding: utf-8 -*-
"""Diagnostico de fronteira do GMM congelado sobre observacoes F10.

Sem fit, sem selecao de candidato, sem otimizacao de limiar.
Usa o tau* OOF ja escolhido. Com pelo menos uma previsao positiva e uma
negativa, o segmento no espaco reduzido cruza s(x)=tau*, por continuidade.
O cruzamento e pontual, e nao deve ser confundido com uma isosuperficie
resolvida numericamente por uma malha finita.
"""
from __future__ import annotations

import numpy as np


def _score(modelo, riscos, pontos):
    coords = np.asarray(pontos, dtype=np.float64).reshape(-1, modelo.means_.shape[1])
    resultado = np.asarray(modelo.predict_proba(coords), dtype=np.float64) @ np.asarray(riscos, dtype=np.float64)
    if not np.all(np.isfinite(resultado)):
        raise ValueError('Scores do modelo GMM final invalidos.')
    return resultado


def diagnosticar(modelo, riscos, coords, scores_f10, tau):
    """Devolve prova numerica de um cruzamento no dominio de F10, se existir.

    O cruzamento so e declarado quando ha valores dos dois lados de tau.
    O mesmo modelo/riscos usados para a malha devem reproduzir os scores F10.
    Nao utiliza rotulos verdadeiros de F10 para selecionar nada.
    """
    X = np.asarray(coords, dtype=np.float64)
    scores = np.asarray(scores_f10, dtype=np.float64).ravel()
    r = np.asarray(riscos, dtype=np.float64).ravel()
    tau = float(tau)
    if (X.ndim != 2 or len(X) != len(scores) or len(X) == 0 or
            X.shape[1] not in (1, 2, 3) or not np.all(np.isfinite(X)) or
            not np.all(np.isfinite(scores)) or not np.all((0 <= scores) & (scores <= 1)) or
            not 0 <= tau <= 1):
        raise ValueError('Entradas F10 invalidas para diagnostico de fronteira.')

    positivos = np.flatnonzero(scores >= tau)
    negativos = np.flatnonzero(scores < tau)
    info = {'n_f10': int(len(X)), 'positivos_f10': int(len(positivos)),
            'negativos_f10': int(len(negativos)),
            'limiar_congelado_oof': tau,
            'score_minimo_observado_f10': float(scores.min()),
            'score_maximo_observado_f10': float(scores.max()),
            'ponto_fronteira': None, 'ponto_positivo': None,
            'ponto_negativo': None, 'cruzamento_confirmado': False,
            'nota': 'Nao ha scores dos dois lados do limiar no F10.'}

    # Consistencia entre probabilidades salvas do F10 e o mesmo GMM do renderizador.
    # Seleciona extremos, fronteiras e amostra regular; nao precisa recalcular tudo.
    auditoria = np.unique(np.r_[np.linspace(0, len(X)-1, min(80, len(X)), dtype=int),
                                    np.argmin(scores), np.argmax(scores),
                                    positivos[:min(32, len(positivos))],
                                    negativos[:min(32, len(negativos))]])
    recalculado = _score(modelo, r, X[auditoria])
    if not np.allclose(recalculado, scores[auditoria], rtol=1e-5, atol=2e-7):
        maior = float(np.max(np.abs(recalculado - scores[auditoria])))
        raise ValueError(f'Scores F10 nao reproduzidos pelo modelo final: erro maximo={maior:.12g}.')

    if not len(positivos) or not len(negativos):
        return info

    # Positivo mais proximo do limiar (para cruzamento local); negativo mais
    # proximo na geometria do embedding em coordenadas normalizadas por amplitude.
    ip = int(positivos[np.argmin(scores[positivos] - tau)])
    escala = np.ptp(X, axis=0)
    escala = np.where(escala > 0, escala, 1.)
    dist = np.sum(np.square((X[negativos] - X[ip]) / escala), axis=1)
    ineg = int(negativos[np.argmin(dist)])
    ponto_pos = X[ip].copy()
    ponto_neg = X[ineg].copy()
    p_pos, p_neg = _score(modelo, r, np.stack([ponto_pos, ponto_neg]))
    # Guardas conservadoras se o score salvo foi arredondado junto do limiar.
    if p_pos < tau and not np.isclose(p_pos, tau, rtol=0, atol=2e-10):
        raise ValueError('Ponto previsto positivo nao supera tau* ao pontuar com GMM final.')
    if p_neg >= tau and not np.isclose(p_neg, tau, rtol=0, atol=2e-10):
        raise ValueError('Ponto previsto negativo nao ficou abaixo de tau* no GMM final.')
    if np.isclose(p_pos, tau, rtol=0, atol=2e-10):
        raiz = ponto_pos.copy()
    elif np.isclose(p_neg, tau, rtol=0, atol=2e-10):
        raiz = ponto_neg.copy()
    else:
        pos, neg = ponto_pos.copy(), ponto_neg.copy()
        # 65 bissecoes independentes de rotulos: garantia numerica de score=limiar.
        for _ in range(65):
            meio = .5 * (pos + neg)
            prob = float(_score(modelo, r, meio[None, :])[0])
            if prob >= tau:
                pos = meio
            else:
                neg = meio
        raiz = .5 * (pos + neg)
    residual = abs(float(_score(modelo, r, raiz[None, :])[0]) - tau)
    if residual > max(1e-10, 1e-7 * max(tau, 1e-6)):
        raise RuntimeError(f'Bissecao nao confirmou fronteira tau*: residual={residual:.12g}')

    info.update(ponto_fronteira=raiz, ponto_positivo=ponto_pos,
                ponto_negativo=ponto_neg, cruzamento_confirmado=True,
                residual_tau=float(residual), indice_positivo=ip, indice_negativo=ineg,
                nota='Cruzamento exato de tau* confirmado entre previsoes F10 de classes opostas; curvas de malha dependem da resolucao.')
    return info
