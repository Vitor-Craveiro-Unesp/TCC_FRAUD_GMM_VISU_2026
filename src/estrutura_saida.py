"""Contrato unico de diretorios do TCC (layout 2).

A raiz guarda somente configuracao.yaml, manifesto.json e subpastas.
Separar geometria de desenvolvimento da avaliacao independente F10.
"""
from pathlib import Path

VERSAO_LAYOUT = 2


def candidato(ensaio, identificador, configuracao):
    """Candidato de ID estavel, permitindo grades com varios parametros por K."""
    k = int(configuracao['gmm']['n_components'])
    return Path(ensaio) / 'candidatos' / f'K{k:02d}_{identificador}'


def vencedor(ensaio):
    return Path(ensaio) / 'vencedor'


def metricas_vencedor(ensaio):
    return vencedor(ensaio) / 'metricas'


def predicoes_vencedor(ensaio):
    return vencedor(ensaio) / 'oof' / 'predicoes.npz'


def imagens_oof(ensaio):
    return vencedor(ensaio) / 'visualizacao' / 'oof'


def imagens_geometria(ensaio):
    return vencedor(ensaio) / 'visualizacao' / 'geometria_desenvolvimento'


def galeria(ensaio):
    return vencedor(ensaio) / 'visualizacao' / 'galeria'


def teste_final(ensaio):
    return vencedor(ensaio) / 'teste_final'
