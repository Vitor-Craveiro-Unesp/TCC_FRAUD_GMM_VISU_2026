"""Paleta deterministica para K variavel, sem alterar as primeiras 10/20 cores."""
from __future__ import annotations

from colorsys import hsv_to_rgb
import matplotlib.pyplot as plt


def cor_gaussiana(indice: int, *, paleta: str = 'tab10'):
    if type(indice) is not int or indice < 0:
        raise ValueError('Indice da gaussiana deve ser inteiro nao negativo.')
    if paleta not in ('tab10', 'tab20'):
        raise ValueError('Paleta deve ser tab10 ou tab20.')
    limite = 10 if paleta == 'tab10' else 20
    if indice < limite:
        return plt.get_cmap(paleta)(indice)
    # Evita reutilizar a mesma cor quando K excede a paleta categorica.
    return (*hsv_to_rgb((indice * 0.61803398875) % 1, 0.72, 0.76), 1.0)
