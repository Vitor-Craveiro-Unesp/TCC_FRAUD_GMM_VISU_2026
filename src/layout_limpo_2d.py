"""Politica de apresentacao para imagens 2D do TCC.

Nao altera curvas, coordenadas, valores, probabilidades, eixos ou metricas.
So atualiza titulos e remove notas do canvas (fig.text), conservando textos
internos dos eixos, como valores KL, tempo e legendas.
"""
from __future__ import annotations

from pathlib import Path


TITULOS = {
    "TSNE02_trustworthiness_vs_perplexity": "Preservação de vizinhanças | Trustworthiness × Perplexity",
    "13_tsne_mapa_TSNE1_TSNE2": "Projeção TSNE1 × TSNE2",
    "14_tsne_diagnostico": "Diagnóstico KL / otimização",
    "12_pca_projecao_PC1_PC2": "Projeção PC1 × PC2",
}


def limpar(fig, destino):
    """Normaliza a apresentacao antes de salvar, sem modificar os dados.

    Os nomes TSNE02/14 nao incluem a dimensao no arquivo; neles, aplicar
    a limpeza so quando os proprios textos da figura identificam 2D.
    Assim nao altera o 1D ja aprovado nem o futuro 3D.
    """
    stem = Path(destino).stem
    if stem not in TITULOS:
        return False

    if stem in {"TSNE02_trustworthiness_vs_perplexity", "14_tsne_diagnostico"}:
        blocos = [t.get_text() for t in fig.texts]
        for ax in fig.axes:
            blocos.extend(t.get_text() for t in ax.texts)
        texto = " ".join(blocos).lower().replace(" ", "")
        # A marca de dimensao aparece nos rodapes e cartoes originais.
        if "dimensão2d" not in texto and "embedding:2d" not in texto:
            return False

    titulo = TITULOS[stem]
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_text(titulo)

    # Os cartoes/legendas dos eixos sao preservados; elimina apenas notas
    # de pagina inseridas com fig.text() (subtitulos e rodapes).
    for t in list(fig.texts):
        if t is not suptitle:
            t.remove()

    # Evitar titulos duplicados ou prefixos tecnicos nos eixos principais.
    axes_com_titulo = [ax for ax in fig.axes if ax.get_title()]
    if suptitle is not None:
        for ax in axes_com_titulo:
            ax.set_title("")
    elif axes_com_titulo:
        axes_com_titulo[0].set_title(titulo)
        for ax in axes_com_titulo[1:]:
            ax.set_title("")
    elif fig.axes:
        fig.axes[0].set_title(titulo)

    return True
