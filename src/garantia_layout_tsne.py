# -*- coding: utf-8 -*-
"""Contrato permanente de renderizacao t-SNE para os YAMLs do TCC.

Apenas inspeciona arquivos Python locais; nao importa modulos de geometria,
nao le dados, nao faz fit, nao altera YAML nem experimentos.

- TSNE02 e TSNE14: layout corrigido aplica-se a 1D, 2D e 3D.
- TSNE13: garantia de layout corrigido existente somente para 1D;
  visualizacao 2D e distinta e precisa da validacao visual propria.
"""
from __future__ import annotations
import ast
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
MARCADOR = 'PATCH_LAYOUT_TSNE1D_OOF_20261103'
REQUISITOS = {
    'src/visualizacao_tsne_pdf.py': {
        'trustworthiness_perplexity': ('fig.tight_layout(rect=(.015, .145, .985, .91))',),
        'diagnostico_kl': ('fig.subplots_adjust(left=.055, right=.945, top=.81, bottom=.30, wspace=.085)',),
    },
    'src/visualizacao_geometrica.py': {
        '_tsne_histograma_1d': ('fig.subplots_adjust(left=.10, right=.96, top=.83, bottom=.24)',),
    },
}


def _verificar_arquivo(path_rel: str, funcoes: dict) -> None:
    path = RAIZ / path_rel
    if not path.is_file():
        raise RuntimeError(f'[LAYOUT TSNE] Fonte ausente: {path_rel}')
    codigo = path.read_text(encoding='utf-8-sig')
    try:
        arvore = ast.parse(codigo, filename=str(path))
    except SyntaxError as exc:
        raise RuntimeError(f'[LAYOUT TSNE] Fonte invalida: {path_rel}: {exc}') from exc
    linhas = codigo.splitlines()
    nomes = {n.name: n for n in arvore.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for funcao, evidencias in funcoes.items():
        n = nomes.get(funcao)
        if n is None:
            raise RuntimeError(f'[LAYOUT TSNE] Funcao ausente: {path_rel}:{funcao}')
        trecho = '\n'.join(linhas[n.lineno - 1:n.end_lineno])
        if MARCADOR not in trecho or any(e not in trecho for e in evidencias):
            raise RuntimeError(
                f'[LAYOUT TSNE] Renderizador nao corrigido: {path_rel}:{funcao}. '
                'Instale a correcao visual aprovada antes de iniciar novo YAML.'
            )


def verificar_layout_tsne(config: dict, *, fase: str = 'antes_execucao') -> bool:
    """Falha fechada para YAML t-SNE se qualquer um dos 3 layouts for legado.

    Os metodos PCA nao dependem desses renderizadores e sao ignorados.
    """
    metodo = str(config.get('reducao', {}).get('metodo', '')).upper()
    if metodo != 'TSNE':
        return True
    for rel, exigencias in REQUISITOS.items():
        _verificar_arquivo(rel, exigencias)
    n = config.get('tsne', {}).get('n_components')
    if isinstance(n, (list, tuple)) and len(n) == 1:
        n = n[0]
    print(f'[GARANTIA LAYOUT TSNE] {fase}: TSNE02/TSNE14 corrigidos; '
          f'TSNE13 1D corrigido; dimensao YAML={n}; FONTES ATUAIS=SIM.', flush=True)
    if n is not None and int(n) >= 2:
        print('[GARANTIA LAYOUT TSNE] TSNE13 2D/3D usa outra rotina; '
              'validar visualmente o respectivo scatter apos gerar.', flush=True)
    return True
