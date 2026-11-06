# -*- coding: utf-8 -*-
r"""Cria COPIA visual do HTML GMM08 3D já gerado, com elipsoides 95% visíveis.

O HTML original, scores, modelo, F10, YAML e manifestos são preservados.
Não precisa executar OOF/F10 ou gerar outra malha. Só ajusta apresentação
HTML nas propriedades de traces Plotly que já contêm geometria FULL 95%.

Comandos (PowerShell na raiz TCC_EXPERIMENTAL_V3):
  python .\PREVISUALIZAR_GMM08_HTML_3D_V5.py --listar
  python .\PREVISUALIZAR_GMM08_HTML_3D_V5.py --html "outputs\...\GMM08_mapa_risco_3d_comparativo_INTERATIVA.html"
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re

NOME = 'GMM08_mapa_risco_3d_comparativo_INTERATIVA.html'
# O Plotly serializa os traces com name/opacity/showscale/visible nesta ordem.
# Substituir somente os elipsoides GMM08 preexistentes, nao as isosuperficies tau*.
PADRAO = re.compile(
    r'"name":"Gaussiana G(?P<k>\d+), 95%","opacity":0\.14,'
    r'"showscale":false,"visible":"legendonly"'
)


def listar(raiz: Path) -> list[Path]:
    return sorted((raiz/'outputs').rglob(NOME), key=lambda p:p.stat().st_mtime, reverse=True)


def criar_previa(raiz: Path, html: Path) -> Path:
    if not html.is_file():
        raise FileNotFoundError(f'HTML não encontrado: {html}')
    if html.name != NOME:
        raise ValueError('Esperado arquivo '+NOME)
    conteudo=html.read_text(encoding='utf-8-sig')
    novo, n=PADRAO.subn(lambda m: (
        f'"name":"G{m.group("k")} — elipsoide FULL 95%",'
        '"opacity":0.23,"showlegend":true,"showscale":false,"visible":true'
    ), conteudo)
    if n<1:
        if 'G1 — elipsoide FULL 95%' in conteudo:
            raise ValueError('Este HTML já parece conter a V5; não precisa criar prévia.')
        raise ValueError('Não reconheci os elipsoides antigos em legendonly. Arquivo preservado.')
    saida=raiz/'outputs'/('validacao_html_gmm08_gaussianas_95_v5_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    saida.mkdir(parents=True,exist_ok=False)
    arq=saida/('GMM08_F10_3D_PREVIA_ELIPSOIDES_FULL_95.html')
    arq.write_text(novo,encoding='utf-8')
    print(f'[PREVIA] Elipsoides reativados: {n}')
    print(f'[GERADO] {arq}')
    print('[PRESERVADO] HTML original, resultados F10, CSVs, metricas e manifestos.')
    print('Obs.: esta previa reexibe os elipsoides preexistentes; novas geracoes V5 também mostram os centros Gk.')
    return arq


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    m=ap.add_mutually_exclusive_group(required=True)
    m.add_argument('--listar',action='store_true')
    m.add_argument('--html',type=Path)
    args=ap.parse_args()
    raiz=Path(__file__).resolve().parent
    if args.listar:
        arqs=listar(raiz)
        if not arqs:
            print('[VAZIO] Nenhum HTML GMM08 3D encontrado em outputs.')
        for i,p in enumerate(arqs,1):
            print(f'{i:02d}. {p.relative_to(raiz)}')
        return
    entrada=args.html if args.html.is_absolute() else raiz/args.html
    criar_previa(raiz,entrada)


if __name__=='__main__':
    main()
