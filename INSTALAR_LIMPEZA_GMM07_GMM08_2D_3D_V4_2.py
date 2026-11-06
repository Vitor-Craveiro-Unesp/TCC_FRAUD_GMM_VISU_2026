# -*- coding: utf-8 -*-
"""V4.2: correção de layout GMM07/GMM08 para 2D e cortes 3D.

Compatível com fontes F10 V2 + V3 que já têm barra de cores em eixo próprio
(`cax`), e com variantes antigas que utilizam `ax=axs`.

Uso (raiz do TCC_EXPERIMENTAL_V3; ambiente .venv ativado):
    python .\\INSTALAR_LIMPEZA_GMM07_GMM08_2D_3D_V4_2.py --checar
    python .\\INSTALAR_LIMPEZA_GMM07_GMM08_2D_3D_V4_2.py --aplicar

Só altera src/gmm_contornos_quatro_niveis.py (após backup).
Não lê F10, não treina nem substitui PNGs/HTMLs antigos.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile

ALVO = Path('src/gmm_contornos_quatro_niveis.py')
TAG = 'GMM_2D_3D_LIMPO_F10_V4_2'
V2 = 'F10_BOUNDARY_CERTIFICATE_V2'


def _replace(src: str, old: str, new: str, nome: str) -> str:
    n = src.count(old)
    if n != 1:
        raise ValueError(f'Incompatível em {nome}: esperada 1 ocorrência; encontrei {n}. Nada foi modificado.')
    return src.replace(old, new, 1)


def _transformar(src: str) -> tuple[str, str]:
    if TAG in src:
        raise ValueError('V4.2 já foi aplicada. Não instale novamente.')
    if 'GMM_2D_3D_LIMPO_F10_V4' in src:
        raise ValueError('Uma V4 anterior já consta no código. Não reaplicar por segurança.')
    if V2 not in src:
        raise ValueError('É necessária a versão EXCLUSIVO F10 V2 instalada.')
    ast.parse(src, filename=str(ALVO))

    sub_v3 = """    # V3: o subtitulo F10 foi retirado SOMENTE dos paineis 1D.
    if d != 1:
        fig.text(.5, .934, f'{origem} | {d}D FULL | K={gm.n_components} | {desc}',
                 ha='center', fontsize=10, color='#50627a')"""
    sub_v2 = """    fig.text(.5, .934, f'{origem} | {d}D FULL | K={gm.n_components} | {desc}',
             ha='center', fontsize=10, color='#50627a')"""
    if sub_v3 in src:
        src = _replace(src, sub_v3,
                       f'    # {TAG}: subtítulo removido dos PNGs 1D/2D/3D.', 'subtítulo V3')
    else:
        src = _replace(src, sub_v2,
                       f'    # {TAG}: subtítulo removido dos PNGs 1D/2D/3D.', 'subtítulo V2')

    footer_v3 = """    # V3: rodape de contagens omitido SOMENTE nos paineis 1D.
    if d != 1:
        fig.text(.5, .048, msg, ha='center', fontsize=8.2, color='#52657d')"""
    footer_v2 = """    fig.text(.5, .048, msg, ha='center', fontsize=8.2, color='#52657d')"""
    if footer_v3 in src:
        src = _replace(src, footer_v3,
                       f'    # {TAG}: rodapé visual removido; métricas e dados preservados.', 'rodapé V3')
    else:
        src = _replace(src, footer_v2,
                       f'    # {TAG}: rodapé visual removido; métricas e dados preservados.', 'rodapé V2')

    src = _replace(src,
        "    fig.text(.5, .884, global_note, ha='center', fontsize=9.3, color='#24364c')",
        "    fig.text(.5, (.884 if d == 1 else .932), global_note, ha='center', fontsize=9.3, color='#24364c')",
        'resumo superior')
    src = _replace(src,
        "    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .85),",
        "    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .85 if d == 1 else .890),",
        'legenda superior')

    # A V3 distribuída tem top=.72 e bottom diferente em 1D.
    # Em nenhuma variante alterar os números da disposição 1D.
    layouts = (
        (
            "                        top=.72,\n"
            "                        bottom=(.12 if d==1 else .16),\n"
            "                        wspace=.28 if d==3 else .33)",
            "                        top=(.72 if d==1 else .77),\n"
            "                        bottom=(.12 if d==1 else .10),\n"
            "                        wspace=.28 if d==3 else .33)",
            'V3 distribuída',
        ),
        (
            "                        top=(.75 if d==1 else .72),\n"
            "                        bottom=(.12 if d==1 else .16),\n"
            "                        wspace=.28 if d==3 else .33)",
            "                        top=(.75 if d==1 else .77),\n"
            "                        bottom=(.12 if d==1 else .10),\n"
            "                        wspace=.28 if d==3 else .33)",
            'V3 interna',
        ),
        (
            "                        top=.72, bottom=.16, wspace=.28 if d==3 else .33)",
            "                        top=(.72 if d==1 else .77),\n"
            "                        bottom=(.16 if d==1 else .10),\n"
            "                        wspace=.28 if d==3 else .33)",
            'V2',
        ),
    )
    compativeis = [(a, b, nome) for a, b, nome in layouts if src.count(a) == 1]
    if len(compativeis) != 1:
        raise ValueError('Layout não corresponde a V2/V3 conhecida; não modificar.')
    antigo, novo, variante = compativeis[0]
    src = _replace(src, antigo, novo, 'margens '+variante)

    # A CORREÇÃO DESTE ERRO: aceitar duas formas V2/V3 de barra de cores.
    # Variante A: colorbar ainda depende de ax=axs; movê-la para cax externo.
    antiga_barra = (
        "    if d > 1:\n"
        "        from matplotlib.cm import ScalarMappable\n"
        "        fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'), ax=axs, shrink=.78,\n"
        "                     pad=.025, label='Risco estimado na malha')\n"
    )
    ancora = (
        "                        wspace=.28 if d==3 else .33)\n"
        "    # Resumo: pontos positivos da malha p/ cada corte, sem refazer classificacoes oficiais"
    )
    if antiga_barra in src:
        src = _replace(src, antiga_barra,
            '    # V4.2: barra deslocada após subplots_adjust.\n',
            'barra antiga ax=axs')
        src = _replace(src, ancora,
            "                        wspace=.28 if d==3 else .33)\n"
            "    if d > 1:\n"
            "        from matplotlib.cm import ScalarMappable\n"
            "        # Barra de cores externa, sem encolher os subplots.\n"
            "        cax = fig.add_axes([.923, .19 if d==3 else .20, .012 if d==3 else .015, .53])\n"
            "        fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),\n"
            "                     cax=cax, label='Risco estimado na malha')\n"
            "    # Resumo: pontos positivos da malha p/ cada corte, sem refazer classificacoes oficiais",
            'reposicionar barra de cores',
        )
        variante_barra = 'ax=axs convertida para eixo externo'
    else:
        # Variante B: V2 instalada já usa fig.add_axes + cax. Era este o caso
        # ignorado pelos instaladores V4/V4.1 e a causa do segundo traceback.
        cax_n = src.count('cax = fig.add_axes(')
        uses_cax = ('cax=cax' in src or 'cax = cax' in src)
        if cax_n != 1 or not uses_cax:
            raise ValueError('Barra de cores não é nem ax=axs nem cax conhecida. Não modificar.')
        bloco = src[src.index('def _painel_png('):src.index('def _html3d(')]
        if not (bloco.index('fig.subplots_adjust(') < bloco.index('cax = fig.add_axes(') < bloco.index('fig.colorbar(')):
            raise ValueError('cax não está após subplots_adjust. Não modificar.')
        if 'ax=axs' in bloco:
            raise ValueError('Duas formas de barra de cores ao mesmo tempo; não modificar.')
        variante_barra = 'cax externo existente preservado (sem duplicação)'

    ast.parse(src, filename=str(ALVO))
    compile(src, str(ALVO), 'exec')
    return src, variante+' / '+variante_barra


def _checar(raiz: Path) -> tuple[Path, str]:
    alvo=raiz/ALVO
    if not alvo.is_file():
        raise FileNotFoundError(f'Arquivo não encontrado: {alvo}')
    for dep in ('src/gmm_fronteira_f10.py','src/visualizacao_geometrica.py'):
        if not (raiz/dep).is_file():
            raise FileNotFoundError('Dependência ausente: '+str(raiz/dep))
    novo, variante=_transformar(alvo.read_text(encoding='utf-8-sig'))
    print('[COMPATÍVEL] F10 V2/V3:',variante)
    print('[VALIDADO] Sintaxe AST e compilação; 1D, cálculos de risco e gaussianas 95% preservados.')
    return alvo, novo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    grupo=parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument('--checar', action='store_true')
    grupo.add_argument('--aplicar', action='store_true')
    args=parser.parse_args()
    raiz=Path(__file__).resolve().parent
    alvo,novo=_checar(raiz)
    if args.checar:
        print('[OK] Somente leitura: arquivo original inalterado.')
        return
    backup=raiz/'outputs'/('backup_gmm2d3d_visual_v4_2_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True,exist_ok=False)
    shutil.copy2(alvo,backup/alvo.name)
    temporario=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',newline='\n',dir=alvo.parent,
                                         prefix='.gmm_v4_2_',suffix='.incompleto',delete=False) as f:
            temporario=Path(f.name)
            f.write(novo)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporario,alvo)
    finally:
        if temporario is not None and temporario.exists():temporario.unlink()
    print('[APLICADO] src/gmm_contornos_quatro_niveis.py')
    print('[BACKUP]',backup)
    print('[PRESERVADO] Resultados OOF/F10, YAML, modelos, CSVs e gráficos anteriores.')


if __name__=='__main__':
    main()
