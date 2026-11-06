# -*- coding: utf-8 -*-
"""Ajuste visual GMM07/GMM08 1D (PCA/t-SNE) sobre F10, após a V2.

Na raiz do TCC_EXPERIMENTAL_V3:
    python .\\INSTALAR_AJUSTE_GMM07_GMM08_1D_V3.py --checar
    python .\\INSTALAR_AJUSTE_GMM07_GMM08_1D_V3.py --aplicar

- Remove SOMENTE nos gráficos 1D: subtítulo F10 e rodapé de contagens.
- Acrescenta ao GMM08 1D intervalos centrais 95% de cada gaussiana FULL,
  com faixas transparentes, bordas tracejadas e legenda dos componentes.
- Mantém o painel de resumo de riscos, legendas de limiares e suas linhas,
  as previsões e os scores, o teste F10 congelado e os gráficos 2D/3D.
- Faz backup do único fonte modificado, não toca em dados, modelos,
  YAML, métricas, outputs passados nem resultados reais.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile

TAG_V2 = 'F10_BOUNDARY_CERTIFICATE_V2'
TAG_V3 = 'GMM_1D_CLEAN_GAUSS95_V3'
ALVO = Path('src/gmm_contornos_quatro_niveis.py')


def trocar_uma_vez(codigo: str, anterior: str, posterior: str, nome: str) -> str:
    qtd = codigo.count(anterior)
    if qtd != 1:
        raise RuntimeError(
            f'INCOMPATIVEL [{nome}]: esperado exatamente 1 bloco, encontrado {qtd}. '
            'Nada foi alterado; envie a mensagem para conferirmos a versão local.'
        )
    return codigo.replace(anterior, posterior, 1)


def transformar(codigo: str) -> str:
    if TAG_V3 in codigo:
        raise RuntimeError('Ajuste V3 já consta no arquivo. Não reaplicar.')
    if TAG_V2 not in codigo:
        raise RuntimeError('É necessário instalar primeiro o pacote EXCLUSIVO F10 V2.')

    # O 1D possui covariância FULL de dimensão 1x1. A seção de
    # concentração central 95% é [mu-z*sigma, mu+z*sigma] com z=Phi^-1(.975).
    # Cores tab10 iguais às elipses 95% exibidas pelo renderizador 2D.
    auxiliar = '''def _intervalos_gaussianos_95_1d(ax, gm, xx):
    """GMM08 1D: equivalentes unidimensionais às elipses FULL 95% do 2D.

    Faixas centrais de CADA gaussiana, não um intervalo de confiança
    estimado dos dados nem um intervalo 95% da mistura inteira.
    """
    # GMM_1D_CLEAN_GAUSS95_V3
    from scipy.stats import norm

    if gm.covariance_type != 'full' or gm.means_.shape[1] != 1:
        raise ValueError('Intervalos gaussianos 1D exigem GMM de covariância FULL em 1D.')

    z = float(norm.ppf(.975))
    handles = []
    for k in range(gm.n_components):
        media = float(gm.means_[k, 0])
        variancia = float(gm.covariances_[k, 0, 0])
        if not (np.isfinite(media) and np.isfinite(variancia) and variancia > 0):
            raise ValueError(f'Parâmetros inválidos na gaussiana G{k+1}.')
        delta = z * np.sqrt(variancia)
        esquerda, direita = media - delta, media + delta
        cor = plt.get_cmap('tab10')(k % 10)
        ax.axvspan(esquerda, direita, fc=cor, ec='none', alpha=.10, zorder=.3)
        ax.axvline(esquerda, color=cor, ls='--', lw=1.05, alpha=.8, zorder=1.3)
        ax.axvline(direita, color=cor, ls='--', lw=1.05, alpha=.8, zorder=1.3)
        handles.append(Line2D([0], [0], color=cor, ls='--', lw=2,
                              label=f'G{k+1} 95%: [{esquerda:.2f}, {direita:.2f}]'))

    # A adição das gaussianas não deve estender o domínio visual além
    # da malha F10 congelada; regiões externas são simplesmente recortadas.
    ax.set_xlim(float(xx[0]), float(xx[-1]))
    ax.legend(handles=handles, loc='upper right',
              title='Gaussianas — intervalo central 95%',
              fontsize=8.0, title_fontsize=8.2, framealpha=.94)


'''
    codigo = trocar_uma_vez(
        codigo,
        'def _painel_png(config, data, escolha, destino, ext, dpi, grafico, origem, mcc, mccnome,\n',
        auxiliar+'def _painel_png(config, data, escolha, destino, ext, dpi, grafico, origem, mcc, mccnome,\n',
        'inserir intervalo gaussiano 1D',
    )
    codigo = trocar_uma_vez(
        codigo,
        "            ax.set_xlabel(nomes[0]); ax.set_ylabel('Score de fraude s(x)')",
        "            ax.set_xlabel(nomes[0]); ax.set_ylabel('Score de fraude s(x)')\n"
        "            if grafico == 'GMM08':\n"
        "                _intervalos_gaussianos_95_1d(ax, gm, xx)",
        'desenhar 95% apenas no mapa de risco GMM08 1D',
    )
    codigo = trocar_uma_vez(
        codigo,
        "    fig.text(.5, .934, f'{origem} | {d}D FULL | K={gm.n_components} | {desc}',\n"
        "             ha='center', fontsize=10, color='#50627a')",
        "    # V3: o subtitulo F10 foi retirado SOMENTE dos paineis 1D.\n"
        "    if d != 1:\n"
        "        fig.text(.5, .934, f'{origem} | {d}D FULL | K={gm.n_components} | {desc}',\n"
        "                 ha='center', fontsize=10, color='#50627a')",
        'retirar subtitulo F10 1D',
    )
    codigo = trocar_uma_vez(
        codigo,
        "    fig.text(.5, .048, msg, ha='center', fontsize=8.2, color='#52657d')",
        "    # V3: rodape de contagens omitido SOMENTE nos paineis 1D.\n"
        "    if d != 1:\n"
        "        fig.text(.5, .048, msg, ha='center', fontsize=8.2, color='#52657d')",
        'retirar rodape 1D',
    )
    codigo = trocar_uma_vez(
        codigo,
        "                        top=.72, bottom=.16, wspace=.28 if d==3 else .33)",
        "                        top=.72,\n"
        "                        bottom=(.12 if d==1 else .16),\n"
        "                        wspace=.28 if d==3 else .33)",
        'reaproveitar espaco vertical 1D',
    )
    ast.parse(codigo, filename=str(ALVO))
    compile(codigo, str(ALVO), 'exec')
    return codigo


def verificar(raiz: Path):
    alvo = raiz / ALVO
    for obrigatorio in (alvo, raiz/'src/gmm_fronteira_f10.py', raiz/'src/visualizacao_geometrica.py'):
        if not obrigatorio.is_file():
            raise FileNotFoundError(f'Arquivo ausente: {obrigatorio}. Confira a pasta do projeto.')
    texto = alvo.read_text(encoding='utf-8-sig')
    resultado = transformar(texto)
    print('[COMPATÍVEL] Patch V3 reconheceu a implementação F10 V2 instalada.')
    print('[COMPATÍVEL] Novo código validado por AST e compilação Python.')
    return alvo, resultado


def aplicar(raiz: Path):
    alvo, novo = verificar(raiz)
    data = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup = raiz/'outputs'/f'backup_gmm07_gmm08_1d_visual_v3_{data}'
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(alvo, backup/alvo.name)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=alvo.parent, prefix='.gmm1d_v3_',
                                         suffix='.incompleto', delete=False) as handle:
            tmp = Path(handle.name)
            handle.write(novo)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, alvo)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
    print('[APLICADO] src/gmm_contornos_quatro_niveis.py')
    print('[BACKUP]', backup)
    print('[PRESERVADOS] GMM07/GMM08 2D/3D, OOF, F10, YAML, métricas, modelos e PNGs antigos.')
    print('IMPORTANTE: PNGs/HTMLs existentes não são substituídos; novos resultados refletem a V3.')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modo = cli.add_mutually_exclusive_group(required=True)
    modo.add_argument('--checar', action='store_true')
    modo.add_argument('--aplicar', action='store_true')
    args = cli.parse_args()
    raiz = Path(__file__).resolve().parent
    if args.aplicar:
        aplicar(raiz)
    else:
        verificar(raiz)
        print('[OK] Somente leitura; nenhum arquivo modificado.')


if __name__ == '__main__':
    main()
