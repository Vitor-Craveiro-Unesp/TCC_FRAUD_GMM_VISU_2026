# -*- coding: utf-8 -*-
r"""V5: mostrar por padrao os elipsoides FULL 95% no HTML 3D do GMM08.

Uso na raiz de TCC_EXPERIMENTAL_V3 (.venv ativo):
  python .\INSTALAR_GMM08_ELIPSOIDES_HTML_3D_V5.py --checar
  python .\INSTALAR_GMM08_ELIPSOIDES_HTML_3D_V5.py --aplicar

Altera EXCLUSIVAMENTE src/gmm_contornos_quatro_niveis.py; backup em outputs.
NAO altera resultados antigos, F10, OOF, modelos, YAML ou PNGs existentes.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile

ARQUIVO = Path('src/gmm_contornos_quatro_niveis.py')
MARCADOR = 'GMM08_HTML_3D_ELIPSOIDES_FULL_95_V5'


# Este bloco é a implementação V2/V3/V4.2 efetivamente presente no renderizador.
ORIGINAL = '''            color=f'hsl({(k*69)%360},60%,45%)'
            fig.add_trace(go.Surface(x=P[:,:,0],y=P[:,:,1],z=P[:,:,2],
                colorscale=[[0,color],[1,color]],opacity=.14,showscale=False,
                name=f'Gaussiana G{k+1}, 95%',visible='legendonly'))'''

CORRIGIDO = '''            # GMM08_HTML_3D_ELIPSOIDES_FULL_95_V5
            # Cor idêntica às seções gaussianas 95% nos PNGs (tab10).
            from matplotlib.colors import to_hex
            color = to_hex(plt.get_cmap('tab10')(k % 10))
            # Elipsoide FULL: (x-mu)^T Sigma^-1 (x-mu) = chi2_3(0.95).
            # Antes o trace era 'legendonly': não aparecia ao abrir o HTML.
            fig.add_trace(go.Surface(x=P[:,:,0], y=P[:,:,1], z=P[:,:,2],
                colorscale=[[0,color],[1,color]], opacity=.23,
                showscale=False, showlegend=True, visible=True,
                name=f'G{k+1} — elipsoide FULL 95%',
                hovertemplate=f'Gaussiana G{k+1} | elipsoide central FULL 95%<extra></extra>'))
            # Centro visível facilita comparar componentes e suas elipsoides.
            fig.add_trace(go.Scatter3d(
                x=[float(gm.means_[k,0])], y=[float(gm.means_[k,1])],
                z=[float(gm.means_[k,2])], mode='markers',
                marker={'size':5,'color':color,'symbol':'diamond',
                        'line':{'color':'#101827','width':1}},
                name=f'Centro G{k+1}',showlegend=False,
                hovertemplate=f'Centro da gaussiana G{k+1}<extra></extra>'))'''


def transformar(texto: str) -> str:
    ast.parse(texto)
    if MARCADOR in texto:
        raise ValueError('A V5 ja consta neste arquivo. Nao reaplicar.')
    inicio = texto.find('\ndef _html3d(')
    if inicio < 0:
        inicio = texto.find('def _html3d(')
    if inicio < 0:
        raise ValueError('Nao existe a funcao _html3d esperada no modulo.')
    fim = texto.find('\ndef gerar(', inicio + 1)
    if fim < 0:
        raise ValueError('Nao foi encontrado o fim de _html3d; nao modificar.')
    bloco = texto[inicio:fim]
    if "if grafico=='GMM08':" not in bloco:
        raise ValueError('Bloco do GMM08 nao encontrado dentro de _html3d.')
    if bloco.count(ORIGINAL) != 1:
        raise ValueError(f'Trecho original esperado nao encontrado exatamente uma vez dentro de _html3d (encontrado {bloco.count(ORIGINAL)}). Nao modificar.')
    if "if np.any(eig<=0): continue" not in bloco:
        raise ValueError('Guarda da matriz de covariancia mudou; nao modificar.')
    # Covariâncias inválidas não devem ser ocultadas silenciosamente.
    bloco = bloco.replace("            if np.any(eig<=0): continue",
        "            if np.any(eig<=0):\n                raise ValueError(f'Covariancia FULL G{k+1} nao positiva definida: nao e possivel desenhar 95%.')", 1)
    bloco = bloco.replace(ORIGINAL, CORRIGIDO, 1)
    novo = texto[:inicio] + bloco + texto[fim:]
    ast.parse(novo)
    compile(novo, str(ARQUIVO), 'exec')
    return novo


def validar(raiz: Path):
    alvo=raiz/ARQUIVO
    if not alvo.is_file():
        raise FileNotFoundError(f'Arquivo nao encontrado: {alvo}')
    texto=alvo.read_text(encoding='utf-8-sig')
    novo=transformar(texto)
    print('[COMPATIVEL] Funcao _html3d reconhecida; codigo V5 compilado.')
    print('[GMM08] Cada componente FULL 95% aparecera em 3D, com legenda, transparencia e centro.')
    print('[PRESERVADOS] GMM04, GMM07, PNGs 1D/2D/3D, superficies tau*, F10, OOF.')
    return alvo, novo


def aplicar(raiz:Path):
    alvo,novo=validar(raiz)
    backup=raiz/'outputs'/('backup_gmm08_elipsoides_html_v5_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True,exist_ok=False)
    shutil.copy2(alvo, backup/alvo.name)
    temp=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='\n',
              dir=alvo.parent,prefix='.gmm08_v5_',suffix='.incompleto',delete=False) as f:
            temp=Path(f.name)
            f.write(novo)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, alvo)
    finally:
        if temp is not None and temp.exists(): temp.unlink()
    print('[APLICADO]',ARQUIVO)
    print('[BACKUP]',backup)
    print('[AVISO] HTMLs ja gerados sao preservados; V5 aparece nas novas geracoes.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modo=parser.add_mutually_exclusive_group(required=True)
    modo.add_argument('--checar',action='store_true')
    modo.add_argument('--aplicar',action='store_true')
    args=parser.parse_args()
    raiz=Path(__file__).resolve().parent
    if args.checar:
        validar(raiz)
        print('[OK] Somente leitura; nenhuma alteracao.')
    else: aplicar(raiz)


if __name__=='__main__':
    main()
