# -*- coding: utf-8 -*-
"""Hotfix pontual: retira a barra de cores de cima da malha GMM07/GMM08.

Executar na raiz do TCC:
  python CORRIGIR_LAYOUT_GMM07_GMM08_4_NIVEIS.py --checar
  python CORRIGIR_LAYOUT_GMM07_GMM08_4_NIVEIS.py --aplicar

Nao reexecuta modelo ou toca em resultados. Faz backup do fonte alterado.
"""
from pathlib import Path
import argparse
import datetime
import sys

REL = Path('src/gmm_contornos_quatro_niveis.py')
ANCHORS = (
    (
        "        titulo += f' | {nomes[z]} fixada na media das coordenadas da etapa = {medias[z]:.5g}'",
        "        titulo += f'\\n{nomes[z]} = {medias[z]:.4g} (media da etapa)'",
    ),
    (
        "    if d > 1:\n"
        "        from matplotlib.cm import ScalarMappable\n"
        "        fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'), ax=axs, shrink=.78,\n"
        "                     pad=.025, label='Risco estimado na malha')\n"
        "    fig.subplots_adjust(left=.07 if d!=3 else .055, right=.96 if d==1 else .945,\n"
        "                        top=.72, bottom=.16, wspace=.33)",
        "    # HOTFIX_GMM_QUATRO_NIVEIS_BARRA_FORA_DOS_EIXOS_20261104\n"
        "    fig.subplots_adjust(left=.07 if d!=3 else .055,\n"
        "                        right=.96 if d==1 else (.885 if d==2 else .895),\n"
        "                        top=.72, bottom=.16, wspace=.28 if d==3 else .33)\n"
        "    if d > 1:\n"
        "        from matplotlib.cm import ScalarMappable\n"
        "        # Eixo independente: nao ocupa espaco de nenhum painel de risco.\n"
        "        cax = fig.add_axes([.92, .20, .013, .50])\n"
        "        fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),\n"
        "                     cax=cax, label='Risco estimado na malha')",
    ),
)

def main():
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--checar',action='store_true')
    g.add_argument('--aplicar',action='store_true')
    a=p.parse_args()
    path=Path.cwd()/REL
    if not path.is_file():
        print('[BLOQUEADO] Fonte ausente: '+str(path), file=sys.stderr)
        return 2
    original=path.read_bytes()
    s=original.decode('utf-8-sig').replace('\r\n','\n')
    if 'HOTFIX_GMM_QUATRO_NIVEIS_BARRA_FORA_DOS_EIXOS_20261104' in s:
        print('[OK] Hotfix de layout ja instalado. Nao reaplicar.')
        return 0
    try:
        for old,new in ANCHORS:
            n=s.count(old)
            if n!=1:
                raise ValueError(f'Ancora encontrada {n} vezes, esperado 1: {old[:80]}')
            s=s.replace(old,new,1)
        compile(s,str(path),'exec')
    except Exception as e:
        print('[BLOQUEADO] Codigo diferente do esperado; nada modificado: '+str(e),file=sys.stderr)
        return 2
    print('[OK] Hotfix de layout compativel e sintaxe validada.')
    if a.checar:
        print('[CHECAR] Nenhum arquivo alterado.')
        return 0
    backup=Path.cwd()/'outputs'/('backup_layout_4_niveis_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True,exist_ok=False)
    (backup/path.name).write_bytes(original)
    newline='\r\n' if b'\r\n' in original else '\n'
    payload=s.replace('\n',newline).encode('utf-8')
    if original.startswith(b'\xef\xbb\xbf'):payload=b'\xef\xbb\xbf'+payload
    try: path.write_bytes(payload)
    except Exception:
        path.write_bytes(original)
        raise
    print('[APLICADO] Barra de cores fora dos painéis; títulos 3D em duas linhas.')
    print('[BACKUP]',backup)
    print('Nenhum treinamento, YAML, F10 ou resultado antigo alterado.')
    return 0

if __name__=='__main__':raise SystemExit(main())
