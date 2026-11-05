# -*- coding: utf-8 -*-
r"""Instala GMM07/GMM08 4 niveis no TCC_EXPERIMENTAL_V3, sem treinamento.

Extrair ZIP na RAIZ DO TCC e executar (PowerShell, .venv ativada):
 python .\INSTALAR_GMM07_GMM08_4_NIVEIS.py --checar
 python .\INSTALAR_GMM07_GMM08_4_NIVEIS.py --aplicar

O programa recusa ancoras incompatíveis. O backup preserva os fontes originais.
NÃO altera PNGs/HTML antigos; a mudança vale para novas gerações.
"""
from __future__ import annotations
import argparse
import datetime
from pathlib import Path
import sys

VERSAO = 'GMM_QUATRO_NIVEIS_F1F9_F10_V1'
FILES = (
    'src/visualizacao_gmm07.py',
    'src/visualizacao_geometrica.py',
    'src/visualizacao_geometria_tsne3d.py',
)
NEW_FILE = 'src/gmm_contornos_quatro_niveis.py'
SRC_FILE = 'PACOTE_GMM_QUATRO_NIVEIS/gmm_contornos_quatro_niveis.py'


def trocar_exato(s: str, a: str, b: str, rel: str) -> str:
    n = s.count(a)
    if n != 1:
        raise ValueError(f'{rel}: ancora esperada uma vez, encontrada {n}: {a[:110]!r}')
    return s.replace(a,b,1)


def editar(rel, texto):
    if VERSAO in texto:
        raise ValueError(f'{rel} ja recebeu este ajuste; nao aplicar novamente.')
    if rel == FILES[0]:
        old = ("def gerar(config, data, escolha, destino, ext, dpi, csv, *, avaliacao):\n"
               '    """Renderiza exatamente uma figura comparativa com dois cortes por dimensão."""\n')
        new = ("def gerar(config, data, escolha, destino, ext, dpi, csv, *, avaliacao):\n"
               f"    # {VERSAO}\n"
               '    """Renderiza UM campo de risco F1-F9/F10, quatro niveis diagnosticados."""\n'
               '    from src.gmm_contornos_quatro_niveis import gerar as novo_gerador\n'
               "    return novo_gerador(config, data, escolha, destino, ext, dpi, csv, tipo='GMM07', avaliacao=avaliacao)\n")
        return trocar_exato(texto,old,new,rel)
    if rel == FILES[1]:
        old='    # PCA3D_GMM08_OFICIAL: apenas PCA 3D usa o novo PNG + HTML.\n'
        new=(f'    # {VERSAO}\n'
             '    # Um campo da etapa (F1-F9 ou F10) com quatro niveis reais.\n'
             '    from src.gmm_contornos_quatro_niveis import gerar as novo_gerador\n'
             "    return novo_gerador(config, data, escolha, dest, ext, dpi, csv, tipo='GMM08', avaliacao=avaliacao)\n"
             +old)
        return trocar_exato(texto,old,new,rel)
    if rel == FILES[2]:
        old=("    for geom in (False,True):\n"
             '        _painel_fronteiras(X,y,modelo_gmm,secoes,tau,mccfix,mccstar,rotulo_origem,destino,int(dpi),geom)\n'
             '        _html_fronteiras(X,y,modelo_gmm,malha,tau,mccfix,mccstar,rotulo_origem,destino,geom)\n')
        new=(old
             +f'    # {VERSAO}\n'
             '    # Substitui apenas os graficos 07/08 desta pasta de STAGING F10.\n'
             '    # O manifesto SHA e produzido abaixo, depois dos novos PNG/HTML.\n'
             '    from src.gmm_contornos_quatro_niveis import substituir_png_html_tsne3d\n'
             '    substituir_png_html_tsne3d(X, y, modelo_gmm, riscos, destino,\n'
             '        tau=tau, origem=origem, resolucao_cortes=int(resolucao_cortes), dpi=int(dpi))\n')
        return trocar_exato(texto,old,new,rel)
    raise ValueError('Arquivo fora da lista permitida')


def codificar(como_original, texto):
    quebra = '\r\n' if b'\r\n' in como_original else '\n'
    b = texto.replace('\n',quebra).encode('utf-8')
    return (b'\xef\xbb\xbf' + b) if como_original.startswith(b'\xef\xbb\xbf') else b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    grp=parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--checar',action='store_true')
    grp.add_argument('--aplicar',action='store_true')
    parser.add_argument('--raiz',type=Path,default=Path.cwd())
    a=parser.parse_args()
    raiz=a.raiz.resolve()
    entrada=Path(__file__).resolve().parent/SRC_FILE
    out=raiz/NEW_FILE
    originais={};novos={}
    try:
        if not (raiz/'src/orquestracao_oof.py').is_file():
            raise FileNotFoundError('Nao esta na raiz do TCC (falta src/orquestracao_oof.py).')
        if out.exists():
            raise FileExistsError('O modulo novo ja existe; ajuste possivelmente instalado: '+str(out))
        novo_mod=entrada.read_bytes()
        compile(novo_mod.decode('utf-8-sig'),str(out),'exec')
        for rel in FILES:
            p=raiz/rel
            byte=p.read_bytes()
            texto=byte.decode('utf-8-sig').replace('\r\n','\n')
            atualizado=editar(rel,texto)
            compile(atualizado,str(p),'exec')
            originais[rel]=byte
            novos[rel]=codificar(byte,atualizado)
            print('[OK] Compativel:',rel)
        print('[OK] Novo modulo:',NEW_FILE)
    except Exception as e:
        print('[BLOQUEADO] Nenhum fonte alterado:',e,file=sys.stderr)
        return 2
    if a.checar:
        print('Checagem concluida. NENHUM arquivo foi alterado.')
        return 0
    pasta=(raiz/'outputs'/('backup_gmm_4_niveis_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    pasta.mkdir(parents=True,exist_ok=False)
    for rel,content in originais.items():
        backup=pasta/rel
        backup.parent.mkdir(parents=True,exist_ok=True)
        backup.write_bytes(content)
    escritos=[]
    try:
        out.write_bytes(novo_mod)
        escritos.append(NEW_FILE)
        for rel,content in novos.items():
            (raiz/rel).write_bytes(content)
            escritos.append(rel)
    except Exception:
        for rel in escritos:
            if rel==NEW_FILE:
                (raiz/rel).unlink(missing_ok=True)
            else:
                (raiz/rel).write_bytes(originais[rel])
        print('[ROLLBACK] Fonte restaurado; backup permanece em',pasta,file=sys.stderr)
        raise
    print('[APLICADO] GMM07/GMM08 1D/2D/3D; PCA/t-SNE; OOF F1-F9 e F10 congelado.')
    print('Backup dos fontes antigos:',pasta)
    print('Nenhum treinamento, YAML, resultado OOF/F10 ou PNG antigo foi alterado.')
    return 0

if __name__=='__main__':
    sys.exit(main())
