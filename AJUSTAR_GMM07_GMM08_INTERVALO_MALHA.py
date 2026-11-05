# -*- coding: utf-8 -*-
"""Ajuste SOMENTE visual do GMM07/GMM08 (1D/2D/3D).
Uso na raiz do TCC: python AJUSTAR_GMM07_GMM08_INTERVALO_MALHA.py --checar
                 python AJUSTAR_GMM07_GMM08_INTERVALO_MALHA.py --aplicar
Nao treina, nao le dados, nao modifica parametros cientificos.
"""
import argparse
from pathlib import Path
from datetime import datetime
import sys

M = 'DIAGNOSTICO_INTERVALO_GMM07_GMM08_V1'
ARQUIVOS = ['src/visualizacao_geometrica.py', 'src/visualizacao_gmm07.py',
            'src/visualizacao_gmm07_3d.py', 'src/visualizacao_gmm08_3d.py']

def trocar(s, velho, novo):
    qtd = s.count(velho)
    if qtd != 1:
        raise ValueError(f'Ancora alterada, localizada {qtd} vez(es): {velho[:70]!r}')
    return s.replace(velho, novo, 1)

HELPER = '''# DIAGNOSTICO_INTERVALO_GMM07_GMM08_V1
# Usa os scores que JA existem; nao muda limiares nem preditor.
def _anotar_intervalo_malha(ax, campo, limiar):
    scores = np.asarray(campo, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError('Malha de scores invalida.')
    minimo, maximo = float(scores.min()), float(scores.max())
    positivos = int(np.count_nonzero(scores >= float(limiar)))
    total = int(scores.size)
    ax.text(.985, .985,
            f's(x) na malha: min={minimo:.7f}  max={maximo:.7f}\\n'
            f'Positivas previstas na malha: {positivos}/{total} ({100*positivos/total:.2f}%)',
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
            color='#24374b', zorder=12,
            bbox=dict(facecolor='white', edgecolor='#ccd4df', alpha=.93,
                      boxstyle='round,pad=.22'))
    return minimo, maximo, positivos, total


'''

def alterar(rel, s):
    if rel == ARQUIVOS[0]:
        s = trocar(s, 'def _mudanca_classes(score, tau_star):',
                     HELPER + 'def _mudanca_classes(score, tau_star):')
        s = trocar(s, "        titulo=f'A — Corte fixo τ = 0,50' if j==0 else",
                  '        # '+M+' GMM08 1D\n        _anotar_intervalo_malha(ax, campo, tau)\n'
                  "        titulo=f'A — Corte fixo τ = 0,50' if j==0 else")
        s = trocar(s, '            ax.set(xlabel=nomes[par[0]], ylabel=nomes[par[1]],',
                  '            # '+M+' GMM08 2D / TSNE 3D\n'
                  '            _anotar_intervalo_malha(ax, campo, tau)\n'
                  '            ax.set(xlabel=nomes[par[0]], ylabel=nomes[par[1]],')
    elif rel == ARQUIVOS[1]:
        s = trocar(s, '            if not cruzou:\n',
                  '            # '+M+' GMM07 1D / 2D\n'
                  '            geo._anotar_intervalo_malha(ax, campo, limiar)\n'
                  '            if not cruzou:\n')
    elif rel == ARQUIVOS[2]:
        s = trocar(s, '            ax.set_xlim(xx[0,0],xx[0,-1])',
                  '            # '+M+' GMM07 PCA 3D\n'
                  '            geo._anotar_intervalo_malha(ax, risco, limiar)\n'
                  '            ax.set_xlim(xx[0,0],xx[0,-1])')
        s = trocar(s, "              f'. Faixa dos riscos: [{minimo_r:.7f}, {maximo_r:.7f}].')",
                  "              f'. Faixa dos riscos: [{minimo_r:.7f}, {maximo_r:.7f}].'\n"
                  "              f' Malha volumetrica s(x): [{vmin:.7f}, {vmax:.7f}].')")
    elif rel == ARQUIVOS[3]:
        s = trocar(s, '            ax.set_xlim(xx[0,0],xx[0,-1]);ax.set_ylim(yy[0,0],yy[-1,0])',
                  '            # '+M+' GMM08 PCA 3D\n'
                  '            geo._anotar_intervalo_malha(ax, risk, threshold)\n'
                  '            ax.set_xlim(xx[0,0],xx[0,-1]);ax.set_ylim(yy[0,0],yy[-1,0])')
        s = trocar(s, "              f'. Intervalo possivel: [{menor_risco:.7f}, {maior_risco:.7f}].')",
                  "              f'. Intervalo possivel: [{menor_risco:.7f}, {maior_risco:.7f}].'\n"
                  "              f' Malha volumetrica s(x): [{float(risco.min()):.7f}, {float(risco.max()):.7f}].')")
    return s

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raiz', type=Path, default=Path.cwd())
    modo = ap.add_mutually_exclusive_group(required=True)
    modo.add_argument('--checar', action='store_true')
    modo.add_argument('--aplicar', action='store_true')
    args = ap.parse_args()
    raiz = args.raiz.resolve()
    originais, novos = {}, {}
    try:
        for rel in ARQUIVOS:
            p = raiz / rel
            original = p.read_bytes()
            s = original.decode('utf-8-sig').replace('\r\n', '\n')
            if M in s:
                raise ValueError(f'O ajuste ja existe (ou foi aplicado parcialmente): {rel}')
            novo = alterar(rel, s)
            compile(novo, str(p), 'exec')
            quebra = '\r\n' if b'\r\n' in original else '\n'
            saida = novo.replace('\n', quebra).encode('utf-8')
            if original.startswith(b'\xef\xbb\xbf'):
                saida = b'\xef\xbb\xbf' + saida
            originais[rel], novos[rel] = original, saida
            print('[OK]', rel, '(sintaxe verificada)')
    except Exception as exc:
        print('[BLOQUEADO] Nenhum arquivo alterado:', exc, file=sys.stderr)
        return 2
    if args.checar:
        print('Verificacao concluida; nenhum arquivo foi modificado.')
        return 0
    pasta = raiz / 'outputs' / ('validacao_gmm07_gmm08_backup_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    pasta.mkdir(parents=True, exist_ok=False)
    for rel, b in originais.items():
        destino = pasta / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(b)
    escritos = []
    try:
        for rel, b in novos.items():
            (raiz / rel).write_bytes(b)
            escritos.append(rel)
    except Exception:
        for rel in escritos:
            (raiz / rel).write_bytes(originais[rel])
        raise
    print('Ajuste aplicado sem treinamento, sem alterar YAML ou resultados antigos.')
    print('Copia dos arquivos originais em:', pasta)
    print('As linhas de fronteira so aparecem onde s(x) atinge o limiar.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
