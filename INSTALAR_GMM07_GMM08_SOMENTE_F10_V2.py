# -*- coding: utf-8 -*-
"""Instalador conservador do contrato GMM07/GMM08 exclusivo do F10.

Uso na raiz do TCC_EXPERIMENTAL_V3:
  python INSTALAR_GMM07_GMM08_SOMENTE_F10_V2.py --checar
  python INSTALAR_GMM07_GMM08_SOMENTE_F10_V2.py --aplicar

Nao treina modelos, nao altera YAML nem reescreve resultados/figuras anteriores.
Verifica as assinaturas do checkpoint 862e52d e cria backup dos arquivos
originais. A instalacao e intencionalmente nao idempotente: uma nova aplicacao
ja instalada nao volta a modificar os arquivos.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
from pathlib import Path
import shutil
import sys

TAG = 'F10_BOUNDARY_CERTIFICATE_V2'
FILES = ('src/visualizacao_geometrica.py', 'src/gmm_contornos_quatro_niveis.py')
NEW = 'src/gmm_fronteira_f10.py'


def trocar_unico(texto, antigo, novo, nome):
    n = texto.count(antigo)
    if n != 1:
        raise RuntimeError(f'Fonte incompativel ({nome}): esperado 1 bloco, encontrado {n}. Nenhum arquivo alterado.')
    return texto.replace(antigo, novo, 1)


def corrigir_geo(src):
    src = trocar_unico(
        src,
        'def _ativo(config, nome):\n',
        'def _ativo(config, nome):\n'
        '    # F10_BOUNDARY_CERTIFICATE_V2: GMM07/GMM08 somente na avaliacao F10.\n'
        '    # As rotinas F10 chamam os renderizadores diretamente e nao dependem daqui.\n'
        "    if nome in ('sensibilidade_fronteira', 'mapa_risco_fronteira'):\n"
        '        return False\n',
        'filtro OOF 07.02')
    return src


def corrigir_common(src):
    src = trocar_unico(
        src,
        '''def _faixa(q, margem):
    a, b = np.quantile(q, [.005, .995])
    amplitude = float(b - a)
    if amplitude <= 1e-12:
        amplitude = max(abs(float(a)), 1.)
    return float(a - amplitude*margem), float(b + amplitude*margem)


def _planos(X, gm, riscos, d, n, margem):
    medias = X.mean(axis=0)
    pares = [(0,)] if d == 1 else ([(0, 1)] if d == 2 else [(0, 1), (0, 2), (1, 2)])
    planos = []
    for par in pares:
        if len(par) == 1:
            xx = np.linspace(*_faixa(X[:, par[0]], margem), n)
            pts = np.tile(medias, (n, 1))
            pts[:, par[0]] = xx
            risco = _pontuar(gm, riscos, pts)
            grade = (xx,)
        else:
            xx, yy = np.meshgrid(np.linspace(*_faixa(X[:, par[0]], margem), n),
                                 np.linspace(*_faixa(X[:, par[1]], margem), n))
            pts = np.tile(medias, (xx.size, 1))
            pts[:, par[0]], pts[:, par[1]] = xx.ravel(), yy.ravel()
            risco = _pontuar(gm, riscos, pts).reshape(xx.shape)
            grade = (xx, yy)
        planos.append((par, grade, risco))
    return planos, medias
''',
        '''def _faixa(q, margem):
    # F10_BOUNDARY_CERTIFICATE_V2: cobrir TODOS os pontos F10, nao percentis.
    # Inclui extremos observados e uma margem em cada direcao.
    a, b = float(np.min(q)), float(np.max(q))
    amplitude = float(b - a)
    if amplitude <= 1e-12:
        amplitude = max(abs(float(a)), 1.)
    return float(a - amplitude*margem), float(b + amplitude*margem)


def _eixo_com_ancora(valores, n, margem, referencia):
    eixo = np.linspace(*_faixa(valores, margem), n)
    if referencia is not None:
        extras = np.atleast_1d(np.asarray(referencia, dtype=float))
        extras = extras[(extras >= eixo[0]) & (extras <= eixo[-1])]
        # Acrescenta nos exatos para manter extremos e certificados, sem perder
        # resolucao da malha regular nem alterar a ordem monotona dos eixos.
        eixo = np.unique(np.concatenate((eixo, extras)))
    return eixo


def _planos(X, gm, riscos, d, n, margem, *, ancora=None, referencias=None):
    # Com F10+ e F10- o ponto de fronteira e obtido por bissecao no MESMO GMM.
    # Os tres planos em 3D passam por este ponto, nao por uma media arbitraria.
    medias = X.mean(axis=0) if ancora is None else np.asarray(ancora, dtype=float).ravel()
    pares = [(0,)] if d == 1 else ([(0, 1)] if d == 2 else [(0, 1), (0, 2), (1, 2)])
    planos = []
    for par in pares:
        def refs(j):
            pontos = [] if ancora is None else [medias[j]]
            if referencias is not None:
                pontos.extend(float(v[j]) for v in referencias)
            return pontos if pontos else None
        if len(par) == 1:
            xx = _eixo_com_ancora(X[:, par[0]], n, margem, refs(par[0]))
            pts = np.tile(medias, (len(xx), 1))
            pts[:, par[0]] = xx
            risco = _pontuar(gm, riscos, pts)
            grade = (xx,)
        else:
            e0 = _eixo_com_ancora(X[:, par[0]], n, margem, refs(par[0]))
            e1 = _eixo_com_ancora(X[:, par[1]], n, margem, refs(par[1]))
            xx, yy = np.meshgrid(e0, e1)
            pts = np.tile(medias, (xx.size, 1))
            pts[:, par[0]], pts[:, par[1]] = xx.ravel(), yy.ravel()
            risco = _pontuar(gm, riscos, pts).reshape(xx.shape)
            grade = (xx, yy)
        planos.append((par, grade, risco))
    return planos, medias
''', 'faixa completa F10 e planos por fronteira')

    src = trocar_unico(src, 'titulo += f\'\\n{nomes[z]} = {medias[z]:.4g} (media da etapa)\'',
                      'titulo += f\'\\n{nomes[z]} = {medias[z]:.4g} (valor fixado)\'',
                      'rotulo do plano')
    src = trocar_unico(
        src,
        'planos, medias, lo, hi, *, substituir_staging=False, avaliacao_diagnostico=None):',
        'planos, medias, lo, hi, *, substituir_staging=False, avaliacao_diagnostico=None, certificado=None):',
        'assinatura PNG')
    src = trocar_unico(
        src,
        '''        ax.set_title(_rotulos_planos(par, nomes, medias), fontsize=11, pad=10)''',
        '''        # F10_BOUNDARY_CERTIFICATE_V2: ponto REAL da funcao score=tau*.
        # No 3D os planos passam por esse ponto, pois suas dimensoes restantes
        # foram fixadas na propria coordenada da raiz de bissecao.
        if certificado is not None and certificado['cruzamento_confirmado']:
            raiz = certificado['ponto_fronteira']
            if d == 1:
                ax.scatter([raiz[0]], [tau], marker='X', s=120,
                           color=COLORS['oof'], edgecolors='white', linewidths=.7,
                           zorder=14)
            else:
                ax.scatter([raiz[par[0]]], [raiz[par[1]]], marker='X', s=105,
                           color=COLORS['oof'], edgecolors='white', linewidths=.7,
                           zorder=14)
        ax.set_title(_rotulos_planos(par, nomes, medias), fontsize=11, pad=10)''',
        'marcador certificado PNG')
    src = trocar_unico(
        src,
        "    fig.text(.5, .048, msg, ha='center', fontsize=8.2, color='#52657d')",
        "    if origem == 'F10' and certificado is not None:\n"
        "        if certificado['cruzamento_confirmado']:\n"
        "            msg += ' | X vermelho: fronteira τ* confirmada por bissecao F10.'\n"
        "        elif certificado['positivos_f10']:\n"
        "            msg += ' | Positivos F10 sem par negativo: contorno nao garantido.'\n"
        "    fig.text(.5, .048, msg, ha='center', fontsize=8.2, color='#52657d')",
        'nota certificado PNG')
    src = trocar_unico(
        src,
        '*, substituir_staging=False):\n    """HTML offline com a malha VOLUMÉTRICA',
        '*, substituir_staging=False, certificado=None):\n    """HTML offline com a malha VOLUMÉTRICA',
        'assinatura HTML')
    src = trocar_unico(
        src,
        "    if grafico=='GMM08':\n        for k in range(gm.n_components):",
        "    # F10_BOUNDARY_CERTIFICATE_V2: marcacao exata de cruzamento em 3D.\n"
        "    if certificado is not None and certificado['cruzamento_confirmado']:\n"
        "        raiz = certificado['ponto_fronteira']\n"
        "        fig.add_trace(go.Scatter3d(\n"
        "            x=[raiz[0]], y=[raiz[1]], z=[raiz[2]], mode='markers',\n"
        "            marker={'size':8, 'color':COLORS['oof'], 'symbol':'diamond'},\n"
        "            name='Cruzamento real τ* confirmado por bissecao F10'))\n"
        "    if grafico=='GMM08':\n        for k in range(gm.n_components):",
        'marcador HTML 3D')
    # Executar a identificacao de origem antes de construir malha.
    src = trocar_unico(
        src,
        '''    planos,medias=_planos(X,gm,riscos,d,n,margem)
    lo = min(float(np.min(item[2])) for item in planos)
    hi = max(float(np.max(item[2])) for item in planos)
    origem,mcc,nome_mcc=_fonte(destino,avaliacao,y,data,tau)
    filename = _painel_png(config,data,escolha,destino,ext,dpi,tipo,origem,mcc,nome_mcc,
                          planos,medias,lo,hi,substituir_staging=substituir_staging,
                          avaliacao_diagnostico=avaliacao if origem=='OOF'
                          and isinstance(avaliacao,dict) and 'p' in avaliacao else None)''',
        '''    origem,mcc,nome_mcc=_fonte(destino,avaliacao,y,data,tau)
    certificado = None
    if origem == 'F10':
        from src.gmm_fronteira_f10 import diagnosticar
        certificado = diagnosticar(gm, riscos, X, data['scores'], tau)
    referencias = ([certificado['ponto_positivo'], certificado['ponto_negativo']]
                   if certificado is not None and certificado['cruzamento_confirmado'] else None)
    planos,medias=_planos(X,gm,riscos,d,n,margem,
                          ancora=(certificado['ponto_fronteira']
                                  if certificado is not None else None),
                          referencias=referencias)
    lo = min(float(np.min(item[2])) for item in planos)
    hi = max(float(np.max(item[2])) for item in planos)
    filename = _painel_png(config,data,escolha,destino,ext,dpi,tipo,origem,mcc,nome_mcc,
                          planos,medias,lo,hi,substituir_staging=substituir_staging,
                          avaliacao_diagnostico=avaliacao if origem=='OOF'
                          and isinstance(avaliacao,dict) and 'p' in avaliacao else None,
                          certificado=certificado)''',
        'chamada renderizador origem F10')
    src = trocar_unico(
        src,
        '''        _html3d(config,data,escolha,dest,tipo,origem,mcc,planos,lo,hi,
                substituir_staging=substituir_staging)''',
        '''        _html3d(config,data,escolha,dest,tipo,origem,mcc,planos,lo,hi,
                substituir_staging=substituir_staging, certificado=certificado)''',
        'chamada HTML 3D')
    return src


def montar(raiz):
    for f in FILES:
        if not (raiz/f).is_file():
            raise FileNotFoundError('Arquivo obrigatorio ausente: '+f)
    if not (raiz/'src/visualizacao_integrada.py').is_file():
        raise FileNotFoundError('src/visualizacao_integrada.py ausente. Pasta incorreta?')
    if not (raiz/'src/visualizacao_geometria_tsne3d.py').is_file():
        raise FileNotFoundError('src/visualizacao_geometria_tsne3d.py ausente.')
    if (raiz/NEW).exists():
        raise RuntimeError('Modulo de fronteira F10 ja existe; nao sobrescrever. Confira versoes.')
    textos = {f:(raiz/f).read_text(encoding='utf-8-sig') for f in FILES}
    if any(TAG in t for t in textos.values()):
        raise RuntimeError('Ajuste F10 V2 aparentemente ja instalado. Nenhuma alteracao.')
    novos = {
        FILES[0]: corrigir_geo(textos[FILES[0]]),
        FILES[1]: corrigir_common(textos[FILES[1]]),
        NEW: (Path(__file__).resolve().parent/'gmm_fronteira_f10.py').read_text(encoding='utf-8-sig'),
    }
    for nome, conteudo in novos.items():
        ast.parse(conteudo, filename=nome)
    return novos


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    m=ap.add_mutually_exclusive_group(required=True)
    m.add_argument('--checar',action='store_true')
    m.add_argument('--aplicar',action='store_true')
    arg=ap.parse_args()
    raiz=Path(__file__).resolve().parent
    print('=== Contrato GMM07/GMM08 EXCLUSIVO F10 — V2 ===')
    novos=montar(raiz)
    for f in novos:
        print('[COMPATIVEL]', f, '(AST valido)')
    if arg.checar:
        print('[OK] Verificacao somente leitura. Nenhum arquivo modificado.')
        return
    base=raiz/'outputs'/('backup_gmm_so_f10_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    base.mkdir(parents=True,exist_ok=False)
    for f in FILES:
        dst=base/f
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(raiz/f,dst)
    print('Backup dos arquivos originais:',base)
    try:
        for f,conteudo in novos.items():
            dst=raiz/f
            tmp=dst.with_name(dst.name+'.F10V2.incompleto')
            with tmp.open('x',encoding='utf-8',newline='\n') as h:
                h.write(conteudo)
            tmp.replace(dst)
            print('[ATUALIZADO]',f)
    except Exception:
        # Restauracao imediata, sem apagar o backup.
        for f in FILES:
            shutil.copy2(base/f,raiz/f)
        (raiz/NEW).unlink(missing_ok=True)
        raise
    print('[INSTALADO] Somente F10 para GMM07/GMM08; dominio completo F10,')
    print('            marcacao de cruzamento verdadeiro quando existem F10+ e F10-.')
    print('OOF, modelos, resultados, YAML, F10 salvo e PNGs antigos: PRESERVADOS.')


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print('[BLOQUEADO]',str(exc),file=sys.stderr)
        sys.exit(1)
