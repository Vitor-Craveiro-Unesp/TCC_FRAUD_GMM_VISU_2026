"""Restaurar SOMENTE os rodapes PCA 2D de GMM04, GMM07 e GMM08.

Executar a partir da raiz do TCC_EXPERIMENTAL_V3:
    python restaurar_rodapes_pca2d.py

Faz backup de todos os arquivos alterados. Nao executa treinamento,
nao altera YAML e nao regrava PNGs ja gerados.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import sys

ROOT = Path.cwd()
FILES = {
    'GMM04': ROOT / 'src' / 'visualizacao_gmm04.py',
    'GMM07': ROOT / 'src' / 'visualizacao_gmm07.py',
    'GMM08': ROOT / 'src' / 'visualizacao_geometrica.py',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def patch_gmm04(src):
    tag = '# RODAPE_RESTAURADO_PCA2D_GMM04'
    if tag in src:
        return src
    anchor = '    geo._guardar(fig,destino,f\'GMM04_dominancia_{d}d\',ext,dpi)'
    require(src.count(anchor) == 1, 'GMM04: bloco de salvamento nao localizado unicamente.')
    # Rodape existe apenas para d=2, preservando PCA 1D e 3D.
    addition = '''    # RODAPE_RESTAURADO_PCA2D_GMM04
    if d == 2:
        fig.text(
            .5, .045,
            'Regiões coloridas: gaussiana dominante  |  '
            'Curvas tracejadas: isodensidades gaussianas 50% / 95%  |  '
            '×: médias gaussianas',
            ha='center', fontsize=9.3, color='#46556e'
        )

'''
    # Colocar antes de subplots_adjust para deixar o layout consciente do rodape.
    end = src.index(anchor)
    start = src.rfind('    fig.subplots_adjust(', 0, end)
    require(start > 0 and (end-start) < 350, 'GMM04: layout 2D nao identificado perto do salvamento.')
    layout = src[start:end]
    # Em versoes com margens pequenas, abrir espaco para o rodape.
    match = re.search(r'\bbottom=(\.\d+)(?=\s*,\s*wspace=)', layout)
    require(match is not None, 'GMM04: margem inferior nao encontrada.')
    original = float(match.group(1))
    if original < .14:
        layout = layout[:match.start()] + 'bottom=.145 if d == 2 else ' + match.group(1) + layout[match.end():]
    return src[:start] + addition + layout + src[end:]


def patch_gmm07(src):
    tag = '# RODAPE_RESTAURADO_PCA2D_GMM07'
    if tag in src:
        return src
    anchor = "    nome = f'GMM07_sensibilidade_{dimensoes}d_comparativo'"
    require(src.count(anchor) == 1, 'GMM07: marcador de saida nao encontrado unicamente.')
    start = src.index('    direcao = (')
    end = src.index(anchor)
    nearby = src[start:end]
    # Se o rodape antigo nao estiver condicionado a !=2,
    # a nova linha repetiria a mesma informacao. Bloquear nesse caso.
    if ('fig.text(' in nearby and 'mcc_fixo:.7f' in nearby and 'mcc_otimo:.7f' in nearby and 'direcao' in nearby):
        print('GMM07: rodape ja presente; preservado.')
        return src
    require('if dimensoes != 2:' in nearby or 'if dimensoes not in (2,)' in nearby,
            'GMM07: rodape antigo nao esta condicionado a excluir 2D; '
            'verifique o trecho direcao/fig.text antes de continuar.')
    addition = '''    # RODAPE_RESTAURADO_PCA2D_GMM07
    if dimensoes == 2:
        fig.text(
            .5, .052,
            f'{nome_mcc}(τ=0.5000000) = {mcc_fixo:.7f}   |   '
            f'{nome_mcc}(τ*={tau_star:.7f}) = {mcc_otimo:.7f}   |   '
            f'Mudança: {direcao}',
            ha='center', fontsize=10.1,
            color='#1b3048', fontweight='bold'
        )

'''
    src = src.replace(anchor, addition + anchor, 1)
    # Ajustar apenas a margem inferior do PCA 2D quando ainda estreita.
    layout_pattern = (r'(fig\.subplots_adjust\(left=\.075,right=\.88 if dimensoes==2 else \.985,\s*'
                      r'top=\.76,bottom=)(\.\d+)(,wspace=\.22\))')
    found = re.search(layout_pattern, src)
    require(found is not None, 'GMM07: margem da figura 2D nao localizada.')
    if float(found.group(2)) < .16:
        replacement = found.group(1) + '.16 if dimensoes==2 else ' + found.group(2) + found.group(3)
        src = src[:found.start()] + replacement + src[found.end():]
    src = src.replace('cax=fig.add_axes([.905,.12,.012,.64])',
                      'cax=fig.add_axes([.905,.16,.012,.60])', 1)
    return src


def patch_gmm08(src):
    tag = '# RODAPE_RESTAURADO_PCA2D_GMM08'
    if tag in src:
        return src
    anchor = "    _guardar(fig, dest, f'GMM08_mapa_risco_{dim}d_comparativo'"
    require(src.count(anchor) == 1, 'GMM08: layout atual de PCA 2D nao localizado.')
    addition = '''    # RODAPE_RESTAURADO_PCA2D_GMM08
    if dim == 2:
        direcao_2d = (
            'legítima → fraude' if tau_star < .5 else
            'fraude → legítima' if tau_star > .5 else
            'nenhuma mudança'
        )
        fig.text(
            .49, .050,
            f'{nome_mcc}(τ=0.5000000) = {mcc_fixo:.7f}   |   '
            f'{nome_mcc}(τ*={tau_star:.7f}) = {mcc_otimo:.7f}   |   '
            f'Hachura: mudança de classe ({direcao_2d})',
            fontsize=10.0, ha='center',
            color='#23334a', fontweight='bold'
        )

'''
    ponto_saida = src.index(anchor)
    ponto_layout = src.rfind('    fig.subplots_adjust(', 0, ponto_saida)
    require(ponto_layout >= 0 and ponto_saida - ponto_layout < 750,
            'GMM08: nao foi possivel identificar o layout antes do salvamento.')
    src = src[:ponto_layout] + addition + src[ponto_layout:]
    src = src.replace('bottom=.105 if dim == 2 else .145',
                      'bottom=.155 if dim == 2 else .145', 1)
    src = src.replace('fig.add_axes([.905, .105, .012, .655])',
                      'fig.add_axes([.905, .155, .012, .605])', 1)
    return src


def main():
    require(all(path.is_file() for path in FILES.values()),
            'Execute na raiz do TCC_EXPERIMENTAL_V3 (com pasta src).')
    functions = {'GMM04': patch_gmm04, 'GMM07': patch_gmm07, 'GMM08': patch_gmm08}
    prepared = {}
    for key, path in FILES.items():
        raw = path.read_bytes()
        bom = raw.startswith(b'\xef\xbb\xbf')
        content = raw.decode('utf-8-sig')
        newline = '\r\n' if b'\r\n' in raw else '\n'
        # Normalizar apenas para edicao; depois repor terminacoes originais.
        normal = content.replace('\r\n', '\n')
        altered = functions[key](normal)
        compile(altered, str(path), 'exec')
        if key != 'GMM07' or altered != normal:
            assert altered.count('# RODAPE_RESTAURADO_PCA2D_' + key) == 1
        out = altered.replace('\n', newline).encode('utf-8')
        if bom:
            out = b'\xef\xbb\xbf' + out
        prepared[key] = (path, raw, out)
        print(key + ': OK - validacao sintatica aprovada')

    suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
    for key, (path, raw, out) in prepared.items():
        if raw == out:
            print(key + ': ja atualizado; nenhuma modificacao')
            continue
        backup = path.with_name(path.name + '.ANTES_RODAPE_' + suffix + '.bak')
        with backup.open('xb') as handle:
            handle.write(raw)
        path.write_bytes(out)
        print(key + ': atualizado. Backup: ' + str(backup))
    print('\nConcluido. YAML e PNGs existentes permanecem intocados.')
    print('Os rodapes aparecerão apenas nos PNGs regenerados apos esta alteracao.')


if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        print('ALTERACAO BLOQUEADA: ' + str(ex), file=sys.stderr)
        sys.exit(1)
