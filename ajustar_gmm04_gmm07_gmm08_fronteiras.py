# -*- coding: utf-8 -*-
"""Ajustes VISUAIS GMM04/07/08. Sem treinamento, YAML, OOF ou F10.

Uso: python ajustar_gmm04_gmm07_gmm08_fronteiras.py --verificar
     python ajustar_gmm04_gmm07_gmm08_fronteiras.py --aplicar

O instalador valida sintaxe, cria backups e bloqueia alteracoes inesperadas.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
from pathlib import Path
import sys

STAMP = '# AJUSTE_FRONTEIRAS_TCC_20261028_V1'


def no_function_literal(node, word: str) -> bool:
    return any(isinstance(sub, ast.Constant) and isinstance(sub.value, str) and word in sub.value
               for sub in ast.walk(node))


def remove_ranges(source: str, nodes: list[ast.AST], comentario: str) -> str:
    lines = source.splitlines(keepends=True)
    spans = {(n.lineno, n.end_lineno) for n in nodes}
    # Reject overlapping AST edits. Remove from bottom upwards.
    previous = len(lines) + 1
    for start, end in sorted(spans, reverse=True):
        if end >= previous:
            raise RuntimeError('Blocos de edicao sobrepostos.')
        indent = lines[start - 1][:len(lines[start - 1]) - len(lines[start - 1].lstrip())]
        lines[start - 1:end] = [indent + 'pass  # ' + comentario + '\n']
        previous = start
    return ''.join(lines)


def patch_gmm04(source: str) -> tuple[str, str]:
    # This is only the lower caption, NOT contours, ellipses or labels.
    tree = ast.parse(source)
    removals = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not (isinstance(call.func, ast.Attribute) and call.func.attr == 'text'):
            continue
        if not (isinstance(call.func.value, ast.Name) and call.func.value.id == 'fig'):
            continue
        if no_function_literal(call, 'Regiões coloridas:'):
            removals.append(node)
    if len(removals) > 1:
        raise RuntimeError('GMM04: mais de uma nota inferior; conferir manualmente.')
    if not removals:
        return source, 'Sem frase inferior GMM04 localizada (possivelmente ja removida).'
    return remove_ranges(source, removals, 'GMM04: nota inferior removida'), 'GMM04: rodape removido.'


def patch_gmm07(source: str) -> tuple[str, str]:
    tree = ast.parse(source)
    # Only within the 2D rendering method, no changes to the 3D renderer.
    methods = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == '_grafico']
    if len(methods) != 1:
        raise RuntimeError('GMM07: funcao _grafico nao encontrada exatamente uma vez.')
    method = methods[0]
    remove_ellipses = []
    remove_legend = []
    for node in ast.walk(method):
        if not isinstance(node, ast.If):
            continue
        cond = ast.unparse(node.test)
        if not ('dimensoes' in cond and '== 2' in cond):
            continue
        original = ast.get_source_segment(source, node) or ''
        if 'Ellipse(' in original and 'ax.add_patch' in original:
            remove_ellipses.append(node)
        if no_function_literal(node, 'Elipses gaussianas 95%'):
            remove_legend.append(node)
    if len(remove_ellipses) > 1 or len(remove_legend) > 1:
        raise RuntimeError('GMM07: foram encontradas multiplas estruturas; nada alterado.')
    if bool(remove_ellipses) != bool(remove_legend):
        raise RuntimeError('GMM07: apenas uma das estruturas existe (ellipses/legenda); nada alterado.')
    if not remove_ellipses:
        # Distinguish already updated from unexpected version.
        if 'Elipses gaussianas 95%' in source or ('Ellipse(' in (ast.get_source_segment(source, method) or '')):
            raise RuntimeError('GMM07: formato desconhecido de elipses. Solicite alteracao especifica.')
        return source, 'GMM07 2D: elipses e legenda ja ausentes.'
    patched = remove_ranges(source, remove_ellipses + remove_legend,
                            'GMM07 2D: elipses e legenda removidas; fronteiras mantidas')
    return patched, 'GMM07 PCA 2D: elipses removidas, fronteiras e scores preservados.'


def patch_gmm08(source: str) -> tuple[str, str]:
    tree = ast.parse(source)
    notes = ('Hachura: mudança de classe', 'Planos com coordenada omitida', 'Elipses tracejadas:')
    nodes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        fn = node.value.func
        if (isinstance(fn, ast.Attribute) and fn.attr == 'text'
                and isinstance(fn.value, ast.Name) and fn.value.id == 'fig'
                and any(no_function_literal(node.value, word) for word in notes)):
            nodes.append(node)
    if len(nodes) > 3:
        raise RuntimeError('GMM08: rodapes duplicados; conferir manualmente.')
    if nodes:
        source = remove_ranges(source, nodes, 'GMM08: rodape redundante removido')

    # The first level is the source of truth: the three posterior risk weights.
    # Introduce a clear, dynamic status on the 3D HTML, not fabricated surfaces.
    if 'AJUSTE_STATUS_GMM08_PCA3D_V1' not in source:
        anchor = "    html=html.replace('__METRIC__',html_lib.escape(origem_mcc))"
        if source.count(anchor) != 1:
            raise RuntimeError('GMM08: insercao de status HTML nao encontrada; nada alterado.')
        insertion = '''
    # AJUSTE_STATUS_GMM08_PCA3D_V1
    riscos_gaussianos = np.asarray(rh.riscos_.to_numpy(dtype=float))
    if not np.isfinite(riscos_gaussianos).all() or len(riscos_gaussianos) != modelo.n_components:
        raise ValueError('Riscos gaussianos invalidos para diagnostico das fronteiras.')
    menor_risco = float(riscos_gaussianos.min())
    maior_risco = float(riscos_gaussianos.max())
    def explicar_limiar(limiar, nome):
        if limiar < menor_risco:
            return f'{nome}: sem fronteira; abaixo de todos os riscos gaussianos'
        if limiar > maior_risco:
            return f'{nome}: sem fronteira; acima de todos os riscos gaussianos'
        if not (float(risco.min()) < limiar < float(risco.max())):
            return f'{nome}: sem cruzamento na malha calculada'
        return f'{nome}: fronteira presente na malha'
    status = (explicar_limiar(0.5, 'Tau = 0,50') + ' | ' +
              explicar_limiar(tau, 'Tau* = ' + f'{tau:.7f}') +
              f'. Intervalo possivel: [{menor_risco:.7f}, {maior_risco:.7f}].')
    html = html.replace('__STATUS_FRONTEIRAS__', html_lib.escape(status))
'''
        source = source.replace(anchor, anchor + insertion, 1)
        html_anchor = '</header><div class="layout">'
        # Unique in the source; CSS-only addition in the HTML template.
        if source.count(html_anchor) != 1:
            raise RuntimeError('GMM08: template HTML inesperado; nada alterado.')
        source = source.replace(html_anchor,
                                '<p class="status-fronteira" role="status">__STATUS_FRONTEIRAS__</p>\n' + html_anchor, 1)
        css_anchor = '.layout{display:grid;grid-template-columns:'
        if source.count(css_anchor) != 1:
            raise RuntimeError('GMM08: folha de estilo inesperada; nada alterado.')
        source = source.replace(css_anchor,
                                '.status-fronteira{font-size:13px;color:#44566d;margin:13px 0 0;line-height:1.5}\n' + css_anchor, 1)
    js_original = "el.disabled=true;el.parentElement.title='Limiar fora da faixa da malha';"
    js_novo = "el.disabled=true;el.checked=false;el.parentElement.title='Sem superficie disponivel para esse limiar';"
    if js_original in source:
        source = source.replace(js_original, js_novo, 1)
    elif js_novo not in source:
        raise RuntimeError('GMM08: comportamento de controles indisponiveis mudou.')
    return source, 'GMM08: PNG limpo, HTML com status e controles indisponiveis desmarcados.'


def patch_gmm07_3d(source: str) -> tuple[str, str]:
    if 'AJUSTE_STATUS_GMM07_PCA3D_V1' in source:
        return source, 'GMM07 3D: explicacao de limiares ja presente.'
    # This is only a status banner; it does not change any trace or threshold.
    anchor = "        html=html.replace(key,value)"
    if source.count(anchor) != 1:
        raise RuntimeError('GMM07 3D: substituicao de template nao encontrada.')
    insertion = '''
    # AJUSTE_STATUS_GMM07_PCA3D_V1
    riscos_gaussianos = np.asarray(rh.riscos_.to_numpy(dtype=float))
    if not np.isfinite(riscos_gaussianos).all() or len(riscos_gaussianos) != ncomp:
        raise ValueError('Riscos gaussianos invalidos para GMM07.')
    minimo_r = float(riscos_gaussianos.min())
    maximo_r = float(riscos_gaussianos.max())
    def explicar_limiar(limiar, nome):
        if limiar < minimo_r:
            return f'{nome}: sem fronteira; abaixo de todos os riscos gaussianos'
        if limiar > maximo_r:
            return f'{nome}: sem fronteira; acima de todos os riscos gaussianos'
        if not (vmin < limiar < vmax):
            return f'{nome}: sem cruzamento na malha 3D'
        return f'{nome}: fronteira presente na malha'
    resumo = (explicar_limiar(0.5, 'Tau = 0,50') + ' | ' +
              explicar_limiar(tau, 'Tau* = ' + f'{tau:.7f}') +
              f'. Faixa dos riscos: [{minimo_r:.7f}, {maximo_r:.7f}].')
    html = html.replace('__STATUS_FRONTEIRAS__', escape(resumo))
'''
    source = source.replace(anchor, anchor + insertion, 1)
    template_anchor = '</header><div class="layout">'
    if source.count(template_anchor) != 1:
        raise RuntimeError('GMM07 3D: cabecalho HTML inesperado.')
    source = source.replace(template_anchor,
                            '<p class="status-fronteira" role="status">__STATUS_FRONTEIRAS__</p>\n' + template_anchor, 1)
    css_anchor = '.layout{display:grid;grid-template-columns:'
    if source.count(css_anchor) != 1:
        raise RuntimeError('GMM07 3D: CSS inesperado.')
    source = source.replace(css_anchor,
                            '.status-fronteira{font-size:13px;color:#44566d;margin:13px 0 0;line-height:1.5}\n' + css_anchor, 1)
    js_original = "el.disabled=true;el.parentElement.title='Sem geometria disponivel nesse corte';"
    js_novo = "el.disabled=true;el.checked=false;el.parentElement.title='Sem superficie disponivel para esse limiar';"
    if js_original in source:
        source = source.replace(js_original, js_novo, 1)
    elif js_novo not in source:
        raise RuntimeError('GMM07 3D: comportamento de controles indisponiveis mudou.')
    return source, 'GMM07 3D: HTML explica as fronteiras ausentes e desmarca controles indisponiveis.'


MODULES = {
    'visualizacao_gmm04.py': patch_gmm04,
    'visualizacao_gmm07.py': patch_gmm07,
    'visualizacao_gmm07_3d.py': patch_gmm07_3d,
    'visualizacao_gmm08_3d.py': patch_gmm08,
}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verificar', action='store_true')
    group.add_argument('--aplicar', action='store_true')
    args = parser.parse_args()

    raiz = Path.cwd()
    if not (raiz / 'src' / 'visualizacao_geometrica.py').is_file():
        parser.error('Execute na raiz do TCC_EXPERIMENTAL_V3 (src/ ausente).')
    updates = []
    errors = []
    for filename, apply in MODULES.items():
        path = raiz / 'src' / filename
        if not path.is_file():
            errors.append(f'{filename}: arquivo nao encontrado')
            continue
        raw = path.read_bytes()
        bom = b'\xef\xbb\xbf' if raw.startswith(b'\xef\xbb\xbf') else b''
        try:
            text = raw.decode('utf-8-sig')
            new_text, message = apply(text)
            compile(new_text, str(path), 'exec')
            # Ensure no accidental loss of executable code via the AST removals.
            updates.append((path, raw, bom, new_text, message))
        except Exception as e:
            errors.append(f'{filename}: {type(e).__name__}: {e}')
    for _, raw, _, new_text, message in updates:
        print(message)
    if errors:
        print('\nBLOQUEADO; NENHUM ARQUIVO MODIFICADO:', *errors, sep='\n  - ', file=sys.stderr)
        return 1
    if args.verificar:
        print('\nPREFLIGHT APROVADO. Simulacao das modificacoes. Nenhum arquivo alterado.')
        return 0
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    written = []
    try:
        for path, raw, bom, new_text, message in updates:
            new_raw = bom + new_text.encode('utf-8')
            if new_raw == raw:
                print('JA ATUALIZADO:', path.name)
                continue
            backup = path.with_name(path.name + '.ANTES_AJUSTE_FRONTEIRAS_' + stamp + '.bak')
            if backup.exists():
                raise RuntimeError('Backup ja existe; parada preventiva.')
            backup.write_bytes(raw)
            path.write_bytes(new_raw)
            written.append((path, raw, backup))
            print('ATUALIZADO:', path.name, '| BACKUP:', backup.name)
    except Exception:
        for path, raw, _ in reversed(written):
            path.write_bytes(raw)
        raise
    print('\nCONCLUIDO: somente codigo visual atualizado. YAML, OOF e F10 preservados.')
    print('ATENCAO: PNGs/HTMLs antigos nao sao modificados; devem ser regenerados.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
