# -*- coding: utf-8 -*-
"""Finaliza PRE01 V2 a partir dos artefatos ja calculados; sem ajuste de modelos."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import html
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from src import pre01_v2_diagnostico as old

ONE_LINE = re.compile(
    r'(?m)^(?P<indent>[ \t]*)permitido\s*=\s*arq\.stem\s+in\s+'
    r'\(oof\.ARQUIVOS\[n\]\s+for\s+n\s+in\s+ativos_oof\)[^\n]*$'
)


def require(c, msg):
    if not c:
        raise RuntimeError(msg)


def patched_integrated_source(original):
    matches = list(ONE_LINE.finditer(original))
    require(len(matches) == 1,
            'Filtro OOF nao encontrado uma unica vez em src/visualizacao_integrada.py; nada alterado.')
    m = matches[0]
    if ('PRE01_V2_IGNORAR_PNG' in original and old.FILENAME in m.group(0)
            and f"permitido = permitido and arq.name != {old.OLD_PNG!r}" in original):
        return original
    indent = m.group('indent')
    start = 'permitido=arq.stem in (oof.ARQUIVOS[n] for n in ativos_oof)'
    substitute = (
        f'{indent}# PRE01_V2_IGNORAR_PNG\n'
        f'{indent}{start} or arq.name == {old.FILENAME!r}\n'
        f'{indent}permitido = permitido and arq.name != {old.OLD_PNG!r}'
    )
    modified = original[:m.start()] + substitute + original[m.end():]
    ast.parse(modified)
    return modified


def patched_v2_module(original):
    # Substitui apenas a funcao que falhou; altera o modulo para futuras execucoes.
    pattern = re.compile(r'(?ms)^def future_integration\(root\):\n.*?(?=^def run\(root,k,n_init,tipo\):)')
    m = pattern.search(original)
    require(m is not None, 'Funcao future_integration nao identificada; arquivos preservados.')
    substitute = '''def future_integration(root):
    """Integra PRE01, inclusive em projetos que ja receberam o PRE01 V1."""
    import ast
    p = root / 'src' / 'visualizacao_integrada.py'
    texto = p.read_text(encoding='utf-8')
    padrao = re.compile(
        r'(?m)^(?P<indent>[ \\t]*)permitido\\s*=\\s*arq\\.stem\\s+in\\s+'
        r'\\(oof\\.ARQUIVOS\\[n\\]\\s+for\\s+n\\s+in\\s+ativos_oof\\)[^\\n]*$'
    )
    resultados = list(padrao.finditer(texto))
    exig(len(resultados) == 1, 'Filtro OOF inesperado; nenhuma alteracao aplicada.')
    m = resultados[0]
    if ('PRE01_V2_IGNORAR_PNG' in texto and FILENAME in m.group(0)
            and 'permitido = permitido and arq.name != ' + repr(OLD_PNG) in texto):
        ast.parse(texto)
        return p, texto.encode('utf-8')
    espaco = m.group('indent')
    novo = (
        espaco + '# PRE01_V2_IGNORAR_PNG\\n'
        + espaco + "permitido=arq.stem in (oof.ARQUIVOS[n] for n in ativos_oof) or arq.name == " + repr(FILENAME) + '\\n'
        + espaco + 'permitido = permitido and arq.name != ' + repr(OLD_PNG)
    )
    resultado = texto[:m.start()] + novo + texto[m.end():]
    ast.parse(resultado)
    return p, resultado.encode('utf-8')


'''
    patched = original[:m.start()] + substitute + original[m.end():]
    ast.parse(patched)
    return patched


def find_staged_bundle(root, k, n_init, tipo):
    outputs = root / 'outputs'
    for p in sorted(outputs.glob('validacao_PRE01_V2_*'),
                    key=lambda q: q.stat().st_mtime if q.is_dir() else -1,
                    reverse=True):
        if not p.is_dir():
            continue
        mp = p / old.META
        if not mp.is_file():
            continue
        try:
            meta = json.loads(mp.read_text(encoding='utf-8-sig'))
            if (meta.get('tipo') != tipo or int(meta.get('k', -1)) != k
                    or int(meta.get('n_init', -1)) != n_init
                    or meta.get('f10_usado') is not False):
                continue
            expected = [old.FILENAME, old.CSV_SUM, old.CSV_DETAIL,
                        old.CSV_FOLDS, old.CSV_PCA, old.META]
            require(all((p/f).is_file() and (p/f).stat().st_size > 0 for f in expected),
                    f'Backup PRE01 V2 parcial em {p}.')
            return p, meta, expected
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    raise RuntimeError('Nenhum backup completo de PRE01 V2 K=4/n_init=5 encontrado em outputs/validacao_PRE01_V2_*. '
                       'Os arquivos calculados nao podem ser recuperados automaticamente.')


def check_staged(p, meta):
    s = pd.read_csv(p/old.CSV_SUM)
    d = pd.read_csv(p/old.CSV_DETAIL)
    f = pd.read_csv(p/old.CSV_FOLDS)
    pcs = pd.read_csv(p/old.CSV_PCA)
    require(len(s) > 0 and len(d) == len(s)*9,
            'CSV de folds/variaveis incompleto; backup nao confiavel.')
    require(set(d['fold_validacao'].astype(int)) == set(range(1,10)),
            'CSV PRE01 V2 sem os nove folds.')
    require(len(f)==9 and set(f['fold'].astype(int))==set(range(1,10)),
            'CSV de resumo dos folds incompleto.')
    require(len(pcs)==27, 'CSV de estabilidade PCA deveria ter 27 linhas (9 folds * 3 PCs).')
    require(len(s)==int(meta['total_features']), 'Total de variaveis difere do manifesto PRE01.')
    page = (p/old.FILENAME).read_text(encoding='utf-8')
    require('table-details' in page and 'table-summary' in page and 'PRE01_DATA' in page,
            'HTML completo nao passou na verificacao de integridade.')
    return len(s), len(d)


def load_trio(meta,k,n_init,tipo):
    experiments = meta.get('experimentos') or {}
    require(all(f'{d}D' in experiments for d in (1,2,3)),
            'Manifesto de calculo PRE01 V2 sem as tres pastas de origem.')
    trio = []
    for d in (1,2,3):
        exp = Path(experiments[f'{d}D']).resolve()
        require(exp.is_dir(),f'Experimento desapareceu: {exp}')
        c=old.candidate(exp,k,n_init,tipo)
        require(c is not None and c['dim']==d,
                f'Experimento PCA {d}D alterado ou nao concluido; nada modificado.')
        require(c['meta'].get('sha256_parquet')==meta['sha256_parquet'] and
                c['meta'].get('sha256_folds')==meta['sha256_folds'],
                f'Hashes dos dados/folds divergem no PCA {d}D.')
        trio.append(c)
    require(len({c['key'] for c in trio})==1,
            'Configuracoes PCA 1D/2D/3D nao equivalentes.')
    ids, folds=old.check_indices(trio)
    require(len(ids)==int(meta['total_observacoes']),
            'Tamanho OOF diferente da origem do diagnostico PRE01 V2.')
    require((folds!=10).all(), 'F10 detectado. Operacao bloqueada.')
    return trio


def run(root,k,n_init,tipo):
    require((root/'src'/'orquestracao_oof.py').is_file(),
            'Execute na raiz do TCC_EXPERIMENTAL_V3.')
    require((root/'src'/'pre01_v2_diagnostico.py').is_file(),
            'Modulo PRE01 V2 nao encontrado em src/.')
    staging,meta,filenames=find_staged_bundle(root,k,n_init,tipo)
    print('[RECUPERACAO] Arquivos pre-calculados:',staging,flush=True)
    nfeatures,nrows=check_staged(staging,meta)
    print(f'[DADOS] {nfeatures} variaveis, {nrows} linhas (fold, variavel) e 27 registros PCA; SEM NOVOS FITS.',flush=True)
    trio=load_trio(meta,k,n_init,tipo)
    for t in trio:
        print(f"[ORIGEM] PCA {t['dim']}D: {t['folder']}",flush=True)

    # Preparar e validar codigo-fonte/HTML antes de tocar nos arquivos reais.
    integrated = root/'src'/'visualizacao_integrada.py'
    v2mod = root/'src'/'pre01_v2_diagnostico.py'
    src_integrated=patched_integrated_source(integrated.read_text(encoding='utf-8'))
    src_v2mod=patched_v2_module(v2mod.read_text(encoding='utf-8'))
    edits = {integrated:src_integrated.encode('utf-8'),
             v2mod:src_v2mod.encode('utf-8')}
    blobs={f:(staging/f).read_bytes() for f in filenames}
    for t in trio:
        gallery=t['gallery']
        html_new=old.new_gallery_index(gallery.read_text(encoding='utf-8'))
        require(old.NEW_MARKER in html_new, 'Marcador PRE01 V2 nao foi incluido.')
        view=t['folder']/'vencedor'/'visualizacao'
        for destfolder in (view/'oof',view/'galeria'/'oof'):
            for name,data in blobs.items():
                edits[destfolder/name]=data
        edits[gallery]=html_new.encode('utf-8')

    links=''.join(
        f'<li><a href="{html.escape((t["folder"] / "vencedor" / "visualizacao" / "galeria" / "oof" / old.FILENAME).relative_to(root / "outputs").as_posix(),quote=True)}">'
        f'PCA {t["dim"]}D — Abrir PRE01 V2 completo</a></li>'
        for t in trio
    )
    panel=root/'outputs'/'ABRIR_PRE01_V2_TRES_GALERIAS.html'
    panel_html=('''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>PRE01 V2 | PCA 1D / 2D / 3D</title><style>body{background:#f5f7fb;color:#20324a;font:17px system-ui;max-width:960px;margin:65px auto}li{padding:9px}a{color:#245c98}</style></head><body><h1>PRE01 — Diagnostico completo da padronizacao</h1><p>Variaveis completas, 9 folds de desenvolvimento, medias, variancias, escalas e PCA diagnostico. F10 reservado.</p><ul>'''+links+'''</ul></body></html>''')
    edits[panel]=panel_html.encode('utf-8')

    # Backup de cada arquivo sobrescrito, rollback se alguma validacao falhar.
    backup=root/'outputs'/('validacao_PRE01_V2_INTEGRACAO_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True,exist_ok=False)
    completed=[]
    try:
        for dest, data in edits.items():
            b=old.backup_one(dest,root,backup)
            completed.append((dest,b))
            old.write_atomic(dest,data)
        for t in trio:
            g=t['gallery']
            path=g.parent/'oof'/old.FILENAME
            require(path.is_file() and path.stat().st_size>1000,
                    f'HTML PRE01 ausente: {path}')
            require(old.NEW_MARKER in g.read_text(encoding='utf-8'),
                    f'Link PRE01 V2 ausente no PCA {t["dim"]}D.')
            print(f'[OK] PCA {t["dim"]}D: cartao PRE01 V2 e HTML completo presentes.',flush=True)
    except Exception:
        for dest,copy in reversed(completed):
            if copy is not None: shutil.copy2(copy,dest)
            else:dest.unlink(missing_ok=True)
        raise
    print('[SUCESSO] PRE01 V2 nas tres galerias. Gráfico PNG antigo preservado no disco, oculto da pagina.',flush=True)
    print('PAINEL='+str(panel),flush=True)
    print('BACKUP='+str(backup),flush=True)
    print('[PRESERVADO] OOF, PCA/GMM existentes e F10. Nada foi treinado.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raiz',required=True,type=Path)
    p.add_argument('--k',type=int,default=4)
    p.add_argument('--n-init',type=int,default=5)
    p.add_argument('--tipo',choices=['ensaio','oficial'],default='ensaio')
    a=p.parse_args()
    run(a.raiz.resolve(),a.k,a.n_init,a.tipo)


if __name__ == '__main__':
    main()
