# Corrige somente o layout das legendas do grafico 07 (fold OOF)
# e substitui as copias PNG nas 3 galerias compativeis mais recentes.
# NAO executa OOF, NAO reajusta modelos, NAO le o dataset e NAO acessa F10.
[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$raiz = (Get-Location).Path
$python = Join-Path $raiz '.venv\Scripts\python.exe'
$fonte = Join-Path $raiz 'src\visualizacao_oof.py'
$outputs = Join-Path $raiz 'outputs'
foreach ($alvo in @($python, $fonte, $outputs)) {
    if (-not (Test-Path -LiteralPath $alvo)) { throw "Nao encontrado: $alvo" }
}

# Nao alterar o gerador enquanto o OOF estiver em execucao.
$ativos = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'src\.orquestracao_oof' })
if ($ativos.Count -gt 0) {
    throw 'Existe um processo OOF rodando. Conclua a execucao antes de corrigir as galerias.'
}

$arquivoTemporario = Join-Path $env:TEMP ("corrigir_legendas_fold_oof_" + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.py')
$scriptPython = @'
from __future__ import annotations

import copy
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback

import yaml

ROOT = Path.cwd()
OUTPUTS = ROOT / 'outputs'
SOURCE = ROOT / 'src' / 'visualizacao_oof.py'
NAME = '07_metricas_por_fold.png'


def die(message):
    raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def timestamp(path):
    m = re.fullmatch(r'(ensaio|experimento)_pca_gmm_(\d{8}_\d{6})', path.name)
    if not m:
        return None
    return datetime.datetime.strptime(m.group(2), '%Y%m%d_%H%M%S')


def find_groups():
    groups = {}
    for path in OUTPUTS.iterdir():
        if not path.is_dir():
            continue
        moment = timestamp(path)
        if moment is None:
            continue
        try:
            config = yaml.safe_load((path / 'configuracao.yaml').read_text(encoding='utf-8-sig'))
            manifest = json.loads((path / 'manifesto.json').read_text(encoding='utf-8-sig'))
            dim_raw = config['pca']['n_components']
            dim = int(dim_raw[0] if isinstance(dim_raw, list) else dim_raw)
            k_raw = config['gmm']['n_components']
            k = int(k_raw[0] if isinstance(k_raw, list) else k_raw)
            if dim not in (1, 2, 3) or k != 3:
                continue
            if config['reducao']['metodo'].upper() != 'PCA':
                continue
            if config['gmm']['covariance_type'].lower() != 'full':
                continue
            if int(config['divisao']['qual_fold_teste']) != 10:
                continue
            if manifest.get('status') != 'concluido_06_03':
                continue
            if manifest.get('visualizacoes') != 'concluidas':
                continue
            if manifest.get('tipo') not in ('ensaio', 'oficial'):
                continue
            oof = path / 'vencedor' / 'visualizacao' / 'oof' / NAME
            gal = path / 'vencedor' / 'visualizacao' / 'galeria' / 'oof' / NAME
            index = path / 'vencedor' / 'visualizacao' / 'galeria' / 'index.html'
            if not (oof.is_file() and gal.is_file() and index.is_file()):
                continue
            normalized = copy.deepcopy(config)
            normalized['pca']['n_components'] = ['DIMENSAO_VARIAVEL']
            normalized_txt = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
            key = hashlib.sha256(normalized_txt.encode('utf-8')).hexdigest()
            groups.setdefault(key, {}).setdefault(dim, []).append((moment, path, config))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, yaml.YAMLError):
            continue
    complete = []
    for group in groups.values():
        if all(i in group for i in (1, 2, 3)):
            combo = {dim: max(group[dim], key=lambda v: v[0]) for dim in (1, 2, 3)}
            complete.append(combo)
    if not complete:
        die('Nao encontrei um trio PCA 1D/2D/3D concluido com config identica exceto n_components. Nenhum PNG foi alterado.')
    return max(complete, key=lambda g: min(g[d][0] for d in (1, 2, 3)))


def source_patch():
    raw = SOURCE.read_bytes()
    txt = raw.decode('utf-8-sig')
    start = txt.find('def _plot_metricas_por_fold(')
    end = txt.find('\ndef gerar_graficos_oof(', start)
    if start < 0 or end < 0:
        die('Nao encontrei a funcao _plot_metricas_por_fold. Codigo nao alterado.')
    frag = txt[start:end]
    old_adjust = 'fig.subplots_adjust(left=.13, right=.965, top=.88, bottom=.075, hspace=.49)'
    new_adjust = 'fig.subplots_adjust(left=.13, right=.76, top=.88, bottom=.075, hspace=.49)'
    old_legend = "ax.legend(frameon=False, fontsize=9, loc='upper right')"
    new_legend = ("ax.legend(frameon=False, fontsize=9, loc='upper left', "
                  "bbox_to_anchor=(1.02, 1.0), borderaxespad=0)")
    if frag.count(old_adjust) == 1 and frag.count(old_legend) == 3:
        updated = frag.replace(old_adjust, new_adjust).replace(old_legend, new_legend)
    elif frag.count(new_adjust) == 1 and frag.count(new_legend) == 3 and old_legend not in frag:
        print('[CODIGO] Legendas ja corrigidas; nenhuma nova edicao necessaria.')
        return raw, raw, False
    else:
        die('O codigo do grafico 07 e diferente do checkpoint esperado; arquivo preservado.')
    new_txt = txt[:start] + updated + txt[end:]
    compile(new_txt, str(SOURCE), 'exec')
    bom = b'\xef\xbb\xbf' if raw.startswith(b'\xef\xbb\xbf') else b''
    result = bom + new_txt.encode('utf-8')
    return raw, result, True


def atomic_copy(origin, destination):
    temp = destination.with_name(destination.name + '.corrigindo.tmp')
    try:
        shutil.copy2(origin, temp)
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def main():
    trio = find_groups()
    print('TRIO COMPATIVEL SELECIONADO:')
    for dim in (1, 2, 3):
        moment, path, _ = trio[dim]
        print(f'  PCA {dim}D: {path}')

    root_backup = OUTPUTS / ('validacao_legendas_fold_oof_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    root_backup.mkdir(parents=True, exist_ok=False)
    print('BACKUPS E PREVIAS:', root_backup)

    original, new_source, modified = source_patch()
    source_backup = root_backup / 'visualizacao_oof.py.ANTES_LEGENDA.bak'
    source_backup.write_bytes(original)
    source_applied = False
    replacements = []
    try:
        if modified:
            temp_source = SOURCE.with_name(SOURCE.name + '.atualizando.tmp')
            try:
                temp_source.write_bytes(new_source)
                os.replace(temp_source, SOURCE)
            finally:
                if temp_source.exists():
                    temp_source.unlink()
            source_applied = True
            print('[CODIGO] Legendas movidas para coluna direita, fora dos eixos.')

        importlib.invalidate_caches()
        from src import visualizacao_oof as voof
        from src import estrutura_saida as layout

        previews = []
        for dim in (1, 2, 3):
            _, exp, _ = trio[dim]
            cfg, grade, escolha, y, p = voof._ler_resultados(exp)
            if int(escolha['configuracao']['reducao']['n_components']) != dim:
                die(f'Dimensionalidade do vencedor nao bate com PCA {dim}D: {exp}')
            preview_dir = root_backup / f'PCA_{dim}D' / 'preview'
            preview_dir.mkdir(parents=True, exist_ok=True)
            dpi = int(cfg['visualizacao']['padrao'].get('dpi', 180))
            voof._plot_metricas_por_fold(exp, cfg, y, p, escolha, preview_dir, 'png', dpi, False)
            preview = preview_dir / NAME
            if not preview.is_file() or preview.stat().st_size < 10000:
                die(f'Previa ausente ou invalida para PCA {dim}D')
            if preview.read_bytes()[:8] != b'\x89PNG\r\n\x1a\n':
                die(f'Cabecalho PNG invalido para PCA {dim}D')
            previews.append((dim, exp, preview))
            print(f'[PREVIA PCA {dim}D] gerada a partir do NPZ OOF existente.')

        # Criar backups de TODOS os PNGs antes de substituir qualquer arquivo.
        updates = []
        for dim, exp, preview in previews:
            targets = [
                ('oof', exp / 'vencedor' / 'visualizacao' / 'oof' / NAME),
                ('galeria', exp / 'vencedor' / 'visualizacao' / 'galeria' / 'oof' / NAME),
            ]
            for kind, target in targets:
                if not target.is_file():
                    die('PNG esperado nao encontrado: ' + str(target))
                backup = root_backup / f'PCA_{dim}D' / 'originais' / f'{kind}_{NAME}'
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
                if sha256(backup) != sha256(target):
                    die('Falha na verificacao do backup: ' + str(target))
                updates.append((dim, preview, target, backup))

        for dim, preview, target, backup in updates:
            atomic_copy(preview, target)
            replacements.append((target, backup))
            if sha256(preview) != sha256(target):
                die('Divergencia apos substituicao: ' + str(target))
            print(f'[ATUALIZADO PCA {dim}D] {target}')

        galleries = [str(trio[d][1] / 'vencedor' / 'visualizacao' / 'galeria' / 'index.html') for d in (1, 2, 3)]
        (root_backup / 'galerias.txt').write_text('\n'.join(galleries), encoding='utf-8')
        print('\nCONCLUIDO: 3 figuras, 6 copias PNG atualizadas, fonte corrigida.')
        print('F1-F9 reutilizados. Nenhum modelo treinado. F10 intacto.')
        print('Para atualizar a imagem no navegador: Ctrl+F5.')
        print('GALERIAS_TXT=' + str(root_backup / 'galerias.txt'))
    except Exception:
        print('[ERRO] Restaurando imagens alteradas nesta tentativa...', file=sys.stderr)
        for target, backup in reversed(replacements):
            try:
                atomic_copy(backup, target)
            except Exception as exc:
                print('Falha de restauracao:', target, str(exc), file=sys.stderr)
        if source_applied:
            try:
                SOURCE.write_bytes(original)
            except Exception as exc:
                print('Falha restaurando codigo:', str(exc), file=sys.stderr)
        raise

if __name__ == '__main__':
    main()
'@

$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($arquivoTemporario, $scriptPython, $utf8)

try {
    & $python $arquivoTemporario
    $codigo = $LASTEXITCODE
    if ($codigo -ne 0) { throw "Falha ao corrigir as legendas. Codigo de saida: $codigo" }
    Write-Host "`nLEGENDAS CORRIGIDAS EM PCA 1D, PCA 2D E PCA 3D!" -ForegroundColor Green
    Write-Host 'Atualize o navegador com Ctrl+F5. Nenhum treino/OOF/F10 executado.'
}
finally {
    if (Test-Path -LiteralPath $arquivoTemporario) {
        Remove-Item -LiteralPath $arquivoTemporario -Force
    }
}
