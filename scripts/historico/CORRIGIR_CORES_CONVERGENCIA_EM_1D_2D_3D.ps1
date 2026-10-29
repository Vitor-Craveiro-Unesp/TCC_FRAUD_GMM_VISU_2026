# Ajuste exclusivamente visual do GMM18 nos tres experimentos PCA.
# Modifica src/visualizacao_gmm18.py e regenera os PNGs a partir dos CSVs EXISTENTES.
# Nao treina modelos, nao recalcula OOF, nao acessa F10 e nao altera YAML.

$ErrorActionPreference = 'Continue'  # Avisos nao fatais do Matplotlib em stderr.
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$raiz = (Get-Location).Path
$python = Join-Path $raiz '.venv\Scripts\python.exe'
$fonte = Join-Path $raiz 'src\visualizacao_gmm18.py'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Ambiente Python .venv nao encontrado. Execute na raiz do TCC.'
}
if (-not (Test-Path -LiteralPath $fonte -PathType Leaf)) {
    throw 'src/visualizacao_gmm18.py nao encontrado. Execute na raiz do TCC.'
}
$carimbo = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$saida = Join-Path $raiz "outputs\validacao_cores_convergencia_em_$carimbo"
New-Item -ItemType Directory -Path $saida -Force | Out-Null
$arquivoPython = Join-Path $saida 'corrigir_cores_convergencia_em.py'

$codigo = @'
from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
import yaml

RAIZ = Path.cwd().resolve()
BACKUP = Path(sys.argv[1]).resolve()
FONTE = RAIZ / 'src' / 'visualizacao_gmm18.py'
NOME = 'GMM18_convergencia_em_inicializacoes'
# Cores de alto contraste; linhas/markers complementam as cores.
CORES = {1: '#B12A90', 2: '#D55E00', 3: '#0057B8'}
EXPERIMENTOS = {
    1: 'experimento_pca_gmm_20261028_113443',
    2: 'experimento_pca_gmm_20261028_113726',
    3: 'experimento_pca_gmm_20261028_114038',
}


def alterar_uma_vez(texto: str, antigo: str, novo: str) -> str:
    ocorrencias = texto.count(antigo)
    if ocorrencias == 0 and novo in texto:
        return texto  # Reexecutar e seguro.
    if ocorrencias != 1:
        raise RuntimeError(f'Ancora ausente/ambigua (encontradas {ocorrencias}): {antigo[:80]}')
    return texto.replace(antigo, novo, 1)


def preparar_fonte(original: str) -> str:
    novo = original
    novo = alterar_uma_vez(
        novo,
        "cores = ['#96a7bb', '#d19b69', '#7599ac', '#b9a1bf', '#a6ad8f']",
        "cores = ['#B12A90', '#D55E00', '#0057B8', '#009E73', '#7F3C8D']"
    )
    novo = alterar_uma_vez(
        novo,
        "cor = '#235a93' if selecionada else cores[(indice - 1) % len(cores)]",
        "cor = cores[(indice - 1) % len(cores)]"
    )
    novo = alterar_uma_vez(
        novo,
        "ls='-' if selecionada else '--',",
        "ls='-' if selecionada else ('-.' if indice % 2 else '--'),"
    )
    novo = alterar_uma_vez(
        novo,
        'linewidth=2.8 if selecionada else 1.65,',
        'linewidth=3.2 if selecionada else 2.2,'
    )
    novo = alterar_uma_vez(
        novo,
        'alpha=1 if selecionada else .78, label=rotulo,',
        'alpha=1.0, label=rotulo,'
    )
    compile(novo, str(FONTE), 'exec')
    return novo


def carregar(dim: int):
    base = RAIZ / 'outputs' / EXPERIMENTOS[dim]
    if not base.is_dir():
        raise FileNotFoundError(f'Experimento PCA {dim}D inexistente: {base}')
    cfg = yaml.safe_load((base / 'configuracao.yaml').read_text(encoding='utf-8-sig'))
    partes = cfg['pca']['n_components']
    if isinstance(partes, list):
        assert partes == [dim], f'Dimensao PCA divergente na pasta {base}'
    else:
        assert int(partes) == dim, f'Dimensao PCA divergente na pasta {base}'
    ncomp = cfg['gmm']['n_components']
    assert ncomp == [3] or ncomp == 3, 'Esperado GMM K=3.'
    assert str(cfg['gmm']['covariance_type']).lower() == 'full', 'Esperado GMM FULL.'
    assert int(cfg['divisao']['qual_fold_teste']) == 10, 'F10 nao esta reservado.'

    pasta_geo = base / 'vencedor' / 'visualizacao' / 'geometria_desenvolvimento'
    pasta_galeria = base / 'vencedor' / 'visualizacao' / 'galeria' / 'geometria_desenvolvimento'
    csv = pasta_geo / (NOME + '.csv')
    origens = [pasta_geo / (NOME + '.png'), pasta_galeria / (NOME + '.png')]
    if not csv.is_file():
        raise FileNotFoundError(f'CSV de convergencia nao encontrado: {csv}')
    for png in origens:
        if not png.is_file():
            raise FileNotFoundError(f'PNG nao encontrado: {png}')

    t = pd.read_csv(csv)
    requeridas = {'inicializacao','iteracao','lower_bound_medio_por_amostra',
                  'inicializacao_selecionada','convergiu'}
    assert requeridas.issubset(t.columns), f'Colunas ausentes em {csv}: {requeridas - set(t.columns)}'
    assert set(t['inicializacao'].astype(int)) == {1, 2, 3}, f'CSV sem tres inicializacoes: {csv}'
    assert np.isfinite(t['lower_bound_medio_por_amostra'].to_numpy(dtype=float)).all(), 'Lower bounds invalidos.'
    escolheu = t['inicializacao_selecionada'].astype(str).str.lower().isin(['true', '1'])
    ids_escolhidos = set(t.loc[escolheu, 'inicializacao'].astype(int))
    assert len(ids_escolhidos) == 1, f'Selecao inconsistente no CSV: {ids_escolhidos}'
    melhor = next(iter(ids_escolhidos))
    for idx, parte in t.groupby('inicializacao'):
        iteracoes = parte['iteracao'].astype(int).tolist()
        assert iteracoes == list(range(1, len(iteracoes) + 1)), f'Iteracoes inconsistentes em Init {idx}'
    dpi = int(cfg['visualizacao']['padrao'].get('dpi', 180))
    return t, melhor, dpi, origens


def gerar_png(t: pd.DataFrame, melhor: int, dim: int, dpi: int, saida: Path):
    fig, ax = plt.subplots(figsize=(12.5, 7.2), facecolor='white')
    fig.subplots_adjust(left=.135, right=.96, top=.78, bottom=.175)
    for indice, h in t.groupby('inicializacao', sort=True):
        indice = int(indice)
        h = h.sort_values('iteracao')
        xs = h['iteracao'].to_numpy(dtype=int)
        valores = h['lower_bound_medio_por_amostra'].to_numpy(dtype=float)
        selecionada = indice == melhor
        cor = CORES[indice]
        convergiu = str(h['convergiu'].iloc[-1]).strip().lower() in ('true','1')
        rotulo = f'Init {indice}' + (' — selecionada' if selecionada else '')
        if not convergiu:
            rotulo += ' (não convergiu)'
        ax.plot(xs, valores, color=cor,
                ls='-' if selecionada else ('-.' if indice % 2 else '--'),
                linewidth=3.2 if selecionada else 2.2, alpha=1,
                label=rotulo, zorder=4 if selecionada else 2)
        if not selecionada:
            ax.scatter(xs, valores,
                       marker='x' if indice % 2 == 0 else '+',
                       s=42, color=cor, linewidths=1.5, zorder=10)
        ax.scatter([xs[-1]], [valores[-1]], s=70 if selecionada else 35,
                   color=cor, edgecolors='white', linewidths=1,
                   zorder=6 if selecionada else 3)
    ax.set_xlabel('Iteração do algoritmo EM', fontsize=11)
    ax.set_ylabel('Log-verossimilhança média (lower bound)', fontsize=10.8)
    ax.set_xlim(left=1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.7f}'))
    ax.grid(axis='both', color='#dbe3ed', alpha=.7, lw=.8)
    ax.set_axisbelow(True)
    for lateral in ('top','right'):
        ax.spines[lateral].set_visible(False)
    fig.suptitle('CONVERGÊNCIA DO ALGORITMO EM', fontsize=16,
                 fontweight='bold', y=.968, color='#20324a')
    fig.text(.5, .895, f'{dim}D | FULL | K=3 | 3 inicializações',
             ha='center', fontsize=10.5, color='#61728a')
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .864), ncol=3, frameon=False, fontsize=9.5)
    h_melhor = t[t['inicializacao'] == melhor]
    fim = float(h_melhor['lower_bound_medio_por_amostra'].iloc[-1])
    iters = int(h_melhor['iteracao'].iloc[-1])
    fig.text(.135, .085,
             f'Melhor inicialização: {melhor}  |  '
             f'Lower bound final = {fim:.7f}  |  Iterações = {iters}',
             fontsize=9.8, color='#26374d', weight='bold')
    saida.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(saida, dpi=dpi, bbox_inches='tight')
    finally:
        plt.close(fig)
    if saida.stat().st_size < 15000:
        raise RuntimeError(f'PNG de visualizacao muito pequeno: {saida}')


def main():
    original = FONTE.read_text(encoding='utf-8')
    novo = preparar_fonte(original)
    tarefas = []
    # 1. Validar todas as fontes OOF/CSV antes de tocar em qualquer arquivo.
    for dim in (1, 2, 3):
        t, melhor, dpi, destinos = carregar(dim)
        previa = BACKUP / f'PCA_{dim}D_GMM18_PREVIA.png'
        gerar_png(t, melhor, dim, dpi, previa)
        tarefas.append((dim, previa, destinos))
        print(f'[PREVIA] PCA {dim}D, melhor Init {melhor}: {previa}', flush=True)

    # 2. Salvar backups de TODOS os arquivos que serao alterados.
    backup_fonte = BACKUP / 'visualizacao_gmm18.py.ANTES'
    shutil.copy2(FONTE, backup_fonte)
    backups = []
    for dim, previa, destinos in tarefas:
        for posicao, destino in enumerate(destinos):
            backup_png = BACKUP / f'PCA_{dim}D_{"GEO" if posicao == 0 else "GALERIA"}.png.ANTES'
            shutil.copy2(destino, backup_png)
            backups.append((destino, backup_png))

    # 3. Alterar codigo e substituir apenas os PNGs, reversivel se algo falhar.
    tocados = []
    fonte_alterada = False
    try:
        if novo != original:
            FONTE.write_text(novo, encoding='utf-8')
            fonte_alterada = True
            print('[FONTE] Cores atualizadas em src/visualizacao_gmm18.py', flush=True)
        else:
            print('[FONTE] Cores ja estavam atualizadas.', flush=True)
        for dim, previa, destinos in tarefas:
            for destino in destinos:
                shutil.copyfile(previa, destino)
                tocados.append(destino)
            print(f'[CORRIGIDO] PCA {dim}D: imagem geometria e copia da galeria', flush=True)
    except Exception:
        print('[ERRO] Restaurando arquivos originais a partir dos backups...', file=sys.stderr)
        for destino, backup_png in backups:
            if destino in tocados:
                shutil.copy2(backup_png, destino)
        if fonte_alterada:
            shutil.copy2(backup_fonte, FONTE)
        raise

    print('\nCONCLUIDO: 3 graficos EM atualizados, 6 PNGs e fonte corrigidos.', flush=True)
    print('Cores: Init 1 = magenta; Init 2 = laranja; Init 3 = azul-escuro.', flush=True)
    print('Backups e previas:', BACKUP, flush=True)
    print('Nenhum GMM/OOF treinado; YAML e F10 preservados.', flush=True)


if __name__ == '__main__':
    main()
'@

$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($arquivoPython, $codigo, $utf8)
Push-Location $raiz
try {
    & $python $arquivoPython $saida
    if ($LASTEXITCODE -ne 0) {
        throw "Falha na alteracao de cores. Arquivos anteriores preservados ou restaurados. Validacao: $saida"
    }
}
finally {
    Pop-Location
}
Write-Host "`nCORES DA CONVERGENCIA EM ATUALIZADAS NAS 3 GALERIAS!" -ForegroundColor Green
Write-Host "Backups e previas: $saida" -ForegroundColor Cyan
