$ErrorActionPreference = 'Stop'

$raiz = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $raiz

$python = Join-Path $raiz '.venv\Scripts\python.exe'
$script = Join-Path $raiz 'scripts\execucao\v7_galeria_tsne1d_f10.py'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python do ambiente virtual nao encontrado: $python"
}
if (-not (Test-Path -LiteralPath $script -PathType Leaf)) {
    throw "Script da V7 nao encontrado: $script"
}

$modulo = 'scripts.execucao.v7_galeria_tsne1d_f10'

foreach ($modo in @('--planejar', '--executar', '--validar')) {
    Write-Host "`n=== V7: $modo ===" -ForegroundColor Cyan

    $original = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $python -B -m $modulo $modo
        $codigo = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $original
    }

    Write-Host "V7_$($modo.TrimStart('-'))_EXIT_CODE=$codigo"
    if ($codigo -ne 0) {
        throw "V7 interrompida em $modo. Preserve as saidas e envie o traceback."
    }
}

$galeria = Join-Path $raiz `
    'outputs\ensaio_tsne_gmm_20261102_134606\vencedor\teste_final\galeria'
$pagina = Join-Path $galeria 'index.html'
$manifesto = Join-Path $galeria 'manifesto_v7.json'

if (-not (Test-Path -LiteralPath $pagina -PathType Leaf)) {
    throw 'Galeria V7 nao existe apos validacao.'
}
if (-not (Test-Path -LiteralPath $manifesto -PathType Leaf)) {
    throw 'Manifesto V7 nao existe apos validacao.'
}

$meta = Get-Content -LiteralPath $manifesto -Raw -Encoding UTF8 | ConvertFrom-Json
if ($meta.status -ne 'APROVADA' -or $meta.total_graficos -ne 12) {
    throw 'Manifesto V7 nao confirma aprovacao de todos os graficos.'
}

Start-Process -FilePath (Resolve-Path -LiteralPath $pagina).Path

Write-Host "`n[OK] GALERIA TSNE F10 CONSOLIDADA E ABERTA." -ForegroundColor Green
Write-Host 'V7_GRAFICOS_VERIFICADOS=12/12'
Write-Host 'V7_PRE01_INTERATIVO=APROVADO'
Write-Host 'V7_STATUS=APROVADA'
Write-Host 'V7_HTML_ABERTO=SIM'
Write-Host 'OOF_REEXECUTADO=NAO'
Write-Host 'MODELO_TSNE_GMM_REAJUSTADO=NAO'
Write-Host 'PREVISOES_F10_RECALCULADAS=NAO'
Write-Host 'LIMIAR_F10_REOTIMIZADO=NAO'
Write-Host 'V7_EXIT_CODE=0'
Write-Host "GALERIA_HTML=$pagina"
