$ErrorActionPreference = 'Stop'
$raiz = 'C:\Users\Vitor Craveiro\Desktop\ClusterPosMac\MEU PROJETO\TCC_EXPERIMENTAL_V3'
Set-Location -LiteralPath $raiz
$python = Join-Path $raiz '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Python do projeto nao encontrado.' }
$modulo = 'scripts.execucao.v4_graficos_preditivos_tsne1d_f10'
foreach ($modo in @('--planejar', '--executar')) {
    Write-Host "`n=== V4: $modo ===" -ForegroundColor Cyan
    $anterior = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $python -B -m $modulo $modo
        $codigo = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $anterior
    }
    Write-Host "V4_${modo}_EXIT_CODE=$codigo"
    if ($codigo -ne 0) { throw "V4 falhou em $modo. Preserve os arquivos e envie o traceback." }
}
$saida = Join-Path $raiz 'outputs\ensaio_tsne_gmm_20261102_134606\vencedor\teste_final\graficos_preditivos_v4'
if (-not (Test-Path -LiteralPath (Join-Path $saida 'manifesto_v4.json'))) { throw 'Manifesto V4 nao encontrado.' }
Start-Process explorer.exe -ArgumentList "`"$saida`""
Write-Host '[OK] V4 finalizada; pasta dos seis PNGs aberta.' -ForegroundColor Green
Write-Host 'OOF_REEXECUTADO=NAO'
Write-Host 'F10_SCORES_RECALCULADOS=NAO'
Write-Host 'V4_EXECUCAO_EXIT_CODE=0'
