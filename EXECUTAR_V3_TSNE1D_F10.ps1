$ErrorActionPreference = 'Stop'

$raiz = 'C:\Users\Vitor Craveiro\Desktop\ClusterPosMac\MEU PROJETO\TCC_EXPERIMENTAL_V3'
Set-Location -LiteralPath $raiz

$python = Join-Path $raiz '.venv\Scripts\python.exe'
$modulo = 'scripts.execucao.v3_avaliacao_f10_tsne1d'
$script = Join-Path $raiz 'scripts\execucao\v3_avaliacao_f10_tsne1d.py'

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Python .venv nao encontrado.'
}
if (-not (Test-Path -LiteralPath $script)) {
    throw 'Extraia o ZIP V3 para a raiz do TCC antes de executar.'
}

Write-Host "`n=== V3 TSNE1D: PLANEJAMENTO ===" -ForegroundColor Cyan
$anterior = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $python -B -m $modulo --planejar
    $planejamento = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $anterior
}
Write-Host "V3_PLANEJAMENTO_EXIT_CODE=$planejamento"
if ($planejamento -ne 0) {
    throw 'V3 planejamento falhou; nao executar F10.'
}

Write-Host "`n=== V3 TSNE1D: F10 PAREADO OOS ===" -ForegroundColor Cyan
$anterior = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $python -B -m $modulo --executar --lote 500
    $resultado = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $anterior
}
Write-Host "V3_EXECUCAO_EXIT_CODE=$resultado"
if ($resultado -ne 0) {
    throw 'V3 interrompida. Preserve qualquer pasta .avaliacao_f10_v3_incompleta e envie o erro.'
}

$destino = Join-Path $raiz 'outputs\ensaio_tsne_gmm_20261102_134606\vencedor\teste_final\avaliacao_f10_v3'
$metricas = Join-Path $destino 'metricas_f10.json'
$manifesto = Join-Path $destino 'manifesto_v3.json'

if (-not (Test-Path -LiteralPath $metricas) -or -not (Test-Path -LiteralPath $manifesto)) {
    throw 'Resultado V3 incompleto.'
}

$m = Get-Content -LiteralPath $metricas -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host "`n=== METRICAS F10 TSNE + GMM 1D ===" -ForegroundColor Green
Write-Host ('AP       = {0:F7}' -f $m.ap)
Write-Host ('MCC      = {0:F7}' -f $m.mcc_limiar_oof)
Write-Host ('LogLoss  = {0:F7}' -f $m.logloss)
Write-Host ('Score    = {0:F7}' -f $m.score_composto_limiar_oof)
Write-Host 'V3_STATUS=APROVADA'
Write-Host 'OOF_REEXECUTADO=NAO'
Write-Host 'F10_LIMIAR_REOTIMIZADO=NAO'
Write-Host 'GRAFICOS_F10=A_PARTIR_DA_V4'

Start-Process explorer.exe -ArgumentList ('"' + $destino + '"')
