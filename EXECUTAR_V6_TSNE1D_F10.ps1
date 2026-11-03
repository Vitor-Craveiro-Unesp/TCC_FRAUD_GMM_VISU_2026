$ErrorActionPreference = 'Stop'

$raiz = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $raiz

$python = Join-Path $raiz '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Python do .venv nao localizado.'
}

$modulo = 'scripts.execucao.v6_diagnosticos_tsne1d_f10'

foreach ($modo in @('--planejar', '--executar')) {
    Write-Host "`n=== V6: $modo ===" -ForegroundColor Cyan
    $anterior = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $python -B -m $modulo $modo
        $resultado = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $anterior
    }
    Write-Host "V6_$($modo)_EXIT_CODE=$resultado"
    if ($resultado -ne 0) {
        throw "V6 interrompida em $modo. Preserve a pasta parcial e envie o traceback."
    }
}

$teste = Join-Path $raiz 'outputs\ensaio_tsne_gmm_20261102_134606\vencedor\teste_final'
$saida = Join-Path $teste 'diagnosticos_tsne_v6'
$html = Join-Path $saida 'PRE01_F10\PRE01_F10_diagnostico_padronizacao_completo.html'
$manifesto = Join-Path $saida 'manifesto_v6.json'
$png1 = Join-Path $saida '13_tsne_mapa_TSNE1.png'
$png2 = Join-Path $saida '14_tsne_diagnostico.png'

foreach ($arquivo in @($html, $manifesto, $png1, $png2)) {
    if (-not (Test-Path -LiteralPath $arquivo)) {
        throw "Saida V6 obrigatoria ausente: $arquivo"
    }
}

Start-Process -FilePath (Resolve-Path -LiteralPath $html).Path
Write-Host "`n[OK] V6 APROVADA: TSNE13, TSNE14 e PRE01 F10." -ForegroundColor Green
Write-Host 'V6_GRAFICOS_VERIFICADOS=2/2'
Write-Host 'V6_PRE01_HTML_ABERTO=SIM'
Write-Host 'OOF_REEXECUTADO=NAO'
Write-Host 'TSNE_GMM_REAJUSTADOS=NAO'
Write-Host 'V3_PREDICOES_RECALCULADAS=NAO'
Write-Host 'V6_EXIT_CODE=0'
