$ErrorActionPreference = 'Stop'

$raiz = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $raiz
$python = Join-Path $raiz '.venv\Scripts\python.exe'
$script = Join-Path $raiz 'scripts\execucao\v5_geometria_f10_tsne1d.py'

if (-not (Test-Path -LiteralPath $python)) { throw 'Python do ambiente .venv ausente.' }
if (-not (Test-Path -LiteralPath $script)) { throw 'Script V5 ausente: $script' }

foreach ($modo in @('--planejar','--executar')) {
    Write-Host "`n=== V5: $modo ===" -ForegroundColor Cyan
    $anterior = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $python -B -m scripts.execucao.v5_geometria_f10_tsne1d $modo
        $codigo = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $anterior }
    Write-Host "V5_${modo}_EXIT_CODE=$codigo"
    if ($codigo -ne 0) { throw "V5 falhou em $modo; preserve qualquer pasta parcial e envie o traceback." }
}

$destino = Join-Path $raiz 'outputs\ensaio_tsne_gmm_20261102_134606\vencedor\teste_final\graficos_geometricos_v5'
$arquivos = @(
  'GMM01_densidade_marginal_1d_TSNE1.png',
  'GMM04_dominancia_1d.png',
  'GMM07_sensibilidade_1d_comparativo.png',
  'GMM08_mapa_risco_1d_comparativo.png',
  'manifesto_v5.json'
)
foreach ($nome in $arquivos) {
    $caminho = Join-Path $destino $nome
    if (-not (Test-Path -LiteralPath $caminho)) { throw "Artefato V5 nao encontrado: $nome" }
}

Start-Process explorer.exe -ArgumentList "`"$destino`""
Write-Host "`n[OK] V5 GEOMETRIA CONCLUIDA (4/4)." -ForegroundColor Green
Write-Host 'OOF_REEXECUTADO=NAO'
Write-Host 'GMM_TSNE_REAJUSTADOS=NAO'
Write-Host 'F10_REOTIMIZACAO_TAU=NAO'
Write-Host 'V5_EXIT_CODE=0'
