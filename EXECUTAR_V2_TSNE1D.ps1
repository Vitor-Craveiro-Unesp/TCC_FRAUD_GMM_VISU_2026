$ErrorActionPreference = 'Stop'
$raiz = 'C:\Users\Vitor Craveiro\Desktop\ClusterPosMac\MEU PROJETO\TCC_EXPERIMENTAL_V3'
$downloads = Join-Path $env:USERPROFILE 'Downloads'
Set-Location -LiteralPath $raiz
$script = Join-Path $raiz 'scripts\execucao\v2_ajuste_final_tsne1d.py'
if (-not (Test-Path -LiteralPath $script)) {
    $zip = Get-ChildItem -LiteralPath $downloads -File -Filter 'V2_AJUSTE_FINAL_TSNE_GMM_1D*.zip' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $zip) { throw 'Baixe o ZIP V2 em Downloads antes.' }
    Expand-Archive -LiteralPath $zip.FullName -DestinationPath $raiz
}
$python = Join-Path $raiz '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Ambiente Python do projeto nao localizado.' }
Write-Host "`n=== V2 - PLANEJAR ===" -ForegroundColor Cyan
$anterior = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try { & $python -B -m scripts.execucao.v2_ajuste_final_tsne1d --planejar; $estado = $LASTEXITCODE } finally { $ErrorActionPreference = $anterior }
Write-Host "V2_PLANEJAMENTO_EXIT_CODE=$estado"
if ($estado -ne 0) { throw 'V2 nao aprovada no planejamento. Confira o traceback.' }
Write-Host "`n=== V2 - AJUSTAR SOMENTE F1..F9 ===" -ForegroundColor Cyan
$anterior = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try { & $python -B -m scripts.execucao.v2_ajuste_final_tsne1d --executar; $estado = $LASTEXITCODE } finally { $ErrorActionPreference = $anterior }
Write-Host "V2_EXECUCAO_EXIT_CODE=$estado"
if ($estado -ne 0) { throw 'V2 nao concluida. Preserve quaisquer arquivos .incompleto_v2 e envie o erro.' }
$dir = Join-Path $raiz 'outputs\ensaio_tsne_gmm_20261102_134606\vencedor\teste_final\modelo_final_f1_f9'
$modelo = Join-Path $dir 'modelo_final_f1_f9.joblib'
$manifesto = Join-Path $dir 'manifesto_modelo_final.json'
if (-not (Test-Path -LiteralPath $modelo) -or -not (Test-Path -LiteralPath $manifesto)) {
    throw 'Modelo/manifesto V2 nao encontrados apos execucao.'
}
Write-Host "`n[OK] V2 FINALIZADA - MODELO TSNE 1D SALVO" -ForegroundColor Green
Write-Host "MODELO_FINAL=$modelo"
Write-Host 'F10_AVALIADO=NAO'
Write-Host 'OOF_REEXECUTADO=NAO'
Write-Host 'V2_EXECUCAO_EXIT_CODE=0'
