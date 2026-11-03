$ErrorActionPreference = 'Stop'
$raiz = 'C:\Users\Vitor Craveiro\Desktop\ClusterPosMac\MEU PROJETO\TCC_EXPERIMENTAL_V3'
Set-Location -LiteralPath $raiz
$python = Join-Path $raiz '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Ambiente Python nao encontrado.' }
$modulo = 'scripts.validacao.v1_auditoria_f10_tsne_1d'
Write-Host "`n=== V1 PLANEJAMENTO TSNE F10 (SOMENTE LEITURA) ===" -ForegroundColor Cyan
$old = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $python -B -m $modulo --planejar
    $code1 = $LASTEXITCODE
} finally { $ErrorActionPreference = $old }
Write-Host "V1_PLANEJAMENTO_EXIT_CODE=$code1"
if ($code1 -ne 0) { throw 'A V1 bloqueou a auditoria; veja o motivo acima.' }
Write-Host "`n=== V1 REGISTRAR RELATORIO (SEM ALTERAR EXPERIMENTOS) ===" -ForegroundColor Cyan
$old = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $python -B -m $modulo --registrar
    $code2 = $LASTEXITCODE
} finally { $ErrorActionPreference = $old }
Write-Host "V1_REGISTRO_EXIT_CODE=$code2"
if ($code2 -ne 0) { throw 'Falha no registro V1; veja o motivo acima.' }
Write-Host '[OK] V1 concluida. Envie as linhas V1_STATUS e V1_RELATORIO.' -ForegroundColor Green
