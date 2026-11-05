# Checkpoint local restrito aos fontes GMM07/GMM08 e ao YAML da execucao atual.
# Execute SOMENTE depois de inspecionar os PNGs sinteticos e aprovar o layout.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$staged = @(git diff --cached --name-only)
if ($LASTEXITCODE -ne 0) { throw 'Git nao esta disponivel nesta pasta.' }
if ($staged.Count -gt 0 -and @($staged | Where-Object { $_.Trim() }).Count -gt 0) {
    Write-Host 'Ja existem alteracoes preparadas para commit. Nao vou misturar com o checkpoint.' -ForegroundColor Yellow
    git diff --cached --name-only
    throw 'Revise primeiro o Git staging area.'
}

$arquivos = @(
 'src/visualizacao_gmm07.py',
 'src/visualizacao_geometrica.py',
 'src/visualizacao_geometria_tsne3d.py',
 'src/visualizacao_gmm07_3d.py',
 'src/visualizacao_gmm08_3d.py',
 'src/gmm_contornos_quatro_niveis.py',
 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
)
foreach ($arquivo in $arquivos) {
    if (-not (Test-Path -LiteralPath $arquivo -PathType Leaf)) {
        throw "Arquivo esperado ausente: $arquivo"
    }
}
Write-Host '=== ARQUIVOS QUE SERAO SALVOS ==='
$arquivos | ForEach-Object { Write-Host " - $_" }
Write-Host '=== ATENCAO: NENHUM OUTRO ZIP/RELATORIO/ARQUIVO SERA ADICIONADO ==='
$confirm = Read-Host "Confirmou visualmente as imagens sinteticas? Digite SIM para preparar o checkpoint"
if ($confirm -cne 'SIM') { Write-Host 'Cancelado sem modificar o indice Git.'; exit 0 }

git add -- $arquivos
if ($LASTEXITCODE -ne 0) { throw 'Falha ao preparar arquivos no Git.' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Falha em git diff --cached --check.' }
Write-Host '=== DIFERENCAS PREPARADAS ==='
git diff --cached --stat
$commit = Read-Host 'Digite COMMIT para salvar localmente (sem push)'
if ($commit -cne 'COMMIT') {
    Write-Host 'Commit nao criado. Seus arquivos estao no staging; use git restore --staged -- <arquivo> para retirar se necessario.'
    exit 0
}
git commit -m "feat(vis): quatro niveis GMM07 GMM08 e configuracao atual"
if ($LASTEXITCODE -ne 0) { throw 'Commit local falhou.' }
Write-Host '=== CHECKPOINT LOCAL CRIADO ==='
git log -1 --oneline
Write-Host 'Nenhum push foi realizado. Os demais arquivos nao rastreados continuam fora do commit.'
