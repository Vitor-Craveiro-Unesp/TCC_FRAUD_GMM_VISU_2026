#Requires -Version 5.1
<#
Checkpoint do TCC_EXPERIMENTAL_V3 sem alterar main nem sobrescrever branches.
Inclui as modificacoes versionaveis e scripts atuais; respeita .gitignore.
Resultados volumosos em outputs/data permanecem locais.
Execute na raiz do projeto com PowerShell 5.1 ou 7.
#>
[CmdletBinding()]
param()

# Git escreve informacoes no stderr mesmo quando tem sucesso.
# Evitar que o PowerShell 5 trate isso como falha (verificamos LASTEXITCODE).
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
if (Test-Path Variable:\PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Confirmar-UltimoGit([string]$etapa) {
    if ($LASTEXITCODE -ne 0) {
        throw "Falha em $etapa (Git retornou codigo $LASTEXITCODE). Nenhum push forcado foi realizado."
    }
}

function Eh-Proibido([string]$nome) {
    $p = $nome.Replace('\','/')
    if ($p -match '(?i)^(outputs|data|backups|temp|\.venv|venv|__pycache__|\.git|\.vscode|\.pytest_cache)/') { return $true }
    if ($p -match '(?i)(^|/)(\.env($|\.)|id_rsa($|\.)|id_ed25519($|\.)|credentials[^/]*|secrets?[^/]*)') { return $true }
    if ($p -match '(?i)\.(zip|rar|7z|tar|gz|parquet|arrow|feather|npy|npz|pkl|pickle|joblib|sqlite|sqlite3|db|pem|key|p12|pfx|bak|log)$') { return $true }
    return $false
}

# 1. Validar contexto e repositorio
if (-not (Test-Path -LiteralPath '.\src\orquestracao_oof.py')) {
    throw 'Execute na raiz de TCC_EXPERIMENTAL_V3 (precisa conter src/orquestracao_oof.py).'
}
if (-not (Test-Path -LiteralPath '.\.git')) {
    throw 'A pasta atual nao parece ser um repositorio Git.'
}
$raiz = (Get-Location).Path
$topo = (& git rev-parse --show-toplevel)
Confirmar-UltimoGit 'localizar a raiz Git'
$topoLimpo = ((($topo | Select-Object -First 1) -as [string]) -replace '[\\/]+$', '')
if ([IO.Path]::GetFullPath($topoLimpo) -ne [IO.Path]::GetFullPath($raiz)) {
    throw 'A pasta atual deve ser exatamente a raiz do repositorio Git.'
}
$remote = (& git remote get-url github-tcc)
Confirmar-UltimoGit 'identificar github-tcc'
$remote = ($remote | Select-Object -First 1).Trim()
if ($remote -notmatch '(?i)github\.com[:/]Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026(?:\.git)?/?$') {
    throw "O remoto github-tcc nao e o repositorio esperado: $remote"
}
$naoResolvidos = @(& git diff --name-only --diff-filter=U)
Confirmar-UltimoGit 'conferir conflitos'
if ($naoResolvidos.Count -gt 0) {
    throw 'Existem arquivos em conflito (merge). Resolva antes do checkpoint.'
}
Write-Host ''
Write-Host '=== CHECKPOINT: PRE01 V2 / PCA 1D-2D-3D / GMM K4 NINIT5 ===' -ForegroundColor Cyan
Write-Host "Projeto: $raiz"
Write-Host "GitHub: $remote"
Write-Host "Branch de origem: $(& git branch --show-current)"
Write-Host 'Estado anterior:' -ForegroundColor Cyan
& git status --short
Confirmar-UltimoGit 'listar estado inicial'

# 2. Criar branch exclusiva, sem rebase, merge nem force push
$ramo = 'checkpoint/pre01v2-pca123-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& git switch -c $ramo
Confirmar-UltimoGit 'criar branch de checkpoint'

# 3. Adicionar todas as mudancas versionaveis (respeitando .gitignore)
& git add -A -- .
Confirmar-UltimoGit 'preparar arquivos'

# Desfazer staging de possiveis arquivos pesados/sensiveis fora do .gitignore
$excluidos = New-Object System.Collections.Generic.List[string]
$staged = @(& git diff --cached --name-only)
Confirmar-UltimoGit 'auditar staging'
foreach ($nome in $staged) {
    if ([string]::IsNullOrWhiteSpace($nome)) { continue }
    $p = $nome.Trim()
    $excluir = Eh-Proibido $p
    if (-not $excluir) {
        $full = Join-Path $raiz ($p.Replace('/',[IO.Path]::DirectorySeparatorChar))
        if (Test-Path -LiteralPath $full -PathType Leaf) {
            if ((Get-Item -LiteralPath $full).Length -gt 25MB) { $excluir = $true }
        }
    }
    if ($excluir) {
        & git restore --staged -- $p
        Confirmar-UltimoGit "retirar arquivo nao recomendado: $p"
        [void]$excluidos.Add($p)
    }
}

# 4. Mostrar o que de fato sera versionado
$staged = @(& git diff --cached --name-only)
Confirmar-UltimoGit 'conferir staging final'
if ($staged.Count -eq 0) {
    Write-Host 'Nao ha novas alteracoes versionaveis a salvar.' -ForegroundColor Yellow
    Write-Host 'A nova branch foi criada localmente; nao foi feito commit nem push.'
    Write-Host 'Verifique git status --short para arquivos ignorados ou excluidos.'
    exit 0
}
Write-Host ''
Write-Host ('Arquivos versionaveis preparados: ' + $staged.Count) -ForegroundColor Green
& git diff --cached --stat
Confirmar-UltimoGit 'mostrar resumo do commit'
if ($excluidos.Count -gt 0) {
    Write-Host ''
    Write-Host 'Nao incluidos por seguranca (continuam no computador):' -ForegroundColor Yellow
    $excluidos | ForEach-Object { Write-Host "  - $_" }
}

# 5. Commit na branch de checkpoint
& git commit -m 'checkpoint: PRE01 V2 HTML completo, galerias PCA 1D 2D 3D e scripts'
Confirmar-UltimoGit 'criar commit local'
$sha = ((& git rev-parse HEAD) | Select-Object -First 1).Trim()
Confirmar-UltimoGit 'obter SHA local'

# 6. Push nao destrutivo para github-tcc
& git push -u github-tcc ("HEAD:refs/heads/" + $ramo)
Confirmar-UltimoGit 'enviar branch ao GitHub'

# 7. Verificar que o SHA publicado confere com o commit local
$resposta = @(& git ls-remote --heads github-tcc ("refs/heads/" + $ramo))
Confirmar-UltimoGit 'verificar o checkpoint no GitHub'
if ($resposta.Count -ne 1 -or -not $resposta[0].StartsWith($sha + "`t")) {
    throw "Push executado, mas a verificacao do SHA remoto nao foi conclusiva. Local=$sha."
}
$url = 'https://github.com/Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026/tree/' + $ramo
Write-Host ''
Write-Host '========================================================' -ForegroundColor Green
Write-Host 'GIT LOCAL E GITHUB CONFIRMADOS!' -ForegroundColor Green
Write-Host "BRANCH=$ramo"
Write-Host "COMMIT=$sha"
Write-Host "GITHUB=$url"
Write-Host '========================================================' -ForegroundColor Green
Write-Host 'Observacao: outputs/, data/, backups/ e .venv/ NAO sao enviados ao GitHub.' -ForegroundColor Yellow
Write-Host 'Se precisar guardar os resultados, faca backup separado da pasta outputs/.' -ForegroundColor Yellow
Write-Host ''
& git status --short
