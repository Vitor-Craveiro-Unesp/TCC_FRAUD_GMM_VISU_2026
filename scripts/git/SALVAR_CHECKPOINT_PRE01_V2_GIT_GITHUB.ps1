#Requires -Version 5.1
<#
TCC_EXPERIMENTAL_V3 - checkpoints Git com selecao explicita.
Nao executa commit/push por padrao, nao altera main nem faz force push.
Execute na raiz do projeto. Consulte os exemplos no final deste arquivo.

MODOS:
  Previa          : somente leitura (padrao).
  CriarCheckpoint : cria branch local nova e commit SOMENTE dos arquivos informados.
  Publicar        : publica HEAD em uma branch REMOTA NOVA, mediante autorizacao.

Importante: backups/, outputs/, data/ e .venv/ NAO sao backups no GitHub.
#>
[CmdletBinding()]
param(
    [ValidateSet('Previa', 'CriarCheckpoint', 'Publicar')]
    [string]$Modo = 'Previa',

    [string[]]$Arquivos = @(),

    [switch]$ConfirmarCommit,
    [switch]$ConfirmarPush,

    # Para Publicar, indicar um novo nome: checkpoint/tcc-YYYYMMDD-HHMMSS.
    [string]$RamoPublicacao = '',

    # Confirma que o commit examinado e exatamente o commit a publicar.
    [string]$ShaEsperado = '',

    [string]$MensagemCommit = 'checkpoint: auditoria e organizacao controlada TCC'
)

# Avisos de conversao LF/CRLF podem aparecer em stderr mesmo com exit code 0.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
if (Test-Path Variable:\PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Executar-Git {
    param([string]$Etapa, [string[]]$Argumentos)
    & git @Argumentos
    $codigo = $LASTEXITCODE
    if ($codigo -ne 0) { throw "Falha ao $Etapa (git exit code $codigo)." }
}

function Consultar-Git {
    param([string]$Etapa, [string[]]$Argumentos)
    $saida = @(& git @Argumentos)
    $codigo = $LASTEXITCODE
    if ($codigo -ne 0) { throw "Falha ao $Etapa (git exit code $codigo)." }
    return $saida
}

function Eh-Proibido {
    param([string]$Caminho)
    $p = $Caminho.Replace('\', '/')
    if ($p -match '(?i)^(outputs|data|backups|temp|terceiraconversa|envio|\.venv|venv|\.git|\.vscode|__pycache__|\.pytest_cache)(/|$)') { return $true }
    if ($p -match '(?i)(^|/)(\.env($|\.)|\.rhistory$|id_rsa($|\.)|id_ed25519($|\.)|credentials[^/]*|secrets?[^/]*)') { return $true }
    if ($p -match '(?i)\.(zip|rar|7z|tar|gz|parquet|arrow|feather|npy|npz|pkl|pickle|joblib|sqlite|sqlite3|db|pem|key|p12|pfx|bak|log)$') { return $true }
    return $false
}

function Normalizar-Caminho {
    param([string]$Caminho)
    if ([string]::IsNullOrWhiteSpace($Caminho)) { throw 'Um caminho informado esta vazio.' }
    $p = $Caminho.Replace('\', '/')
    while ($p.StartsWith('./')) { $p = $p.Substring(2) }
    if ([IO.Path]::IsPathRooted($Caminho) -or $p.StartsWith('/') -or $p -match '(^|/)\.\.(/|$)' -or $p -match '[\r\n\x00]' -or $p.Contains(':')) {
        throw "Caminho inseguro ou fora da raiz: $Caminho"
    }
    if ($p.EndsWith('/') -or $p -eq '.' -or $p.Contains('*') -or $p.Contains('?')) {
        throw "Informe arquivos individuais (sem pastas ou curingas): $Caminho"
    }
    if (Eh-Proibido $p) { throw "Arquivo bloqueado por seguranca: $p" }
    $full = [IO.Path]::GetFullPath((Join-Path $script:Raiz ($p.Replace('/', [IO.Path]::DirectorySeparatorChar))))
    $prefixo = $script:Raiz.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $full.StartsWith($prefixo, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Arquivo fora da raiz: $p"
    }
    if ((Test-Path -LiteralPath $full) -and (Test-Path -LiteralPath $full -PathType Container)) {
        throw "Pastas nao sao permitidas na selecao; informe arquivos: $p"
    }
    # --no-index verifica tambem arquivos ja rastreados, que o .gitignore sozinho nao protege.
    & git check-ignore --no-index -q -- $p
    $check = $LASTEXITCODE
    if ($check -eq 0) { throw "Arquivo ignorado por .gitignore (inclusive se ja rastreado): $p" }
    if ($check -ne 1) { throw "Falha ao consultar .gitignore para $p (codigo $check)." }
    if (Test-Path -LiteralPath $full -PathType Leaf) {
        if ((Get-Item -LiteralPath $full).Length -gt 25MB) { throw "Arquivo maior que 25 MB: $p" }
    } else {
        # Exclusoes de arquivos previamente rastreados devem ser auditadas fora desta ferramenta.
        throw "Arquivo ausente: $p. Esta ferramenta NAO prepara exclusoes."
    }
    return $p
}

# Contexto Git: estritamente a raiz correta do projeto.
if (-not (Test-Path -LiteralPath '.\src\orquestracao_oof.py' -PathType Leaf)) {
    throw 'Execute na raiz de TCC_EXPERIMENTAL_V3 (src/orquestracao_oof.py ausente).'
}
$script:Raiz = [IO.Path]::GetFullPath((Get-Location).Path).TrimEnd('\', '/')
$topo = @(Consultar-Git 'obter a raiz Git' @('rev-parse', '--show-toplevel'))[0]
if ([IO.Path]::GetFullPath($topo).TrimEnd('\', '/') -ine $script:Raiz) {
    throw 'Execute exatamente na raiz do repositorio Git.'
}
$remote = @(Consultar-Git 'identificar github-tcc' @('remote', 'get-url', 'github-tcc'))[0].Trim()
if ($remote -notmatch '(?i)^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026(?:\.git)?/?$') {
    throw "O remoto github-tcc nao corresponde ao repositorio esperado: $remote"
}
$remotePush = @(Consultar-Git 'identificar destino de push github-tcc' @('remote', 'get-url', '--push', 'github-tcc'))[0].Trim()
if ($remotePush -ne $remote) {
    throw "O destino do push difere do remoto de leitura; verificar manualmente: $remotePush"
}
$branchAtual = @(Consultar-Git 'obter branch atual' @('branch', '--show-current'))[0]
if ([string]::IsNullOrWhiteSpace($branchAtual)) { throw 'HEAD destacado: selecione primeiro uma branch local.' }
$shaInicial = @(Consultar-Git 'obter SHA atual' @('rev-parse', 'HEAD'))[0].Trim()
$conflitos = @(Consultar-Git 'verificar conflitos' @('ls-files', '-u'))
if ($conflitos.Count -gt 0) { throw 'Existem conflitos de merge; resolva-os antes de continuar.' }

Write-Host "`n=== CHECKPOINT TCC: $Modo ===" -ForegroundColor Cyan
Write-Host "Raiz:  $script:Raiz"
Write-Host "Branch: $branchAtual"
Write-Host "HEAD:   $shaInicial"
Write-Host "Remoto: $remote"

if ($Modo -eq 'Publicar') {
    if (-not $ConfirmarPush) { throw 'Publicacao bloqueada: informe -ConfirmarPush explicitamente.' }
    if ($Arquivos.Count -gt 0) { throw 'Modo Publicar nao recebe -Arquivos; publica somente HEAD ja commitado.' }
    if ([string]::IsNullOrWhiteSpace($RamoPublicacao)) { throw 'Informe -RamoPublicacao com um nome NOVO checkpoint/...' }
    if ([string]::IsNullOrWhiteSpace($ShaEsperado) -or $ShaEsperado -cne $shaInicial) {
        throw "SHA esperado ausente ou diferente. Confira HEAD e informe -ShaEsperado $shaInicial"
    }
    if ($RamoPublicacao -notmatch '^checkpoint/[a-zA-Z0-9][a-zA-Z0-9._/-]*$' -or $RamoPublicacao.Contains('..') -or $RamoPublicacao.Contains('//')) {
        throw 'Nome de branch remota invalido ou fora de checkpoint/.'
    }
    & git check-ref-format --branch $RamoPublicacao | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Nome de branch rejeitado pelo Git.' }
    $ignoradosCommitados = @(Consultar-Git 'verificar ignorados rastreados antes da publicacao' @('ls-files', '-ci', '--exclude-standard'))
    if ($ignoradosCommitados.Count -gt 0) {
        Write-Warning "O commit atual ainda rastreia $($ignoradosCommitados.Count) caminhos ignorados (podem ser historicos). Revise antes de publicar."
    }
    $pendencias = @(Consultar-Git 'verificar estado de trabalho' @('status', '--porcelain'))
    if ($pendencias.Count -gt 0) { throw 'Ha alteracoes locais pendentes. Revise-as antes da publicacao.' }
    $ref = 'refs/heads/' + $RamoPublicacao
    $jaExiste = @(Consultar-Git 'consultar branch remota' @('ls-remote', '--heads', 'github-tcc', $ref))
    if ($jaExiste.Count -ne 0) { throw "A branch remota ja existe. NADA sera sobrescrito: $RamoPublicacao" }
    Write-Host "`nPUBLICANDO SOMENTE o commit existente $shaInicial em NOVA branch $RamoPublicacao" -ForegroundColor Yellow
    Executar-Git 'publicar branch nova' @('push', '-u', 'github-tcc', ('HEAD:' + $ref))
    $conferencia = @(Consultar-Git 'conferir SHA remoto' @('ls-remote', '--heads', 'github-tcc', $ref))
    if ($conferencia.Count -ne 1 -or -not ([string]$conferencia[0]).StartsWith($shaInicial + "`t")) {
        throw "Push realizado, mas a verificacao do SHA remoto falhou. CONFIRA MANUALMENTE: $ref"
    }
    Write-Host "PUBLICACAO CONFIRMADA: $RamoPublicacao ($shaInicial)" -ForegroundColor Green
    exit 0
}

# Previa e CriarCheckpoint: a selecao deve ser individual e nao inclui exclusoes.
$selecionados = New-Object System.Collections.Generic.List[string]
foreach ($entrada in $Arquivos) {
    # -File no Windows PowerShell recebe listas como uma string separada por virgulas.
    # Nomes com virgulas devem ser avaliados em comando interativo, nao via -File.
    foreach ($a in $entrada.Split(',')) {
        $p = Normalizar-Caminho $a.Trim()
        if (-not $selecionados.Contains($p)) { [void]$selecionados.Add($p) }
    }
}
Write-Host "`nEstado atual:" -ForegroundColor Cyan
Executar-Git 'exibir status' @('status', '--short')
$ignoradosRastreados = @(Consultar-Git 'listar rastreados ignorados' @('ls-files', '-ci', '--exclude-standard'))
if ($ignoradosRastreados.Count -gt 0) {
    Write-Host "`nATENCAO: $($ignoradosRastreados.Count) arquivos ignorados continuam rastreados. Nenhum sera preparado automaticamente." -ForegroundColor Yellow
}
if ($selecionados.Count -gt 0) {
    Write-Host "`nSelecao explicita (sem exclusoes):" -ForegroundColor Cyan
    foreach ($p in $selecionados) {
        $s = @(Consultar-Git "consultar $p" @('status', '--short', '--', $p))
        $estado = if ($s.Count -eq 0) { '(sem alteracao)' } else { $s -join '; ' }
        Write-Host " - $p $estado"
    }
}
if ($Modo -eq 'Previa') {
    Write-Host "`nPREVIA: nenhum arquivo foi adicionado, nenhum commit ou push foi realizado." -ForegroundColor Green
    Write-Host 'Para criar um checkpoint, informe -Modo CriarCheckpoint -Arquivos <arquivos> -ConfirmarCommit.'
    exit 0
}

# Para criar commit, exige selecao e aprovacao explicita.
if (-not $ConfirmarCommit) { throw 'Commit bloqueado: informe -ConfirmarCommit explicitamente.' }
if ($selecionados.Count -eq 0) { throw 'Commit bloqueado: selecione arquivos com -Arquivos.' }
if ([string]::IsNullOrWhiteSpace($MensagemCommit)) { throw 'Informe uma mensagem de commit.' }
& git diff --cached --quiet
$indexCodigo = $LASTEXITCODE
if ($indexCodigo -eq 1) { throw 'O staging ja contem alteracoes. Nao misturaremos alteracoes de outros comandos.' }
if ($indexCodigo -ne 0) { throw 'Nao foi possivel verificar se o staging esta limpo.' }
$haMudancas = $false
foreach ($p in $selecionados) {
    if (@(Consultar-Git "status do arquivo $p" @('status', '--porcelain', '--', $p)).Count -gt 0) { $haMudancas = $true }
}
if (-not $haMudancas) { throw 'Nenhum dos arquivos selecionados apresenta alteracao a registrar.' }
$ramoNovo = 'checkpoint/tcc-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6)
Write-Host "`nCriando SOMENTE commit local na branch NOVA: $ramoNovo" -ForegroundColor Yellow
Executar-Git 'criar branch local' @('switch', '-c', $ramoNovo)
$branchCriada = $true
$indexAlterado = $false
try {
    foreach ($p in $selecionados) {
        $indexAlterado = $true
        Executar-Git "adicionar apenas $p" @('add', '--', $p)
    }
    $staged = @(Consultar-Git 'auditar arquivos preparados' @('-c', 'core.quotePath=false', 'diff', '--cached', '--name-only'))
    if ($staged.Count -eq 0) { throw 'Nenhuma alteracao permaneceu no staging.' }
    foreach ($p in $staged) {
        if (-not $selecionados.Contains(([string]$p).Trim())) {
            throw "Arquivo nao autorizado entrou no staging: $p"
        }
    }
    $exclusoes = @(Consultar-Git 'verificar exclusoes no staging' @('diff', '--cached', '--name-only', '--diff-filter=D'))
    if ($exclusoes.Count -gt 0) { throw 'Exclusoes nao sao autorizadas neste procedimento.' }
    Write-Host "`nResumo EXATO do que sera incluido no commit:" -ForegroundColor Cyan
    Executar-Git 'exibir diff preparado' @('diff', '--cached', '--stat')
    Executar-Git 'criar commit local' @('commit', '-m', $MensagemCommit)
    $indexAlterado = $false
    $sha = @(Consultar-Git 'obter SHA do checkpoint' @('rev-parse', 'HEAD'))[0].Trim()
    Write-Host "`nCHECKPOINT LOCAL CONCLUIDO: $ramoNovo / $sha" -ForegroundColor Green
    Write-Host 'Nenhum push foi realizado. Para publicar, use -Modo Publicar -RamoPublicacao <nome novo> -ConfirmarPush.'
} catch {
    if ($indexAlterado) {
        Write-Warning 'Tentando retirar somente os arquivos selecionados do staging (arquivos locais preservados).'
        & git restore --staged -- @($selecionados.ToArray())
        if ($LASTEXITCODE -ne 0) { Write-Warning 'Nao foi possivel limpar o staging automaticamente. Examine git status.' }
    }
    Write-Warning "Falha na criacao do checkpoint. A branch $ramoNovo pode existir localmente. Nenhum push foi realizado."
    throw
}

<# EXEMPLOS (PowerShell, a partir da raiz):
   # Somente leitura / inspecao:
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\git\SALVAR_CHECKPOINT_PRE01_V2_GIT_GITHUB.ps1

   # Previa da selecao:
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\git\SALVAR_CHECKPOINT_PRE01_V2_GIT_GITHUB.ps1 -Modo Previa -Arquivos README.md,requirements.txt

   # Criar SOMENTE novo commit local, depois de conferir a selecao:
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\git\SALVAR_CHECKPOINT_PRE01_V2_GIT_GITHUB.ps1 -Modo CriarCheckpoint -Arquivos README.md,requirements.txt -ConfirmarCommit

   # Publicar o HEAD em uma branch REMOTA QUE AINDA NAO EXISTE (acao separada):
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\git\SALVAR_CHECKPOINT_PRE01_V2_GIT_GITHUB.ps1 -Modo Publicar -RamoPublicacao checkpoint/tcc-NOVO-20261029 -ShaEsperado <SHA_COMPLETO_DE_HEAD> -ConfirmarPush
#>
