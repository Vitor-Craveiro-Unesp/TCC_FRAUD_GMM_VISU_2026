#Requires -Version 5.1
<#
Organizacao conservadora da raiz TCC_EXPERIMENTAL_V3.
Padrao: somente PREVIA. -Aplicar move arquivos; -Desfazer reverte a ultima rodada.
Nao remove nem modifica: src, tests, data, outputs, .venv, passos, Temp, TerceiraConversa.
Nao move: .gitignore, README.md, requirements.txt, YAML, CONTEXTO_RETOMADA_TCC.md,
          RODAR_NOVO_TESTE_PCA_1D_2D_3D_COM_PRE01_V2.ps1, SALVAR_CHECKPOINT_PRE01_V2_GIT_GITHUB.ps1.
Os scripts historicos sao arquivados em conjunto para manter dependencias de co-localizacao.
Execute NO DIRETORIO RAIZ (Get-Location) do projeto.
#>
[CmdletBinding(DefaultParameterSetName='Preview')]
param(
    [Parameter(ParameterSetName='Apply')][switch]$Aplicar,
    [Parameter(ParameterSetName='Undo')][switch]$Desfazer
)

$ErrorActionPreference = 'Stop'
$raiz = (Get-Location).Path
if (-not (Test-Path -LiteralPath (Join-Path $raiz '.git') -PathType Container) -or
    -not (Test-Path -LiteralPath (Join-Path $raiz 'src\orquestracao_oof.py') -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $raiz 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml') -PathType Leaf)) {
    throw 'Abra o PowerShell na raiz de TCC_EXPERIMENTAL_V3 (pasta com .git, src e YAML).'
}

$manifestBase = Join-Path $raiz 'backups\organizacao_raiz'
$manifNome = 'movimentos.csv'

function Get-HashSeguro([string]$caminho) {
    return (Get-FileHash -LiteralPath $caminho -Algorithm SHA256).Hash
}

function Get-RotaRelativa([string]$base, [string]$arquivo) {
    $prefixo = $base.TrimEnd('\') + '\'
    if (-not $arquivo.StartsWith($prefixo,[System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Caminho externo ao projeto: $arquivo"
    }
    return $arquivo.Substring($prefixo.Length)
}

# Os diretorios backups e .venv NAO sao incluidos no Git, como ja definido no .gitignore.
# Nao executar 'git add' ou 'git push' aqui: o usuario revisa as mudancas primeiro.
$tracked = @{}
& git -C $raiz ls-files | ForEach-Object { $tracked[$_.Replace('/','\')] = $true }
if ($LASTEXITCODE -ne 0) { throw 'Nao foi possivel ler os arquivos rastreados pelo Git.' }

function Move-ComControle([string]$origemRel, [string]$destRel, [bool]$ehGit) {
    $origem = Join-Path $raiz $origemRel
    $destino = Join-Path $raiz $destRel
    if (-not (Test-Path -LiteralPath $origem -PathType Leaf)) { throw "Origem ausente: $origemRel" }
    if (Test-Path -LiteralPath $destino) { throw "Destino ja existe, nao sobrescrever: $destRel" }
    New-Item -ItemType Directory -Path (Split-Path -Parent $destino) -Force | Out-Null
    if ($ehGit -and -not $destRel.StartsWith('backups\', [StringComparison]::OrdinalIgnoreCase) -and
        -not $origemRel.StartsWith('backups\', [StringComparison]::OrdinalIgnoreCase)) {
        & git -C $raiz mv -- $origemRel.Replace('\','/') $destRel.Replace('\','/')
        if ($LASTEXITCODE -ne 0) { throw "git mv falhou: $origemRel" }
    } else {
        Move-Item -LiteralPath $origem -Destination $destino -ErrorAction Stop
    }
}

if ($Desfazer) {
    $rodadas = @(Get-ChildItem -LiteralPath $manifestBase -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName $manifNome) } |
        Sort-Object Name -Descending)
    if ($rodadas.Count -eq 0) { throw 'Nenhuma rodada de organizacao anterior foi encontrada.' }
    $manifesto = Join-Path $rodadas[0].FullName $manifNome
    $itens = @(Import-Csv -LiteralPath $manifesto -Delimiter ';')
    if ($itens.Count -eq 0) { throw 'Manifesto sem movimentos.' }
    Write-Host "[DESFAZER] Registro: $manifesto" -ForegroundColor Yellow
    [array]::Reverse($itens)
    $restaurados = 0
    foreach ($m in $itens) {
        $antigo = Join-Path $raiz $m.Origem
        $atual = Join-Path $raiz $m.Destino
        if (-not (Test-Path -LiteralPath $atual -PathType Leaf)) {
            Write-Warning "Ja ausente; nao restaurado: $($m.Destino)"; continue
        }
        if (Test-Path -LiteralPath $antigo) {
            Write-Warning "Origem ja existe; nao sobrescrever: $($m.Origem)"; continue
        }
        if ((Get-HashSeguro $atual) -ne $m.SHA256) {
            Write-Warning "Arquivo foi alterado depois da organizacao; manter: $($m.Destino)"; continue
        }
        Move-ComControle $m.Destino $m.Origem ($m.Rastreado -eq 'True')
        $restaurados++
    }
    Write-Host "[DESFEITO] $restaurados de $($itens.Count) movimentos revertidos. Confira 'git status --short'." -ForegroundColor Green
    exit 0
}

# Referencias dos scripts atuais: evita mover um arquivo diretamente citado pelo codigo principal.
$protegidos = @(
    'RODAR_NOVO_TESTE_PCA_1D_2D_3D_COM_PRE01_V2.ps1',
    'SALVAR_CHECKPOINT_PRE01_V2_GIT_GITHUB.ps1',
    '.gitignore', 'README.md', 'requirements.txt',
    'CONTEXTO_RETOMADA_TCC.md',
    'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
)
$textoAtivo = New-Object System.Text.StringBuilder
$pastasAtivas = @('src','tests')
foreach ($p in $pastasAtivas) {
    $dir = Join-Path $raiz $p
    if (-not (Test-Path -LiteralPath $dir -PathType Container)) { continue }
    foreach ($f in (Get-ChildItem -LiteralPath $dir -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in @('.py','.ps1','.yaml','.yml') -and $_.FullName -notmatch '__pycache__' })) {
        try { [void]$textoAtivo.AppendLine([System.IO.File]::ReadAllText($f.FullName)) } catch {}
    }
}
foreach ($p in @('RODAR_NOVO_TESTE_PCA_1D_2D_3D_COM_PRE01_V2.ps1',
                  'SALVAR_CHECKPOINT_PRE01_V2_GIT_GITHUB.ps1',
                  'config_TCC_EXPERIMENTAL_V3_atualizado.yaml')) {
    $f=Join-Path $raiz $p
    if (Test-Path -LiteralPath $f -PathType Leaf) {
        try { [void]$textoAtivo.AppendLine([System.IO.File]::ReadAllText($f)) } catch {}
    }
}
$referencias = $textoAtivo.ToString()

$plano = New-Object System.Collections.Generic.List[object]
$pulados = New-Object System.Collections.Generic.List[string]
foreach ($f in (Get-ChildItem -LiteralPath $raiz -File)) {
    $n = $f.Name
    if ($n -in $protegidos) { continue }
    $dest = $null
    if ($n -match '(?i)^BACKUP_CODIGO_TCC_.*\.zip$') {
        $dest = 'backups\arquivo_historico_raiz\'+$n
    } elseif ($n -match '(?i)\.bak$' -or $n -match '(?i)\.zip$') {
        $dest = 'backups\arquivo_historico_raiz\'+$n
    } elseif ($n -match '(?i)^(CHECKPOINT_TCC_|CHECKLIST_AUDITORIA_|README_REESTRUTURACAO_|LEIA_ME_).*\.(md|txt)$') {
        $dest = 'docs\historico\'+$n
    } elseif ($n -match '(?i)^(ajustar_|auditar_|comparar_|ensaios_|instalar_|restaurar_|validar_|verificar_|visualizacao_).*\.py$') {
        $dest = 'scripts\historico\'+$n
    } elseif ($n -match '(?i)^(AUDITAR_|ADICIONAR_|CORRIGIR_|Executar_Comparacao_|FINALIZAR_|INSTALAR_|RECUPERAR_|RODAR_TESTE_|rodar_ensaio_|comparar_|criar_passos_).*\.ps1$') {
        $dest = 'scripts\historico\'+$n
    }
    if ($null -eq $dest) { continue }
    if ($dest.StartsWith('scripts\', [System.StringComparison]::OrdinalIgnoreCase) -and
        $referencias.IndexOf($n,[System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
        [void]$pulados.Add("$n (referenciado pelo codigo principal)")
        continue
    }
    $fullDest = Join-Path $raiz $dest
    if (Test-Path -LiteralPath $fullDest) {
        [void]$pulados.Add("$n (destino ja existe)")
        continue
    }
    $registro = [pscustomobject]@{
        Origem=$n
        Destino=$dest
        Rastreado=([string]($tracked.ContainsKey($n)))
        SHA256=Get-HashSeguro $f.FullName
    }
    [void]$plano.Add($registro)
}

Write-Host ''
Write-Host '==========================================================' -ForegroundColor Cyan
Write-Host 'ORGANIZACAO DA RAIZ DO TCC - PREVIA / EXECUCAO CONTROLADA' -ForegroundColor Cyan
Write-Host '==========================================================' -ForegroundColor Cyan
foreach ($m in $plano) { Write-Host ('  '+$m.Origem+'  ->  '+$m.Destino) }
Write-Host "`nArquivos planejados: $($plano.Count)"
if ($pulados.Count -gt 0) {
    Write-Host "`nArquivos preservados por precaucao:" -ForegroundColor Yellow
    foreach ($s in $pulados) { Write-Host ('  '+$s) }
}
if (-not $Aplicar) {
    Write-Host "`n[PREVIA] Nada foi movido. Execute com -Aplicar para organizar." -ForegroundColor Yellow
    Write-Host 'Comando: powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\ORGANIZAR_RAIZ_TCC_SEGURO.ps1 -Aplicar'
    exit 0
}

if ($plano.Count -eq 0) { Write-Host 'Nada a mover.'; exit 0 }
$rodada = Join-Path $manifestBase (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
New-Item -ItemType Directory -Path $rodada -Force | Out-Null
$manifestoFinal = Join-Path $rodada $manifNome
$concluidos = New-Object System.Collections.Generic.List[object]
foreach ($m in $plano) {
    try {
        Move-ComControle $m.Origem $m.Destino ($m.Rastreado -eq 'True')
        [void]$concluidos.Add($m)
        $concluidos | Export-Csv -LiteralPath $manifestoFinal -Delimiter ';' -NoTypeInformation -Encoding UTF8
        Write-Host ('[MOVIDO] '+$m.Origem) -ForegroundColor Green
    } catch {
        Write-Warning "[PAROU] $($m.Origem): $($_.Exception.Message)"
        Write-Warning "Movimentos anteriores ficam preservados no manifesto $manifestoFinal. Use -Desfazer se quiser revertê-los."
        throw
    }
}
Write-Host "`n[CONCLUIDO] $($concluidos.Count) arquivos movidos. Nenhum dado ou experimento modificado." -ForegroundColor Green
Write-Host "[REGISTRO] $manifestoFinal"
Write-Host '[GIT] Alteracoes locais ainda NAO foram enviadas ao GitHub.'
& git -C $raiz status --short
