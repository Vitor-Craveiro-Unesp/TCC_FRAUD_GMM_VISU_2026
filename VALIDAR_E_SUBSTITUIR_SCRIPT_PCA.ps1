# Validacao e substituicao controlada do executor PCA 1D/2D/3D.
# Salve este arquivo NA RAIZ do TCC_EXPERIMENTAL_V3.
# Testa os tres ensaios reais em F1-F9. Nao executa F10.
# So substitui o original se todas as verificacoes forem aprovadas.
[CmdletBinding()]
param(
    [ValidateRange(20, 1000000)]
    [int]$PorFold = 1000,

    [ValidateSet('Automatica', 'Principal')]
    [string]$OrigemConfiguracao = 'Automatica'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$raiz = $PSScriptRoot
$nome = 'RODAR_NOVO_TESTE_PCA_1D_2D_3D_COM_PRE01_V2'
$pastaScripts = Join-Path $raiz 'scripts\execucao'
$original = Join-Path $pastaScripts ($nome + '.ps1')
$revisado = Join-Path $pastaScripts ($nome + '_REVISADO.ps1')
$cfg = Join-Path $raiz 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
$python = Join-Path $raiz '.venv\Scripts\python.exe'
$outputs = Join-Path $raiz 'outputs'
$psWindows = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'

function Exigir-Arquivo([string]$Caminho) {
    if (-not (Test-Path -LiteralPath $Caminho -PathType Leaf)) {
        throw "Arquivo nao encontrado: $Caminho"
    }
}

foreach ($f in @($original, $revisado, $cfg, $python, $psWindows)) {
    Exigir-Arquivo $f
}
if (-not (Test-Path -LiteralPath (Join-Path $raiz 'src\orquestracao_oof.py') -PathType Leaf)) {
    throw 'Este validador deve ser salvo na raiz do TCC_EXPERIMENTAL_V3.'
}
if (-not (Test-Path -LiteralPath $outputs -PathType Container)) {
    throw 'A pasta outputs/ nao foi encontrada. Interrompendo sem modificar scripts.'
}

# Evita trocar um arquivo que nao seja a versao realmente examinada na auditoria.
$hashAntigoEsperado = '076bf5ba1e9a06c3fe8951210c2b785f6214c170f3831b975dee6d4603f2aa05'
$hashNovoEsperado = '072410a550fb30f8129aaa351bf2e15ab029d207c52452a0a7c2401dea99afec'
$hashAntigo = (Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash.ToLowerInvariant()
$hashNovo = (Get-FileHash -LiteralPath $revisado -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hashAntigo -ne $hashAntigoEsperado) {
    throw "O script antigo nao corresponde ao arquivo auditado (SHA256=$hashAntigo). Nada foi alterado."
}
if ($hashNovo -ne $hashNovoEsperado) {
    throw "O script revisado nao corresponde ao arquivo auditado (SHA256=$hashNovo). Nada foi alterado."
}

# Validacao sintatica do PowerShell, antes de qualquer treinamento.
$tokens = $null
$erros = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    $revisado, [ref]$tokens, [ref]$erros
) | Out-Null
if ($erros.Count -gt 0) {
    $erros | Format-List | Out-Host
    throw 'Falha na sintaxe PowerShell do arquivo revisado. Original preservado.'
}
Write-Host '[OK] Sintaxe PowerShell e hashes dos dois scripts conferidos.' -ForegroundColor Green

# Backups do script e YAML ANTES do teste real.
$identificador = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$pastaBackup = Join-Path $raiz ("backups\scripts\validacao_$identificador")
New-Item -ItemType Directory -Path $pastaBackup -ErrorAction Stop | Out-Null
$backupOriginal = Join-Path $pastaBackup ($nome + '_ORIGINAL.ps1')
$backupNovo = Join-Path $pastaBackup ($nome + '_REVISADO.ps1')
$backupYaml = Join-Path $pastaBackup 'YAML_PRINCIPAL_ANTES.yaml'
$log = Join-Path $pastaBackup 'teste_revisado.log'
Copy-Item -LiteralPath $original -Destination $backupOriginal
Copy-Item -LiteralPath $revisado -Destination $backupNovo
Copy-Item -LiteralPath $cfg -Destination $backupYaml
$hashYamlAntes = (Get-FileHash -LiteralPath $cfg -Algorithm SHA256).Hash

# Registramos as pastas de controle existentes para distinguir o novo teste.
$controlesAnteriores = @{}
Get-ChildItem -LiteralPath $outputs -Directory -Filter 'NOVO_TESTE_PCA_1D_2D_3D_K4_NINIT5_*' |
    ForEach-Object { $controlesAnteriores[$_.FullName] = $true }

Write-Host ''
Write-Host '[TESTE REAL] Executando o script REVISADO (PCA 1D, 2D e 3D, PRE01 V2).' -ForegroundColor Cyan
Write-Host "[BACKUP] $pastaBackup"
Write-Host '[ATENCAO] Os ensaios criarao novos resultados em outputs/. F10 segue reservado.'

# Processo filho: Bypass apenas para ESTA execucao, sem alterar a politica global.
# O novo script continua com seu nome _REVISADO durante todo o teste.
$codigo = -999
$preferenciaAntes = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'  # Stderr do processo nativo nao deve interromper o log.
    & $psWindows -NoProfile -ExecutionPolicy Bypass -File $revisado `
        -PorFold $PorFold -OrigemConfiguracao $OrigemConfiguracao -Abrir Nenhum 2>&1 |
        Tee-Object -FilePath $log
    $codigo = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $preferenciaAntes
}

# Nao trocamos o arquivo se o YAML nao tiver sido restaurado byte a byte.
$hashYamlDepois = (Get-FileHash -LiteralPath $cfg -Algorithm SHA256).Hash
if ($hashYamlDepois -ne $hashYamlAntes) {
    throw "YAML principal ficou diferente. Original PRESERVADO. Backup em: $backupYaml"
}
if ($codigo -ne 0) {
    throw "Teste do revisado falhou (codigo $codigo). Original PRESERVADO. Log: $log"
}

$novosControles = @(Get-ChildItem -LiteralPath $outputs -Directory -Filter 'NOVO_TESTE_PCA_1D_2D_3D_K4_NINIT5_*' |
    Where-Object { -not $controlesAnteriores.ContainsKey($_.FullName) })
if ($novosControles.Count -ne 1) {
    throw "Esperada exatamente uma nova pasta de controle; encontradas $($novosControles.Count). Original PRESERVADO."
}
$resumoArquivo = Join-Path $novosControles[0].FullName 'RESUMO_TRES_GALERIAS.json'
Exigir-Arquivo $resumoArquivo
$resumo = Get-Content -LiteralPath $resumoArquivo -Raw -Encoding UTF8 | ConvertFrom-Json

if ($resumo.status_geral -ne 'concluido' -or $resumo.pre01_v2_confirmado -ne $true) {
    throw 'O PRE01 V2 nao foi confirmado no resumo. Original PRESERVADO.'
}
if ($resumo.f10_usado -ne $false -or [int]$resumo.k -ne 4 -or
    [int]$resumo.n_init -ne 5 -or [int]$resumo.por_fold -ne $PorFold) {
    throw 'O resumo nao corresponde aos parametros esperados ou registra uso de F10. Original PRESERVADO.'
}
Exigir-Arquivo ([string]$resumo.painel)

$galerias = @()
foreach ($dim in @('1D', '2D', '3D')) {
    $propriedade = $resumo.novo_trio.PSObject.Properties[$dim]
    if ($null -eq $propriedade) {
        throw "Resumo sem ensaio PCA $dim. Original PRESERVADO."
    }
    $pasta = [string]$propriedade.Value
    $galeria = Join-Path $pasta 'vencedor\visualizacao\galeria\index.html'
    $manifesto = Join-Path $pasta 'manifesto.json'
    Exigir-Arquivo $galeria
    Exigir-Arquivo $manifesto
    $meta = Get-Content -LiteralPath $manifesto -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($meta.status -ne 'concluido_06_03' -or $meta.visualizacoes -ne 'concluidas' -or
        [int]$meta.registros_desenvolvimento -ne (9 * $PorFold)) {
        throw "Manifesto PCA $dim nao confirma sucesso e quantidade de registros. Original PRESERVADO."
    }
    $galerias += $galeria
}

# Protecao contra alteracao do script revisado durante o proprio teste.
if ((Get-FileHash -LiteralPath $revisado -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hashNovoEsperado) {
    throw 'O arquivo REVISADO mudou durante o teste. Original PRESERVADO.'
}
Write-Host '[OK] 3 ensaios, 3 galerias, PRE01 V2 e YAML restaurado: verificados.' -ForegroundColor Green

# Substituicao: guardar o antigo sob um nome temporario, instalar o novo,
# conferir hash e SO ENTAO excluir a copia temporaria (backup permanente permanece).
$antigoTemporario = Join-Path $pastaScripts ('.' + $nome + '_ANTIGO_TEMP_' + $identificador + '.ps1')
try {
    Move-Item -LiteralPath $original -Destination $antigoTemporario -ErrorAction Stop
    Move-Item -LiteralPath $revisado -Destination $original -ErrorAction Stop
    if ((Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hashNovoEsperado) {
        throw 'Hash incorreto apos a substituicao.'
    }
}
catch {
    $motivo = $_.Exception.Message
    try {
        if (Test-Path -LiteralPath $antigoTemporario -PathType Leaf) {
            if (Test-Path -LiteralPath $original -PathType Leaf) {
                if (-not (Test-Path -LiteralPath $revisado)) {
                    Move-Item -LiteralPath $original -Destination $revisado -ErrorAction Stop
                } else {
                    throw 'Original e revisado ocupados: recuperacao manual necessaria.'
                }
            }
            Move-Item -LiteralPath $antigoTemporario -Destination $original -ErrorAction Stop
        }
        elseif (-not (Test-Path -LiteralPath $original)) {
            Copy-Item -LiteralPath $backupOriginal -Destination $original -ErrorAction Stop
        }
    }
    catch {
        throw "Falha na substituicao E na recuperacao automatica: $motivo. Verifique $pastaBackup e os scripts."
    }
    throw "Falha na substituicao: $motivo. O original foi recuperado."
}

# Nao prejudica a instalacao se a remocao desta copia extra falhar.
try {
    Remove-Item -LiteralPath $antigoTemporario -Force -ErrorAction Stop
}
catch {
    Write-Warning "Novo script instalado, mas sobrou o original temporario: $antigoTemporario"
}

Write-Host ''
Write-Host '[SUCESSO] Revisado APROVADO e renomeado para o nome original.' -ForegroundColor Green
Write-Host "[ATIVO] $original"
Write-Host "[BACKUP ORIGINAL] $backupOriginal"
Write-Host "[LOG] $log"
Write-Host ''
Write-Host '[GALERIAS] Abrindo as 3 galerias do NOVO teste:' -ForegroundColor Cyan
foreach ($galeria in $galerias) {
    try { Invoke-Item -LiteralPath $galeria -ErrorAction Stop }
    catch { Write-Warning "Galeria pronta, mas a abertura no navegador falhou: $galeria" }
}
