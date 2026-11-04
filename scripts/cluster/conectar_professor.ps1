# Uso exclusivo pelo professor no computador dele, com chave SSH privada LOCAL.
# O endereco e porta abaixo provem do guia historico (2014); confirme com a TI.
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z_][A-Za-z0-9_.-]*$')]
    [string]$Usuario,
    [Parameter(Mandatory = $true)]
    [string]$ChavePrivada,
    [string]$Servidor = '200.145.184.171',
    [ValidateRange(1,65535)]
    [int]$Porta = 40665
)
$ErrorActionPreference = 'Stop'
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    throw 'OpenSSH Client nao encontrado no Windows.'
}
if (-not (Test-Path -LiteralPath $ChavePrivada -PathType Leaf)) {
    throw 'Caminho da chave privada inexistente. Nao envie a chave ao ChatGPT/GitHub.'
}
Write-Host 'Verifique a autenticidade da impressao digital do servidor com o administrador antes de aceitar uma nova chave de host.'
Write-Host 'Nenhuma credencial e salva por este script.'
& ssh -p $Porta -i $ChavePrivada -o 'IdentitiesOnly=yes' -o 'StrictHostKeyChecking=ask' "$Usuario@$Servidor"
exit $LASTEXITCODE
