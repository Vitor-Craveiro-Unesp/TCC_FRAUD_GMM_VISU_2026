# TCC_EXPERIMENTAL_V3 - dois ensaios INDEPENDENTES PCA 2D e PCA 3D, GMM K=3.
# Na raiz do projeto, execute SEPARADAMENTE:
# powershell -NoProfile -ExecutionPolicy Bypass -File .\rodar_ensaio_PCA_2D_ou_3D_K3.ps1 -Dim 2
# powershell -NoProfile -ExecutionPolicy Bypass -File .\rodar_ensaio_PCA_2D_ou_3D_K3.ps1 -Dim 3
# F1-F9, F10 reservado, mesma configuracao exceto dimensionalidade.
# Cada chamada cria seu proprio output e sua propria galeria. Nenhum HTML de comparacao.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(2, 3)]
    [int]$Dim,
    [ValidateRange(20, 1000000)]
    [int]$PorFold = 1000
)

$ErrorActionPreference = 'Continue'  # Avisos do Matplotlib em stderr nao devem encerrar o PowerShell.
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$raiz = (Get-Location).Path
$python = Join-Path $raiz '.venv\Scripts\python.exe'
$yaml = Join-Path $raiz 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
$outputs = Join-Path $raiz 'outputs'
$orquestrador = Join-Path $raiz 'src\orquestracao_oof.py'

foreach ($arquivo in @($python, $yaml, $orquestrador)) {
    if (-not (Test-Path -LiteralPath $arquivo -PathType Leaf)) {
        throw "Arquivo nao encontrado: $arquivo. Execute na raiz do TCC_EXPERIMENTAL_V3."
    }
}
if (-not (Test-Path -LiteralPath $outputs -PathType Container)) {
    throw 'Pasta outputs nao encontrada.'
}

# Registro das pastas FINALIZADAS anteriores: uma nova pasta por rodada.
$anteriores = @(
    Get-ChildItem -LiteralPath $outputs -Directory -Filter 'ensaio_pca_gmm_*' |
        ForEach-Object { $_.FullName }
)

$original = [System.IO.File]::ReadAllBytes($yaml)
$carimbo = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$backup = "$yaml.ANTES_ENSAIO_PCA${Dim}D_K3_$carimbo.bak"
[System.IO.File]::WriteAllBytes($backup, $original)

Write-Host ''
Write-Host "===== NOVO ENSAIO PCA ${Dim}D + GMM K=3 =====" -ForegroundColor Cyan
Write-Host "Limite por fold: $PorFold | Folds: F1-F9 | F10 reservado"
Write-Host "Backup YAML: $backup"

try {
    # Programa Python INLINE: altera somente os dois n_components, preserva comentarios,
    # BOM e todos os demais campos; compara os objetos YAML antes de gravar.
    $preparar = @'
from pathlib import Path
from copy import deepcopy
import re
import sys
import yaml

p = Path('config_TCC_EXPERIMENTAL_V3_atualizado.yaml')
dim = int(sys.argv[1])
assert dim in (2, 3)
original = p.read_bytes()
texto = original.decode('utf-8-sig')
base = yaml.safe_load(texto)

assert base['reducao']['metodo'].upper() == 'PCA', 'Metodo atual nao e PCA.'
assert int(base['divisao']['qual_fold_teste']) == 10, 'F10 nao esta reservado no YAML.'
assert base['execucao'].get('diretorio_retomada') is None, 'Retomada explicita habilitada.'
assert base['execucao']['proteger_resultados'] is True, 'Protecao de resultados desativada.'
assert Path(base['saidas']['diretorio_base']) == Path('outputs'), 'Saida nao e outputs/.'
assert base['gmm']['covariance_type'].lower() == 'full', 'GMM nao usa covariancia FULL.'
assert base['pca']['n_components'] is not None

# Bloquear reaproveitamento involuntario de execucao incompleta equivalente.
esperado = deepcopy(base)
esperado['pca']['n_components'] = [dim]
esperado['gmm']['n_components'] = [3]
for pasta in Path('outputs').glob('em_andamento_ensaio_pca_gmm_*'):
    arq = pasta / 'configuracao.yaml'
    if arq.is_file():
        try:
            config_anterior = yaml.safe_load(arq.read_text(encoding='utf-8-sig'))
        except (OSError, yaml.YAMLError):
            continue
        if config_anterior == esperado:
            raise RuntimeError('Ja existe ensaio inacabado com esta configuracao: ' + str(pasta))

def alterar(texto, secao, numero):
    linhas = texto.splitlines(keepends=True)
    dentro = False
    achados = 0
    for i, linha in enumerate(linhas):
        sem_fim = linha.rstrip('\r\n')
        if re.fullmatch(rf'{re.escape(secao)}[ \t]*:[ \t]*(?:#.*)?', sem_fim):
            dentro = True
            continue
        if dentro and re.match(r'^[A-Za-z_][A-Za-z_0-9-]*[ \t]*:', sem_fim):
            dentro = False
        if dentro:
            m = re.fullmatch(
                r'([ \t]+n_components[ \t]*:[ \t]*)'
                r'(?:\[[^\]]+\]|[0-9]+)'
                r'([ \t]*(?:#.*)?)',
                sem_fim
            )
            if m:
                linhas[i] = m.group(1) + f'[{numero}]' + m.group(2) + linha[len(sem_fim):]
                achados += 1
    if achados != 1:
        raise RuntimeError(f'Esperado 1 campo {secao}.n_components, encontrados {achados}.')
    return ''.join(linhas)

novo = alterar(alterar(texto, 'pca', dim), 'gmm', 3)
cfg = yaml.safe_load(novo)
assert cfg == esperado, 'Mudancas no YAML excederiam PCA dimensionalidade e GMM K.'
bom = b'\xef\xbb\xbf' if original.startswith(b'\xef\xbb\xbf') else b''
p.write_bytes(bom + novo.encode('utf-8'))

print('CONFIGURACAO PRONTA')
print('PCA:', cfg['pca']['n_components'])
print('GMM K:', cfg['gmm']['n_components'])
print('Covariancia:', cfg['gmm']['covariance_type'])
print('n_init:', cfg['gmm']['n_init'])
print('RUS:', cfg['amostragem']['modo'])
print('Semente:', cfg['experimento']['semente'])
print('Fracao-base:', cfg['dados']['fracao_base'])
print('Todos os demais parametros: preservados')
print('F10: reservado')
'@

    $preparar | & $python - $Dim
    if ($LASTEXITCODE -ne 0) {
        throw 'Preparacao do YAML falhou; nenhum ensaio iniciado.'
    }

    Write-Host "`n[PLANEJAMENTO PCA ${Dim}D]" -ForegroundColor Cyan
    & $python -u -m src.orquestracao_oof --planejar
    if ($LASTEXITCODE -ne 0) {
        throw 'Planejamento falhou; treinamento nao iniciado.'
    }

    Write-Host "`n[EXECUCAO PCA ${Dim}D | K=3 | F1-F9]" -ForegroundColor Cyan
    & $python -u -m src.orquestracao_oof --ensaio --por-fold $PorFold
    if ($LASTEXITCODE -ne 0) {
        throw 'Execucao falhou; confira outputs/em_andamento_*. Nao exclua arquivos.'
    }
}
finally {
    # Restauracao byte a byte da configuracao ORIGINAL, inclusive se ocorrer erro.
    try {
        [System.IO.File]::WriteAllBytes($yaml, $original)
        Write-Host "`nYAML ORIGINAL RESTAURADO (byte a byte)." -ForegroundColor Green
    }
    catch {
        Write-Host "FALHA CRITICA RESTAURANDO YAML! USE O BACKUP: $backup" -ForegroundColor Red
        throw
    }
}

$novas = @(
    Get-ChildItem -LiteralPath $outputs -Directory -Filter 'ensaio_pca_gmm_*' |
        Where-Object { $anteriores -notcontains $_.FullName }
)
if ($novas.Count -ne 1) {
    throw "Esperava exatamente uma pasta nova, encontrei $($novas.Count). Confira outputs/."
}

$pasta = $novas[0].FullName
$galeria = Join-Path $pasta 'vencedor\visualizacao\galeria\index.html'
if (-not (Test-Path -LiteralPath $galeria -PathType Leaf)) {
    throw "Ensaio terminou, mas a galeria nao foi encontrada: $galeria"
}

# Conferir a configuracao gravada DENTRO do output.
$validar = @'
from pathlib import Path
import sys
import yaml
import json

pasta = Path(sys.argv[1]); dim = int(sys.argv[2]); por_fold = int(sys.argv[3])
config = yaml.safe_load((pasta / 'configuracao.yaml').read_text(encoding='utf-8-sig'))
assert config['pca']['n_components'] == [dim], 'PCA salvo diverge.'
assert config['gmm']['n_components'] == [3], 'GMM salvo diverge de K3.'
assert config['gmm']['covariance_type'].lower() == 'full', 'GMM salvo nao e FULL.'
assert int(config['divisao']['qual_fold_teste']) == 10, 'F10 nao esta reservado.'
manifesto = json.loads((pasta / 'manifesto.json').read_text(encoding='utf-8-sig'))
assert manifesto.get('status') == 'concluido_06_03', 'OOF nao concluiu.'
assert manifesto.get('visualizacoes') == 'concluidas', 'Visualizacoes incompletas.'
assert manifesto.get('tipo') == 'ensaio', 'Rodada nao e ensaio.'
print(f'OUTPUT VALIDADO: PCA {dim}D, GMM K=3 FULL, ensaio ate {por_fold} por fold.')
print('GALERIA:', pasta / 'vencedor' / 'visualizacao' / 'galeria' / 'index.html')
'@
$validar | & $python - $pasta $Dim $PorFold
if ($LASTEXITCODE -ne 0) {
    throw 'Output gerado, mas validacao da configuracao falhou. Verifique antes de comparar.'
}

# Registrar um caminho simples de cada galeria (sem criar galeria comparativa).
$ponteiro = Join-Path $outputs "ULTIMA_GALERIA_PCA${Dim}D_K3.txt"
[System.IO.File]::WriteAllText($ponteiro, $galeria, (New-Object System.Text.UTF8Encoding($false)))

Write-Host ''
Write-Host "PCA ${Dim}D CONCLUIDO E VALIDADO!" -ForegroundColor Green
Write-Host "OUTPUT: $pasta"
Write-Host "GALERIA: $galeria"
Write-Host "F10 nao utilizado. YAML principal restaurado."
Start-Process -FilePath $galeria
