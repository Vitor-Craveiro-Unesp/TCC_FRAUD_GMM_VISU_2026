# TCC_EXPERIMENTAL_V3
# Comparacao FINAL de desenvolvimento: PCA 1D, 2D, 3D com GMM K=3 FULL.
# OOF integral em F1-F9; F10 NAO e acessado.
# Rode a partir da raiz do projeto, no PowerShell do Windows.

$ErrorActionPreference = 'Continue'   # Warnings em stderr (Matplotlib) nao devem parar o script.
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$raiz = (Get-Location).Path
$python = Join-Path $raiz '.venv\Scripts\python.exe'
$yaml = Join-Path $raiz 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
$outputs = Join-Path $raiz 'outputs'
$orquestrador = Join-Path $raiz 'src\orquestracao_oof.py'
foreach ($p in @($python, $yaml, $orquestrador)) {
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) {
        throw "Arquivo nao encontrado: $p. Execute na raiz do projeto."
    }
}
if (-not (Test-Path -LiteralPath $outputs -PathType Container)) {
    throw "Pasta outputs nao encontrada: $outputs"
}

# Um checkpoint proprio da comparacao, sem apagar saidas anteriores.
$pastaComparacao = Join-Path $outputs ("COMPARACAO_FINAL_PCA_1D_2D_3D_K3_" + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $pastaComparacao -ErrorAction Stop | Out-Null
$originalBytes = [System.IO.File]::ReadAllBytes($yaml)
$backupYAML = Join-Path $pastaComparacao 'CONFIGURACAO_ORIGINAL_BACKUP.yaml'
[System.IO.File]::WriteAllBytes($backupYAML, $originalBytes)
$hashOriginal = (Get-FileHash -LiteralPath $yaml -Algorithm SHA256).Hash

# Arquivo auxiliar temporario: muda semanticamente apenas
# dados.fracao_base=1.0 (mesmo valor nas tres execucoes),
# gmm.n_components=[3] (mesmo valor nas tres execucoes) e
# pca.n_components=[1]/[2]/[3] (unica diferenca ENTRE as execucoes).
$preparador = Join-Path $pastaComparacao '_preparar_dimensao.py'
$codigoPreparador = @'
from copy import deepcopy
from pathlib import Path
import sys
import yaml

origem = Path(sys.argv[1])
destino = Path(sys.argv[2])
dim = int(sys.argv[3])
assert dim in (1, 2, 3)
base = yaml.safe_load(origem.read_text(encoding='utf-8-sig'))
assert base['reducao']['metodo'].upper() == 'PCA', 'Metodo diferente de PCA.'
assert int(base['divisao']['qual_fold_teste']) == 10, 'F10 nao reservado.'
assert base['gmm']['covariance_type'].lower() == 'full', 'GMM deve usar FULL.'
assert base['execucao']['proteger_resultados'] is True, 'Protecao de outputs desativada.'
assert base['execucao'].get('diretorio_retomada') is None, 'Retomada explicita ativa.'
assert Path(base['saidas']['diretorio_base']) == Path('outputs'), 'Diretorio de saida inesperado.'
config = deepcopy(base)
config['dados']['fracao_base'] = 1.0
config['gmm']['n_components'] = [3]
config['pca']['n_components'] = [dim]
# Nunca alterar divisao, semente, amostragem, metricas ou visualizacao.
assert int(config['divisao']['qual_fold_teste']) == 10
assert config['gmm']['covariance_type'].lower() == 'full'
destino.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
assert yaml.safe_load(destino.read_text(encoding='utf-8')) == config
print(f'CONFIGURACAO: PCA {dim}D; K=3; FULL; 100% do desenvolvimento; F10 reservado.')
print('Semente:', config['experimento']['semente'])
print('Amostragem:', config['amostragem']['modo'])
print('RUS razao:', config['amostragem']['razao_negativos_positivos'])
'@
[System.IO.File]::WriteAllText($preparador, $codigoPreparador, (New-Object System.Text.UTF8Encoding($false)))

$execucoes = @()
try {
    foreach ($dim in @(1,2,3)) {
        Write-Host "`n=========================================================" -ForegroundColor Cyan
        Write-Host "PCA ${dim}D + GMM K=3 FULL | OOF INTEGRAL F1-F9" -ForegroundColor Cyan
        Write-Host "=========================================================" -ForegroundColor Cyan

        # Restaurar a base original antes de preparar cada dimensao.
        [System.IO.File]::WriteAllBytes($yaml, $originalBytes)
        $antes = @(Get-ChildItem -LiteralPath $outputs -Directory -Filter 'experimento_pca_gmm_*' | ForEach-Object { $_.FullName })

        & $python $preparador $backupYAML $yaml $dim
        if ($LASTEXITCODE -ne 0) { throw "Preparacao do PCA ${dim}D falhou." }

        & $python -u -m src.orquestracao_oof --planejar
        if ($LASTEXITCODE -ne 0) { throw "Planejamento do PCA ${dim}D falhou." }

        & $python -u -m src.orquestracao_oof --executar
        if ($LASTEXITCODE -ne 0) {
            throw "Execucao PCA ${dim}D interrompida. Resultados existentes preservados; confira outputs/."
        }

        $novas = @(Get-ChildItem -LiteralPath $outputs -Directory -Filter 'experimento_pca_gmm_*' |
            Where-Object { $antes -notcontains $_.FullName })
        if ($novas.Count -ne 1) {
            throw "PCA ${dim}D: esperado um output novo, encontrados $($novas.Count)."
        }
        $pasta = $novas[0].FullName
        $galeria = Join-Path $pasta 'vencedor\visualizacao\galeria\index.html'
        $configSalva = Join-Path $pasta 'configuracao.yaml'
        if (-not (Test-Path -LiteralPath $galeria -PathType Leaf)) {
            throw "OOF PCA ${dim}D terminou, mas galeria ainda nao existe: $galeria"
        }
        if (-not (Test-Path -LiteralPath $configSalva -PathType Leaf)) {
            throw "Configuracao do PCA ${dim}D nao encontrada: $configSalva"
        }
        $execucoes += [PSCustomObject]@{ Dim=$dim; Pasta=$pasta; Galeria=$galeria }
        Write-Host "PCA ${dim}D: GALERIA GERADA - $galeria" -ForegroundColor Green

        # Ja na proxima iteracao, a configuracao original sera recuperada.
        [System.IO.File]::WriteAllBytes($yaml, $originalBytes)
        $hashRestaurado = (Get-FileHash -LiteralPath $yaml -Algorithm SHA256).Hash
        if ($hashRestaurado -ne $hashOriginal) { throw 'Falha na restauracao do YAML original.' }
    }
}
finally {
    [System.IO.File]::WriteAllBytes($yaml, $originalBytes)
    $hashFinal = (Get-FileHash -LiteralPath $yaml -Algorithm SHA256).Hash
    if ($hashFinal -eq $hashOriginal) {
        Write-Host "`nYAML ORIGINAL RESTAURADO byte a byte." -ForegroundColor Green
    } else {
        Write-Host "ERRO: YAML pode nao ter sido restaurado. Backup: $backupYAML" -ForegroundColor Red
    }
}

if ($execucoes.Count -ne 3) { throw "A comparacao ficou incompleta: $($execucoes.Count)/3 galerias." }

# Auditoria das tres configuracoes efetivamente persistidas em outputs/.
$auditoria = Join-Path $pastaComparacao '_auditar_comparabilidade.py'
$codigoAuditoria = @'
from copy import deepcopy
from pathlib import Path
import json
import sys
import yaml

assert len(sys.argv) == 4
referencia = None
for dim, pasta_str in enumerate(sys.argv[1:], start=1):
    pasta = Path(pasta_str)
    cfg = yaml.safe_load((pasta / 'configuracao.yaml').read_text(encoding='utf-8-sig'))
    manifest = json.loads((pasta / 'manifesto.json').read_text(encoding='utf-8-sig'))
    assert cfg['reducao']['metodo'].upper() == 'PCA'
    assert cfg['pca']['n_components'] == [dim], f'Dimensao errada: PCA {dim}D'
    assert cfg['gmm']['n_components'] == [3], 'K GMM diferente de 3.'
    assert cfg['gmm']['covariance_type'].lower() == 'full'
    assert cfg['dados']['fracao_base'] == 1.0, 'Nao usou 100% do desenvolvimento.'
    assert int(cfg['divisao']['qual_fold_teste']) == 10
    assert manifest.get('tipo') == 'oficial', 'Rodada ainda e ensaio amostral.'
    assert manifest.get('status') == 'concluido_06_03'
    assert manifest.get('visualizacoes') == 'concluidas'
    assert (pasta / 'vencedor' / 'visualizacao' / 'galeria' / 'index.html').is_file()
    comp = deepcopy(cfg)
    comp['pca']['n_components'] = ['DIMENSAO_VARIAVEL']
    if referencia is None:
        referencia = comp
    else:
        assert comp == referencia, 'Hiperparametros diferem alem da dimensionalidade PCA.'
    print(f'PCA {dim}D: VALIDADO | FULL | K=3 | F1-F9 | F10 reservado')
print('COMPARABILIDADE VALIDADA: os tres YAMLs diferem somente em pca.n_components.')
'@
[System.IO.File]::WriteAllText($auditoria, $codigoAuditoria, (New-Object System.Text.UTF8Encoding($false)))
& $python $auditoria $execucoes[0].Pasta $execucoes[1].Pasta $execucoes[2].Pasta
if ($LASTEXITCODE -ne 0) {
    throw "Galerias geradas, mas auditoria de igualdade dos hiperparametros falhou. Confira $pastaComparacao"
}

# Registrar manifestos de caminhos e gerar uma pagina inicial de navegacao.
$registro = Join-Path $pastaComparacao 'CAMINHOS_DAS_GALERIAS.json'
$execucoes | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $registro -Encoding UTF8

$cartoes = @()
foreach ($e in $execucoes) {
    $href = [System.Uri]::new($e.Galeria).AbsoluteUri
    $href = [System.Net.WebUtility]::HtmlEncode($href)
    $pastaLegivel = [System.Net.WebUtility]::HtmlEncode($e.Pasta)
    $cartoes += @"
    <section class="card"><h2>PCA $($e.Dim)D</h2>
      <p>GMM K=3 · FULL · OOF F1–F9</p>
      <a href="$href" target="_blank" rel="noopener">Abrir galeria PCA $($e.Dim)D</a>
      <small>$pastaLegivel</small>
    </section>
"@
}

$html = @"
<!doctype html><html lang="pt-BR"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Comparacao PCA 1D 2D 3D - GMM K3</title>
<style>
*{box-sizing:border-box}body{font:16px system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#f4f7fb;color:#14263e}
main{max-width:1200px;margin:0 auto;padding:44px 25px}h1{font-size:34px;margin:0 0 8px}
.sub{color:#506581;line-height:1.5;margin-bottom:35px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}
.card{background:white;border:1px solid #dbe3ee;border-radius:14px;padding:24px;box-shadow:0 3px 12px #17274010}
.card h2{font-size:26px;margin:0 0 10px}.card p{color:#5b6d83;font-size:14px}
.card a{display:inline-block;background:#1e4978;color:white;text-decoration:none;padding:12px 16px;border-radius:8px;font-weight:650;margin:14px 0}
.card a:hover{background:#163a60}.card small{display:block;overflow-wrap:anywhere;color:#69798e;font-size:11px}
footer{margin-top:30px;color:#62748a;font-size:13px;line-height:1.5}
@media(max-width:800px){.grid{grid-template-columns:1fr}main{padding:25px 15px}}
</style></head><body><main>
<h1>Comparação final de desenvolvimento — PCA 1D × 2D × 3D</h1>
<p class="sub">Três execuções OOF independentes usando 100% do desenvolvimento F1–F9. Todos os hiperparâmetros são idênticos, exceto a dimensão PCA (1, 2 ou 3). GMM K=3, covariância FULL. O F10 não foi utilizado.</p>
<div class="grid">$($cartoes -join "`n")</div>
<footer>As galerias individuais contêm os gráficos pertinentes a cada dimensão. Esta página reúne apenas a navegação, sem alterar modelos ou resultados.</footer>
</main></body></html>
"@
$indice = Join-Path $pastaComparacao 'ABRIR_TRES_GALERIAS.html'
[System.IO.File]::WriteAllText($indice, $html, (New-Object System.Text.UTF8Encoding($false)))

Write-Host "`n============================================" -ForegroundColor Green
Write-Host "COMPARACAO PCA 1D / 2D / 3D CONCLUIDA!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host "Auditoria: comparabilidade aprovada; F10 preservado."
Write-Host "Galeria central: $indice"
Start-Process -FilePath $indice
