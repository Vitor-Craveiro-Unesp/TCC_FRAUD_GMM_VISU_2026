# PRE01: inclui diagnostico de padronizacao nos ensaios PCA 1D/2D/3D K4 n_init5.
# Sem treinamento PCA/GMM, sem OOF, sem F10.
$ErrorActionPreference = 'Continue'
$raiz = (Get-Location).Path
$python = Join-Path $raiz '.venv\Scripts\python.exe'
$modulo = Join-Path $raiz 'src\visualizacao_padronizacao_folds.py'
$saida = Join-Path $raiz 'outputs'
$nome = 'PRE01_estabilidade_padronizacao_folds.png'
$metaNome = 'PRE01_estabilidade_padronizacao_folds_manifesto.json'

if (-not (Test-Path -LiteralPath (Join-Path $raiz 'src\orquestracao_oof.py'))) {
    throw 'Execute este comando na raiz TCC_EXPERIMENTAL_V3.'
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Python da venv nao encontrado.'
}
if (-not (Test-Path -LiteralPath $modulo -PathType Leaf)) {
    throw 'Modulo PRE01 nao encontrado em src. Execute primeiro o instalador ADICIONAR_DIAGNOSTICO_PADRONIZACAO_PCA_1D_2D_3D.ps1 -K 4 -NInit 5 -Tipo ensaio.'
}
if (-not (Test-Path -LiteralPath $saida -PathType Container)) { throw 'Pasta outputs inexistente.' }

$log = Join-Path $saida ('PRE01_ENSAIOS_STERR_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
Write-Host '[PRE01] Procurando trio de ensaios compativeis: PCA 1D/2D/3D, K4, n_init5.' -ForegroundColor Cyan
Write-Host '[PRE01] Reutilizando pre-processamento e indices de F1-F9. Nenhum PCA/GMM sera treinado.' -ForegroundColor Cyan

# O processo Python escreve eventuais avisos do Matplotlib no log; nao interrompe o PowerShell.
& $python -u -m src.visualizacao_padronizacao_folds --raiz "$raiz" --k 4 --n-init 5 --tipo ensaio 2> "$log"
$codigo = $LASTEXITCODE
if ($codigo -ne 0) {
    Write-Host '[FALHA] O PRE01 nao foi incluido. Veja o erro abaixo:' -ForegroundColor Red
    if (Test-Path -LiteralPath $log) { Get-Content -LiteralPath $log -Tail 80 }
    throw "PRE01 falhou com codigo de saida $codigo. Nenhum treinamento OOF foi executado."
}

$manifestos = @(Get-ChildItem -LiteralPath $saida -Directory -Filter 'validacao_padronizacao_folds_*' |
    Sort-Object LastWriteTime -Descending |
    ForEach-Object { Join-Path $_.FullName $metaNome } |
    Where-Object { Test-Path -LiteralPath $_ })
if ($manifestos.Count -eq 0) { throw 'PRE01 terminou, mas manifesto do diagnostico nao foi encontrado.' }
$meta = Get-Content -LiteralPath $manifestos[0] -Raw -Encoding UTF8 | ConvertFrom-Json
if ($meta.tipo -ne 'ensaio' -or $meta.k -ne 4 -or $meta.n_init -ne 5) {
    throw 'O manifesto mais recente nao corresponde a ensaio K4 n_init5.'
}

$cartoes = @()
foreach ($d in @(1,2,3)) {
    $id = "${d}D"
    $exp = [string]($meta.experimentos.$id)
    if (-not $exp) { throw "Manifesto nao possui PCA $id." }
    $indice = Join-Path $exp 'vencedor\visualizacao\galeria\index.html'
    $png = Join-Path $exp ('vencedor\visualizacao\galeria\oof\' + $nome)
    if (-not (Test-Path -LiteralPath $png -PathType Leaf)) {
        throw "PNG do PCA $id nao encontrado: $png"
    }
    if (-not (Test-Path -LiteralPath $indice -PathType Leaf)) { throw "Galeria do PCA $id nao encontrada." }
    $conteudo = [System.IO.File]::ReadAllText($indice)
    if ($conteudo -notlike '*PRE01_DIAGNOSTICO_PADRONIZACAO_FOLDS*') {
        throw "Grafico nao foi incluido no HTML da galeria PCA $id."
    }
    $pngUri = ([System.Uri]$png).AbsoluteUri
    $galUri = ([System.Uri]$indice).AbsoluteUri
    $cartoes += @"
<section><h2>PCA $id</h2><p><a href="$galUri" target="_blank">Abrir galeria completa $id</a> &nbsp; | &nbsp; <a href="$pngUri" target="_blank">Abrir PRE01 em tamanho original</a></p>
<a href="$pngUri" target="_blank"><img alt="PRE01 PCA $id" src="$pngUri"></a></section>
"@
    Write-Host "[CONFIRMADO] PCA ${id}: $png" -ForegroundColor Green
}

$painel = Join-Path $saida ('PAINEL_PRE01_ENSAIOS_K4_NINIT5_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.html')
$html = @"
<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>PRE01 — PCA 1D 2D 3D</title>
<style>body{font:16px system-ui,Arial,sans-serif;background:#f4f7fb;color:#24344b;max-width:1540px;margin:28px auto;padding:0 18px}h1{font-size:27px}p{color:#586b83}a{color:#205b9d}section{background:white;border:1px solid #dce4ef;border-radius:12px;margin:24px 0;padding:18px}img{display:block;width:100%;height:auto;object-fit:contain;border-radius:6px}</style>
</head><body><h1>PRE01 — Estabilidade da padronizacao</h1><p>Ensaios PCA 1D, 2D e 3D | GMM K=4 | n_init=5 | F1-F9 | F10 reservado.</p>
<p>A padronizacao antecede o PCA. Assim, para os mesmos dados e folds, este diagnostico deve ser identico nas tres galerias. Zero indica parametros iguais a referencia F1-F9; o grafico avalia estabilidade, nao acuracia preditiva.</p>
$($cartoes -join "`n")
</body></html>
"@
[System.IO.File]::WriteAllText($painel,$html,[System.Text.UTF8Encoding]::new($false))
Write-Host "`n[PRONTO] PRE01 confirmado nas tres galerias." -ForegroundColor Green
Write-Host "PAINEL: $painel"
try {
    Start-Process -FilePath $painel -ErrorAction Stop
    Write-Host '[NAVEGADOR] Abertura solicitada ao Windows.' -ForegroundColor Green
} catch {
    Write-Host 'Nao foi possivel abrir automaticamente; copie o caminho PAINEL e abra o HTML manualmente.' -ForegroundColor Yellow
}
