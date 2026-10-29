# TCC_EXPERIMENTAL_V3
# Teste PCA 1D, 2D, 3D, GMM K=4, FULL, 5 inicializacoes; abre painel.
# Retoma o ensaio PCA1D anterior se sua configuracao bater exatamente.
# Padrao: ensaio rapido com 1000 observacoes/fold. Use -Completo para OOF integral.
# O F10 NUNCA e utilizado e os outputs anteriores nao sao sobrescritos.
# Rode a partir da raiz do projeto, no PowerShell do Windows.

[CmdletBinding()]
param(
    [switch]$Rapido,
    [switch]$Completo,
    [ValidateRange(20, 1000000)]
    [int]$PorFold = 1000,
    [switch]$SemRetomar1D
)

if ($Completo -and $Rapido) { throw 'Escolha somente -Completo ou -Rapido.' }
if (-not $Completo) { $Rapido = $true } # Por padrao: teste rapido.
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
$modo = if ($Rapido) { 'ensaio' } else { 'completo' }
$pastaComparacao = Join-Path $outputs ("COMPARACAO_PCA_1D_2D_3D_K4_NINIT5_${modo}_" + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $pastaComparacao -ErrorAction Stop | Out-Null
$originalBytes = [System.IO.File]::ReadAllBytes($yaml)
$backupYAML = Join-Path $pastaComparacao 'CONFIGURACAO_ORIGINAL_BACKUP.yaml'
[System.IO.File]::WriteAllBytes($backupYAML, $originalBytes)
$hashOriginal = (Get-FileHash -LiteralPath $yaml -Algorithm SHA256).Hash

# GMM01 3D do projeto ainda assumia K=3 em varios trechos.
# Ajustar de maneira conservadora para usar todos os 4 componentes.
$adaptador = Join-Path $pastaComparacao '_adaptar_gmm01_3d_k_dinamico.py'
$codigoAdaptador = @'
# -*- coding: utf-8 -*-
"""Adapta o gerador GMM01 PCA3D validado K=3 para K dinamico (incluindo K=4).

Atua somente no codigo-fonte, sem treinar ou gerar resultados.
Uso: python _adaptar_gmm01_3d_para_k4.py <caminho_codigo> <pasta_backup>
"""
from __future__ import annotations
import ast
from datetime import datetime
from pathlib import Path
import re
import shutil
import sys


def mudar_bloco(codigo, funcao, trocas):
    inicio = codigo.find('def ' + funcao + '(')
    if inicio < 0:
        raise RuntimeError('Funcao esperada nao encontrada: ' + funcao)
    fim = codigo.find('\ndef ', inicio + 5)
    if fim < 0:
        fim = len(codigo)
    bloco = codigo[inicio:fim]
    for anterior, novo, esperado in trocas:
        ocorrencias = bloco.count(anterior)
        if ocorrencias != esperado:
            raise RuntimeError(f'{funcao}: ancora incompatível (esperado {esperado}, encontrado {ocorrencias}): {anterior[:75]}')
        bloco = bloco.replace(anterior, novo)
    return codigo[:inicio] + bloco + codigo[fim:]


def adaptar(codigo):
    if 'K4_DINAMICO_VALIDADO_20261028' in codigo:
        ast.parse(codigo)
        return codigo, False
    if "COLORS = ('#1685bb', '#f0842d', '#25a277')" not in codigo:
        raise RuntimeError('Paleta GMM01 inesperada; adaptação bloqueada.')
    codigo = codigo.replace(
        "COLORS = ('#1685bb', '#f0842d', '#25a277')",
        "COLORS = ('#1685bb', '#f0842d', '#25a277', '#c22c70', '#735bd0', '#8c564b')",
        1,
    )
    codigo = mudar_bloco(codigo, '_validar', [
        (
            'if mu.shape!=(3,3) or cov.shape!=(3,3,3) or weights.shape!=(3,):',
            'if (mu.ndim != 2 or mu.shape[1] != 3 or len(mu) < 1 or\n'
            '            cov.shape != (len(mu), 3, 3) or weights.shape != (len(mu),)):', 1),
        ('GMM01 3D: este layout validado requer K=3 e PCA 3D.',
         'GMM01 3D: requer K positivo e PCA 3D.', 1),
    ])
    codigo = mudar_bloco(codigo, '_densidade_componentes', [
        ('for k in range(3)', 'for k in range(len(mu))', 1),
    ])
    codigo = mudar_bloco(codigo, '_planos', [
        ('for h in range(3)', 'for h in range(len(mu))', 1),
        ('for k in range(3)', 'for k in range(len(mu))', 1),
    ])
    codigo = mudar_bloco(codigo, '_cortes', [
        ('plt.subplots(1,3,figsize=(19.5,6.4)',
         'plt.subplots(1,len(mu),figsize=(6.5*len(mu),6.4)', 1),
        ('for k in range(3)', 'for k in range(len(mu))', 2),
        ('ncol=3,loc=', 'ncol=min(4,len(mu)),loc=', 1),
    ])
    codigo = mudar_bloco(codigo, '_geometria_png', [
        ('for k in range(3)', 'for k in range(len(mu))', 1),
        ("'FULL | K=3 | elipsoides de concentração gaussiana de 95%'",
         "f'FULL | K={len(mu)} | elipsoides de concentração gaussiana de 95%'", 1),
    ])
    codigo = mudar_bloco(codigo, '_malha_3d', [
        ('for k in range(3)', 'for k in range(len(mu))', 1),
    ])
    codigo = mudar_bloco(codigo, '_interativo_html', [
        ('for k in range(3)', 'for k in range(len(mu))', 1),
        ('[False,True]*3', '[False,True]*len(mu)', 1),
        ('[True,True]*3', '[True,True]*len(mu)', 2),
        ("html.replace('__CHART__',chart)",
         "html.replace('__CHART__',chart).replace('K=3',f'K={len(mu)}')", 1),
    ])
    if 'def _geometria_interativa_html(' in codigo:
        codigo = mudar_bloco(codigo, '_geometria_interativa_html', [
            ('for k in range(3)', 'for k in range(len(mu))', 1),
            ("pagina.replace('__GRAFICO__', grafico)",
             "pagina.replace('__GRAFICO__', grafico).replace('K=3',f'K={len(mu)}')", 1),
        ])
        # A visibilidade HTML deve contemplar 1+2*K camadas.
        inicio = codigo.find('function visiveis(tipo){parar();')
        fim = codigo.find('Plotly.restyle(plot,{visible:v});}', inicio)
        if inicio < 0 or fim < 0:
            raise RuntimeError('Controles HTML da geometria PCA3D nao encontrados.')
        fim += len('Plotly.restyle(plot,{visible:v});}')
        codigo = (codigo[:inicio] +
            "function visiveis(tipo){parar();const n=plot.data.length;"
            "const v=Array.from({length:n},(_,i)=>tipo==='todos'?true:"
            "(tipo==='gaussianas'?i>0:i===0));Plotly.restyle(plot,{visible:v});}"
            + codigo[fim:])
    codigo = codigo.replace('Gera, para PCA 3D / GMM FULL / K=3:',
                           'Gera, para PCA 3D / GMM FULL / K dinamico:', 1)
    codigo = codigo.replace('"""Nove seções da MESMA densidade conjunta: 3 pares x 3 alturas."""',
                           '"""Secoes da MESMA densidade conjunta: 3 pares x K alturas."""', 1)
    codigo = '# K4_DINAMICO_VALIDADO_20261028\n' + codigo
    ast.parse(codigo)
    return codigo, True


def main():
    fonte = Path(sys.argv[1]).resolve()
    pasta = Path(sys.argv[2]).resolve()
    if not fonte.is_file():
        raise FileNotFoundError(fonte)
    original = fonte.read_text(encoding='utf-8-sig')
    novo, modificado = adaptar(original)
    if modificado:
        pasta.mkdir(parents=True, exist_ok=True)
        copia = pasta / ('visualizacao_gmm01_3d.py.ANTES_K4_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        shutil.copy2(fonte, copia)
        fonte.write_text(novo, encoding='utf-8')
        print('GMM01 3D: preparado para quatro gaussianas (K dinamico).')
        print('BACKUP:', copia)
    else:
        print('GMM01 3D: adaptacao K dinamico ja estava presente.')

if __name__ == '__main__':
    main()

'@
[System.IO.File]::WriteAllText($adaptador, $codigoAdaptador, (New-Object System.Text.UTF8Encoding($false)))
$moduloGMM01 = Join-Path $raiz 'src\visualizacao_gmm01_3d.py'
if (-not (Test-Path -LiteralPath $moduloGMM01 -PathType Leaf)) {
    throw 'Modulo GMM01 3D nao encontrado. Nenhum treinamento iniciado.'
}
& $python $adaptador $moduloGMM01 $pastaComparacao
if ($LASTEXITCODE -ne 0) {
    throw 'Modulo GMM01 3D ainda nao aceita K=4. Treinamento bloqueado para evitar uma galeria incompleta.'
}

# Arquivo auxiliar temporario: muda semanticamente apenas
# dados.fracao_base=1.0 no modo completo (mesmo valor nas tres),
# gmm.n_components=[4] e gmm.n_init=5 nas tres execucoes,
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
modo = str(sys.argv[4])
assert dim in (1, 2, 3) and modo in ('completo', 'ensaio')
base = yaml.safe_load(origem.read_text(encoding='utf-8-sig'))
assert base['reducao']['metodo'].upper() == 'PCA', 'Metodo diferente de PCA.'
assert int(base['divisao']['qual_fold_teste']) == 10, 'F10 nao reservado.'
assert base['gmm']['covariance_type'].lower() == 'full', 'GMM deve usar FULL.'
assert base['execucao']['proteger_resultados'] is True, 'Protecao de outputs desativada.'
assert base['execucao'].get('diretorio_retomada') is None, 'Retomada explicita ativa.'
assert Path(base['saidas']['diretorio_base']) == Path('outputs'), 'Diretorio de saida inesperado.'
config = deepcopy(base)
if modo == 'completo':
    config['dados']['fracao_base'] = 1.0
config['gmm']['n_components'] = [4]
config['gmm']['n_init'] = 5
config['pca']['n_components'] = [dim]
# Nunca alterar divisao, semente, amostragem, metricas ou visualizacao.
assert int(config['divisao']['qual_fold_teste']) == 10
assert config['gmm']['covariance_type'].lower() == 'full'
destino.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
assert yaml.safe_load(destino.read_text(encoding='utf-8')) == config
print(f'CONFIGURACAO: PCA {dim}D; K=4; 5 inicializacoes; FULL; modo={modo}; F10 reservado.')
print('Semente:', config['experimento']['semente'])
print('Amostragem:', config['amostragem']['modo'])
print('RUS razao:', config['amostragem']['razao_negativos_positivos'])
print('n_init:', config['gmm']['n_init'])
'@
[System.IO.File]::WriteAllText($preparador, $codigoPreparador, (New-Object System.Text.UTF8Encoding($false)))

$execucoes = @()
$dimensoesPendentes = @(1,2,3)
# O ensaio 1D foi treinado ate o fim em 21:30, mas 07.01 foi interrompido.
# Reutiliza o OOF ja existente SOMENTE se o YAML salvo for exatamente o esperado.
# O caminho pode nao existir em outra maquina; nesse caso roda PCA1D do zero.
$candidatoReutilizavel = Join-Path $outputs 'ensaio_pca_gmm_20261028_213039'
if ($Rapido -and -not $SemRetomar1D -and
    (Test-Path -LiteralPath (Join-Path $candidatoReutilizavel 'manifesto.json') -PathType Leaf)) {

    $esperado1D = Join-Path $pastaComparacao 'CONFIG_ESPERADA_PCA1D.yaml'
    & $python $preparador $backupYAML $esperado1D 1 $modo
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao preparar configuracao esperada para retomada 1D.' }

    $verificador = Join-Path $pastaComparacao '_validar_retomada_1d.py'
    $codigoVerificador = @'
import json, sys, yaml
from pathlib import Path
pasta = Path(sys.argv[1])
config = yaml.safe_load((pasta/'configuracao.yaml').read_text(encoding='utf-8-sig'))
esperado = yaml.safe_load(Path(sys.argv[2]).read_text(encoding='utf-8-sig'))
manifesto = json.loads((pasta/'manifesto.json').read_text(encoding='utf-8-sig'))
assert config == esperado, 'Configuracao 1D anterior difere da nova comparacao; nao reutilizar.'
assert manifesto.get('status') == 'concluido_06_03', 'OOF PCA1D incompleto.'
assert manifesto.get('tipo') == 'ensaio', 'O experimento nao e um ensaio.'
assert int(manifesto.get('registros_desenvolvimento', -1)) == 9000, 'Tamanho do ensaio anterior inesperado.'
assert (pasta/'vencedor'/'oof'/'predicoes.npz').is_file(), 'Predicoes OOF do vencedor ausentes.'
print('[RETOMADA] PCA 1D anterior compativel: 9000 predições OOF ja calculadas.')
'@
    [System.IO.File]::WriteAllText($verificador, $codigoVerificador, (New-Object System.Text.UTF8Encoding($false)))
    & $python $verificador $candidatoReutilizavel $esperado1D
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[RETOMADA] PCA 1D: completando apenas graficos 07.01, 07.02 e 07.03." -ForegroundColor Cyan
        & $python -u -m src.orquestracao_oof --so-visualizacoes --experimento $candidatoReutilizavel
        if ($LASTEXITCODE -ne 0) {
            throw 'Retomada dos graficos PCA 1D falhou; OOF original preservado.'
        }
        $galeria1DRetomada = Join-Path $candidatoReutilizavel 'vencedor\visualizacao\galeria\index.html'
        if (-not (Test-Path -LiteralPath $galeria1DRetomada -PathType Leaf)) {
            throw 'Galeria do PCA 1D nao foi criada apos a retomada.'
        }
        $execucoes += [PSCustomObject]@{ Dim=1; Pasta=$candidatoReutilizavel; Galeria=$galeria1DRetomada }
        $dimensoesPendentes = @(2,3)
        Write-Host '[RETOMADA] PCA 1D pronto sem repetir os 9 folds.' -ForegroundColor Green
    } else {
        Write-Host '[AVISO] PCA 1D anterior incompativel; um novo PCA 1D sera executado.' -ForegroundColor Yellow
    }
}

try {
    foreach ($dim in $dimensoesPendentes) {
        Write-Host "`n=========================================================" -ForegroundColor Cyan
        Write-Host "PCA ${dim}D + GMM K=4 FULL | n_init=5 | $modo | F1-F9" -ForegroundColor Cyan
        Write-Host "=========================================================" -ForegroundColor Cyan

        # Restaurar a base original antes de preparar cada dimensao.
        [System.IO.File]::WriteAllBytes($yaml, $originalBytes)
        $filtro = if ($Rapido) { 'ensaio_pca_gmm_*' } else { 'experimento_pca_gmm_*' }
        $antes = @(Get-ChildItem -LiteralPath $outputs -Directory -Filter $filtro | ForEach-Object { $_.FullName })

        & $python $preparador $backupYAML $yaml $dim $modo
        if ($LASTEXITCODE -ne 0) { throw "Preparacao do PCA ${dim}D falhou." }

        & $python -u -m src.orquestracao_oof --planejar
        if ($LASTEXITCODE -ne 0) { throw "Planejamento do PCA ${dim}D falhou." }

        if ($Rapido) {
            & $python -u -m src.orquestracao_oof --ensaio --por-fold $PorFold
        }
        else {
            & $python -u -m src.orquestracao_oof --executar
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Execucao PCA ${dim}D interrompida. Resultados existentes preservados; confira outputs/."
        }

        $novas = @(Get-ChildItem -LiteralPath $outputs -Directory -Filter $filtro |
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
# F10 nao e carregado pelo orquestrador em modo desenvolvimento.
$auditoria = Join-Path $pastaComparacao '_auditar_comparabilidade.py'
$codigoAuditoria = @'
from copy import deepcopy
from pathlib import Path
import json
import sys
import yaml

assert len(sys.argv) == 5
modo = sys.argv[1]
referencia = None
for dim, pasta_str in enumerate(sys.argv[2:], start=1):
    pasta = Path(pasta_str)
    cfg = yaml.safe_load((pasta / 'configuracao.yaml').read_text(encoding='utf-8-sig'))
    manifest = json.loads((pasta / 'manifesto.json').read_text(encoding='utf-8-sig'))
    assert cfg['reducao']['metodo'].upper() == 'PCA'
    assert cfg['pca']['n_components'] == [dim], f'Dimensao errada: PCA {dim}D'
    assert cfg['gmm']['n_components'] == [4], 'K GMM diferente de 4.'
    assert cfg['gmm']['n_init'] == 5, 'Esperadas cinco inicializacoes EM.'
    assert cfg['gmm']['covariance_type'].lower() == 'full'
    if modo == 'completo':
        assert cfg['dados']['fracao_base'] == 1.0, 'Nao usou 100% do desenvolvimento.'
    assert int(cfg['divisao']['qual_fold_teste']) == 10
    assert manifest.get('tipo') == ('ensaio' if modo == 'ensaio' else 'oficial'), 'Tipo de rodada inesperado.'
    assert manifest.get('status') == 'concluido_06_03'
    assert manifest.get('visualizacoes') == 'concluidas'
    assert (pasta / 'vencedor' / 'visualizacao' / 'galeria' / 'index.html').is_file()
    comp = deepcopy(cfg)
    comp['pca']['n_components'] = ['DIMENSAO_VARIAVEL']
    if referencia is None:
        referencia = comp
    else:
        assert comp == referencia, 'Hiperparametros diferem alem da dimensionalidade PCA.'
    print(f'PCA {dim}D: VALIDADO | FULL | K=4 | n_init=5 | F1-F9 | F10 reservado')
print('COMPARABILIDADE VALIDADA: os tres YAMLs diferem somente em pca.n_components.')
'@
[System.IO.File]::WriteAllText($auditoria, $codigoAuditoria, (New-Object System.Text.UTF8Encoding($false)))
& $python $auditoria $modo $execucoes[0].Pasta $execucoes[1].Pasta $execucoes[2].Pasta
if ($LASTEXITCODE -ne 0) {
    throw "Galerias geradas, mas auditoria de igualdade dos hiperparametros falhou. Confira $pastaComparacao"
}

# PRE01 para este trio. -Tipo ensaio no modo rapido evita selecionar galerias oficiais antigas.
$instaladorPRE01 = Join-Path $raiz 'ADICIONAR_DIAGNOSTICO_PADRONIZACAO_PCA_1D_2D_3D.ps1'
if (Test-Path -LiteralPath $instaladorPRE01 -PathType Leaf) {
    $tipoPRE01 = if ($Rapido) { 'ensaio' } else { 'oficial' }
    Write-Host "`n[PRE01] Inserindo diagnostico no trio $tipoPRE01 / K=4 / n_init=5" -ForegroundColor Cyan
    # O instalador faz auditoria dos mesmos indices OOF entre as tres galerias.
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $instaladorPRE01 -K 4 -NInit 5 -Tipo $tipoPRE01
    if ($LASTEXITCODE -ne 0) {
        throw 'OOF e galerias concluidos, mas PRE01 falhou. Os resultados foram preservados.'
    }
    foreach ($e in $execucoes) {
        $textoGaleria = [System.IO.File]::ReadAllText($e.Galeria)
        if (-not $textoGaleria.Contains('PRE01_DIAGNOSTICO_PADRONIZACAO_FOLDS')) {
            throw "PRE01 nao foi inserido na galeria PCA $($e.Dim)D: $($e.Galeria)"
        }
    }
    Write-Host '[PRE01] Confirmado no HTML das tres galerias.' -ForegroundColor Green
} else {
    Write-Host '[AVISO] Instalador PRE01 nao encontrado: painel sera aberto sem o novo diagnostico.' -ForegroundColor Yellow
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
      <p>GMM K=4 · n_init=5 · FULL · F1–F9</p>
      <a href="$href" target="_blank" rel="noopener">Abrir galeria PCA $($e.Dim)D</a>
      <small>$pastaLegivel</small>
    </section>
"@
}

$html = @"
<!doctype html><html lang="pt-BR"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Comparacao PCA 1D 2D 3D - GMM K4 NINIT5</title>
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
<h1>Comparação PCA 1D × 2D × 3D — GMM K=4 e n_init=5</h1>
<p class="sub">Três execuções independentes do desenvolvimento F1–F9. Todos os hiperparâmetros são idênticos entre as três, exceto a dimensão PCA (1, 2 ou 3). GMM K=4, cinco inicializações EM, covariância FULL. O F10 não foi utilizado. Modo: $modo.</p>
<div class="grid">$($cartoes -join "`n")</div>
<footer>As galerias individuais contêm os gráficos pertinentes a cada dimensão. Esta página reúne apenas a navegação, sem alterar modelos ou resultados.</footer>
</main></body></html>
"@
$indice = Join-Path $pastaComparacao 'ABRIR_TRES_GALERIAS.html'
[System.IO.File]::WriteAllText($indice, $html, (New-Object System.Text.UTF8Encoding($false)))

Write-Host "`n============================================" -ForegroundColor Green
Write-Host "COMPARACAO PCA 1D / 2D / 3D K4 NINIT5 CONCLUIDA!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host "Auditoria: comparabilidade aprovada; F10 preservado."
Write-Host "Modo: $modo"
Write-Host "GMM: K=4 | n_init=5 | FULL | F10 reservado"
Write-Host "Galeria central: $indice"
Start-Process -FilePath $indice
Write-Host "`n[PRONTO] Painel das tres galerias aberto: $indice" -ForegroundColor Green
