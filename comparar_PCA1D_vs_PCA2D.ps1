# TCC_EXPERIMENTAL_V3 — novo ensaio PCA 1D comparavel ao PCA 2D
# Execute na raiz do projeto em PowerShell.
# Nao executa F10; preserva os ensaios anteriores.

$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$python = '.\.venv\Scripts\python.exe'
$yaml = '.\config_TCC_EXPERIMENTAL_V3_atualizado.yaml'

if (-not (Test-Path -LiteralPath $python)) { throw 'Ambiente virtual nao encontrado.' }
if (-not (Test-Path -LiteralPath $yaml)) { throw 'YAML principal nao encontrado.' }
if (-not (Test-Path -LiteralPath '.\src\orquestracao_oof.py')) {
    throw 'Execute na raiz do TCC_EXPERIMENTAL_V3.'
}

New-Item -ItemType Directory -Path '.\outputs' -Force | Out-Null

# 1. Encontrar o ultimo ensaio PCA 2D com OOF e galeria concluidos.
$referencia = @'
from pathlib import Path
import json
import yaml

opcoes = sorted(Path('outputs').glob('ensaio_pca_gmm_*'),
                key=lambda p: p.stat().st_mtime, reverse=True)
for pasta in opcoes:
    arq = pasta / 'configuracao.yaml'
    man = pasta / 'manifesto.json'
    gal = pasta / 'vencedor' / 'visualizacao' / 'galeria' / 'index.html'
    if not (arq.is_file() and man.is_file() and gal.is_file()):
        continue
    try:
        cfg = yaml.safe_load(arq.read_text(encoding='utf-8-sig'))
        m = json.loads(man.read_text(encoding='utf-8-sig'))
        if (cfg['reducao']['metodo'].upper() == 'PCA'
                and cfg['pca']['n_components'] == [2]
                and m.get('status') == 'concluido_06_03'
                and m.get('visualizacoes') == 'concluidas'):
            print(str(pasta.resolve()))
            break
    except (ValueError, KeyError, TypeError, OSError):
        continue
else:
    raise RuntimeError(
        'Nao encontrei um ensaio PCA 2D com galeria concluida em outputs/. '
        'Confirme primeiro a pasta do PCA 2D.')
'@ | & $python -

if ($LASTEXITCODE -ne 0) { throw 'Nao foi possivel localizar a referencia PCA 2D.' }
$ref2d = ($referencia | Select-Object -Last 1).Trim()
Write-Host "Referencia PCA 2D: $ref2d" -ForegroundColor Cyan

# 2. Comparar TODO o YAML com o usado no PCA 2D e trocar SOMENTE [2] por [1].
$preparo = @'
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import re
import sys
import yaml

pasta2 = Path(sys.argv[1])
yaml2 = yaml.safe_load((pasta2 / 'configuracao.yaml').read_text(encoding='utf-8-sig'))
arquivo = Path('config_TCC_EXPERIMENTAL_V3_atualizado.yaml')
original = arquivo.read_bytes()
bom = original.startswith(b'\xef\xbb\xbf')
texto = original.decode('utf-8-sig')
atual = yaml.safe_load(texto)

if yaml2['pca']['n_components'] != [2]:
    raise RuntimeError('Referencia nao possui PCA [2].')
if yaml2['reducao']['metodo'].upper() != 'PCA':
    raise RuntimeError('Referencia nao e PCA.')

alvo = deepcopy(yaml2)
alvo['pca']['n_components'] = [1]
if atual != yaml2 and atual != alvo:
    topicos = sorted(k for k in set(atual) | set(yaml2)
                     if atual.get(k) != yaml2.get(k)
                     and not (k == 'pca' and atual.get(k) == alvo.get(k)))
    raise RuntimeError(
        'O YAML local difere da configuracao do ensaio 2D alem da dimensionalidade. '
        'Nenhum arquivo foi alterado. Secoes divergentes: ' + str(topicos))

if atual == yaml2:
    linhas = texto.splitlines(keepends=True)
    secao_pca = False
    alteracoes = 0
    for i, linha in enumerate(linhas):
        limpo = linha.rstrip('\r\n')
        if re.match(r'^pca\s*:', limpo):
            secao_pca = True
            continue
        if secao_pca and re.match(r'^[^\s#][^:]*:', limpo):
            secao_pca = False
        if secao_pca:
            m = re.fullmatch(
                r'([ \t]+n_components[ \t]*:[ \t]*)\[[ \t]*2[ \t]*\]([ \t]*(?:#.*)?)',
                limpo)
            if m:
                linhas[i] = m.group(1) + '[1]' + m.group(2) + linha[len(limpo):]
                alteracoes += 1
    if alteracoes != 1:
        raise RuntimeError('Nao encontrei uma unica linha pca.n_components: [2].')
    novo = ''.join(linhas)
    if yaml.safe_load(novo) != alvo:
        raise RuntimeError('Troca PCA 2D para 1D alteraria outro parametro.')

    backup = arquivo.with_name(
        arquivo.name + '.antes_comparacao_1D_' +
        datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.bak')
    with backup.open('xb') as f:
        f.write(original)
    arquivo.write_bytes((b'\xef\xbb\xbf' if bom else b'') + novo.encode('utf-8'))
    print('BACKUP YAML:', backup.resolve())
else:
    print('YAML ja estava em PCA [1] com parametros equivalentes ao 2D.')

final = yaml.safe_load(arquivo.read_text(encoding='utf-8-sig'))
assert final == alvo
if final['execucao'].get('diretorio_retomada') is not None:
    raise RuntimeError('Diretorio de retomada explicito configurado: revise antes de executar.')
if final['execucao']['proteger_resultados'] is not True:
    raise RuntimeError('Protecao de resultados deve estar ativada.')

print('PCA:', final['pca']['n_components'])
print('FRACAO:', final['dados']['fracao_base'])
print('SEMENTE:', final['experimento']['semente'])
print('K GMM:', final['gmm']['n_components'])
print('INICIALIZACOES:', final['gmm']['n_init'])
print('F10: RESERVADO')
'@ | & $python - $ref2d

$preparo | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -ne 0) { throw 'Preparacao do PCA 1D falhou. Nenhum ensaio foi iniciado.' }

# 3. Planejar e conferir sintaxe.
& $python -m py_compile .\src\orquestracao_oof.py .\src\visualizacao_geometrica.py .\src\visualizacao_integrada.py
if ($LASTEXITCODE -ne 0) { throw 'Erro de sintaxe no pipeline.' }
& $python -m src.orquestracao_oof --planejar
if ($LASTEXITCODE -ne 0) { throw 'Planejamento falhou; ensaio nao iniciado.' }

# 4. Executar de modo protegido em uma nova pasta, registrar log.
$anteriores = @(Get-ChildItem '.\outputs' -Directory -Filter 'ensaio_pca_gmm_*' |
    ForEach-Object { $_.FullName })
$log = Join-Path (Resolve-Path '.\outputs').Path (
    'execucao_PCA1D_comparavel_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
Write-Host 'Iniciando ensaio PCA 1D. OOF F1-F9, SEM F10...' -ForegroundColor Cyan
& $python -u -m src.orquestracao_oof --executar 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) {
    Write-Host "Log preservado: $log" -ForegroundColor Yellow
    throw 'Execucao interrompida. Nao exclua resultados; caso OOF tenha terminado, use --so-visualizacoes.'
}

$novos = @(Get-ChildItem '.\outputs' -Directory -Filter 'ensaio_pca_gmm_*' |
    Where-Object { $anteriores -notcontains $_.FullName })
if ($novos.Count -ne 1) {
    throw 'Nao foi possivel identificar uma unica nova pasta de ensaio 1D. Confira o log.'
}
$ref1d = $novos[0].FullName

# 5. Conferir que os dados e hiperparametros sao os mesmos, salvo PCA 1D/2D.
$comparacao = @'
from pathlib import Path
from copy import deepcopy
import json
import sys
import yaml

p1, p2 = (Path(v) for v in sys.argv[1:3])
c1 = yaml.safe_load((p1 / 'configuracao.yaml').read_text(encoding='utf-8-sig'))
c2 = yaml.safe_load((p2 / 'configuracao.yaml').read_text(encoding='utf-8-sig'))
assert c1['pca']['n_components'] == [1]
assert c2['pca']['n_components'] == [2]
c2['pca']['n_components'] = [1]
assert c1 == c2, 'As configuracoes diferem em outros parametros!'

m1 = json.loads((p1 / 'manifesto.json').read_text(encoding='utf-8-sig'))
m2 = json.loads((p2 / 'manifesto.json').read_text(encoding='utf-8-sig'))
for chave in ('sha256_parquet', 'sha256_folds', 'registros_desenvolvimento'):
    assert m1[chave] == m2[chave], f'Diferenca em {chave}: comparar as amostras.'
for man in (m1, m2):
    assert man.get('status') == 'concluido_06_03'
    assert man.get('visualizacoes') == 'concluidas'

print('COMPARABILIDADE: YAML igual exceto PCA; mesma base, folds e tamanho de amostra.')
print('DIMENSOES: PCA 1D versus PCA 2D')
for titulo, pasta in (('PCA 1D', p1), ('PCA 2D', p2)):
    selecao = json.loads((pasta/'vencedor'/'metricas'/'melhor_configuracao.json').read_text(encoding='utf-8-sig'))
    met = json.loads((pasta/'vencedor'/'metricas'/'metricas.json').read_text(encoding='utf-8-sig'))
    assert selecao.get('teste_final_usado') is False
    print('\n'+titulo, '| vencedor:', selecao['id_candidato'])
    for chave in ('ap', 'mcc', 'logloss', 'score', 'limiar_mcc'):
        print(f'  {chave}: {float(met[chave]):.7f}')
    print('  galeria:', (pasta/'vencedor'/'visualizacao'/'galeria'/'index.html').resolve())
'@ | & $python - $ref1d $ref2d

$comparacao | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -ne 0) {
    throw 'Os ensaios existem, mas a auditoria comparativa encontrou uma divergencia.'
}

# 6. Abrir ambas as galerias.
$galeria1 = Join-Path $ref1d 'vencedor\visualizacao\galeria\index.html'
$galeria2 = Join-Path $ref2d 'vencedor\visualizacao\galeria\index.html'
if (-not ((Test-Path -LiteralPath $galeria1) -and (Test-Path -LiteralPath $galeria2))) {
    throw 'Galeria 1D ou 2D nao encontrada.'
}
Start-Process -FilePath $galeria1
Start-Process -FilePath $galeria2
Write-Host "`nAS DUAS GALERIAS FORAM ABERTAS!" -ForegroundColor Green
Write-Host "PCA 1D: $galeria1"
Write-Host "PCA 2D: $galeria2"
Write-Host "LOG: $log"
Write-Host 'F10 nao utilizado.'
