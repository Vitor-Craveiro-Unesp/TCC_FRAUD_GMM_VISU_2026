# Novo ensaio PCA 1D / 2D / 3D com GMM K4, n_init=5 e PRE01 V2.
# Execute na raiz de TCC_EXPERIMENTAL_V3. Nao usa F10.
[CmdletBinding()]
param(
    [ValidateRange(20,1000000)]
    [int]$PorFold = 1000,
    [ValidateSet('Automatica', 'Principal', 'Arquivo')]
    [string]$OrigemConfiguracao = 'Automatica',
    [string]$ArquivoConfiguracao = '',
    [ValidateSet('Individuais', 'Painel', 'Ambos', 'Nenhum')]
    [string]$Abrir = 'Individuais'
)

$ErrorActionPreference = 'Stop'
$raiz = (Get-Location).Path
$py = Join-Path $raiz '.venv\Scripts\python.exe'
$src = Join-Path $raiz 'src\orquestracao_oof.py'
$cfg = Join-Path $raiz 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { throw 'Python .venv nao encontrado na raiz.' }
if (-not (Test-Path -LiteralPath $src -PathType Leaf)) { throw 'Execute na raiz TCC_EXPERIMENTAL_V3.' }
if (-not (Test-Path -LiteralPath $cfg -PathType Leaf)) { throw 'YAML principal nao encontrado.' }
if (-not (Test-Path -LiteralPath (Join-Path $raiz 'src\pre01_v2_diagnostico.py') -PathType Leaf)) {
    throw 'PRE01 V2 nao instalado. O modulo src\pre01_v2_diagnostico.py e necessario.'
}

# Resolucao explicita da origem. Nenhuma configuracao sera modificada nesta etapa.
if ($OrigemConfiguracao -eq 'Arquivo') {
    if ([string]::IsNullOrWhiteSpace($ArquivoConfiguracao)) {
        throw 'Informe -ArquivoConfiguracao quando usar -OrigemConfiguracao Arquivo.'
    }
    if (-not (Test-Path -LiteralPath $ArquivoConfiguracao -PathType Leaf)) {
        throw "Arquivo YAML de referencia nao encontrado: $ArquivoConfiguracao"
    }
    $arquivoParaDriver = (Resolve-Path -LiteralPath $ArquivoConfiguracao).Path
} else {
    if (-not [string]::IsNullOrWhiteSpace($ArquivoConfiguracao)) {
        throw '-ArquivoConfiguracao so pode ser informado com -OrigemConfiguracao Arquivo.'
    }
    $arquivoParaDriver = '-'
}

# Bloquear duas execucoes simultaneas que alterariam o mesmo YAML temporariamente.
$executando = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'src[.]orquestracao_oof|_novo_teste_pca_driver' })
if ($executando.Count -gt 0) {
    throw 'Ja existe execucao do pipeline PCA em andamento. Nao execute dois ensaios em paralelo.'
}

$destino = Join-Path $raiz 'outputs'
if (-not (Test-Path $destino -PathType Container)) { New-Item -Path $destino -ItemType Directory | Out-Null }
$runner = Join-Path $destino '_novo_teste_pca_1d_2d_3d_k4_ninit5.py'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$conteudo = @'
# -*- coding: utf-8 -*-
"""Novo trio OOF amostral PCA 1/2/3D, GMM K=4 n_init=5 e PRE01 V2.

Instalado temporariamente por um PowerShell na raiz do TCC_EXPERIMENTAL_V3.
O computador do usuario executa os ajustes; o script nao utiliza F10.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import yaml

ROOT = Path.cwd().resolve()
OUT = ROOT / 'outputs'
CFG_FILE = ROOT / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
STAMP = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
CONTROL = OUT / ('NOVO_TESTE_PCA_1D_2D_3D_K4_NINIT5_' + STAMP)
BY_FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
CONFIG_MODE = sys.argv[2] if len(sys.argv) > 2 else 'Automatica'
CONFIG_PATH = sys.argv[3] if len(sys.argv) > 3 else '-'
OPEN_MODE = sys.argv[4] if len(sys.argv) > 4 else 'Individuais'
DIMS = (1, 2, 3)


def require(cond, message):
    if not cond:
        raise RuntimeError(message)


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def single(x):
    if isinstance(x, list):
        if len(x) != 1:
            return None
        x = x[0]
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def intended(cfg):
    try:
        return (cfg['reducao']['metodo'].upper() == 'PCA'
                and single(cfg['gmm']['n_components']) == 4
                and int(cfg['gmm']['n_init']) == 5
                and cfg['gmm']['covariance_type'].lower() == 'full'
                and cfg['amostragem']['modo'].upper() == 'RUS'
                and int(cfg['amostragem']['razao_negativos_positivos']) == 16
                and int(cfg['experimento']['semente']) == 42
                and int(cfg['divisao']['qual_fold_teste']) == 10
                and single(cfg['pca']['n_components']) in DIMS)
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


def find_template(original_cfg):
    # Retornar configuracao, caminho de referencia e bytes exatos para auditoria.
    if CONFIG_MODE == 'Principal':
        print('[BASE] Origem escolhida: YAML principal.', flush=True)
        return deepcopy(original_cfg), CFG_FILE
    if CONFIG_MODE == 'Arquivo':
        source = Path(CONFIG_PATH)
        require(source.is_file(), f'YAML de referencia ausente: {source}')
        cfg = read_yaml(source)
        require(isinstance(cfg, dict), f'YAML de referencia invalido: {source}')
        print(f'[BASE] Origem escolhida: arquivo {source}', flush=True)
        return cfg, source
    require(CONFIG_MODE == 'Automatica', f'OrigemConfiguracao desconhecida: {CONFIG_MODE}')
    # Comportamento anterior preservado: ensaio PRE01 V2 mais recente, se existir.
    choices = []
    for p in OUT.glob('ensaio_pca_gmm_*'):
        if not p.is_dir():
            continue
        try:
            cfg = read_yaml(p/'configuracao.yaml')
            meta = read_json(p/'manifesto.json')
            page = p/'vencedor'/'visualizacao'/'galeria'/'index.html'
            if (intended(cfg) and meta.get('status') == 'concluido_06_03'
                    and meta.get('visualizacoes') == 'concluidas'
                    and meta.get('tipo') == 'ensaio'
                    and page.is_file()
                    and 'PRE01_V2_DIAGNOSTICO_INTERATIVO' in page.read_text(encoding='utf-8')):
                choices.append((p.stat().st_mtime, p, cfg))
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError):
            continue
    if choices:
        _, p, cfg = max(choices, key=lambda t: t[0])
        source = p/'configuracao.yaml'
        print(f'[BASE] Ultimo ensaio com PRE01 V2: {p.name}', flush=True)
    else:
        cfg = deepcopy(original_cfg)
        source = CFG_FILE
        print('[BASE] Nenhum ensaio PRE01 V2 anterior localizado; usando YAML principal.', flush=True)
    require(isinstance(cfg, dict), 'Configuracao-base invalida.')
    return cfg, source


def make_cfg(template, dim):
    cfg = deepcopy(template)
    cfg['dados']['fracao_base'] = 1.0  # --ensaio limita a 1000/fold; nao altera F10
    cfg['reducao']['metodo'] = 'PCA'
    cfg['pca']['n_components'] = [dim]
    cfg['gmm']['n_components'] = [4]
    cfg['gmm']['n_init'] = 5
    cfg['gmm']['covariance_type'] = 'full'
    cfg['amostragem']['modo'] = 'RUS'
    cfg['amostragem']['razao_negativos_positivos'] = 16
    cfg['experimento']['semente'] = 42
    cfg['divisao']['qual_fold_teste'] = 10
    # Nao retomar ensaios inacabados antigos: o usuario pediu um NOVO trio.
    cfg['execucao']['retomar'] = False
    cfg['execucao']['diretorio_retomada'] = None
    require(intended(cfg) and single(cfg['pca']['n_components']) == dim,
            'Configuracao incorreta para PCA ' + str(dim))
    return cfg


def comparable(cfg):
    q = deepcopy(cfg)
    q['pca']['n_components'] = '__DIMENSAO__'
    return json.dumps(q, ensure_ascii=False, sort_keys=True, default=str)


def atomic_bytes(path, blob):
    tmp = path.with_name(path.name+'.novo_teste_tmp')
    try:
        tmp.write_bytes(blob)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def run_python(label, arguments):
    log = CONTROL/(label+'_stderr.log')
    env = os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1', MPLBACKEND='Agg')
    print('\n'+'='*60, flush=True)
    print('[EXECUTAR]', label, ' | stderr registrado em:', log.name, flush=True)
    print('='*60, flush=True)
    with log.open('w', encoding='utf-8') as out:
        result = subprocess.run([sys.executable, '-u', *arguments], cwd=ROOT, env=env,
                                stderr=out, check=False)
    raw = log.read_text(encoding='utf-8',errors='replace')
    if result.returncode != 0:
        print(f'[ERRO] {label}: Python retornou {result.returncode}.', flush=True)
        if raw.strip():
            print('[TRACEBACK / STDERR]\n'+raw[-16000:], flush=True)
        raise RuntimeError(f'{label} falhou. Nada foi apagado; consulte {log}.')
    warnings = sum('findfont' in line for line in raw.splitlines())
    if warnings:
        print(f'[AVISO NAO FATAL] {warnings} mensagens findfont registradas.', flush=True)
    others = [line for line in raw.splitlines() if line.strip() and 'findfont' not in line]
    if others:
        print('[STDERR] '+'\n'.join(others[-10:]),flush=True)


def check_folder(path, dim, config):
    meta = read_json(path/'manifesto.json')
    frozen = read_yaml(path/'configuracao.yaml')
    gallery = path/'vencedor'/'visualizacao'/'galeria'/'index.html'
    require(meta.get('tipo')=='ensaio' and meta.get('status')=='concluido_06_03'
            and meta.get('visualizacoes')=='concluidas', f'Ensaio {dim}D incompleto.')
    require(int(meta.get('registros_desenvolvimento',0))==9*BY_FOLD,
            f'Ensaio {dim}D nao possui {9*BY_FOLD} registros.')
    require(intended(frozen) and single(frozen['pca']['n_components']) == dim,
            f'Config congelada inesperada PCA {dim}D.')
    require(comparable(frozen)==comparable(config),
            f'Config congelada diferente no PCA {dim}D.')
    require(gallery.is_file(), f'Galeria PCA {dim}D ausente: {gallery}')
    pred = path/'vencedor'/'oof'/'predicoes.npz'
    require(pred.is_file(), f'Predicoes OOF PCA {dim}D ausentes.')
    with np.load(pred, allow_pickle=False) as z:
        rows=np.asarray(z['linha_original'],dtype=np.int64)
        folds=np.asarray(z['fold'],dtype=np.int64)
        require(int(z['fold_esperado'].item())==0,'Nao e OOF consolidado.')
    require(len(rows)==9*BY_FOLD and np.all(np.diff(rows)>0),
            f'IDs OOF invalidos no PCA {dim}D.')
    kinds, counts = np.unique(folds,return_counts=True)
    require(np.array_equal(kinds,np.arange(1,10)) and np.all(counts==BY_FOLD),
            f'Folds invalidos no PCA {dim}D.')
    return dict(path=path, gallery=gallery, cfg=frozen, meta=meta, rows=rows, folds=folds)


def build_panel(trio, pre01_ok):
    cards = []
    for d in DIMS:
        p=trio[d]['gallery'].relative_to(OUT).as_posix()
        pre=trio[d]['gallery'].parent/'oof'/'PRE01_diagnostico_padronizacao_completo.html'
        prelink=''
        if pre01_ok and pre.is_file():
            r=pre.relative_to(OUT).as_posix()
            prelink=f'<a class="pre" href="{escape(r,quote=True)}" target="_blank" rel="noopener">Abrir PRE01 V2 completo</a>'
        cards.append(f'''<section class="pane"><div class="top"><h2>PCA {d}D</h2>
        <div class="actions"><a href="{escape(p,quote=True)}" target="_blank" rel="noopener">Abrir galeria em nova guia</a>{prelink}</div></div>
        <iframe src="{escape(p,quote=True)}" loading="lazy" title="Galeria PCA {d}D"></iframe></section>''')
    page='''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Novo teste PCA 1D/2D/3D — GMM K4</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fb;color:#233551;font:15px system-ui,Segoe UI,Arial,sans-serif}
header{background:#20314d;color:#fff;padding:24px 4%}h1{margin:0 0 8px;font-size:26px}header p{margin:0;color:#dbe7f8}
main{max-width:1640px;margin:24px auto;padding:0 18px}.pane{border:1px solid #dae1eb;background:white;border-radius:14px;padding:16px;margin-bottom:26px;box-shadow:0 2px 9px #18324b12}
.top{display:flex;gap:15px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-bottom:12px}h2{margin:0;font-size:23px}.actions{display:flex;flex-wrap:wrap;gap:8px}
a{background:#285c98;color:white;padding:9px 13px;border-radius:7px;text-decoration:none;font-weight:600;font-size:13px}a.pre{background:#26786d}a:hover{filter:brightness(.89)}
iframe{width:100%;height:770px;border:1px solid #d6e0eb;border-radius:8px;background:white}
footer{padding:15px 4% 30px;color:#52647b;font-size:13px}</style></head><body>
<header><h1>Novo teste PCA 1D · 2D · 3D</h1><p>GMM K=4 | 5 inicializações EM | FULL | RUS 16:1 | semente 42 | '''+f'{BY_FOLD}'+''' registros/fold | F1–F9 | F10 reservado</p></header>
<main>'''+''.join(cards)+'''</main><footer>Resultados OOF amostrais (exploratórios). O teste final F10 não foi usado.</footer></body></html>'''
    filename='PAINEL_NOVO_TESTE_PCA_K4_NINIT5_'+STAMP+'.html'
    path=OUT/filename
    atomic_bytes(path,page.encode('utf-8'))
    return path


def main():
    require(BY_FOLD>=20,'PorFold invalido; use pelo menos 20 observacoes por fold.')
    for required in [CFG_FILE, ROOT/'src'/'orquestracao_oof.py',
                     ROOT/'src'/'pre01_v2_diagnostico.py', ROOT/'src'/'visualizacao_integrada.py',
                     ROOT/'src'/'visualizacao_gmm01_3d.py']:
        require(required.is_file(), f'Arquivo obrigatorio nao encontrado: {required}')
    # GMM01 dinamico e necessario para representar integralmente as quatro gaussianas.
    src=(ROOT/'src'/'visualizacao_gmm01_3d.py').read_text(encoding='utf-8-sig')
    require('K4_DINAMICO_VALIDADO_20261028' in src or 'len(mu)' in src,
            'Modulo GMM01 3D nao aparenta suportar K4. Preserve as imagens e aplique a adaptacao K4 anterior.')
    OUT.mkdir(parents=True,exist_ok=True)
    CONTROL.mkdir(parents=False,exist_ok=False)
    original=CFG_FILE.read_bytes()
    (CONTROL/'YAML_PRINCIPAL_ORIGINAL.yaml').write_bytes(original)
    templ, template_source = find_template(yaml.safe_load(original.decode('utf-8-sig')))
    source_blob = original if template_source.resolve() == CFG_FILE.resolve() else template_source.read_bytes()
    templ = yaml.safe_load(source_blob.decode('utf-8-sig'))
    require(isinstance(templ, dict), f'YAML-base invalido: {template_source}')
    (CONTROL/'YAML_BASE_SELECIONADO.yaml').write_bytes(source_blob)
    print(f'[BASE] Modo={CONFIG_MODE} | arquivo={template_source}', flush=True)
    print(f'[BASE] SHA256={sha256(source_blob).hexdigest()}', flush=True)
    configs={d:make_cfg(templ,d) for d in DIMS}
    assert len({comparable(c) for c in configs.values()})==1
    trio={}
    try:
        for d in DIMS:
            # NOVA rodada sempre, nao retoma OOF antigo.
            before={p.resolve() for p in OUT.glob('ensaio_pca_gmm_*') if p.is_dir()}
            yml=yaml.safe_dump(configs[d],allow_unicode=True,sort_keys=False)
            atomic_bytes(CFG_FILE,yml.encode('utf-8'))
            print(f'\n[INICIO] NOVO PCA {d}D: K4, n_init5, {BY_FOLD} por fold.',flush=True)
            run_python(f'PCA_{d}D', ['-m','src.orquestracao_oof','--ensaio','--por-fold',str(BY_FOLD)])
            now={p.resolve() for p in OUT.glob('ensaio_pca_gmm_*') if p.is_dir()}
            new=sorted(now-before)
            require(len(new)==1, f'Esperada uma nova pasta ensaio para PCA {d}D; obtidas {len(new)}. Confira outputs/.')
            trio[d]=check_folder(new[0],d,configs[d])
            print(f'[OK] PCA {d}D: OOF completo e galeria em {new[0].name}',flush=True)
    finally:
        # O YAML principal fica identico byte a byte ao anterior, inclusive em erro.
        atomic_bytes(CFG_FILE,original)
        require(CFG_FILE.read_bytes()==original,'Falha ao restaurar YAML original.')
        print('[YAML] Configuracao principal restaurada byte a byte.',flush=True)
    require(len(trio)==3, 'As tres dimensoes precisam estar completas antes de integrar o PRE01 V2.')
    ref=trio[1]
    for d in (2,3):
        x=trio[d]
        require(np.array_equal(x['rows'],ref['rows']) and np.array_equal(x['folds'],ref['folds']),
                f'PCA {d}D usa amostra ou divisao diferente do PCA 1D.')
        for h in ('sha256_parquet','sha256_folds'):
            require(x['meta'].get(h)==ref['meta'].get(h),
                    f'Hash {h} diferente entre PCA {d}D e PCA 1D.')
    print('[AUDITORIA] Mesmo conjunto de observacoes, folds e hiperparametros nas tres dimensoes.',flush=True)
    # Se PRE01 V2 falhar, manter galerias/OOF e sinalizar erro real ao chamador.
    pre01_ok=False
    pre01_error=None
    try:
        run_python('PRE01_V2',['-m','src.pre01_v2_diagnostico',
                               '--raiz',str(ROOT),'--k','4','--n-init','5','--tipo','ensaio'])
        for d in DIMS:
            page=trio[d]['gallery']
            text=page.read_text(encoding='utf-8')
            pre=page.parent/'oof'/'PRE01_diagnostico_padronizacao_completo.html'
            require('PRE01_V2_DIAGNOSTICO_INTERATIVO' in text and pre.is_file(),
                    f'PRE01 V2 ausente da nova galeria PCA {d}D.')
        pre01_ok=True
        print('[PRE01 V2] Diagnostico completo confirmado nas 3 galerias.',flush=True)
    except Exception as ex:
        pre01_error=str(ex)
        print('[FALHA PARCIAL] PRE01 V2 nao foi confirmado: '+pre01_error,flush=True)
        print('[FALHA PARCIAL] OOF e galerias preservados; consulte o log PRE01_V2_stderr.log.',flush=True)
    path=build_panel(trio,pre01_ok)
    summary={'tipo':'ensaio','k':4,'n_init':5,'por_fold':BY_FOLD,'f10_usado':False,
             'pre01_v2_confirmado':pre01_ok,'pre01_v2_erro':pre01_error,
             'status_geral':'concluido' if pre01_ok else 'parcial_pre01_v2',
             'origem_configuracao_modo':CONFIG_MODE,
             'origem_configuracao_arquivo':str(template_source),
             'sha256_yaml_base_selecionado':sha256(source_blob).hexdigest(),
             'novo_trio':{f'{d}D':str(trio[d]['path']) for d in DIMS},
             'painel':str(path),'sha256_yaml_original':sha256(original).hexdigest()}
    (CONTROL/'RESUMO_TRES_GALERIAS.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    status_label = '[CONCLUIDO]' if pre01_ok else '[PARCIAL]'
    print('\n'+'='*65+'\n'+status_label+' NOVOS TESTES PCA 1D, 2D, 3D\nPAINEL='+str(path),flush=True)
    for d in DIMS:
        print(f'GALERIA_{d}D='+str(trio[d]['gallery']),flush=True)
    print('PRE01_V2='+('CONFIRMADO' if pre01_ok else 'PENDENTE'),flush=True)
    print('F10=RESERVADO',flush=True)
    if OPEN_MODE not in ('Individuais', 'Painel', 'Ambos', 'Nenhum'):
        raise RuntimeError(f'Modo de abertura desconhecido: {OPEN_MODE}')
    if os.environ.get('TCC_NAO_ABRIR_NAVEGADOR') == '1':
        print('[NAVEGADOR] Abertura desabilitada por TCC_NAO_ABRIR_NAVEGADOR=1.',flush=True)
    elif OPEN_MODE == 'Nenhum':
        print('[NAVEGADOR] Abertura desabilitada por -Abrir Nenhum.',flush=True)
    else:
        targets=[]
        if OPEN_MODE in ('Individuais', 'Ambos'):
            targets += [(f'PCA {d}D', trio[d]['gallery']) for d in DIMS]
        if OPEN_MODE in ('Painel', 'Ambos'):
            targets.append(('Painel consolidado', path))
        for label, target in targets:
            try:
                os.startfile(str(target))
                print(f'[NAVEGADOR] Solicitada abertura: {label} | {target}',flush=True)
            except (OSError, AttributeError) as ex:
                print(f'[NAVEGADOR] Nao foi possivel abrir {label}: {ex} | {target}',flush=True)
    if not pre01_ok:
        print('[FALHA PARCIAL] Tres ensaios concluidos, mas PRE01 V2 pendente. Codigo de saida 2.',flush=True)
        return 2
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print('\n[FALHA] '+str(e)+'\n',flush=True)
        traceback.print_exc(file=sys.stdout)
        print('[INSTRUCAO] OOF/galerias existentes preservados; consulte o diretorio '+str(CONTROL),flush=True)
        sys.exit(1)

'@
[System.IO.File]::WriteAllText($runner, $conteudo, $utf8)

$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
$env:MPLBACKEND='Agg'
$ErrorActionPreference='Continue'  # O driver escreve erros Python no stdout; avisos vao para log.
Write-Host "[NOVO TESTE] PCA 1D/2D/3D | GMM K4 | n_init=5 | PorFold=$PorFold | Origem=$OrigemConfiguracao | Abrir=$Abrir" -ForegroundColor Cyan
& $py -u $runner $PorFold $OrigemConfiguracao $arquivoParaDriver $Abrir
$saida = $LASTEXITCODE
if ($saida -eq 2) {
    Write-Host '[FALHA PARCIAL] Tres ensaios concluiram, mas PRE01 V2 falhou. Consulte o resumo e os logs em outputs/NOVO_TESTE_PCA_*.' -ForegroundColor Yellow
    exit 2
}
if ($saida -ne 0) {
    Write-Host '[FALHA] Consulte o traceback acima e os logs em outputs/NOVO_TESTE_PCA_*.' -ForegroundColor Red
    exit $saida
}
Write-Host '[OK] Tres ensaios e PRE01 V2 confirmados. Verifique os caminhos das galerias acima.' -ForegroundColor Green
