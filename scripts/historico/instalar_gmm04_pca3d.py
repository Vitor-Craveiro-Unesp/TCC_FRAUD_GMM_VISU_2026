# -*- coding: utf-8 -*-
"""Integra GMM04 HTML 3D ao gerador e a galeria, com backup e rollback.

Executar na raiz do TCC_EXPERIMENTAL_V3. Nao le dados, nao treina GMM,
nao altera YAML, nao executa F10, nao modifica outputs anteriores.
"""
from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
import re
import shutil

raiz = Path.cwd()
modulo = raiz / 'src' / 'visualizacao_gmm04.py'
galeria = raiz / 'src' / 'visualizacao_integrada.py'
extra_origem = Path(__file__).resolve().with_name('visualizacao_gmm04_3d.py')
extra_destino = raiz / 'src' / 'visualizacao_gmm04_3d.py'
MARCA_MODULO = '# PCA3D_GMM04_INTERATIVO_OFICIAL_V1'
MARCA_GALERIA = '# PCA3D_GMM04_GALERIA_V1'

for p in (modulo, galeria, extra_origem):
    if not p.is_file():
        raise SystemExit(f'Arquivo obrigatorio ausente: {p}. Nada alterado.')
if extra_destino.exists():
    raise SystemExit('O modulo GMM04 3D ja existe; instalacao repetida bloqueada.')

src = modulo.read_text(encoding='utf-8-sig')
g = galeria.read_text(encoding='utf-8-sig')
if MARCA_MODULO in src or MARCA_GALERIA in g:
    raise SystemExit('GMM04 3D ja integrado; nada alterado.')
if ("# PCA3D_GMM01_GALERIA_V1" not in g or
        "FORMATS = ('.png', '.pdf', '.svg', '.html')" not in g):
    raise SystemExit('A galeria ainda nao possui o suporte HTML do GMM01. Nada alterado.')

# Modificacao minima: manter intactas todas as rotinas de dominancia
# e adicionar o HTML SOMENTE depois do PNG oficial quando dimensao==3.
arvore = ast.parse(src)
fns = [n for n in arvore.body if isinstance(n, ast.FunctionDef) and n.name == 'gerar']
if len(fns) != 1 or not isinstance(fns[0].body[-1], ast.Expr):
    raise SystemExit('GMM04.gerar() diferente do esperado. Nada alterado.')
if not (isinstance(fns[0].body[-1].value, ast.Call)
        and isinstance(fns[0].body[-1].value.func, ast.Attribute)
        and fns[0].body[-1].value.func.attr == '_guardar'):
    raise SystemExit('GMM04.gerar() nao termina no salvamento de PNG. Nada alterado.')
insert = '''    # PCA3D_GMM04_INTERATIVO_OFICIAL_V1
    # GMM ja ajustado: apenas visualizacao; 1D e 2D ficam intactos.
    if d == 3:
        from src.visualizacao_gmm04_3d import gerar_interativo
        gerar_interativo(config, data, destino, salvar_csv=csv)
'''
linhas = src.splitlines(keepends=True)
linhas.insert(fns[0].end_lineno, '\n' + insert)
novo_modulo = ''.join(linhas)
compile(novo_modulo, str(modulo), 'exec')

# Exigir novo HTML no caso PCA 3D quando GMM04 estiver habilitado.
retornos = list(re.finditer(
    r'(?m)^        return imagens_oof[ \t]*,[ \t]*imagens_geo[ \t]*$', g
))
if len(retornos) != 1:
    raise SystemExit('Completude da galeria nao localizada unicamente. Nada alterado.')
completude = '''        # PCA3D_GMM04_GALERIA_V1: exigir HTML apenas no PCA 3D.
        if (escolha['configuracao']['metodo'].upper() == 'PCA'
                and int(escolha['configuracao']['reducao']['n_components']) == 3
                and 'regioes_dominancia' in esperado_geo):
            imagens_geo = imagens_geo and (
                geo_dir / 'GMM04_dominancia_3d_INTERATIVA.html'
            ).is_file()

'''
g = g[:retornos[0].start()] + completude + g[retornos[0].start():]

# Lista branca: nao inclui HTML generico ou HTML de outra dimensao.
permitidos = list(re.finditer(r'(?m)^            if permitido:[ \t]*$', g))
if len(permitidos) != 1:
    raise SystemExit('Lista branca da galeria nao encontrada unicamente. Nada alterado.')
inclusao = '''                # PCA3D_GMM04_GALERIA_V1: HTML de dominancia do vencedor PCA3D.
                if ('regioes_dominancia' in ativos_geo
                        and escolha['configuracao']['metodo'].upper() == 'PCA'
                        and int(escolha['configuracao']['reducao']['n_components']) == 3
                        and arq.name == 'GMM04_dominancia_3d_INTERATIVA.html'):
                    permitido = True

'''
g = g[:permitidos[0].start()] + inclusao + g[permitidos[0].start():]
if g.count(MARCA_GALERIA) != 2 or novo_modulo.count(MARCA_MODULO) != 1:
    raise SystemExit('Verificacao de marcadores falhou. Nada alterado.')
compile(g, str(galeria), 'exec')
compile(extra_origem.read_text(encoding='utf-8-sig'), str(extra_origem), 'exec')

instante = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
bak_modulo = modulo.with_name(modulo.name + '.ANTES_PCA3D_GMM04_' + instante + '.bak')
bak_galeria = galeria.with_name(galeria.name + '.ANTES_PCA3D_GMM04_' + instante + '.bak')
if bak_modulo.exists() or bak_galeria.exists():
    raise SystemExit('Conflito com arquivo de backup. Nada alterado.')
bak_modulo.write_bytes(modulo.read_bytes())
bak_galeria.write_bytes(galeria.read_bytes())
try:
    shutil.copy2(extra_origem, extra_destino)
    modulo.write_text(novo_modulo, encoding='utf-8')
    galeria.write_text(g, encoding='utf-8')
    compile(modulo.read_text(encoding='utf-8'), str(modulo), 'exec')
    compile(galeria.read_text(encoding='utf-8'), str(galeria), 'exec')
except Exception:
    modulo.write_bytes(bak_modulo.read_bytes())
    galeria.write_bytes(bak_galeria.read_bytes())
    extra_destino.unlink(missing_ok=True)
    raise

print('GMM04 3D OFICIAL + HTML: INSTALADOS COM SUCESSO')
print('Backup GMM04:', bak_modulo)
print('Backup da galeria:', bak_galeria)
print('Novo modulo:', extra_destino)
print('YAML, F10 e resultados existentes preservados.')
