# -*- coding: utf-8 -*-
"""Integra GMM08 PCA3D no pipeline e galeria, com backup e rollback.
Execute na raiz do projeto. Nao roda OOF, nem F10.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import ast
import re
import shutil

raiz=Path.cwd()
modulo=raiz/'src'/'visualizacao_geometrica.py'
galeria=raiz/'src'/'visualizacao_integrada.py'
extra=Path(__file__).resolve().with_name('visualizacao_gmm08_3d.py')
destino=raiz/'src'/'visualizacao_gmm08_3d.py'
marca_mod='# PCA3D_GMM08_OFICIAL'
marca_gal='# PCA3D_GMM08_GALERIA'

for p in (modulo,galeria,extra):
    if not p.is_file():raise SystemExit(f'Arquivo ausente: {p}. Nada alterado.')
if destino.exists():raise SystemExit('Modulo GMM08 3D existente: nao sobrescrever.')
sm=modulo.read_text(encoding='utf-8-sig')
sg=galeria.read_text(encoding='utf-8-sig')
if marca_mod in sm or marca_gal in sg:
    raise SystemExit('GMM08 PCA3D ja integrado. Nada alterado.')
if ('# PCA3D_GMM07_V2_GALERIA' not in sg or
        "FORMATS = ('.png', '.pdf', '.svg', '.html')" not in sg):
    raise SystemExit('Galeria precisa estar na versao com GMM07 V2. Nada alterado.')
# Localizar a funcao por AST; inserir apenas apos docstring, sem mudar 1D e 2D.
a=ast.parse(sm)
f=[n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='_fronteira']
if len(f)!=1 or not isinstance(f[0].body[0],ast.Expr):
    raise SystemExit('Funcao _fronteira nao localizada unicamente.')
fn=f[0]
if not (isinstance(fn.body[0].value,ast.Constant) and isinstance(fn.body[0].value.value,str)):
    raise SystemExit('Docstring de _fronteira nao localizada.')
linhas=sm.splitlines(keepends=True)
pos=fn.body[0].end_lineno
inserir='''    # PCA3D_GMM08_OFICIAL: apenas PCA 3D usa o novo PNG + HTML.
    if (data['coord'].shape[1] == 3
            and list(data['coord'].columns) == ['PC1','PC2','PC3']):
        from src.visualizacao_gmm08_3d import gerar_3d
        return gerar_3d(config,data,escolha,dest,ext,dpi,csv,avaliacao=avaliacao)

'''
sm=''.join(linhas[:pos])+inserir+''.join(linhas[pos:])
if sm.count(marca_mod)!=1:raise SystemExit('Marcador de gerador invalido.')
# Completeness check
retornos=list(re.finditer(r'(?m)^        return imagens_oof[ \t]*,[ \t]*imagens_geo[ \t]*$',sg))
if len(retornos)!=1:raise SystemExit('Retorno de completude nao localizado unicamente.')
completude='''        # PCA3D_GMM08_GALERIA: HTML obrigatorio somente para GMM08 PCA3D habilitado.
        if (escolha['configuracao']['metodo'].upper() == 'PCA'
                and int(escolha['configuracao']['reducao']['n_components']) == 3
                and 'mapa_risco_fronteira' in esperado_geo):
            imagens_geo = imagens_geo and (
                geo_dir / 'GMM08_mapa_risco_3d_comparativo_INTERATIVA.html'
            ).is_file()

'''
sg=sg[:retornos[0].start()]+completude+sg[retornos[0].start():]
# HTML whitelist
permitidos=list(re.finditer(r'(?m)^            if permitido:[ \t]*$',sg))
if len(permitidos)!=1:raise SystemExit('Lista branca da galeria nao encontrada unicamente.')
lista='''                # PCA3D_GMM08_GALERIA: somente HTML esperado do vencedor PCA3D.
                if ('mapa_risco_fronteira' in ativos_geo
                        and escolha['configuracao']['metodo'].upper() == 'PCA'
                        and int(escolha['configuracao']['reducao']['n_components']) == 3
                        and arq.name == 'GMM08_mapa_risco_3d_comparativo_INTERATIVA.html'):
                    permitido = True

'''
sg=sg[:permitidos[0].start()]+lista+sg[permitidos[0].start():]
if sg.count(marca_gal)!=2:raise SystemExit('Marcadores inconsistentes.')
for p,s in ((modulo,sm),(galeria,sg),(extra,extra.read_text(encoding='utf-8-sig'))):
    compile(s,str(p),'exec')
stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
bak1=modulo.with_name(modulo.name+'.ANTES_GMM08_3D_'+stamp+'.bak')
bak2=galeria.with_name(galeria.name+'.ANTES_GMM08_3D_'+stamp+'.bak')
bak1.write_bytes(modulo.read_bytes());bak2.write_bytes(galeria.read_bytes())
try:
    shutil.copy2(extra,destino)
    modulo.write_text(sm,encoding='utf-8')
    galeria.write_text(sg,encoding='utf-8')
    compile(modulo.read_text(encoding='utf-8'),str(modulo),'exec')
    compile(galeria.read_text(encoding='utf-8'),str(galeria),'exec')
except Exception:
    modulo.write_bytes(bak1.read_bytes())
    galeria.write_bytes(bak2.read_bytes())
    destino.unlink(missing_ok=True)
    raise
print('GMM08 PCA3D: INSTALADO COM SUCESSO')
print('Backup geometria:',bak1)
print('Backup galeria:',bak2)
print('Novo modulo:',destino)
print('YAML, F10 e ensaios existentes preservados.')
