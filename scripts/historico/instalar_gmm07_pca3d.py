# -*- coding: utf-8 -*-
"""Instala GMM07 V2 PCA 3D com backup, verificacao e rollback.

Executar na raiz de TCC_EXPERIMENTAL_V3. Nao treina ou utiliza F10.
"""
from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
import re
import shutil

raiz=Path.cwd()
modulo=raiz/'src'/'visualizacao_gmm07.py'
galeria=raiz/'src'/'visualizacao_integrada.py'
extra=Path(__file__).resolve().with_name('visualizacao_gmm07_3d.py')
destino=raiz/'src'/'visualizacao_gmm07_3d.py'
marca_mod='# PCA3D_GMM07_V2_OFICIAL'
marca_gal='# PCA3D_GMM07_V2_GALERIA'

for p in (modulo, galeria, extra):
    if not p.is_file():
        raise SystemExit(f'Arquivo obrigatorio ausente: {p}. Nada alterado.')
if destino.exists():
    raise SystemExit('Modulo GMM07 3D ja existe. Instalacao repetida bloqueada.')
sm=modulo.read_text(encoding='utf-8-sig')
sg=galeria.read_text(encoding='utf-8-sig')
if marca_mod in sm or marca_gal in sg:
    raise SystemExit('GMM07 V2 ja integrado. Nada alterado.')
if ("# PCA3D_GMM04_GALERIA_V1" not in sg or
        "FORMATS = ('.png', '.pdf', '.svg', '.html')" not in sg):
    raise SystemExit('Galeria nao esta na versao posterior ao GMM04. Nada alterado.')

# Localizar por AST o unico ponto que gera o PNG do GMM07; modificar apenas
# a chamada em PCA3D, mantendo o caminho original integral para PCA1D/2D.
arvore=ast.parse(sm)
funcoes=[n for n in arvore.body if isinstance(n,ast.FunctionDef) and n.name=='gerar']
if len(funcoes)!=1:
    raise SystemExit('Funcao gerar do GMM07 nao localizada unicamente. Nada alterado.')
alvos=[]
for n in ast.walk(funcoes[0]):
    if (isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
            and isinstance(n.value.func,ast.Name)
            and n.value.func.id=='_grafico'):
        alvos.append(n)
if len(alvos)!=1:
    raise SystemExit('Chamada oficial _grafico nao localizada unicamente. Nada alterado.')
original=sm.splitlines(keepends=True)
no=alvos[0]
if not (no.end_lineno and no.lineno>0):
    raise SystemExit('Posicoes AST nao encontradas. Nada alterado.')
substituicao='''    # PCA3D_GMM07_V2_OFICIAL: somente o 3D usa o PNG V2 + HTML rotativo.
    if data['coord'].shape[1] == 3:
        from src.visualizacao_gmm07_3d import gerar_3d
        nome = gerar_3d(config, data, escolha, planos, medias, destino,
                        ext, dpi, avaliacao, mcc_fixo, mcc_otimo)
    else:
        nome = _grafico(config,data,escolha,planos,medias,destino=destino,ext=ext,dpi=dpi,
                        mcc_fixo=mcc_fixo,mcc_otimo=mcc_otimo,avaliacao=avaliacao)
'''
novo_m=''.join(original[:no.lineno-1])+substituicao+''.join(original[no.end_lineno:])
compile(novo_m,str(modulo),'exec')
if novo_m.count(marca_mod)!=1:
    raise SystemExit('Marcador do gerador incorreto. Nada alterado.')

retornos=list(re.finditer(r'(?m)^        return imagens_oof[ \t]*,[ \t]*imagens_geo[ \t]*$',sg))
if len(retornos)!=1:
    raise SystemExit('Retorno de completude da galeria nao encontrado unicamente. Nada alterado.')
completude='''        # PCA3D_GMM07_V2_GALERIA: HTML obrigatorio so para PCA3D com GMM07 ativo.
        if (escolha['configuracao']['metodo'].upper() == 'PCA'
                and int(escolha['configuracao']['reducao']['n_components']) == 3
                and 'sensibilidade_fronteira' in esperado_geo):
            imagens_geo = imagens_geo and (
                geo_dir / 'GMM07_sensibilidade_3d_comparativo_INTERATIVA.html'
            ).is_file()

'''
sg=sg[:retornos[0].start()]+completude+sg[retornos[0].start():]

permitidos=list(re.finditer(r'(?m)^            if permitido:[ \t]*$',sg))
if len(permitidos)!=1:
    raise SystemExit('Lista branca da galeria nao encontrada unicamente. Nada alterado.')
lista='''                # PCA3D_GMM07_V2_GALERIA: incorporar apenas o HTML do vencedor PCA3D.
                if ('sensibilidade_fronteira' in ativos_geo
                        and escolha['configuracao']['metodo'].upper() == 'PCA'
                        and int(escolha['configuracao']['reducao']['n_components']) == 3
                        and arq.name == 'GMM07_sensibilidade_3d_comparativo_INTERATIVA.html'):
                    permitido = True

'''
sg=sg[:permitidos[0].start()]+lista+sg[permitidos[0].start():]
if sg.count(marca_gal)!=2:
    raise SystemExit('Marcadores da galeria inconsistentes. Nada alterado.')
compile(sg,str(galeria),'exec')
compile(extra.read_text(encoding='utf-8-sig'),str(extra),'exec')

stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
bak1=modulo.with_name(modulo.name+'.ANTES_GMM07_V2_'+stamp+'.bak')
bak2=galeria.with_name(galeria.name+'.ANTES_GMM07_V2_'+stamp+'.bak')
if bak1.exists() or bak2.exists():
    raise SystemExit('Backup com nome duplicado. Nada alterado.')
bak1.write_bytes(modulo.read_bytes())
bak2.write_bytes(galeria.read_bytes())
try:
    shutil.copy2(extra,destino)
    modulo.write_text(novo_m,encoding='utf-8')
    galeria.write_text(sg,encoding='utf-8')
    compile(modulo.read_text(encoding='utf-8'),str(modulo),'exec')
    compile(galeria.read_text(encoding='utf-8'),str(galeria),'exec')
except Exception:
    modulo.write_bytes(bak1.read_bytes())
    galeria.write_bytes(bak2.read_bytes())
    destino.unlink(missing_ok=True)
    raise
print('GMM07 V2 PCA 3D: INSTALADO COM SUCESSO')
print('Backup GMM07:',bak1)
print('Backup galeria:',bak2)
print('Novo modulo:',destino)
print('YAML, F10 e ensaios anteriores preservados.')
