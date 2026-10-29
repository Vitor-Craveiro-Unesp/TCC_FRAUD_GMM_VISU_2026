# -*- coding: utf-8 -*-
"""Instala o GMM01 3D e o reconhecimento dos complementos na galeria.

Executar na raiz de TCC_EXPERIMENTAL_V3.
Backups antes das alteracoes; nunca executa treinamento nem modifica o YAML.
"""
from __future__ import annotations
import ast
from pathlib import Path
from datetime import datetime
import shutil

raiz=Path.cwd()
src=raiz/'src'
original=src/'visualizacao_gmm01.py'
galeria_path=src/'visualizacao_integrada.py'
extra_dest=src/'visualizacao_gmm01_3d.py'
extra_src=Path(__file__).resolve().with_name('visualizacao_gmm01_3d.py')
for arq in (original,galeria_path,extra_src):
    if not arq.is_file():
        raise SystemExit(f'Arquivo obrigatorio ausente: {arq}')
if extra_dest.exists():
    raise SystemExit('Modulo adicional ja existe; preservado. Nenhuma alteracao.')


def substituir_uma_vez(texto,antigo,novo,rotulo):
    if texto.count(antigo)!=1:
        raise SystemExit(f'Nao foi possivel localizar UNICAMENTE: {rotulo}. Nada alterado.')
    return texto.replace(antigo,novo,1)

texto=original.read_text(encoding='utf-8-sig')
marca='# PCA3D_GMM01_EXTRAS_OFICIAIS_V1'
if marca in texto:
    raise SystemExit('GMM01 3D ja integrado. Nenhuma alteracao.')
if 'calcular_marginal' not in texto or 'densidade_marginal_' not in texto:
    raise SystemExit('Modulo GMM01 diferente do esperado. Nada alterado.')
arvore=ast.parse(texto)
funcoes=[f for f in arvore.body if isinstance(f,ast.FunctionDef) and f.name=='gerar']
if len(funcoes)!=1:
    raise SystemExit('Funcao gerar nao localizada de forma unica.')
linhas=texto.splitlines(keepends=True)
insercao='''\n    # PCA3D_GMM01_EXTRAS_OFICIAIS_V1
    # A extensao e executada somente apos as marginais PCA 3D.
    if d == 3:
        from src.visualizacao_gmm01_3d import gerar_complementares
        gerar_complementares(config, data, destino, ext, dpi, csv)
'''
linhas.insert(funcoes[0].end_lineno,insercao)
novo_gmm=''.join(linhas)
compile(novo_gmm,str(original),'exec')

# Incluir apenas as cinco visualizacoes adicionais do GMM01 no PCA 3D.
galeria=galeria_path.read_text(encoding='utf-8-sig')
marca_gallery='# PCA3D_GMM01_GALERIA_V1'
if marca_gallery in galeria:
    raise SystemExit('GMM01 PCA3D ja reconhecido pela galeria. Nada alterado.')
galeria=substituir_uma_vez(
    galeria,"FORMATS = ('.png', '.pdf', '.svg')",
    "FORMATS = ('.png', '.pdf', '.svg', '.html')",
    'extensoes FORMATS')

antigo="else '<div class=\"sem-previa\">Visualização em PDF — clique para abrir</div>')"
novo=("else ('<div class=\"sem-previa\">Visualização 3D interativa — clique para abrir</div>' "
      "if caminho.suffix.lower()=='.html' else "
      "'<div class=\"sem-previa\">Visualização em PDF — clique para abrir</div>'))")
galeria=substituir_uma_vez(galeria,antigo,novo,'legenda da miniatura HTML')

nomes=(
    'GMM01_conjunta_corte_PC1_PC2.png',
    'GMM01_conjunta_corte_PC1_PC3.png',
    'GMM01_conjunta_corte_PC2_PC3.png',
    'GMM01_geometria_gaussianas_FULL_3d_95.png',
    'GMM01_densidade_conjunta_PCA3D_INTERATIVA.html',
)
lista='\n'.join(f"                '{n}'," for n in nomes)

completo='''        # PCA3D_GMM01_GALERIA_V1: verificar os cinco anexos GMM01.
        if (escolha['configuracao']['metodo'].upper() == 'PCA'
                and int(escolha['configuracao']['reducao']['n_components']) == 3
                and 'densidades_marginais' in esperado_geo):
            anexos_gmm01_3d = (
__NOMES__
            )
            imagens_geo = imagens_geo and all(
                (geo_dir / nome).is_file() for nome in anexos_gmm01_3d
            )

'''.replace('__NOMES__',lista)
# Localizar o retorno independentemente dos espacos.
import re

retornos = list(re.finditer(
    r'(?m)^        return imagens_oof[ \t]*,[ \t]*imagens_geo[ \t]*$',
    galeria
))

if len(retornos) != 1:
    raise SystemExit(
        'Completude GMM01 nao encontrada unicamente. Nada alterado.'
    )

posicao = retornos[0].start()
galeria = galeria[:posicao] + completo + galeria[posicao:]

permitidos='\n'.join(f"                            '{n}'," for n in nomes)
selecao='''                # PCA3D_GMM01_GALERIA_V1: arquivos extras da dimensao 3.
                if ('densidades_marginais' in ativos_geo
                        and escolha['configuracao']['metodo'].upper() == 'PCA'
                        and int(escolha['configuracao']['reducao']['n_components']) == 3
                        and arq.name in {
__NOMES__
                        }):
                    permitido = True

'''.replace('__NOMES__',permitidos)
galeria=substituir_uma_vez(galeria,'            if permitido:\n',
                          selecao+'            if permitido:\n','lista branca da galeria')
compile(galeria,str(galeria_path),'exec')
if galeria.count(marca_gallery)!=2:
    raise RuntimeError('O integrador nao recebeu as duas marcacoes previstas.')
compile(extra_src.read_text(encoding='utf-8-sig'),str(extra_src),'exec')

stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
bak_gmm=original.with_name(original.name+'.ANTES_EXTRAS3D_'+stamp+'.bak')
bak_gallery=galeria_path.with_name(galeria_path.name+'.ANTES_EXTRAS3D_'+stamp+'.bak')
if bak_gmm.exists() or bak_gallery.exists():
    raise RuntimeError('Backup ja existente, instalacao interrompida.')
bak_gmm.write_bytes(original.read_bytes())
bak_gallery.write_bytes(galeria_path.read_bytes())
try:
    shutil.copy2(extra_src,extra_dest)
    original.write_text(novo_gmm,encoding='utf-8')
    galeria_path.write_text(galeria,encoding='utf-8')
    compile(original.read_text(encoding='utf-8'),str(original),'exec')
    compile(galeria_path.read_text(encoding='utf-8'),str(galeria_path),'exec')
except Exception:
    original.write_bytes(bak_gmm.read_bytes())
    galeria_path.write_bytes(bak_gallery.read_bytes())
    extra_dest.unlink(missing_ok=True)
    raise

print('GMM01 3D OFICIAL + GALERIA: INSTALADOS COM SUCESSO')
print('Backups:',bak_gmm.name,'|',bak_gallery.name)
print('Modulo:',extra_dest)
print('YAML, folds, F10 e ensaios anteriores preservados.')
