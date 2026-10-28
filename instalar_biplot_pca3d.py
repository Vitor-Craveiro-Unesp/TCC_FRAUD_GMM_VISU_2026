# -*- coding: utf-8 -*-
"""Integra coeficientes PCA3D e Biplot HTML ao pipeline, com backups.

Execute a partir da raiz TCC_EXPERIMENTAL_V3. Nao modifica YAML, F10,
outputs existentes nem geradores PCA1D/PCA2D.
"""
from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
import re
import shutil

raiz = Path.cwd()
modulo_biplot = raiz / 'src' / 'visualizacao_pca_biplot.py'
modulo_galeria = raiz / 'src' / 'visualizacao_integrada.py'
complementos_src = Path(__file__).resolve().with_name('visualizacao_pca3d_complementos.py')
complementos_dst = raiz / 'src' / 'visualizacao_pca3d_complementos.py'
marca_biplot = '# PCA3D_BIPLOT_COEFICIENTES_OFICIAIS_V1'
marca_galeria = '# PCA3D_BIPLOT_GALERIA_V1'

for arq in (modulo_biplot, modulo_galeria, complementos_src):
    if not arq.is_file():
        raise SystemExit(f'Arquivo obrigatorio ausente: {arq}')
if complementos_dst.exists():
    raise SystemExit(f'Modulo adicional ja existe: {complementos_dst}. Nada alterado.')

orig_biplot = modulo_biplot.read_text(encoding='utf-8-sig')
orig_galeria = modulo_galeria.read_text(encoding='utf-8-sig')
if marca_biplot in orig_biplot or marca_galeria in orig_galeria:
    raise SystemExit('Complementos PCA3D ja integrados, instalacao repetida bloqueada.')

# Instrumentar somente a funcao gerar() existente, imediatamente antes do return.
arvore = ast.parse(orig_biplot)
funcoes = [f for f in arvore.body if isinstance(f, ast.FunctionDef) and f.name == 'gerar']
if len(funcoes) != 1 or not isinstance(funcoes[0].body[-1], ast.Return):
    raise SystemExit('Funcao gerar() nao termina com return esperado. Nada alterado.')
if not all(s in orig_biplot for s in ('_mapa_cores_variaveis_pca', 'tabela_correlacoes', 'modelo', 'coord')):
    raise SystemExit('Interface do modulo biplot diferente do esperado. Nada alterado.')

antes_return = '''    # PCA3D_BIPLOT_COEFICIENTES_OFICIAIS_V1
    # Reutiliza dados, indices e PCA ja ajustados; PCA 1D/2D ficam intactos.
    if dim == 3:
        from src.visualizacao_pca3d_complementos import gerar_complementares
        gerar_complementares(
            config, data, destino, ext, dpi, csv,
            idx=idx, tabela_correlacoes=tab,
            mapa_cores=_mapa_cores_variaveis_pca(x.columns),
        )

'''
linhas = orig_biplot.splitlines(keepends=True)
linhas.insert(funcoes[0].body[-1].lineno - 1, antes_return)
novo_biplot = ''.join(linhas)
compile(novo_biplot, str(modulo_biplot), 'exec')

# 07.03: a instalacao anterior do GMM01 ja acrescentou suporte a HTML.
if "'.html'" not in orig_galeria or '# PCA3D_GMM01_GALERIA_V1' not in orig_galeria:
    raise SystemExit('Integracao GMM01 3D e suporte HTML ausentes. Nada alterado.')

# Verificar o retorno no escopo da funcao interna completos().
retornos = list(re.finditer(
    r'(?m)^        return imagens_oof[ \t]*,[ \t]*imagens_geo[ \t]*$',
    orig_galeria,
))
if len(retornos) != 1:
    raise SystemExit('Retorno de completude nao encontrado unicamente. Nada alterado.')
bloco_completude = '''        # PCA3D_BIPLOT_GALERIA_V1: exigir HTML e coeficientes quando PCA3D.
        if (escolha['configuracao']['metodo'].upper() == 'PCA'
                and int(escolha['configuracao']['reducao']['n_components']) == 3
                and 'pca_biplot' in esperado_geo):
            imagens_geo = imagens_geo and all(
                (geo_dir / nome).is_file() for nome in (
                    f'PCA_COEFICIENTES_PC1_PC2_PC3{ext}',
                    'PCA_BIPLOT_3D_INTERATIVO.html',
                )
            )

'''
pos = retornos[0].start()
novo_galeria = orig_galeria[:pos] + bloco_completude + orig_galeria[pos:]

permitidos = list(re.finditer(r'(?m)^            if permitido:[ \t]*$', novo_galeria))
if len(permitidos) != 1:
    raise SystemExit('Regra de inclusao na galeria nao encontrada unicamente. Nada alterado.')
bloco_inclusao = '''                # PCA3D_BIPLOT_GALERIA_V1: somente dois arquivos oficiais.
                if ('pca_biplot' in ativos_geo
                        and escolha['configuracao']['metodo'].upper() == 'PCA'
                        and int(escolha['configuracao']['reducao']['n_components']) == 3
                        and arq.name in {
                            'PCA_COEFICIENTES_PC1_PC2_PC3' +
                            '.' + config['visualizacao']['padrao'].get('formato', 'png').lower(),
                            'PCA_BIPLOT_3D_INTERATIVO.html',
                        }):
                    permitido = True

'''
pos = permitidos[0].start()
novo_galeria = novo_galeria[:pos] + bloco_inclusao + novo_galeria[pos:]
if novo_galeria.count(marca_galeria) != 2:
    raise SystemExit('Insercoes de galeria incompletas. Nada alterado.')
compile(novo_galeria, str(modulo_galeria), 'exec')
compile(complementos_src.read_text(encoding='utf-8-sig'), str(complementos_src), 'exec')

stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
bak_biplot = modulo_biplot.with_name(modulo_biplot.name + '.ANTES_PCA3D_BIPLOT_' + stamp + '.bak')
bak_galeria = modulo_galeria.with_name(modulo_galeria.name + '.ANTES_PCA3D_BIPLOT_' + stamp + '.bak')
for bak in (bak_biplot, bak_galeria):
    if bak.exists():
        raise SystemExit('Backup ja existente; nada alterado.')
bak_biplot.write_bytes(modulo_biplot.read_bytes())
bak_galeria.write_bytes(modulo_galeria.read_bytes())
try:
    shutil.copy2(complementos_src, complementos_dst)
    modulo_biplot.write_text(novo_biplot, encoding='utf-8')
    modulo_galeria.write_text(novo_galeria, encoding='utf-8')
    compile(modulo_biplot.read_text(encoding='utf-8'), str(modulo_biplot), 'exec')
    compile(modulo_galeria.read_text(encoding='utf-8'), str(modulo_galeria), 'exec')
except Exception:
    modulo_biplot.write_bytes(bak_biplot.read_bytes())
    modulo_galeria.write_bytes(bak_galeria.read_bytes())
    complementos_dst.unlink(missing_ok=True)
    raise

print('BIPLOT PCA3D + COEFICIENTES: INTEGRADOS COM SUCESSO')
print('Backup biplot:', bak_biplot.name)
print('Backup galeria:', bak_galeria.name)
print('Modulo 3D:', complementos_dst)
print('YAML, dados, F10 e ensaios existentes preservados.')
