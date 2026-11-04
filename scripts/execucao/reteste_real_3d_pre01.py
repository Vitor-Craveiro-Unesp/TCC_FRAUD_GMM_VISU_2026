# -*- coding: utf-8 -*-
"""Integra PRE01 aprovado 1D nas galerias 3D com mesma base/folds.

Somente copia diagnosticos existentes e atualiza um HTML OOF local;
nao ajusta modelos ou acessa rotulos de F10.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import shutil
import os
import numpy as np

MARCADOR='<!-- PRE01_3D_PADRONIZACAO -->'
NOME='PRE01_diagnostico_padronizacao_completo.html'


def exigir(condicao,msg):
    if not condicao:raise RuntimeError(msg)


def sha(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def fontes(metodo,refs):
    ref=refs[metodo]
    return ref/'vencedor'/'visualizacao'/'galeria'/'oof'


def conferir_pre01(metodo,pasta,refs):
    src=fontes(metodo,refs)
    dest=pasta/'vencedor'/'visualizacao'/'galeria'/'oof'
    exigir((src/NOME).is_file(),f'PRE01 do 1D ausente: {src/NOME}')
    exigir((dest/NOME).is_file(),f'PRE01 3D ausente: {dest/NOME}')
    files=sorted(src.glob('PRE01*'))
    exigir(len(files)>=1,'Referencia PRE01 1D incompleta.')
    for f in files:
        exigir((dest/f.name).is_file() and sha(dest/f.name)==sha(f),
               f'PRE01 com hash divergente: {f.name}')
    g=pasta/'vencedor'/'visualizacao'/'galeria'/'index.html'
    h=pasta/'vencedor'/'teste_final'/'galeria_3pct'/'index.html'
    exigir(MARCADOR in g.read_text(encoding='utf-8'),'PRE01 nao incluido no OOF.')
    exigir(NOME in h.read_text(encoding='utf-8'),'PRE01 nao incluido no F10.')


def incluir_pre01_nos_resultados(metodo,pasta,ids_ref,refs):
    src=fontes(metodo,refs)
    origem=refs[metodo]/'vencedor'/'oof'/'predicoes.npz'
    exigir(origem.is_file(),f'Arquivo de origem PRE01 nao encontrado: {origem}')
    with np.load(origem,allow_pickle=False) as z:
        ids=np.asarray(z['linha_original'],dtype=np.int64)
    exigir(np.array_equal(ids,ids_ref),
           'Os IDs OOF 1D nao coincidem com a amostra 3D; PRE01 nao pode ser reaproveitado.')
    arquivos=sorted(src.glob('PRE01*'))
    exigir(len(arquivos)>=1 and any(p.name==NOME for p in arquivos),
           'PRE01 aprovado do 1D incompleto.')
    galeria=pasta/'vencedor'/'visualizacao'/'galeria'
    oof=galeria/'oof'
    exigir(galeria.is_dir() and oof.is_dir(),
           'Galeria OOF 3D ainda nao foi criada.')
    # Copia somente arquivos diagnosticos sem substituir conteudo preexistente.
    for f in arquivos:
        destino=oof/f.name
        if destino.is_file():
            exigir(sha(destino)==sha(f),'PRE01 destino difere do aprovado: '+str(destino))
        else:
            exigir(not destino.exists(),'PRE01 destino nao e arquivo: '+str(destino))
            shutil.copy2(f,destino)
            exigir(sha(destino)==sha(f),'PRE01 apresentou hash divergente na copia.')
    indice=galeria/'index.html'
    html=indice.read_text(encoding='utf-8')
    if MARCADOR not in html:
        exigir('</body></html>' in html,'Estrutura HTML OOF inesperada; nao editar.')
        backup=galeria/'index_antes_PRE01_3D_backup.html'
        exigir(not backup.exists(),'Backup PRE01 ja existe, porem o marcador sumiu.')
        shutil.copy2(indice,backup)
        card='''<!-- PRE01_3D_PADRONIZACAO -->
<section><h2>PRE01 — Graficos e descricao da padronizacao</h2>
<p>Diagnostico do 1D aprovado, reaproveitado porque o preprocessamento usa as mesmas entradas, folds e features no 3D. Nenhum dado F10 foi empregado no ajuste.</p>
<div class="grade"><article><a href="oof/PRE01_diagnostico_padronizacao_completo.html" target="_blank" rel="noopener">
<div class="sem-previa">Abrir diagnostico de padronizacao interativo</div>
<div>PRE01 — Variaveis, medias, escalas e comparacoes por fold</div></a></article></div></section>
'''
        temporario=galeria/'index.html.pre01tmp'
        exigir(not temporario.exists(),'Temporario de galeria PRE01 existe.')
        temporario.write_text(html.replace('</body></html>',card+'</body></html>'),encoding='utf-8')
        os.replace(temporario,indice)
    conferir_pre01(metodo,pasta,refs)
    print(f'{metodo.upper()}_PRE01_3D=LINKS_OOF_F10_OK',flush=True)
