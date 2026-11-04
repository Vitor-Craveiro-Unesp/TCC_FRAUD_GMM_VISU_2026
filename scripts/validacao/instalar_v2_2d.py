# -*- coding: utf-8 -*-
"""V2: instala apenas adaptacoes 2D e cria dois YAMLs congelados de 3%.

Nao le dados nem F10, nao faz treinamento, nao cria outputs, nao apaga resultados.
Modo --planejar: somente leitura; --aplicar: backup e patch; --validar: leitura.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
FONTE = {
    'src/visualizacao_gmm01.py': '3f7acd7a107903acab49b63dbd3f55b66ecc1c90f7e710d720639c82d79974bb',
    'src/visualizacao_integrada.py': '0634a80d33370a9821cef5e97aab9e54f12b204c8043c9ec4107486451b059b1',
    'src/visualizacao_geometrica.py': '91ccf3e4f54e4fd5048bc6de6e3f754fe65d7bf18566ad8fa1ee9d6cd89f3e36',
}
PATCHES_SHA256 = {
    'src/visualizacao_gmm01.py': '9459d0919963d452ba58c6b9ee3138268bae3014bd644b54caa1d0e192056d5d',
    'src/visualizacao_integrada.py': 'f90d7ac13bc04e067d186731f9bd9cd508f2813fd26873bbf106ed01f18e2b34',
    'src/visualizacao_geometrica.py': '7bb41431e46f7c2e68e083b0753774b44d9dc8482a9a87caf26d65fe23445bef',
}
MARCADOR = 'V2_TCC_TSNE_2D_COMPAT_20261103'
ARQUIVOS_NOVOS = (
    'src/visualizacao_tsne_2d.py',
    'scripts/execucao/experimentos_reais_3pct_2d.py',
)
NOVOS_SHA256 = {
    'src/visualizacao_tsne_2d.py': '724f4720cabfc25781aa59ed8a842bf13fcddb762c1f2f16401512a05d1833d0',
    'scripts/execucao/experimentos_reais_3pct_2d.py': '6cc9ebfd3f0f86b1b84f088079e4cf1d3bdcbff0f471dd18cb2c8f29ea4a0124',
}
CONFIG_DIR = ROOT/'configs'/'experimentos_reais_3pct_2d'


def exigir(cond, message):
    if not cond:
        raise RuntimeError(message)


def digest(b):return sha256(b).hexdigest()


def origem(rel):return ROOT/rel


def estado_rel(rel):
    p=origem(rel)
    exigir(p.is_file(),f'Arquivo fonte ausente: {rel}')
    b=p.read_bytes()
    if digest(b)==FONTE[rel]:return 'ORIGINAL_CONFERIDO'
    if digest(b) == PATCHES_SHA256[rel]:
        return 'ATUALIZADO_V2'
    raise RuntimeError(f'Fonte local diferente do arquivo recebido; NAO sobrescrever: {rel}')


def substituir(texto,antigo,novo,nome):
    exigir(texto.count(antigo)==1,f'Ancora unica nao encontrada em {nome}')
    return texto.replace(antigo,novo,1)


def _patch_gmm01(s,eol):
    s=substituir(s,"'A densidade conjunta do GMM01 exige PCA 2D.'",
                 "'A densidade conjunta do GMM01 exige 2 dimensoes.'",'GMM01 mensagem')
    s=substituir(s,'    if d == 2:'+eol+'        grade_2d',
                 '    if d == 2:'+eol+
                 "        # "+MARCADOR+eol+
                 "        geo._exigir(nomes in (['PC1','PC2'],['TSNE1','TSNE2']),"+eol+
                 "                    'GMM01 2D exige PC1/PC2 ou TSNE1/TSNE2.')"+eol+
                 '        grade_2d','GMM01 dimension')
    s=substituir(s,"ax3d.set_xlabel('PC1', labelpad=9)","ax3d.set_xlabel(nomes[0], labelpad=9)",'GMM01 X')
    s=substituir(s,"ax3d.set_ylabel('PC2', labelpad=9)","ax3d.set_ylabel(nomes[1], labelpad=9)",'GMM01 Y')
    s=substituir(s,"'GMM01_densidade_conjunta_superficie_PCA2D'",
                 "('GMM01_densidade_conjunta_superficie_PCA2D'"+eol+
                 "             if nomes == ['PC1','PC2'] else"+eol+
                 "             'GMM01_densidade_conjunta_superficie_TSNE2D')",'GMM01 stem')
    return s


def _patch_integrada(s,eol):
    pos='        # PCA3D_GMM01_GALERIA_V1: verificar os cinco anexos GMM01.'
    bloco=eol.join([
       '        # '+MARCADOR+' | validar densidades conjuntas TSNE2D',
       "        if (metodo == 'TSNE' and dim == 2 and 'densidades_marginais' in esperado_geo):",
       '            adicionais_tsne_2d = (',
       "                'GMM01_densidade_conjunta_2d_TSNE1_TSNE2',",
       "                'GMM01_densidade_conjunta_superficie_TSNE2D',",
       '            )',
       '            imagens_geo = imagens_geo and all(',
       "                (geo_dir / f'{nome}{ext}').is_file() for nome in adicionais_tsne_2d",
       '            )',
       '',
    ])
    s=substituir(s,pos,bloco+pos,'integrada validacao')
    ini=s.index('                    # Adicionar as duas representacoes conjuntas PCA 2D.')
    fim=s.index('                        if arq.stem in conjuntas_2d:',ini)
    f=s.index('                            permitido = True',fim)+len('                            permitido = True')
    original=s[ini:f]
    bloco2=eol.join([
       '                    # '+MARCADOR+' | conjunta 2D PCA ou TSNE',
       '                    if dim == 2:',
       "                        metodo_2d = escolha['configuracao']['metodo'].upper()",
       "                        eixos_2d = {'PCA': ('PC1','PC2'),",
       "                                   'TSNE': ('TSNE1','TSNE2')}.get(metodo_2d)",
       '                        if eixos_2d is not None:',
       '                            conjuntas_2d = {',
       "                                f'GMM01_densidade_conjunta_2d_{eixos_2d[0]}_{eixos_2d[1]}',",
       "                                f'GMM01_densidade_conjunta_superficie_{metodo_2d}2D'",
       '                            }',
       '                            if arq.stem in conjuntas_2d:',
       '                                permitido = True',
    ])
    s=substituir(s,original,bloco2,'integrada copia')
    return s


def _patch_geometrica(s,eol):
    ancora="    if metodo == 'TSNE' and coord.shape[1] == 3:"
    bloco=eol.join([
        '    # '+MARCADOR+' | TSNE13 2D com titulo e rodape separados',
        "    if metodo == 'TSNE' and coord.shape[1] == 2:",
        '        from src.visualizacao_tsne_2d import gerar_projecao_2d',
        "        gerar_projecao_2d(config,data,dest,ext,dpi,csv,origem='desenvolvimento',metodo='TSNE')",
        '        return',
        '',
    ])
    s=substituir(s,ancora,bloco+ancora,'geometrica TSNE13 2D')
    return s


FUNCOES={'src/visualizacao_gmm01.py':_patch_gmm01,
         'src/visualizacao_integrada.py':_patch_integrada,
         'src/visualizacao_geometrica.py':_patch_geometrica}


def patch_bytes(rel,b):
    texto=b.decode('utf-8-sig')
    # Fontes antigas podem ter CRLF, e visualizacao_geometrica tem CRCRLF.
    eol='\r\r\n' if b'\r\r\n' in b else ('\r\n' if b'\r\n' in b else '\n')
    novo=FUNCOES[rel](texto,eol)
    ast.parse(novo,filename=rel)
    return (b'\xef\xbb\xbf' if b.startswith(b'\xef\xbb\xbf') else b'') + novo.encode('utf-8')


def testar_fonte_modificada(rel):
    b=origem(rel).read_bytes()
    texto=b.decode('utf-8-sig')
    exigir(MARCADOR in texto and digest(b) == PATCHES_SHA256[rel],
           f'Patch V2 alterado ou incompleto: {rel}')
    ast.parse(texto,filename=rel)
    if rel.endswith('visualizacao_gmm01.py'):
        exigir("ax3d.set_xlabel(nomes[0]" in texto and 'superficie_TSNE2D' in texto,
               'GMM01 patch incompleto')
    elif rel.endswith('visualizacao_integrada.py'):
        exigir("adicionais_tsne_2d" in texto and "'TSNE': ('TSNE1','TSNE2')" in texto,
               'Galeria TSNE2D incompleta')
    elif rel.endswith('visualizacao_geometrica.py'):
        exigir("origem='desenvolvimento',metodo='TSNE'" in texto,
               'TSNE13 2D nao conectado')


def conferir_arquivos_novos():
    for rel in ARQUIVOS_NOVOS:
        p=origem(rel)
        exigir(p.is_file(),f'Instale o pacote ZIP com o arquivo novo: {rel}')
        exigir(digest(p.read_bytes()) == NOVOS_SHA256[rel],
               f'Arquivo novo V2 foi modificado: {rel}')
        ast.parse(p.read_text(encoding='utf-8-sig'),filename=rel)


def gerar_yamls():
    from scripts.execucao import experimentos_reais_3pct_2d as runner
    candidatos={}
    for metodo in ('pca','tsne'):
        cfg,grade=runner.preparar_config_original(metodo)
        candidatos[metodo]=(cfg,grade,runner.yaml_texto(cfg).encode('utf-8'))
    return candidatos


def planejar():
    conferir_arquivos_novos()
    for rel in FONTE:
        est=estado_rel(rel)
        print(f'[OK] {rel}: {est}')
    configs=gerar_yamls()
    for metodo,(cfg,grade,raw) in configs.items():
        path=CONFIG_DIR/f'{metodo}_gmm_2d_3pct.yaml'
        if path.exists():
            import yaml
            atual=yaml.safe_load(path.read_text(encoding='utf-8-sig'))
            exigir(atual==cfg,'YAML congelado existente diverge do YAML que seria produzido: '+str(path))
            estado='JA_EXISTE_IGUAL'
        else:
            estado='NOVO_CONGELADO'
        print(f'{metodo.upper()}_YAML={path}')
        print(f'{metodo.upper()}_CANDIDATOS={len(grade)} | {estado}')
        print(f'{metodo.upper()}_SHA256_YAML_LF={digest(raw)}')
    print('TREINAMENTO_EXECUTADO=NAO')
    print('OOF_F10_ACESSADOS=NAO')
    print('V2_PLANO=APROVADO_PARA_INSTALACAO')


def aplicar():
    conferir_arquivos_novos()
    estados={rel:estado_rel(rel) for rel in FONTE}
    configs=gerar_yamls()
    for metodo,(cfg,grade,raw) in configs.items():
        path=CONFIG_DIR/f'{metodo}_gmm_2d_3pct.yaml'
        if path.exists():
            import yaml
            exigir(yaml.safe_load(path.read_text(encoding='utf-8-sig'))==cfg,
                   'YAML congelado diferente do proposto; bloqueado: '+str(path))
    alterados={rel:origem(rel).read_bytes() for rel,estado in estados.items()
               if estado=='ORIGINAL_CONFERIDO'}
    novos={rel:patch_bytes(rel,b) for rel,b in alterados.items()}
    # Tudo conferido antes da primeira escrita.
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    backup=ROOT/'backups'/'v2_2d_instalacao'/stamp
    backup.mkdir(parents=True,exist_ok=False)
    for rel,b in alterados.items():
        p=backup/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    print('BACKUP_V2='+str(backup))
    escritos=[]
    try:
        for rel,b in novos.items():
            path=origem(rel)
            exigir(path.read_bytes()==alterados[rel],f'Fonte mudou durante instalacao: {rel}')
            tmp=path.with_name('.tmp_v2_'+uuid.uuid4().hex+'.py')
            with tmp.open('xb') as fh:fh.write(b)
            tmp.replace(path)
            escritos.append(rel)
            print('[FONTE V2] '+rel)
        CONFIG_DIR.mkdir(parents=True,exist_ok=True)
        for metodo,(cfg,grade,raw) in configs.items():
            path=CONFIG_DIR/f'{metodo}_gmm_2d_3pct.yaml'
            if not path.exists():
                with path.open('xb') as fh:fh.write(raw)
                print('[YAML CONGELADO] '+str(path))
    except Exception:
        for rel in escritos:
            origem(rel).write_bytes(alterados[rel])
        raise
    validar()


def validar():
    conferir_arquivos_novos()
    for rel in FONTE:testar_fonte_modificada(rel)
    from scripts.execucao import experimentos_reais_3pct_2d as runner
    for metodo in ('pca','tsne'):
        cfg,grade=runner.construir_config(metodo)
        exigir(cfg['dados']['fracao_base']==.03 and all(
            int(c['reducao']['n_components'])==2 for c in grade),
            'Configuracao 2D nao homologada.')
        print(f'[VALIDADO] {metodo.upper()} 2D 3% | OOF candidatos={len(grade)}')
    # Exigir bloqueio do treinamento V2 sem HOMOLOGACAO_V3_APROVADA.
    exigir(not runner.LIBERACAO_V3.exists(),
           'Marcador V3 existe: nao foi criado pela V2; revisar antes de continuar.')
    try:
        runner.verificar_autorizacao_v3()
    except RuntimeError as exc:
        exigir('BLOQUEADOS NA V2' in str(exc),'Gate V3 inesperado.')
    else:
        raise RuntimeError('Gate V3 falhou: treino teria sido autorizado.')
    print('EXECUCAO_OOF_F10_2D=BLOQUEADA_ATE_V3')
    print('ARQUIVOS_ANTIGOS_1D=NAO_MODIFICADOS')
    print('YAML_PRINCIPAL=NAO_MODIFICADO')
    print('DADOS_F10=NAO_LIDOS')
    print('V2_STATUS=APROVADA_SEM_TREINAMENTO')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    mx=ap.add_mutually_exclusive_group(required=True)
    mx.add_argument('--planejar',action='store_true')
    mx.add_argument('--aplicar',action='store_true')
    mx.add_argument('--validar',action='store_true')
    a=ap.parse_args(argv)
    try:
        if a.planejar:planejar()
        elif a.aplicar:aplicar()
        else:validar()
        return 0
    except Exception as exc:
        print('V2_STATUS=BLOQUEADO_SEM_SOBRESCRITA',file=sys.stderr)
        print('MOTIVO='+str(exc),file=sys.stderr)
        traceback.print_exc()
        return 1

if __name__=='__main__':raise SystemExit(main())
