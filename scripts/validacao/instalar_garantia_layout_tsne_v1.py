# -*- coding: utf-8 -*-
"""Instalador idempotente da verificacao obrigatoria de layout em novos YAMLs.

--planejar / --aplicar / --validar; nao altera dados, OOF, F10, modelos ou YAML.
Backup original das fontes antes do patch. Falha fechada se fonte divergir.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
import hashlib
from pathlib import Path
import shutil
import sys

RAIZ = Path(__file__).resolve().parents[2]
MARCADOR = 'GUARDA_LAYOUT_TSNE_YAML_V1_20261103'
ALVOS = (
    'src/orquestracao_oof.py',
    'src/visualizacao_geometrica.py',
    'scripts/execucao/experimentos_reais_3pct_1d.py',
)


def exigir(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def patch(rel: str, text: str) -> str:
    if MARCADOR in text:
        exigir(text.count(MARCADOR) == (2 if rel == ALVOS[0] else 1),
               f'Marcadores incompletos/inesperados em {rel}.')
        return text
    novo = text
    if rel == ALVOS[0]:
        # Preflight central: qualquer novo YAML iniciado pelo orquestrador.
        a = "    validar_operacao(config)\n    automatizar_f10 = bool(config.get('execucao', {}).get('finalizacao_pca1d_automatica', False))"
        b = "    validar_operacao(config)\n    # " + MARCADOR + " / CLI novo YAML\n    from src.garantia_layout_tsne import verificar_layout_tsne\n    verificar_layout_tsne(config, fase='YAML_novo')\n    automatizar_f10 = bool(config.get('execucao', {}).get('finalizacao_pca1d_automatica', False))"
        exigir(novo.count(a) == 1, f'Ancora de leitura de YAML inesperada em {rel}')
        novo = novo.replace(a, b)
        # Visualizacao separada / retomada de OOF ja concluido.
        a = "    experimento=Path(experimento).resolve()\n    caminho_manifest=experimento/'manifesto.json'"
        b = "    experimento=Path(experimento).resolve()\n    # " + MARCADOR + " / visualizacao/retomada\n    from src.garantia_layout_tsne import verificar_layout_tsne\n    config_grafico = yaml.safe_load((experimento / 'configuracao.yaml').read_text(encoding='utf-8-sig'))\n    verificar_layout_tsne(config_grafico, fase='antes_galeria_OOF')\n    caminho_manifest=experimento/'manifesto.json'"
        exigir(novo.count(a) == 1, f'Ancora de galeria OOF inesperada em {rel}')
        novo = novo.replace(a, b)
    elif rel == ALVOS[1]:
        a = '    cfg,grade,escolha,y_av_oof,p_av_oof = vis._ler_resultados(experimento)'
        b = (a + '\n    # ' + MARCADOR + ' / geometria direta\n'
             + '    from src.garantia_layout_tsne import verificar_layout_tsne\n'
             + "    verificar_layout_tsne(cfg, fase='antes_geometria_OOF')")
        exigir(novo.count(a) == 1, f'Ancora de geometria inesperada em {rel}')
        novo = novo.replace(a, b)
    elif rel == ALVOS[2]:
        a = '    cfg, grade = imprimir_plano(metodo)\n    print(f\'[{metodo.upper()}] Iniciando OOF/retomada.\', flush=True)'
        b = ("    # " + MARCADOR + " / executor 3%\n"
             + '    from src.garantia_layout_tsne import verificar_layout_tsne\n'
             + '    cfg, grade = imprimir_plano(metodo)\n'
             + "    verificar_layout_tsne(cfg, fase='executor_3pct')\n"
             + "    print(f'[{metodo.upper()}] Iniciando OOF/retomada.', flush=True)")
        exigir(novo.count(a) == 1, f'Ancora de executor 3% inesperada em {rel}')
        novo = novo.replace(a, b)
    else:
        raise RuntimeError(f'Alvo fora do escopo: {rel}')
    ast.parse(novo, filename=rel)
    return novo


def plano():
    if str(RAIZ) not in sys.path:
        sys.path.insert(0, str(RAIZ))
    from src.garantia_layout_tsne import verificar_layout_tsne
    # O hotfix anterior precisa estar instalado e reconhecido antes de colocar o guard.
    verificar_layout_tsne({'reducao': {'metodo': 'TSNE'}, 'tsne': {'n_components': 1}}, fase='preflight_instalador')
    saidas = {}
    for rel in ALVOS:
        f = RAIZ / rel
        exigir(f.is_file(), f'Arquivo ausente, nao instalar parcialmente: {rel}')
        raw = f.read_bytes()
        cod = raw.decode('utf-8-sig').replace('\r\n','\n')
        novo = patch(rel, cod)
        saidas[rel] = (raw, novo, cod != novo)
        print(f'[PLANEJADO] {rel} | patch={"NECESSARIO" if cod != novo else "JA_INSTALADO"}', flush=True)
    return saidas


def aplicar():
    trabalhos = plano()
    alterar = {p: v for p, v in trabalhos.items() if v[2]}
    if not alterar:
        print('GARANTIA_LAYOUT_STATUS=JA_INSTALADA', flush=True)
        return validar()
    now = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup = RAIZ / 'backups' / 'garantia_layout_tsne_yaml' / now
    exigir(not backup.exists(),f'Backup repetido: {backup}')
    for rel, (raw, novo, _) in alterar.items():
        b = backup / rel
        b.parent.mkdir(parents=True, exist_ok=True)
        b.write_bytes(raw)
        exigir(hashlib.sha256(b.read_bytes()).digest() == hashlib.sha256(raw).digest(),
               f'Backup incompleto: {rel}')
    feitos = []
    try:
        for rel,(raw, novo, _) in alterar.items():
            target = RAIZ / rel
            exigir(target.read_bytes() == raw, f'Fonte mudou durante instalacao: {rel}')
            newline = '\r\n' if b'\r\n' in raw else '\n'
            bom = b'\xef\xbb\xbf' if raw.startswith(b'\xef\xbb\xbf') else b''
            bom += novo.replace('\n', newline).encode('utf-8')
            temp = target.with_name(target.name + '.tmp_guarda_layout')
            exigir(not temp.exists(),f'Arquivo temporario suspeito: {temp}')
            temp.write_bytes(bom)
            temp.replace(target)
            feitos.append(rel)
            print(f'[FONTE PROTEGIDA] {rel}', flush=True)
        validar()
    except Exception:
        # No caso de falha local, devolve as fontes da mesma aplicacao.
        for rel in reversed(feitos):
            (RAIZ / rel).write_bytes((backup / rel).read_bytes())
        raise
    print(f'BACKUP_GARANTIA={backup}', flush=True)
    print('GARANTIA_LAYOUT_STATUS=APROVADA', flush=True)
    return 0


def validar():
    trabalhos = plano()
    exigir(not any(diferente for _,_,diferente in trabalhos.values()),
           'Faltam patches. Execute --aplicar antes de --validar.')
    print('GUARDA_NO_YAML=SIM', flush=True)
    print('GUARDA_NA_GALERIA_OOF=SIM', flush=True)
    print('GUARDA_NA_GEOMETRIA=SIM', flush=True)
    print('GUARDA_NO_EXECUTOR_3PCT=SIM', flush=True)
    print('LAYOUT_FUTUROS_YAML=PROTEGIDO', flush=True)
    print('MODELOS_OOF_F10_REEXECUTADOS=NAO', flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--planejar', action='store_true')
    g.add_argument('--aplicar', action='store_true')
    g.add_argument('--validar', action='store_true')
    opts = p.parse_args()
    if opts.planejar:
        plano()
        print('PLANEJAMENTO_SEM_MODIFICACOES=SIM')
        return 0
    return aplicar() if opts.aplicar else validar()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print('GARANTIA_LAYOUT_STATUS=BLOQUEADA')
        print(f'MOTIVO={e}')
        raise
