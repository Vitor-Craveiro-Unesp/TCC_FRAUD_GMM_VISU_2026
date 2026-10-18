"""Passo 06.04: preflight SOMENTE LEITURA para a execucao OOF oficial.

Nao treina modelos, nao le rotulos de F10 para aprendizado, nao modifica YAML,
folds, Parquet ou outputs. A amostra de memoria usa apenas linhas F1..F9.

Uso: python -m src.preflight_oof
     python -m src.preflight_oof --json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from src.orquestracao_oof import _identificar_folds, validar_operacao
from src.validacao_oof import grade_candidatos_oof

YAML_OFICIAL = "config_TCC_EXPERIMENTAL_V3_atualizado.yaml"
GIB = 1024 ** 3


def _recursos_sistema():
    """Memoria do SO, nao depende de instalar pacote adicional."""
    import ctypes
    if os.name == 'nt':
        class MEMSTAT(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_uint32), ('dwMemoryLoad', ctypes.c_uint32),
                ('ullTotalPhys', ctypes.c_uint64), ('ullAvailPhys', ctypes.c_uint64),
                ('ullTotalPageFile', ctypes.c_uint64), ('ullAvailPageFile', ctypes.c_uint64),
                ('ullTotalVirtual', ctypes.c_uint64), ('ullAvailVirtual', ctypes.c_uint64),
                ('ullAvailExtendedVirtual', ctypes.c_uint64),
            ]
        s = MEMSTAT()
        s.dwLength = ctypes.sizeof(s)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            raise OSError('Nao foi possivel consultar a RAM no Windows.')
        return int(s.ullTotalPhys), int(s.ullAvailPhys)
    if sys.platform.startswith('linux'):
        mem = {}
        for l in Path('/proc/meminfo').read_text().splitlines():
            if ':' in l:
                k, v = l.split(':', 1)
                mem[k] = int(v.strip().split()[0]) * 1024
        if 'MemTotal' in mem and 'MemAvailable' in mem:
            return mem['MemTotal'], mem['MemAvailable']
    # Sem memoria disponivel, preflight nao afirmara capacidade.
    return None, None


def _destino_armazenamento(raiz, config):
    pasta = raiz / config['saidas']['diretorio_base']
    alvo = pasta
    while not alvo.exists():
        anterior = alvo.parent
        if anterior == alvo:
            raise FileNotFoundError('Nao encontrei volume para outputs.')
        alvo = anterior
    if not alvo.is_dir():
        raise RuntimeError(f'Destino de saida nao e diretorio: {alvo}')
    return pasta, shutil.disk_usage(alvo).free


def _amostra_desenvolvimento(parquet, folds, *, max_linhas=12_000):
    """Projeção empírica por lotes; amostra linhas F1..F9, sem treinamento."""
    if max_linhas < 100:
        raise ValueError('Amostra muito pequena para medir memoria.')
    import pyarrow.parquet as pq
    arquivo = pq.ParquetFile(parquet)
    partes = []
    lidas = 0
    deslocamento = 0
    # Leitura fisica em lotes; somente indices de desenvolvimento entram no df.
    for lote in arquivo.iter_batches(batch_size=4096):
        fim = deslocamento + lote.num_rows
        pos = np.flatnonzero(folds[deslocamento:fim] != 10)
        if pos.size:
            pos = pos[:max_linhas-lidas]
            partes.append(lote.to_pandas().iloc[pos].copy())
            lidas += len(pos)
        deslocamento = fim
        if lidas >= max_linhas:
            break
    if lidas < 100:
        raise RuntimeError('Amostra de desenvolvimento insuficiente.')
    df = pd.concat(partes, ignore_index=True)
    tamanho = int(df.memory_usage(index=True, deep=True).sum())
    return len(df), tamanho / len(df)


def _estimar_memoria(n_dev, bytes_por_linha, n_colunas):
    """Limite de PLANEJAMENTO heuristico, nao pico RAM garantido."""
    if n_dev <= 0 or bytes_por_linha <= 0 or n_colunas < 3:
        raise ValueError('Amostra ou metadados invalidos.')
    base = int(np.ceil(n_dev * bytes_por_linha))
    # Memoria de multiplas matrizes de treino/intermediarias (codificado,
    # imputado, normalizado, copias). 39 e somente piso de planejamento,
    # nao representa deducao de dimensoes reais apos encoding.
    n_features_estimadas = max(39, (n_colunas - 2) * 2)
    matriz = n_dev * n_features_estimadas * 8
    estimativa = int(3 * base + 8 * matriz + 2 * GIB)
    return {'dataframe_estimado_bytes':base,
            'matriz_features_estimadas':n_features_estimadas,
            'memoria_recomendada_bytes':estimativa}


def _avaliar_recursos(ram_disponivel, disco_livre, memoria_recomendada, tamanho_parquet):
    """Sinaliza riscos sem aprovar automaticamente uma execucao longa."""
    disco_recomendado = max(3*GIB, tamanho_parquet*30)
    avisos=[]
    bloqueios=[]
    if disco_livre < disco_recomendado:
        bloqueios.append('Espaco livre abaixo da reserva prudencial para caches e artefatos.')
    if ram_disponivel is None:
        avisos.append('RAM disponivel nao identificada: requer verificacao manual.')
    elif ram_disponivel < memoria_recomendada:
        avisos.append('RAM livre abaixo da estimativa conservadora; considere o cluster ou libere memoria.')
    return {'disco_recomendado_bytes':disco_recomendado,
            'avisos':avisos,'bloqueios':bloqueios}


def _estado_git(raiz):
    try:
        proc = subprocess.run(['git','-C',str(raiz),'rev-parse','--short','HEAD'],
                              capture_output=True, text=True, timeout=8)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def diagnosticar(raiz: Path, *, amostra=12_000):
    """Prepara relatorio sem criar diretorios nem modificar arquivos.

    Erros estruturais bloqueiam; alertas de RAM pedem avaliacao do usuario.
    """
    raiz = Path(raiz).resolve()
    caminho = raiz/YAML_OFICIAL
    if not caminho.is_file():
        raise FileNotFoundError(f'YAML oficial ausente: {caminho}')
    config = yaml.safe_load(caminho.read_text(encoding='utf-8-sig'))
    if not isinstance(config,dict):
        raise ValueError('YAML oficial nao contem um mapeamento.')
    validar_operacao(config)
    if config['dados']['fracao_base'] != 1.0:
        raise ValueError('A execucao oficial exige dados.fracao_base: 1.0.')
    grade = list(grade_candidatos_oof(config))
    if not grade:
        raise RuntimeError('Grade de candidatos vazia.')
    arquivo = raiz/config['dados']['arquivo']
    if not arquivo.is_file():
        raise FileNotFoundError(f'Dataset original ausente: {arquivo}')
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(arquivo).metadata
    n_linhas = int(meta.num_rows)
    folds, arq_folds, hash_parquet = _identificar_folds(raiz,config,arquivo)
    if len(folds)!=n_linhas:
        raise RuntimeError('Contagem Parquet/particao diverge.')
    n_dev = int((folds!=10).sum())
    n_f10 = int((folds==10).sum())
    if n_dev < 100 or n_f10 < 10:
        raise RuntimeError('Divisao em desenvolvimento e F10 invalida.')
    n_amostra, b_linha = _amostra_desenvolvimento(arquivo,folds,max_linhas=amostra)
    estimativa = _estimar_memoria(n_dev,b_linha,int(meta.num_columns))
    total_ram, ram_livre = _recursos_sistema()
    pasta_output, disco_livre = _destino_armazenamento(raiz,config)
    riscos = _avaliar_recursos(ram_livre,disco_livre,
                               estimativa['memoria_recomendada_bytes'],arquivo.stat().st_size)
    avisos=riscos['avisos']
    bloqueios=riscos['bloqueios']
    if config['execucao']['n_jobs']==-1:
        avisos.append('n_jobs=-1 usa todos os nucleos logicos para bibliotecas numericas; monitore RAM/temperatura.')
    pendentes=[]
    if pasta_output.is_dir():
        pendentes=sorted(p.name for p in pasta_output.glob('em_andamento_*') if p.is_dir())
        if pendentes:
            avisos.append('Existem execucoes em andamento; verifique retomada antes de nova tentativa.')
    git_hash=_estado_git(raiz)
    if git_hash is None:
        avisos.append('Commit Git local nao encontrado; confira backup/checkpoint.')
    retorno={
        'etapa':'06.04_preflight_execucao_oficial',
        'iniciou_treinamento':False,
        'teste_f10_utilizado_em_fit':False,
        'configuracao':caminho.name,
        'metodo':str(config['reducao']['metodo']).upper(),
        'amostragem_modo':config['amostragem']['modo'],
        'n_jobs':config['execucao']['n_jobs'],
        'quantidade_candidatos':len(grade),
        'rodadas_por_candidato':9,
        'total_rodadas':len(grade)*9,
        'parquet_linhas':n_linhas,
        'parquet_colunas':int(meta.num_columns),
        'parquet_bytes':arquivo.stat().st_size,
        'hash_parquet_verificado':hash_parquet,
        'arquivo_folds':str(arq_folds.relative_to(raiz)),
        'desenvolvimento_linhas':n_dev,
        'f10_linhas':n_f10,
        'amostra_memoria_linhas':n_amostra,
        'bytes_por_linha_amostrada':round(b_linha,2),
        **estimativa,
        'ram_total_bytes':total_ram,
        'ram_disponivel_bytes':ram_livre,
        'disco_livre_bytes':disco_livre,
        'disco_recomendado_bytes':riscos['disco_recomendado_bytes'],
        'outputs_existentes_em_andamento':pendentes,
        'git_commit':git_hash,
        'avisos':avisos,
        'bloqueios':bloqueios,
        'situacao':'BLOQUEADO' if bloqueios else 'AGUARDANDO_REVISAO_MANUAL',
    }
    return retorno


def _formatar_gib(n):
    return 'indisponivel' if n is None else f'{n/GIB:.2f} GiB'


def exibir(relatorio):
    print('PASSO 06.04 - PREFLIGHT OOF (SOMENTE LEITURA)')
    print('YAML:',relatorio['configuracao'])
    print('Git:',relatorio['git_commit'] or 'nao encontrado')
    print('Metodo:',relatorio['metodo'],'| Amostragem:',relatorio['amostragem_modo'],
          '| n_jobs:',relatorio['n_jobs'])
    print(f"Grade: {relatorio['quantidade_candidatos']} candidatos x 9 folds = {relatorio['total_rodadas']} rodadas")
    print(f"Parquet: {relatorio['parquet_linhas']:,} linhas | Desenvolvimento: {relatorio['desenvolvimento_linhas']:,} | F10: {relatorio['f10_linhas']:,}")
    print('Folds:',relatorio['arquivo_folds'])
    print(f"Amostra real: {relatorio['amostra_memoria_linhas']:,} linhas, {relatorio['bytes_por_linha_amostrada']:.1f} bytes/linha (deep)")
    for nome,key in [
       ('DataFrame desenvolvimento (estimado)','dataframe_estimado_bytes'),
       ('RAM recomendada p/ planejamento (heuristica)','memoria_recomendada_bytes'),
       ('RAM fisica total','ram_total_bytes'),
       ('RAM disponivel agora','ram_disponivel_bytes'),
       ('Disco livre no volume de outputs','disco_livre_bytes'),
       ('Reserva prudencial de disco','disco_recomendado_bytes')]:
        print(f'{nome}: {_formatar_gib(relatorio[key])}')
    for a in relatorio['avisos']:
        print('[ATENCAO]',a)
    for a in relatorio['bloqueios']:
        print('[BLOQUEIO]',a)
    print('Situacao:',relatorio['situacao'])
    print('Nenhum GMM foi treinado. Nenhum arquivo foi alterado. F10 reservado.')
    print('NAO execute --executar ate avaliar os resultados deste preflight.')


def main(argv=None, *, raiz=None):
    p=argparse.ArgumentParser(description='Preflight 06.04 OOF sem iniciar modelos')
    p.add_argument('--amostra',type=int,default=12000,
                   help='Apenas linhas de desenvolvimento para medir memoria (>=100)')
    p.add_argument('--json',action='store_true',help='Exibir relatorio JSON para auditoria')
    args=p.parse_args(argv)
    if args.amostra<100:
        p.error('--amostra deve ser >=100')
    root=Path(raiz) if raiz else Path(__file__).resolve().parents[1]
    relatorio=diagnosticar(root,amostra=args.amostra)
    if args.json:
        print(json.dumps(relatorio,indent=2,ensure_ascii=False))
    else:
        exibir(relatorio)
    return relatorio


if __name__=='__main__':
    main()
