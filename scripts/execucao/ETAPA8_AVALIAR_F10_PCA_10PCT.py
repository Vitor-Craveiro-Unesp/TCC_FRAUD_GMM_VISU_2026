# -*- coding: utf-8 -*-
"""Etapa 8/9 — avaliacao F10 em UMA amostra estratificada de 10%.

Instalar em scripts/execucao/ETAPA8_AVALIAR_F10_PCA_10PCT.py.

SEM ACESSAR F10 (verificacao somente):
    python scripts/execucao/ETAPA8_AVALIAR_F10_PCA_10PCT.py --verificar

SOMENTE APOS DECISAO EXPLICITA DE AVALIAR O F10:
    python scripts/execucao/ETAPA8_AVALIAR_F10_PCA_10PCT.py --executar-f10 --confirmacao AUTORIZO_AVALIAR_F10_10PCT

O modo --verificar compara todos os hashes congelados e le apenas arquivos
pre-F10. O modo --executar-f10 le TARGET do F10 para aplicar estratificacao
PRE-REGISTRADA, faz inferencia usando os 3 modelos congelados, avalia AP,
LogLoss, MCC, Score e CMs com tau=0.5 e tau* OOF, sem ajustar parametros.

Trava contra repeticao por protocolo; nenhuma galeria/OOF/modelo e alterado.
O F10 e usado uma unica vez como conjunto teste amostral. Nao selecionar
modelo ou modificar experimentos com base nessas metricas.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

# Scripts do mesmo diretorio. NAO mexer no codigo de Etapa 7.
from ETAPA7_CONGELAR_PRE_F10_PCA import (
    coletar_e_verificar, ler_json, sha_arquivo, selecionar_ids_f10_estratificado,
)
from TREINAR_FINAL_PCA_F1_F9_10PCT import verificar_origens
from src.folds import carregar_alvo
from src.treinamento_final_pca import carregar_pacote_confiavel
from src.avaliacao_final_pca import avaliar_f10_em_memoria

ARQ_PROTOCOLO = (
    RAIZ / 'outputs' / 'protocolo_pre_f10_pca_10pct_20261029T211817882335Z'
    / 'manifesto_pre_f10.json'
)
SHA_PROTOCOLO_CONGELADO = '670577480189be6713238bc3ff07731aa7abd6e64fb6d8e47477c398c7afab4e'
TOKEN_AUTORIZACAO = 'AUTORIZO_AVALIAR_F10_10PCT'
MARCA_EXECUCAO = 'registro_execucao_f10.json'
CAMPOS_IMUTAVEIS = (
    'versao', 'situacao', 'carater', 'f10_avaliado',
    'f10_rotulos_lidos', 'f10_predicoes_geradas',
    'dados', 'protocolo_teste_pre_registrado',
    'modelos', 'sha256_codigo_pre_f10',
)


def exigir(condicao: bool, motivo: str) -> None:
    if not condicao:
        raise RuntimeError(motivo)


def relativo(arquivo: Path) -> str:
    return arquivo.relative_to(RAIZ).as_posix()


def preflight() -> tuple[dict, Path]:
    """Verifica integridade de tudo que foi CONGELADO, sem consultar labels F10.

    Revalida o Parquet por hash, o arquivo de folds, hashes de codigo e os
    artefatos de modelo. A funcao da Etapa 7 nao carrega registros F10.
    """
    print('=== ETAPA 8: PREFLIGHT SEM ACESSAR ROTULOS F10 ===', flush=True)
    exigir(sha_arquivo(ARQ_PROTOCOLO) == SHA_PROTOCOLO_CONGELADO,
           'Hash do manifesto pre-F10 alterado! Interrompendo.')
    congelado = ler_json(ARQ_PROTOCOLO)
    exigir(congelado.get('situacao') == 'PREFLIGHT_CONGELADO_SEM_AVALIACAO',
           'Estado pre-F10 incompatível')
    exigir(congelado.get('f10_avaliado') is False, 'F10 ja marcado como avaliado')
    atualizado = coletar_e_verificar()
    for campo in CAMPOS_IMUTAVEIS:
        exigir(congelado.get(campo) == atualizado.get(campo),
               f'Campo congelado divergente do estado atual: {campo}')
    exigir(congelado['protocolo_teste_pre_registrado']['numero_execucoes_f10_planejadas'] == 1,
           'Protocolo nao autoriza mais de uma avaliacao')
    assert len(congelado['modelos']) == 3
    marcador = ARQ_PROTOCOLO.parent / MARCA_EXECUCAO
    exigir(not marcador.exists(),
           f'F10 ja iniciado/avaliado sob este protocolo: {marcador}. Nao repetir.')
    print('[APROVADO] Manifesto pre-F10 SHA-256 inalterado.', flush=True)
    print('[APROVADO] Modelos, limiares, folds, dados e codigo congelados.', flush=True)
    print('[APROVADO] Nenhuma avaliacao F10 anterior registrada neste protocolo.', flush=True)
    return congelado, marcador


def ler_linhas_parquet(parquet: Path, posicoes: np.ndarray) -> pd.DataFrame:
    """Le somente as linhas solicitadas em batches, sem carregar o parquet inteiro."""
    import pyarrow.parquet as pq
    posicoes = np.asarray(posicoes)
    exigir(posicoes.ndim == 1 and len(posicoes) > 0
           and np.issubdtype(posicoes.dtype, np.integer)
           and posicoes[0] >= 0 and np.all(np.diff(posicoes) > 0),
           'Posicoes precisam ser inteiras, unicas, crescentes e nao vazias')
    pf = pq.ParquetFile(parquet)
    exigir(posicoes[-1] < pf.metadata.num_rows, 'Indice excede o Parquet')
    partes = []
    inicio = 0
    encontrados = 0
    for lote in pf.iter_batches(batch_size=20_000):
        fim = inicio + lote.num_rows
        i = int(np.searchsorted(posicoes, inicio, side='left'))
        j = int(np.searchsorted(posicoes, fim, side='left'))
        if i < j:
            locais = posicoes[i:j] - inicio
            partes.append(lote.to_pandas().iloc[locais].copy())
            encontrados += j - i
        inicio = fim
    exigir(encontrados == len(posicoes), 'Leitura incompleta de linhas F10')
    df = pd.concat(partes, ignore_index=True)
    exigir(len(df) == len(posicoes) and df.index.equals(pd.RangeIndex(len(df))),
           'Alinhamento posicional de F10 foi alterado')
    return df


def selecionar_f10(congelado: dict, folds: np.ndarray, y_completo: np.ndarray) -> np.ndarray:
    """Aplica ESTRITAMENTE a funcao ja congelada no codigo da Etapa 7."""
    dados = congelado['dados']
    exigir(len(folds) == dados['numero_total_linhas'] == len(y_completo),
           'Tamanho do alvo ou dos folds divergente do protocolo')
    exigir(int(np.count_nonzero(folds == 10)) == dados['numero_linhas_f10_integral'],
           'Numero de linhas F10 alterado')
    idx_f10 = np.flatnonzero(folds == 10)
    y_f10 = y_completo[idx_f10]
    exigir(np.isin(y_f10, [0, 1]).all() and np.unique(y_f10).size == 2,
           'F10 precisa conter ambas as classes 0/1')
    escolhidos = selecionar_ids_f10_estratificado(idx_f10, y_f10)
    exigir(np.all(folds[escolhidos] == 10), 'Vazamento! Amostra inclui folds de treino')
    esperado_por_classe = {
        c: max(2, round(float(congelado['protocolo_teste_pre_registrado']['fracao_teste'])
                        * int(np.count_nonzero(y_f10 == c))))
        for c in (0, 1)
    }
    for c, n in esperado_por_classe.items():
        exigir(int(np.count_nonzero(y_completo[escolhidos] == c)) == n,
               f'Amostra da classe {c} nao segue o protocolo congelado')
    return escolhidos


def iniciar_execucao(marcador: Path, destino: Path) -> None:
    """Criacao atomica 'x': impede reexecutar F10 por acidente."""
    registro = {
        'estado': 'INICIADO',
        'criado_utc': datetime.now(timezone.utc).isoformat(),
        'protocolo_pre_f10': relativo(ARQ_PROTOCOLO),
        'sha256_protocolo': SHA_PROTOCOLO_CONGELADO,
        'destino': relativo(destino),
        'sha256_codigo_etapa8': sha_arquivo(Path(__file__).resolve()),
        'observacao': 'Nao remover esta trava nem repetir a avaliacao por resultados desfavoraveis.',
    }
    with marcador.open('x', encoding='utf-8') as f:
        json.dump(registro, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def salvar_json_exclusivo(arquivo: Path, obj: dict) -> None:
    with arquivo.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def executar_f10(congelado: dict, marcador: Path) -> None:
    # A TRAVA e escrita antes de ler qualquer rotulo de F10.
    agora = datetime.now(timezone.utc)
    destino = RAIZ / 'outputs' / ('avaliacao_final_f10_pca_10pct_' + agora.strftime('%Y%m%dT%H%M%S%fZ'))
    exigir(not destino.exists(), 'Pasta destino ja existente')
    iniciar_execucao(marcador, destino)
    destino.mkdir(parents=False, exist_ok=False)
    print('\n=== AVALIACAO F10 AMOSTRAL INICIADA ===', flush=True)
    print(f'[DESTINO] {destino}', flush=True)

    # Reconhecer folds com as verificacoes existentes; nunca criar splits.
    fontes, folds = verificar_origens()
    ref_cfg = fontes[1]['config']
    parquet = RAIZ / congelado['dados']['parquet']
    alvo = ref_cfg['dados']['coluna_alvo']
    print('[F10] Lendo somente a coluna-alvo para selecionar a amostra pre-registrada.', flush=True)
    y_completo = carregar_alvo(parquet, alvo)
    selecionadas = selecionar_f10(congelado, folds, y_completo)
    n0 = int(np.count_nonzero(y_completo[selecionadas] == 0))
    n1 = int(np.count_nonzero(y_completo[selecionadas] == 1))
    print(f'[F10] Amostra estratificada fixa: {len(selecionadas)} registros; legitimas={n0}; fraudes={n1}.', flush=True)
    del y_completo
    df = ler_linhas_parquet(parquet, selecionadas)
    exigir(alvo in df.columns, 'Coluna alvo nao encontrada em F10')
    exigir(len(df) == len(selecionadas), 'Parquet F10 com linhas diferentes da amostra')
    exigir(np.isin(df[alvo].to_numpy(), [0, 1]).all(), 'Rotulos F10 invalidos')
    exigir(int((df[alvo] == 0).sum()) == n0 and int((df[alvo] == 1).sum()) == n1,
           'Rotulos de F10 desalinhados apos leitura do Parquet')
    print('[F10] Integridade da amostra e do TARGET confirmada.', flush=True)

    amostra = {
        'tipo': 'AMOSTRA_F10_10PCT_ESTRATIFICADA_PRE_REGISTRADA',
        'n_linhas': int(len(selecionadas)),
        'n_legitimas': n0,
        'n_fraudes': n1,
        'semente': 42,
        'fracao': 0.1,
        'hash_sha256_indices': __import__('hashlib').sha256(selecionadas.tobytes()).hexdigest(),
    }
    salvar_json_exclusivo(destino / 'amostra_f10.json', amostra)
    np.save(destino / 'indices_f10_amostra.npy', selecionadas, allow_pickle=False)

    resumo = []
    for item in sorted(congelado['modelos'], key=lambda x: x['pca_dimensao']):
        dim = item['pca_dimensao']
        print(f'\n=== PCA {dim}D — AVALIACAO FINAL F10 ===', flush=True)
        pasta_modelo = (RAIZ / item['modelo_final']).parent
        pacote = carregar_pacote_confiavel(pasta_modelo)
        exigir(pacote.limiar_oof == item['limiar_mcc_oof_congelado'],
               f'PCA {dim}D: limiar divergente da escolha OOF')
        tabela, metricas = avaliar_f10_em_memoria(
            pacote, df, folds[selecionadas], selecionadas,
        )
        exigir(metricas['limiar_otimizado_no_f10'] is False,
               f'PCA {dim}D: limiar foi otimizado no teste')
        exigir(metricas['limiar_mcc_oof_congelado'] == item['limiar_mcc_oof_congelado'],
               f'PCA {dim}D: tau mudou')
        pasta_dim = destino / f'PCA_{dim}D'
        pasta_dim.mkdir(exist_ok=False)
        tabela.to_csv(pasta_dim / 'predicoes_f10.csv', index=False, float_format='%.17g')
        salvar_json_exclusivo(pasta_dim / 'metricas_f10.json', metricas)
        resumo.append({
            'pca_dimensao': dim,
            'candidato_oof': item['candidato_oof'],
            'limiar_mcc_oof_congelado': item['limiar_mcc_oof_congelado'],
            'ap': metricas['ap'],
            'mcc_limiar_oof': metricas['mcc_limiar_oof'],
            'logloss': metricas['logloss'],
            'score_composto_limiar_oof': metricas['score_composto_limiar_oof'],
            'predicoes_f10': relativo(pasta_dim / 'predicoes_f10.csv'),
            'sha256_predicoes_f10': sha_arquivo(pasta_dim / 'predicoes_f10.csv'),
            'metricas_f10': relativo(pasta_dim / 'metricas_f10.json'),
            'sha256_metricas_f10': sha_arquivo(pasta_dim / 'metricas_f10.json'),
        })
        print(f'[SALVO] PCA {dim}D | AP={metricas["ap"]:.7f} | MCC(tau*)={metricas["mcc_limiar_oof"]:.7f} | '
              f'LogLoss={metricas["logloss"]:.7f} | S={metricas["score_composto_limiar_oof"]:.7f}',
              flush=True)
        del tabela, metricas, pacote
        gc.collect()

    final = {
        'versao': 'PCA_F10_10PCT_V1',
        'carater': 'AVALIACAO_EXPLORATORIA_AMOSTRAL',
        'f10_avaliado': True,
        'rotulos_f10_utilizados_apenas_para_amostragem_e_metricas': True,
        'otimizacao_limiar_em_f10': False,
        'selecionar_vencedor_em_f10': False,
        'referencia_protocolo': relativo(ARQ_PROTOCOLO),
        'sha256_protocolo_pre_f10': SHA_PROTOCOLO_CONGELADO,
        'sha256_codigo_executor_etapa8': sha_arquivo(Path(__file__).resolve()),
        'criado_em_utc': agora.isoformat(),
        'amostra_f10': amostra,
        'resultados_por_dimensao': resumo,
    }
    salvar_json_exclusivo(destino / 'manifesto_avaliacao_f10.json', final)
    print('\n[CONCLUIDO] Avaliacao amostral F10 dos 3 modelos, sem treino e sem reotimizar tau.', flush=True)
    print(f'[MANIFESTO] {destino / "manifesto_avaliacao_f10.json"}', flush=True)
    print('ATENCAO: F10 ja foi acessado. Nao repetir nem usar suas metricas para selecionar outro modelo.', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modo = parser.add_mutually_exclusive_group(required=True)
    modo.add_argument('--verificar', action='store_true', help='Somente preflight; sem ler rotulos F10')
    modo.add_argument('--executar-f10', action='store_true', help='Acessa F10 REAL uma unica vez (exige confirmacao)')
    parser.add_argument('--confirmacao', default=None,
                        help='Exigido em --executar-f10: AUTORIZO_AVALIAR_F10_10PCT')
    args = parser.parse_args()
    if args.executar_f10:
        exigir(args.confirmacao == TOKEN_AUTORIZACAO,
               'Nao foi fornecida autorizacao explicita correta. F10 nao sera acessado.')
    else:
        exigir(args.confirmacao is None, 'Confirmacao so faz sentido com --executar-f10.')
    congelado, marcador = preflight()
    if args.verificar:
        print('\n[PREFLIGHT APROVADO] Nenhum rotulo F10 lido e nenhum arquivo alterado.', flush=True)
        print('A avaliacao real necessita de uma decisao e um comando separados.', flush=True)
    else:
        executar_f10(congelado, marcador)


if __name__ == '__main__':
    main()
