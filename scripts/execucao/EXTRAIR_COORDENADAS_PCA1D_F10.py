# -*- coding: utf-8 -*-
"""Complemento geométrico do PCA 1D, sem repetir a avaliação F10.

--verificar: confere índices, hashes, origem e ausência de sobreposição; não abre
            o Parquet para ler registros, não carrega o modelo joblib e não grava.
--extrair:   relê APENAS FEATURES das MESMAS 18.523 linhas F10, transforma
            usando o pipeline F1-F9 congelado e salva coordenadas PC1 e resumo
            do pré-processamento em vencedor/teste_final/dados_geometria/.

NUNCA treina, ajusta scaler/PCA, chama preditor GMM, lê TARGET do Parquet,
recalcula limiar/métricas, altera arquivos anteriores ou reamostra F10.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

F10 = RAIZ / 'outputs/avaliacao_final_f10_pca_10pct_20261029T232726363532Z'
ENSAIO = RAIZ / 'outputs/ensaio_pca_gmm_20261029_173123'
MODELO = RAIZ / 'outputs/treinamento_final_pca_10pct_20261029_174241/PCA_1D'
SAIDA = ENSAIO / 'vencedor/teste_final/dados_geometria'


def exigir(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise RuntimeError(mensagem)


def ler_json(p: Path) -> dict:
    exigir(p.is_file(), f'Arquivo ausente: {p}')
    obj = json.loads(p.read_text(encoding='utf-8-sig'))
    exigir(isinstance(obj, dict), f'JSON incorreto: {p}')
    return obj


def sha_arquivo(p: Path) -> str:
    exigir(p.is_file(), f'Arquivo ausente: {p}')
    h = sha256()
    with p.open('rb') as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def relativo(p: Path) -> str:
    return p.relative_to(RAIZ).as_posix()


def validar_indices(indices, predicoes: pd.DataFrame, n: int) -> np.ndarray:
    """Indices originais devem ter a mesma ordem que as predições já avaliadas."""
    ids = np.asarray(indices)
    exigir(ids.ndim == 1 and len(ids) == n and len(ids) > 0,
           'Quantidade de indices diferente da amostra F10 original.')
    exigir(np.issubdtype(ids.dtype, np.integer) and np.all(ids >= 0),
           'Indices F10 devem ser inteiros nao negativos.')
    exigir(np.all(np.diff(ids) > 0), 'Indices F10 devem ser unicos e crescentes.')
    exigir(list(predicoes.columns) == ['linha_original', 'fold'],
           'Colunas de alinhamento das predicoes F10 inesperadas.')
    exigir(len(predicoes) == n and predicoes['linha_original'].notna().all()
           and predicoes['fold'].notna().all(), 'Predicoes F10 desalinhadas.')
    exigir(np.array_equal(predicoes['linha_original'].to_numpy(dtype=np.int64), ids)
           and (predicoes['fold'].to_numpy() == 10).all(),
           'Indices do arquivo NPY e das predicoes F10 nao coincidem.')
    return ids.astype(np.int64, copy=False)


def conferir() -> dict:
    """Pré-verificação sem acessar o Parquet por linhas nem carregar o joblib."""
    print('=== PCA 1D / F10 — CONFERENCIA DAS FONTES JA CONGELADAS ===', flush=True)
    f10meta = ler_json(F10 / 'manifesto_avaliacao_f10.json')
    exigir(f10meta.get('f10_avaliado') is True
           and f10meta.get('otimizacao_limiar_em_f10') is False,
           'Manifesto de avaliacao F10 incompativel.')
    exigir(f10meta.get('carater') == 'AVALIACAO_EXPLORATORIA_AMOSTRAL',
           'Avaliacao nao identificada como exploratoria.')
    protocolo = (RAIZ / f10meta['referencia_protocolo']).resolve()
    exigir(protocolo.is_relative_to((RAIZ/'outputs').resolve()),
           'Protocolo fora de outputs.')
    exigir(sha_arquivo(protocolo) == f10meta['sha256_protocolo_pre_f10'],
           'Hash do protocolo pre-F10 divergente.')
    pre = ler_json(protocolo)
    referencias = [x for x in pre['modelos'] if x['pca_dimensao'] == 1]
    exigir(len(referencias) == 1, 'Referencia PCA 1D ambigua.')
    ref = referencias[0]
    exigir((RAIZ / ref['experimento_oof']).resolve() == ENSAIO.resolve(),
           'O F10 original foi avaliado por modelo vinculado a OUTRO ensaio.')
    exigir((RAIZ / ref['modelo_final']).resolve() == (MODELO/'modelo_final_f1_f9.joblib').resolve(),
           'O F10 original utilizou OUTRO modelo final.')
    exigir(sha_arquivo(MODELO/'modelo_final_f1_f9.joblib') == ref['sha256_modelo_final'],
           'Hash do GMM+PCA congelado mudou.')
    exigir(sha_arquivo(MODELO/'manifesto_modelo_final.json') == ref['sha256_manifesto_modelo'],
           'Manifesto do modelo foi alterado.')
    m_model = ler_json(MODELO/'manifesto_modelo_final.json')
    exigir(m_model['sha256_modelo_final'] == ref['sha256_modelo_final']
           and m_model['dimensao_pca'] == 1 and m_model['n_treino_original'] == 166716,
           'Manifesto do PCA 1D inconsistente.')
    escolha = ler_json(ENSAIO/'vencedor/metricas/melhor_configuracao.json')
    exigir(float(ref['limiar_mcc_oof_congelado']) == float(escolha['limiar_mcc_oof'])
           == float(m_model['limiar_mcc_oof_congelado']),
           'Limiar OOF alterado.')
    result = [x for x in f10meta['resultados_por_dimensao'] if x['pca_dimensao'] == 1]
    exigir(len(result) == 1 and result[0]['candidato_oof'] == escolha['id_candidato'],
           'Resultado F10 nao corresponde ao candidato PCA 1D.')
    r = result[0]
    csv = RAIZ / r['predicoes_f10']
    exigir(sha_arquivo(csv) == r['sha256_predicoes_f10'], 'Hash das predicoes F10 mudou.')
    exigir(sha_arquivo(RAIZ / r['metricas_f10']) == r['sha256_metricas_f10'],
           'Hash das metricas F10 mudou.')
    indice = F10 / 'indices_f10_amostra.npy'
    exigir(indice.is_file(), 'indices_f10_amostra.npy ausente. Nao reamostrar.')
    ids = np.load(indice, allow_pickle=False)
    exigir(sha256(ids.tobytes()).hexdigest() == f10meta['amostra_f10']['hash_sha256_indices'],
           'Hash da amostra F10 diferente do protocolo usado na avaliacao.')
    alinhamento = pd.read_csv(csv, usecols=['linha_original', 'fold'])[['linha_original', 'fold']]
    ids = validar_indices(ids, alinhamento, int(f10meta['amostra_f10']['n_linhas']))
    exigir(len(ids) == 18523, 'Quantidade de F10 diferente da avaliacao ja realizada.')
    # Compara arquivo original apenas por hash; sem ler seus registros.
    parquet = (RAIZ / pre['dados']['parquet']).resolve()
    exigir(parquet.is_relative_to(RAIZ.resolve()), 'Parquet fora da raiz do projeto.')
    exigir(sha_arquivo(parquet) == pre['dados']['sha256_parquet'],
           'O Parquet original mudou: extracao bloqueada.')
    exigir(m_model['proveniencia']['sha256_parquet'] == pre['dados']['sha256_parquet'],
           'Parquet do modelo e da avaliacao F10 divergentes.')
    exigir(not SAIDA.exists(), f'Ja existe pasta de geometria: {SAIDA}. Preservada.')
    print(f'[APROVADO] F10 ja avaliado: n={len(ids)}; IDs identicos aos da avaliacao.', flush=True)
    print(f'[APROVADO] PCA 1D: modelo SHA-256, ensaio e tau*={escolha["limiar_mcc_oof"]:.17g}.', flush=True)
    print('[APROVADO] Parquet original integro por hash; sem ler TARGET nem registros F10.', flush=True)
    print('[APROVADO] Nenhuma pasta ou arquivo alterado.', flush=True)
    return {'ids': ids, 'parquet': parquet, 'ref': ref, 'escolha': escolha,
            'f10meta': f10meta, 'pre': pre, 'csv': csv}


def ler_features_posicoes(parquet: Path, posicoes: np.ndarray, colunas: list[str]) -> pd.DataFrame:
    """Usa apenas colunas autorizadas, em batches. Não lê TARGET ou IDs originais."""
    import pyarrow.parquet as pq
    exigir(bool(colunas) and len(colunas) == len(set(colunas)), 'Features duplicadas/vazias.')
    posicoes = np.asarray(posicoes, dtype=np.int64)
    exigir(posicoes.ndim == 1 and len(posicoes) > 0 and np.all(posicoes >= 0)
           and np.all(np.diff(posicoes) > 0), 'Indices originais nao sao validos.')
    arq = pq.ParquetFile(parquet)
    disponiveis = arq.schema_arrow.names
    exigir(all(c in disponiveis for c in colunas), 'Features do treino ausentes no Parquet.')
    exigir(posicoes[-1] < arq.metadata.num_rows, 'Indices excedem a base original.')
    partes = []
    inicio = 0
    count = 0
    for lote in arq.iter_batches(columns=colunas, batch_size=20000):
        fim = inicio + lote.num_rows
        i = int(np.searchsorted(posicoes, inicio, 'left'))
        j = int(np.searchsorted(posicoes, fim, 'left'))
        if i < j:
            locais = (posicoes[i:j] - inicio).astype(np.int64)
            partes.append(lote.to_pandas().iloc[locais].copy())
            count += j-i
        inicio = fim
        if inicio > posicoes[-1]:
            break
    exigir(count == len(posicoes), 'Leitura de F10 incompleta.')
    df = pd.concat(partes, ignore_index=True)
    exigir(list(df.columns) == colunas and len(df) == len(posicoes),
           'Ordem/tamanho das features F10 alterados.')
    return df


def aplicar_transformadores_congelados(pacote, df: pd.DataFrame):
    """Somente .transform; não existe .fit nem cálculo de probabilidade."""
    exigir(isinstance(df, pd.DataFrame) and not df.empty, 'Dados vazios.')
    exigir(list(df.columns) == list(pacote.encoder.colunas_),
           'Ordem/nome das features diferente do PCA 1D treinado.')
    proibidas = [pacote.configuracao['dados']['coluna_alvo'],
                pacote.configuracao['dados'].get('coluna_id'),
                *pacote.configuracao['dados'].get('colunas_excluir', [])]
    exigir(not any(c in df.columns for c in proibidas if c),
           'Features receberam TARGET, ID ou coluna excluida.')
    x_enc = pacote.encoder.transform(df)
    x_z = pacote.imputador.transform(x_enc)
    coordenadas = pacote.pca.transform(x_z)
    exigir(list(coordenadas.columns) == ['PC1'], 'O modelo nao e PCA 1D.')
    exigir(len(coordenadas) == len(df) and coordenadas.index.equals(df.index),
           'Ordem das coordenadas PC1 mudou.')
    exigir(np.isfinite(coordenadas['PC1'].to_numpy(dtype=float)).all(),
           'PC1 contem valores invalidos.')
    return x_z, coordenadas


def resumo_preprocessamento(pacote, df_raw: pd.DataFrame,
                            x_z: pd.DataFrame) -> pd.DataFrame:
    imp = pacote.imputador
    nomes = list(imp.padronizadas_)
    exigir(imp.scaler_ is not None and len(nomes) == len(imp.scaler_.mean_),
           'Scaler treinado nao corresponde as features padronizadas.')
    rows=[]
    fe = pacote.configuracao['preprocessamento']['encodings']['frequency']['sufixo']
    for j,nome in enumerate(nomes):
        z = x_z[nome].to_numpy(dtype=np.float64)
        mu0 = float(imp.scaler_.mean_[j]); sd0 = float(imp.scaler_.scale_[j])
        exigir(np.isfinite(z).all() and sd0 > 0, f'Scaler invalido: {nome}.')
        rows.append({
            'feature': nome,
            'tipo': 'FEWF' if nome.endswith(fe) else 'numerica',
            'n_f10': int(len(z)),
            'media_treino_f1_f9_codificada': mu0,
            'escala_treino_f1_f9': sd0,
            'media_z_f10': float(z.mean()),
            'desvio_z_f10': float(z.std(ddof=0)),
            'variancia_z_f10': float(z.var(ddof=0)),
            'drift_media_em_escala_treino': float(z.mean()),
            'percentual_ausentes_brutos_f10': float(df_raw[nome].isna().mean()*100),
        })
    return pd.DataFrame(rows)


def extrair(ver: dict) -> None:
    # Só agora carrega o modelo local, validado por hashes no --verificar.
    from src.treinamento_final_pca import carregar_pacote_confiavel
    pacote = carregar_pacote_confiavel(MODELO)
    exigir(pacote.n_treino_original == 166716 and pacote.limiar_oof == ver['ref']['limiar_mcc_oof_congelado'],
           'Modelo carregado nao coincide com a selecao original.')
    print('[F10] Lendo exclusivamente features das mesmas 18.523 linhas; TARGET NAO lido.', flush=True)
    df = ler_features_posicoes(ver['parquet'], ver['ids'], list(pacote.encoder.colunas_))
    x_z, coords = aplicar_transformadores_congelados(pacote, df)
    resumo = resumo_preprocessamento(pacote, df, x_z)
    saida = pd.DataFrame({'linha_original': ver['ids'], 'PC1': coords['PC1'].to_numpy(dtype=np.float64)})
    exigir(np.array_equal(saida['linha_original'], ver['ids']), 'Alinhamento PC1 invalido.')
    exigir(np.isfinite(saida['PC1']).all(), 'PC1 nao finito.')
    # Prepara tudo em diretório temporário; não altera saídas anteriores.
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    temporario = Path(tempfile.mkdtemp(prefix='.temporario_pca1d_f10_', dir=SAIDA.parent))
    try:
        arq_coords = temporario/'coordenadas_PC1_f10.csv'
        arq_resumo = temporario/'resumo_padronizacao_f10.csv'
        saida.to_csv(arq_coords, index=False, float_format='%.17g')
        resumo.to_csv(arq_resumo, index=False, float_format='%.17g')
        manifest = {
            'versao': 'PCA1D_F10_GEOMETRIA_DADOS_V1',
            'tipo': 'EXTRACAO_DESCRITIVA_F10_JA_AVALIADO',
            'original_f10': relativo(F10/'manifesto_avaliacao_f10.json'),
            'sha256_manifesto_f10': sha_arquivo(F10/'manifesto_avaliacao_f10.json'),
            'modelo_final_f1_f9': relativo(MODELO/'modelo_final_f1_f9.joblib'),
            'sha256_modelo_final': ver['ref']['sha256_modelo_final'],
            'experimento_oof': relativo(ENSAIO),
            'n': int(len(ver['ids'])),
            'hash_sha256_indices': ver['f10meta']['amostra_f10']['hash_sha256_indices'],
            'limiar_mcc_oof_congelado': ver['ref']['limiar_mcc_oof_congelado'],
            'dimensao': 1,
            'features_lidas_parquet': list(pacote.encoder.colunas_),
            'parquet_target_lido': False,
            'novo_treinamento': False,
            'nova_inferencia_probabilidades': False,
            'nova_avaliacao_f10': False,
            'recalculo_limiar': False,
            'arquivos': {
                'coordenadas_PC1_f10.csv': sha_arquivo(arq_coords),
                'resumo_padronizacao_f10.csv': sha_arquivo(arq_resumo),
            },
        }
        (temporario/'manifesto_geometria_f10.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        exigir(not SAIDA.exists(), 'O destino ja existe; nao sobrescrever.')
        os.replace(temporario, SAIDA)
    except Exception:
        print(f'[ERRO] Etapa parcial preservada para inspecao: {temporario}', flush=True)
        raise
    print(f'[SALVO] Coordenadas PCA 1D de {len(ver["ids"])} linhas: {SAIDA/"coordenadas_PC1_f10.csv"}', flush=True)
    print(f'[SALVO] Diagnostico padronizacao: {SAIDA/"resumo_padronizacao_f10.csv"}', flush=True)
    print(f'[SALVO] Manifesto: {SAIDA/"manifesto_geometria_f10.json"}', flush=True)
    print('[CONCLUIDO] F10 nao foi reavaliado; valores preditivos originais preservados.', flush=True)


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__)
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument('--verificar', action='store_true', help='Somente auditoria; nao le linhas F10')
    g.add_argument('--extrair', action='store_true', help='Le features F10 uma vez, sem TARGET, sem predicao')
    a.add_argument('--confirmacao', default='', help='Token para a extracao de features F10')
    args = a.parse_args()
    if args.extrair:
        exigir(args.confirmacao == 'AUTORIZO_PROJETAR_FEATURES_F10_SEM_AVALIAR',
               'Extração exige confirmacao explicita. --verificar e seguro.')
    estado = conferir()
    if args.extrair:
        extrair(estado)


if __name__ == '__main__':
    main()
