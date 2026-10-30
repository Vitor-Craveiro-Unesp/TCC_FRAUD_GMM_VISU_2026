# -*- coding: utf-8 -*-
"""Etapa 7/9: congela o protocolo PCA/GMM ANTES da primeira avaliacao F10.

Instalar: scripts/execucao/ETAPA7_CONGELAR_PRE_F10_PCA.py
Uso (da raiz):
    python scripts/execucao/ETAPA7_CONGELAR_PRE_F10_PCA.py --verificar
    python scripts/execucao/ETAPA7_CONGELAR_PRE_F10_PCA.py --congelar

--verificar: apenas verificacao, sem alterar nenhum arquivo.
--congelar: executa a verificacao e cria UM manifesto JSON em pasta nova.

Nao extrai linhas de F10 do Parquet, nao le TARGET de F10 e nao faz
predicoes ou treinamento. Calcula SHA-256 do Parquet completo (bytes),
le folds.npy e os artefatos F1-F9 previamente aprovados.

A regra de amostragem F10 e PRE-REGISTRADA, mas nenhuma amostra ou
rotulo F10 e selecionado/lido nesta etapa. O executor F10 ainda nao existe.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

# Reutiliza o mesmo validador F1-F9 que passou no pre-treinamento,
# evitando implementar uma segunda logica divergente.
from TREINAR_FINAL_PCA_F1_F9_10PCT import FONTES, N_ESPERADO, verificar_origens

PASTA_MODELOS = RAIZ / 'outputs' / 'treinamento_final_pca_10pct_20261029_174241'
N_DIMENSOES = (1, 2, 3)
SEMENTE_F10 = 42
FRACAO_F10 = 0.10


def exigir(condicao, mensagem):
    if not condicao:
        raise RuntimeError(mensagem)


def ler_json(caminho: Path) -> dict:
    exigir(caminho.is_file(), f'Arquivo ausente: {caminho}')
    obj = json.loads(caminho.read_text(encoding='utf-8-sig'))
    exigir(isinstance(obj, dict), f'JSON invalido: {caminho}')
    return obj


def sha_arquivo(caminho: Path) -> str:
    exigir(caminho.is_file(), f'Arquivo ausente: {caminho}')
    h = sha256()
    with caminho.open('rb') as entrada:
        for bloco in iter(lambda: entrada.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def relativo(caminho: Path) -> str:
    return caminho.relative_to(RAIZ).as_posix()


def selecionar_ids_f10_estratificado(ids_f10, rotulos_f10) -> np.ndarray:
    """Regra PRE-REGISTRADA para a ETAPA 8 (NAO invocada na etapa 7).

    ids_f10: indices originais, todos autenticados como fold 10 pelo executor.
    rotulos_f10: 0/1, alinhados a esses indices, lidos somente apos autorizacao.
    Devolve indices originais crescentes de 10% por classe, seed 42.
    NAO deve ser chamada na etapa 7: nao existem rotulos F10 nesta etapa.
    """
    ids = np.asarray(ids_f10)
    y = np.asarray(rotulos_f10)
    if (ids.ndim != 1 or y.shape != ids.shape or len(ids) == 0
            or not np.issubdtype(ids.dtype, np.integer)
            or not np.all(ids >= 0) or not np.all(np.diff(ids) > 0)
            or not np.isin(y, [0, 1]).all()):
        raise ValueError('Indices F10 crescentes e labels binarios alinhados obrigatorios.')
    rng = np.random.default_rng(SEMENTE_F10)
    escolha = []
    for classe in (0, 1):
        posicoes = ids[y == classe]
        n = max(2, round(FRACAO_F10 * len(posicoes)))
        if n > len(posicoes):
            raise ValueError('Classe insuficiente para amostra F10 estratificada.')
        escolha.append(rng.choice(posicoes, size=n, replace=False))
    return np.sort(np.concatenate(escolha)).astype(np.int64, copy=False)


def coletar_e_verificar() -> dict:
    print('=== ETAPA 7/9: PROTOCOLO PRE-F10 ===', flush=True)
    print('Sem treino, sem previsao e sem leitura de registros/rotulos F10.', flush=True)
    print('Validando OOF, SHA-256 do Parquet, folds e 3 modelos finais...', flush=True)
    exigir(PASTA_MODELOS.is_dir(), 'Pasta de modelos finais nao encontrada.')

    fontes, folds_originais = verificar_origens()
    exigir(set(fontes) == set(N_DIMENSOES), 'PCA 1D, 2D e 3D obrigatorios.')
    exigir(folds_originais.ndim == 1, 'folds.npy precisa ser um vetor.')
    exigir(np.isin(folds_originais, np.arange(1, 11)).all(), 'Folds fora de 1..10.')

    n_f10 = int(np.count_nonzero(folds_originais == 10))
    n_f1_f9_base = int(np.count_nonzero(folds_originais != 10))
    exigir(n_f10 > 0 and n_f1_f9_base > N_ESPERADO,
           'Quantidade de linhas por fold inconsistente.')

    ref = fontes[1]
    # Confronto de hashes com os valores ja guardados nas execucoes OOF.
    hash_parquet = ref['manifest']['sha256_parquet']
    hash_folds = ref['manifest']['sha256_folds']
    modelos = []

    resumo_anterior = ler_json(PASTA_MODELOS / 'RESUMO_MODELOS_FINAIS.json')
    exigir(resumo_anterior.get('f10_utilizado') is False, 'Resumo aponta F10 utilizado.')
    exigir(len(resumo_anterior.get('modelos', [])) == 3, 'Resumo sem os 3 modelos.')

    for dim in N_DIMENSOES:
        origem = fontes[dim]
        pasta_modelo = PASTA_MODELOS / f'PCA_{dim}D'
        modelo = pasta_modelo / 'modelo_final_f1_f9.joblib'
        arq_manifesto = pasta_modelo / 'manifesto_modelo_final.json'
        manifest = ler_json(arq_manifesto)
        escolha = origem['escolha']
        prov = manifest.get('proveniencia') or {}

        exigir(manifest.get('teste_f10_executado') is False,
               f'PCA {dim}D: F10 foi marcado como executado.')
        exigir(manifest.get('dimensao_pca') == dim, f'PCA {dim}D: dimensao divergente.')
        exigir(manifest.get('n_treino_original') == N_ESPERADO,
               f'PCA {dim}D: quantidade de treino divergente.')
        exigir(float(manifest.get('fracao_base', -1)) == 0.1,
               f'PCA {dim}D: fracao de desenvolvimento divergente.')
        exigir(manifest.get('n_ajuste_gmm') == 14790,
               f'PCA {dim}D: tamanho de amostra RUS inesperado.')
        exigir(manifest.get('id_candidato_oof') == escolha['id_candidato'],
               f'PCA {dim}D: candidato nao e o aprovado em OOF.')
        exigir(manifest.get('limiar_mcc_oof_congelado') == escolha['limiar_mcc_oof'],
               f'PCA {dim}D: limiar tau nao e o aprovado em OOF.')
        exigir(prov.get('experimento_oof') == relativo(origem['pasta']),
               f'PCA {dim}D: proveniencia do OOF divergente.')
        exigir(prov.get('sha256_parquet') == hash_parquet,
               f'PCA {dim}D: hash do Parquet nao coincide.')
        exigir(prov.get('sha256_folds') == hash_folds,
               f'PCA {dim}D: hash dos folds nao coincide.')
        exigir(prov.get('sha256_yaml') == origem['manifest']['sha256_yaml'],
               f'PCA {dim}D: hash de YAML de execucao OOF divergente.')
        hash_npz = sha_arquivo(origem['arquivo_npz'])
        exigir(prov.get('sha256_predicoes_oof') == hash_npz,
               f'PCA {dim}D: as predicoes OOF foram alteradas.')
        hash_modelo = sha_arquivo(modelo)
        exigir(manifest.get('sha256_modelo_final') == hash_modelo,
               f'PCA {dim}D: hash de modelo final divergente.')

        escolhido = [x for x in resumo_anterior['modelos'] if x.get('dimensao') == dim]
        exigir(len(escolhido) == 1, f'PCA {dim}D: resumo final ambiguo.')
        exigir(escolhido[0].get('limiar_mcc_oof') == escolha['limiar_mcc_oof'],
               f'PCA {dim}D: resumo altera tau.')
        exigir(escolhido[0].get('modelo') == relativo(modelo),
               f'PCA {dim}D: resumo indica outro modelo.')

        modelos.append({
            'pca_dimensao': dim,
            'candidato_oof': escolha['id_candidato'],
            'limiar_mcc_oof_congelado': float(escolha['limiar_mcc_oof']),
            'experimento_oof': relativo(origem['pasta']),
            'manifesto_oof': relativo(origem['pasta'] / 'manifesto.json'),
            'sha256_manifesto_oof': sha_arquivo(origem['pasta'] / 'manifesto.json'),
            'melhor_configuracao_oof': relativo(origem['pasta'] / 'vencedor/metricas/melhor_configuracao.json'),
            'sha256_escolha_oof': sha_arquivo(origem['pasta'] / 'vencedor/metricas/melhor_configuracao.json'),
            'predicoes_oof': relativo(origem['arquivo_npz']),
            'sha256_predicoes_oof': hash_npz,
            'modelo_final': relativo(modelo),
            'sha256_modelo_final': hash_modelo,
            'manifesto_modelo': relativo(arq_manifesto),
            'sha256_manifesto_modelo': sha_arquivo(arq_manifesto),
            'registros_treino_f1_f9': N_ESPERADO,
            'registros_ajuste_gmm_rus': manifest['n_ajuste_gmm'],
        })
        print(f'[APROVADO] PCA {dim}D: OOF, modelo, hash e tau congelados.', flush=True)

    caminhos_codigo = [
        'src/treinamento_final_pca.py',
        'src/avaliacao_final_pca.py',
        'src/orquestracao_oof.py',
        'src/folds.py',
        'scripts/execucao/TREINAR_FINAL_PCA_F1_F9_10PCT.py',
        'scripts/execucao/ETAPA7_CONGELAR_PRE_F10_PCA.py',
    ]
    hashes_codigo = {p: sha_arquivo(RAIZ / p) for p in caminhos_codigo}

    manifest = {
        'versao': 'TCC_PCA_PRE_F10_V1',
        'situacao': 'PREFLIGHT_CONGELADO_SEM_AVALIACAO',
        'carater': 'ENSAIO_AMOSTRAL_10_POR_CENTO',
        'f10_avaliado': False,
        'f10_rotulos_lidos': False,
        'f10_predicoes_geradas': False,
        'dados': {
            'parquet': ref['config']['dados']['arquivo'],
            'sha256_parquet': hash_parquet,
            'sha256_folds': hash_folds,
            'numero_total_linhas': int(len(folds_originais)),
            'numero_linhas_f1_f9_integral': n_f1_f9_base,
            'numero_linhas_f10_integral': n_f10,
            'n_treino_amostral_f1_f9': N_ESPERADO,
            'fold_teste': 10,
        },
        'protocolo_teste_pre_registrado': {
            'fold': 10,
            'fracao_teste': FRACAO_F10,
            'semente_amostragem': SEMENTE_F10,
            'amostragem': 'estratificada_por_classe_dentro_do_f10',
            'regra': (
                'Somente na etapa 8, apos autorizacao explicita: separar indices F10 por '
                'TARGET_OMEGA 0 e 1; para cada classe, escolher sem reposicao '
                'max(2, round(0.1 * n_da_classe)) com um unico numpy.random.default_rng(42), '
                'visitando as classes na ordem [0,1]; concatenar e ordenar os indices. '
                'Exigir pelo menos duas linhas de cada classe; nao ajustar o algoritmo '
                'apos observar resultados. Nunca incluir linhas F1..F9 no teste.'
            ),
            'funcao_amostragem_no_codigo_congelado': 'selecionar_ids_f10_estratificado',
            'numero_execucoes_f10_planejadas': 1,
            'relato': 'reportar 1D, 2D e 3D separadamente; F10 nao escolhe vencedor',
            'limiar_de_decisao': 'tau_MCC_OTIMO_OOF_CONGELADO_POR_DIMENSAO',
            'comparar_com_corte_fixo': 0.5,
            'metricas': ['average_precision', 'mcc_em_tau_oof', 'logloss', 'score_composto',
                         'matriz_confusao_tau_0_50', 'matriz_confusao_tau_oof'],
            'proibicoes': ['otimizar_tau_usando_F10', 'selecionar_modelo_por_metricas_F10',
                          'reajustar_transformadores_com_F10', 'reajustar_GMM_com_F10', 'selecionar_vencedor_com_base_em_F10',
                          'refazer_amostragem_apos_ver_resultados'],
        },
        'modelos': modelos,
        'sha256_codigo_pre_f10': hashes_codigo,
    }
    print(f'[APROVADO] F10 tem {n_f10} posicoes nos folds salvos; nenhum rotulo F10 foi lido.', flush=True)
    print('[APROVADO] Protocolo 10% estratificado, semente 42, tau* fixo por modelo.', flush=True)
    return manifest


def congelar(manifest: dict) -> Path:
    agora = datetime.now(timezone.utc)
    destino = RAIZ / 'outputs' / ('protocolo_pre_f10_pca_10pct_' + agora.strftime('%Y%m%dT%H%M%S%fZ'))
    # Exclusividade. Nunca sobrescrever manifestos anteriores ou galerias.
    destino.mkdir(parents=False, exist_ok=False)
    registro = dict(manifest)
    registro['criado_em_utc'] = agora.isoformat()
    arquivo = destino / 'manifesto_pre_f10.json'
    with arquivo.open('x', encoding='utf-8') as f:
        json.dump(registro, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(f'\n[CONGELADO] {arquivo}', flush=True)
    print(f'SHA-256 do manifesto pre-F10: {sha_arquivo(arquivo)}', flush=True)
    return arquivo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument('--verificar', action='store_true', help='Somente leitura, sem gravacoes.')
    modo.add_argument('--congelar', action='store_true', help='Validar e criar UM manifesto novo.')
    args = p.parse_args()
    manifest = coletar_e_verificar()
    if args.congelar:
        congelar(manifest)
    else:
        print('\n[VERIFICACAO APROVADA] Nenhum arquivo foi gerado.', flush=True)
    print('F10 REAL NAO FOI AVALIADO.', flush=True)


if __name__ == '__main__':
    main()
