# -*- coding: utf-8 -*-
"""Etapa 6/9: smoke test real dos modelos finais, SEM LER A BASE ou o F10.

Instalar em: scripts/execucao/VERIFICAR_INFERENCIA_PCA_SEM_F10.py
Executar na raiz: python scripts/execucao/VERIFICAR_INFERENCIA_PCA_SEM_F10.py

Apenas carrega os tres modelos locais confiaveis e as escolhas OOF,
cria 8 observacoes FICTICIAS de features com base no esquema ja aprendido,
exercita prever_sem_rotulos() e simula a API de avaliacao com rotulos
ARTIFICIAIS. Nao acessa data/raw, folds.npy ou observacoes reais de F10.
NAO gera metricas cientificas, nao ajusta modelos, nao salva arquivos,
nao altera codigo, graficos, manifestos nem YAML.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.treinamento_final_pca import carregar_pacote_confiavel
from src.avaliacao_final_pca import avaliar_f10_em_memoria

PASTA_MODELOS = RAIZ / 'outputs' / 'treinamento_final_pca_10pct_20261029_174241'
ORIGENS = {
    1: 'ensaio_pca_gmm_20261029_173123',
    2: 'ensaio_pca_gmm_20261029_172521',
    3: 'ensaio_pca_gmm_20261029_173325',
}
NUM_LINHAS_FICTICIAS = 8


def exigir(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise RuntimeError(mensagem)


def carregar_json(arquivo: Path) -> dict:
    exigir(arquivo.is_file(), f'Arquivo ausente: {arquivo}')
    obj = json.loads(arquivo.read_text(encoding='utf-8-sig'))
    exigir(isinstance(obj, dict), f'JSON invalido: {arquivo}')
    return obj


def _primeiras_categorias(categorias: list | tuple) -> list:
    valores = [x for x in categorias if str(x) != '__TCC_AUSENTE__']
    if not valores:
        valores = list(categorias)
    exigir(bool(valores), 'Nenhuma categoria armazenada no encoder ajustado.')
    return valores


def construir_features_ficticias(pacote, n: int = NUM_LINHAS_FICTICIAS) -> pd.DataFrame:
    """Usa SOMENTE esquema e estatisticas JA APRENDIDAS; nao consulta dataset.

    A origem dos valores e o proprio modelo final; eles sao ARTIFICIAIS,
    nao representam casos reais nem devem gerar metricas publicaveis.
    """
    exigir(isinstance(n, int) and 2 <= n <= 100, 'Quantidade ficticia invalida.')
    encoder = pacote.encoder
    esquema = list(encoder.colunas_)
    exigir(bool(esquema) and len(set(esquema)) == len(esquema), 'Esquema do modelo invalido.')
    dados = {}
    for nome in esquema:
        regra = encoder.regras_.get(nome)
        if regra == 'frequency':
            cats = _primeiras_categorias(list(encoder.freq_[nome]))
            valores = [cats[i % len(cats)] for i in range(n)]
        elif regra == 'binary':
            cats = _primeiras_categorias(list(encoder.binarios_[nome]))
            valores = [cats[i % len(cats)] for i in range(n)]
        elif regra == 'onehot':
            cats = _primeiras_categorias(encoder.categorias_[nome])
            valores = [cats[i % len(cats)] for i in range(n)]
        elif regra in {'numerica', 'ciclica'}:
            # Medianas foram aprendidas no treino F1-F9 e sao numericas.
            mediana = float(pacote.imputador.mediana_.get(nome, 0.0))
            exigir(np.isfinite(mediana), f'Mediana nao finita: {nome}')
            valores = [mediana] * n
        else:
            raise RuntimeError(f'Regra de feature desconhecida: {nome} ({regra})')
        dados[nome] = valores
    resultado = pd.DataFrame(dados, columns=esquema)
    exigir(resultado.index.equals(pd.RangeIndex(n)), 'Indice de dados ficticios alterado.')
    alvo = pacote.configuracao['dados']['coluna_alvo']
    exigir(alvo not in resultado.columns, 'Vazamento de alvo nos dados ficticios.')
    return resultado


def verificar_um(dim: int, esquema_referencia: list[str] | None) -> list[str]:
    origem = ORIGENS[dim]
    pasta = PASTA_MODELOS / f'PCA_{dim}D'
    escolha = carregar_json(
        RAIZ / 'outputs' / origem / 'vencedor' / 'metricas' / 'melhor_configuracao.json'
    )
    manifesto = carregar_json(pasta / 'manifesto_modelo_final.json')
    exigir(manifesto.get('teste_f10_executado') is False, f'PCA {dim}D: F10 registrado como usado')
    exigir(manifesto.get('n_treino_original') == 166716, f'PCA {dim}D: N de treino inesperado')
    exigir(manifesto.get('dimensao_pca') == dim, f'PCA {dim}D: dimensao incorreta')
    exigir(manifesto.get('limiar_mcc_oof_congelado') == escolha.get('limiar_mcc_oof'),
           f'PCA {dim}D: limiar de manifesto e OOF nao coincide')
    exigir(manifesto.get('id_candidato_oof') == escolha.get('id_candidato'),
           f'PCA {dim}D: candidato nao coincide')
    exigir((manifesto.get('proveniencia') or {}).get('experimento_oof') == f'outputs/{origem}',
           f'PCA {dim}D: origem OOF incorreta')

    print(f'PCA {dim}D: carregando artefato confiavel...', flush=True)
    pacote = carregar_pacote_confiavel(pasta)  # valida hash antes de desserializar
    exigir(pacote.limiar_oof == escolha['limiar_mcc_oof'], f'PCA {dim}D: tau interno mudou')
    exigir(pacote.escolha_oof.get('teste_final_usado') is False,
           f'PCA {dim}D: selecao OOF invalida')
    exigir(pacote.configuracao.get('divisao', {}).get('qual_fold_teste') == 10,
           f'PCA {dim}D: F10 nao reservado')
    exigir(int(pacote.pca.parametros['n_components']) == dim,
           f'PCA {dim}D: PCA interno diverge')

    x = construir_features_ficticias(pacote)
    colunas = list(x.columns)
    if esquema_referencia is not None:
        exigir(colunas == esquema_referencia, f'PCA {dim}D: ordem/nomes divergentes entre modelos')

    scores = pacote.prever_sem_rotulos(x)
    exigir(isinstance(scores, pd.Series) and scores.index.equals(x.index),
           f'PCA {dim}D: ordem da previsao alterada')
    ps = scores.to_numpy(dtype=np.float64)
    exigir(len(ps) == NUM_LINHAS_FICTICIAS and np.isfinite(ps).all()
           and np.all((0.0 <= ps) & (ps <= 1.0)),
           f'PCA {dim}D: scores fora de [0,1]')

    # Simulacao end-to-end: os dados e os rotulos abaixo sao FICTICIOS.
    alvo = pacote.configuracao['dados']['coluna_alvo']
    y_ficticio = np.arange(NUM_LINHAS_FICTICIAS, dtype=np.uint8) % 2
    dados_ficticios = x.copy()
    dados_ficticios[alvo] = y_ficticio
    tabela, metricas_ficticias = avaliar_f10_em_memoria(
        pacote,
        dados_ficticios,
        np.full(NUM_LINHAS_FICTICIAS, 10, dtype=np.uint16),
        np.arange(10_000, 10_000 + NUM_LINHAS_FICTICIAS, dtype=np.int64),
    )
    exigir(len(tabela) == NUM_LINHAS_FICTICIAS, f'PCA {dim}D: saida incompleta')
    exigir(np.allclose(tabela['prob_fraude'].to_numpy(), ps, rtol=0, atol=1e-12),
           f'PCA {dim}D: scores diferentes ao repetir inferencia')
    exigir(metricas_ficticias['limiar_otimizado_no_f10'] is False,
           f'PCA {dim}D: houve otimizacao indevida do limiar')
    exigir(metricas_ficticias['limiar_mcc_oof_congelado'] == pacote.limiar_oof,
           f'PCA {dim}D: tau deixou de estar congelado')
    exigir(set(metricas_ficticias['comparacao_limiares']) == {'tau_0_50', 'tau_mcc_oof'},
           f'PCA {dim}D: comparacao dos limiares incompleta')

    # Bloqueios basicos: o preditor nao pode aceitar TARGET nem mudar de esquema.
    try:
        pacote.prever_sem_rotulos(dados_ficticios)
    except ValueError as exc:
        exigir('alvo' in str(exc).lower(), f'PCA {dim}D: bloqueio de alvo inesperado: {exc}')
    else:
        raise RuntimeError(f'PCA {dim}D: preditor aceitou TARGET indevidamente')

    try:
        pacote.prever_sem_rotulos(x.loc[:, list(reversed(colunas))])
    except ValueError:
        pass
    else:
        if len(colunas) > 1:
            raise RuntimeError(f'PCA {dim}D: alteracao da ordem das features nao bloqueada')

    print(
        f'[APROVADO] PCA {dim}D | {NUM_LINHAS_FICTICIAS} linhas FICTICIAS | '
        f'features={len(colunas)} | tau OOF={pacote.limiar_oof:.17g} | '
        'inferencia e avaliacao simulada OK', flush=True,
    )
    del pacote
    gc.collect()
    return colunas


def main() -> None:
    print('=== ETAPA 6: INFERENCIA EM DADOS FICTICIOS ===', flush=True)
    print('Nao le Parquet, folds, TARGET real ou F10. Nao grava arquivos.\n', flush=True)
    exigir(PASTA_MODELOS.is_dir(), f'Pasta de modelos ausente: {PASTA_MODELOS}')
    esquema = None
    for dim in (1, 2, 3):
        esquema = verificar_um(dim, esquema)
    print('\n[CONCLUIDO] 3 modelos e fluxo de avaliacao testados.', flush=True)
    print('ATENCAO: metricas sinteticas NAO sao resultados do TCC. F10 REAL NAO ACESSADO.', flush=True)


if __name__ == '__main__':
    main()
