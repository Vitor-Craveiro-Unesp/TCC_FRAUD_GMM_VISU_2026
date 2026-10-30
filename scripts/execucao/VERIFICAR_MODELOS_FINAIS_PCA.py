# -*- coding: utf-8 -*-
"""Verifica integridade dos 3 modelos finais PCA; nao treina nem acessa F10.

Instalar em scripts/execucao/VERIFICAR_MODELOS_FINAIS_PCA.py.
Executar da raiz do projeto com:
    python scripts/execucao/VERIFICAR_MODELOS_FINAIS_PCA.py

ATENCAO: arquivos joblib usam pickle. Carregue apenas modelos locais confiaveis.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.treinamento_final_pca import carregar_pacote_confiavel

DESTINO = RAIZ / 'outputs' / 'treinamento_final_pca_10pct_20261029_174241'
ORIGENS = {
    1: 'ensaio_pca_gmm_20261029_173123',
    2: 'ensaio_pca_gmm_20261029_172521',
    3: 'ensaio_pca_gmm_20261029_173325',
}


def ler_json(arquivo: Path) -> dict:
    if not arquivo.is_file():
        raise FileNotFoundError(f'Arquivo ausente: {arquivo}')
    return json.loads(arquivo.read_text(encoding='utf-8-sig'))


def exigir(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise RuntimeError(mensagem)


def main() -> None:
    exigir(DESTINO.is_dir(), f'Pasta de modelos nao encontrada: {DESTINO}')
    for dim, origem in ORIGENS.items():
        pasta = DESTINO / f'PCA_{dim}D'
        manifesto = ler_json(pasta / 'manifesto_modelo_final.json')
        escolha = ler_json(
            RAIZ / 'outputs' / origem / 'vencedor' / 'metricas' / 'melhor_configuracao.json'
        )
        print(f'PCA {dim}D: verificando manifesto, SHA-256 e carregamento...', flush=True)
        exigir(manifesto.get('teste_f10_executado') is False, f'PCA {dim}D: campo F10 invalido')
        exigir(manifesto.get('dimensao_pca') == dim, f'PCA {dim}D: dimensao invalida')
        exigir(manifesto.get('n_treino_original') == 166_716, f'PCA {dim}D: n de treino invalido')
        exigir(manifesto.get('fracao_base') == 0.1, f'PCA {dim}D: fracao invalida')
        exigir(manifesto.get('id_candidato_oof') == escolha.get('id_candidato'),
               f'PCA {dim}D: candidato OOF divergente')
        exigir(manifesto.get('limiar_mcc_oof_congelado') == escolha.get('limiar_mcc_oof'),
               f'PCA {dim}D: limiar OOF divergente')
        proveniencia = manifesto.get('proveniencia') or {}
        exigir(proveniencia.get('experimento_oof') == f'outputs/{origem}',
               f'PCA {dim}D: origem OOF divergente')

        pacote = carregar_pacote_confiavel(pasta)  # confere hash ANTES do joblib.load
        exigir(pacote.n_treino_original == 166_716, f'PCA {dim}D: amostra divergente')
        exigir(pacote.limiar_oof == escolha['limiar_mcc_oof'],
               f'PCA {dim}D: limiar divergente no joblib')
        exigir(pacote.escolha_oof['id_candidato'] == escolha['id_candidato'],
               f'PCA {dim}D: candidato divergente no joblib')
        exigir(pacote.configuracao['divisao']['qual_fold_teste'] == 10,
               f'PCA {dim}D: configuracao de F10 divergente')
        print(
            f'[APROVADO] PCA {dim}D | treino={pacote.n_treino_original} | '
            f'RUS={pacote.n_ajuste_gmm} | tau={pacote.limiar_oof:.17g}',
            flush=True,
        )
        del pacote
        gc.collect()

    print('\nINTEGRIDADE DOS 3 MODELOS APROVADA. F10 NAO ACESSADO.', flush=True)


if __name__ == '__main__':
    main()
