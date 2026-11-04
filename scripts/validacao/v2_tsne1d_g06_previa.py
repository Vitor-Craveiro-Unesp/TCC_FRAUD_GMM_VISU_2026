"""V2/G06: previa sintetica do comparativo de hiperparametros t-SNE 1D.

Somente dados artificiais. Nao acessa Parquet, folds reais ou F10,
nao treina t-SNE/GMM e nao modifica arquivos de producao ou YAML.
Os nomes de inicializacao sao acrescentados APENAS durante esta previa,
pois o rotulo compartilhado atualmente omite essa diferenca importante.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, matthews_corrcoef, log_loss

from src import visualizacao_oof as vis


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _rotulo_tsne_com_inicializacao(linha):
    """Versao demonstrativa: distingue candidatos TSNE que teriam rotulos iguais."""
    c = json.loads(linha['configuracao'])
    if c['metodo'].upper() != 'TSNE':
        raise AssertionError('A previa G06 aceita somente candidatos t-SNE.')
    r = c['reducao']
    return (
        f't-SNE {int(r["n_components"])}D · p={r["perplexity"]}'
        f' · init={r["initialization"]} · K={c["gmm"]["n_components"]}'
        f' · {c["modo"]} {c["razao"]}'
    )


def main() -> None:
    raiz = Path.cwd()
    yaml_oficial = raiz / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    renderizador = raiz / 'src/visualizacao_oof.py'
    for caminho in (yaml_oficial, renderizador):
        if not caminho.is_file():
            raise FileNotFoundError(f'Arquivo obrigatorio ausente: {caminho}')
    hashes_antes = {str(p): _hash(p) for p in (yaml_oficial, renderizador)}
    config = deepcopy(yaml.safe_load(yaml_oficial.read_text(encoding='utf-8-sig')))
    config['reducao']['metodo'] = 'TSNE'
    config['tsne']['n_components'] = 1
    config['visualizacao']['graficos']['comparacao_hiperparametros'].update({
        'habilitar': True,
        'metricas': ['ap', 'mcc', 'logloss', 'score'],
        'ordenar_por': 'score',
        'destacar_vencedor': True,
        'maximo_configuracoes': 25,
        'configuracoes_por_pagina': 8,
    })

    # A mesma amostra artificial usada nos graficos 00 a 05.
    rng = np.random.default_rng(20261101)
    folds = np.repeat(np.arange(1, 10, dtype=np.uint8), 40)
    y = np.zeros(360, dtype=np.uint8)
    for k in range(9):
        locais = rng.choice(40, size=3, replace=False)
        y[k * 40 + locais] = 1
    p_base = rng.beta(1.8, 10.0, size=len(y))
    p_base[y == 1] = rng.beta(2.2, 2.8, size=int(y.sum()))
    p_base = np.clip(p_base, 1e-8, 1 - 1e-8)
    assert len(folds) == len(y) == 360 and int(y.sum()) == 27
    assert set(np.unique(folds)) == set(range(1, 10))

    # Seis CANDIDATOS HIPOTETICOS apenas para testar comparacao, layout e rotulos.
    # As perplexidades 5 e 20 NAO foram executadas no OOF oficial.
    # Nenhuma das metricas abaixo representa um ajuste real de t-SNE ou GMM.
    candidatos = []
    fontes = {}
    for i, (perplexidade, inicializacao) in enumerate(
        product((5, 10, 20), ('random', 'pca'))
    ):
        gerador = np.random.default_rng(6200 + i)
        scores = np.clip(
            p_base * (0.94 + 0.028 * i)
            + gerador.normal(loc=0.0, scale=0.013 + 0.003 * i, size=len(y)),
            1e-8, 1 - 1e-8,
        )
        tau_ficticio = 0.35
        ap = float(average_precision_score(y, scores))
        mcc = float(matthews_corrcoef(y, scores >= tau_ficticio))
        ll = float(log_loss(y, scores, labels=[0, 1]))
        score = float((ap + (mcc + 1.0) / 2.0 + np.exp(-ll)) / 3.0)
        cid = f'TSNE1D_SINTETICO_{i+1:02d}'
        candidato = {
            'metodo': 'TSNE',
            'reducao': {
                'perplexity': perplexidade,
                'n_components': 1,
                'initialization': inicializacao,
                'metric': 'euclidean',
                'learning_rate': 'auto',
                'early_exaggeration': 12.0,
                'n_iter': 300,
                'theta': 0.5,
            },
            'modo': 'RUS',
            'razao': 16,
            'gmm': {
                'n_components': 4,
                'covariance_type': 'full',
                'n_init': 5,
            },
            'suavizacao': 0.0,
        }
        fontes[cid] = scores
        candidatos.append({
            'id_candidato': cid,
            'ap': ap,
            'mcc': mcc,
            'logloss': ll,
            'score': score,
            'limiar_mcc': tau_ficticio,
            'n': len(y),
            'prevalencia': float(y.mean()),
            'configuracao': json.dumps(candidato, ensure_ascii=False, sort_keys=True),
        })

    grade = pd.DataFrame(candidatos)
    ordenada = grade.sort_values(
        ['score', 'ap', 'id_candidato'],
        ascending=[False, False, True], kind='stable'
    ).reset_index(drop=True)
    escolhido_id = str(ordenada.iloc[0]['id_candidato'])
    escolha = {'id_candidato': escolhido_id,
               'limiar_mcc_oof': 0.35, 'teste_final_usado': False}
    assert grade['id_candidato'].is_unique and len(grade) == 6
    assert all(json.loads(c)['reducao']['n_components'] == 1
               for c in grade['configuracao'])
    assert len({_rotulo_tsne_com_inicializacao(l) for _, l in grade.iterrows()}) == 6

    destino = raiz / (
        'outputs/validacao_tsne_v2_1d/'
        '06_comparacao_hiperparametros_previa_sintetica'
    )
    destino.mkdir(parents=True, exist_ok=True)

    # Adaptacao temporaria APENAS em memoria. O renderizador PCA nao muda.
    rotulo_original = vis._rotulo_config
    vis._rotulo_config = _rotulo_tsne_com_inicializacao
    try:
        vis._comparacao(config, grade, escolha, destino, 'png', 180, True)
    finally:
        vis._rotulo_config = rotulo_original

    png = destino / '06_comparacao_hiperparametros.png'
    csv = destino / '06_comparacao_hiperparametros_todos.csv'
    for arquivo in (png, csv):
        assert arquivo.is_file() and arquivo.stat().st_size > 0, arquivo
    tabela = pd.read_csv(csv, dtype={'id_candidato': str})
    assert len(tabela) == 6 and tabela['id_candidato'].is_unique
    assert str(tabela.iloc[0]['id_candidato']) == escolhido_id
    assert np.allclose(tabela['score'],
                       (tabela['ap'] + (tabela['mcc'] + 1) / 2
                        + np.exp(-tabela['logloss'])) / 3,
                       atol=1e-10, rtol=0)
    for _, linha in tabela.iterrows():
        probs = fontes[linha['id_candidato']]
        assert np.isclose(linha['ap'], average_precision_score(y, probs))
        assert np.isclose(linha['mcc'], matthews_corrcoef(y, probs >= 0.35))
        assert np.isclose(linha['logloss'], log_loss(y, probs, labels=[0, 1]))
    assert hashes_antes == {str(p): _hash(p) for p in (yaml_oficial, renderizador)}
    assert vis._rotulo_config is rotulo_original

    # O aviso acompanha o arquivo em manifesto separado, sem faixa na figura.
    aviso = destino / 'LEIA_ME_PREVIA_SINTETICA.json'
    conteudo = {
        'tipo': 'PREVIA_SINTETICA_NAO_OFICIAL',
        'grafico': '06_comparacao_hiperparametros',
        'metodo': 'TSNE', 'dimensao': 1, 'folds_ficticios': list(range(1, 10)),
        'candidatos_hipoteticos': 6,
        'perplexidades_hipoteticas': [5, 10, 20],
        'initializations': ['random', 'pca'],
        'oof_real_executado': False, 'f10_real_acessado': False,
        'treinamento_real_executado': False,
        'codigo_pca_alterado': False,
        'observacao': (
            'Rotulos de inicializacao distinguem modelos na previa; '
            'a adaptacao definitiva t-SNE deve ser validada antes de OOF real.'
        ),
    }
    if aviso.exists():
        assert json.loads(aviso.read_text(encoding='utf-8')) == conteudo
    else:
        aviso.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8')
    print('\n=== V2 / TSNE 1D / GRAFICO 06 ===')
    print('N_SINTETICO=360 | FRAUDES_SINTETICAS=27')
    print('CANDIDATOS_HIPOTETICOS=6 | DIMENSAO=1D')
    print('VENCEDOR_SINTETICO=' + escolhido_id)
    print('INICIALIZACAO_IDENTIFICAVEL_NOS_ROTULOS=SIM')
    print('F10_REAL_ACESSADO=NAO')
    print('TREINAMENTO_REAL_EXECUTADO=NAO')
    print('ARQUIVOS_OFICIAIS_ALTERADOS=NAO')
    print('PNG_PARA_AVALIAR=' + str(png))
    print('CSV_PARA_CONFERIR=' + str(csv))
    print('V2_G06_VALIDACAO=APROVADA_TECNICAMENTE_AGUARDANDO_AVALIACAO_VISUAL')


if __name__ == '__main__':
    main()
