# -*- coding: utf-8 -*-
"""Finalizacao oficial t-SNE 3D pos-OOF, F10 exploratorio e metadados.

Reaproveita os contratos t-SNE ja existentes. Nao cria graficos F10:
a integracao visual e executada separadamente. Sem fallback PCA.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import yaml

from src.folds import hash_arquivo
from src.amostragem_f10 import fracao_amostra_f10, selecionar_f10_estratificado
from src.ponte_oof_final_tsne import verificar_oof_concluido_tsne
from src.treinamento_final_tsne import carregar_pacote_confiavel
from src.finalizacao_controlada_tsne import executar_ajuste_final_tsne_em_memoria, persistir_finalizacao_controlada
from src.avaliacao_final_tsne import avaliar_f10_em_memoria

def validar_automacao(config: dict) -> bool:
    """Valida sem abrir dados nem criar saidas; permissao exclusiva TSNE 3D."""
    fracao_amostra_f10(config)
    if str(config.get('reducao', {}).get('metodo', '')).upper() != 'TSNE':
        raise ValueError('Finalizador t-SNE 3D nao aceita PCA.')
    dimensao = config.get('tsne', {}).get('n_components')
    if dimensao not in (3, [3]):
        raise ValueError('Finalizador t-SNE 3D aceita somente dimensao 3; t-SNE 3D ainda nao integrado.')
    fracao = config.get('dados', {}).get('fracao_base')
    if type(fracao) not in (int, float) or float(fracao) not in (0.01, 0.03, 0.1):
        raise ValueError('Finalizacao aceita desenvolvimento 1%, 3% ou 10%.')
    if (config.get('divisao', {}).get('total_folds') != 10 or
            config['divisao'].get('qual_fold_teste') != 10 or
            config['divisao'].get('quantidade_folds_teste') != 1):
        raise ValueError('O F10 deve permanecer o unico fold de teste.')
    gmm = config.get('gmm', {})
    k = gmm.get('n_components')
    if k not in ([4], 4) or gmm.get('n_init') != 5 or str(gmm.get('covariance_type', '')).lower() != 'full':
        raise ValueError('Requer GMM K4 FULL n_init5.')
    am = config.get('amostragem', {})
    if am.get('modo') != 'RUS' or am.get('razao_negativos_positivos') != 16:
        raise ValueError('Requer RUS 16:1.')
    if config.get('experimento', {}).get('semente') != 42:
        raise ValueError('Requer seed42.')
    if config.get('execucao', {}).get('proteger_resultados') is not True:
        raise ValueError('Protecao de resultados obrigatoria.')
    return True


def _ler_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding='utf-8-sig'))


def _salvar_json_novo(p: Path, conteudo: dict) -> None:
    # Modo x impede sobrescrever resultados; arquivos incompletos ficam visiveis.
    with p.open('x', encoding='utf-8') as f:
        json.dump(conteudo, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def _amostra_f10(folds, y, *, config: dict) -> np.ndarray:
    """Regra central do YAML compartilhada com PCA e t-SNE de qualquer dimensao."""
    return selecionar_f10_estratificado(folds, y, config)


def concluir_ensaio(pasta_experimento: str | Path, *, raiz: str | Path | None = None) -> Path:
    """Protege contra duas finalizacoes concorrentes do mesmo experimento."""
    pasta = Path(pasta_experimento).resolve()
    base = Path(raiz).resolve() if raiz is not None else Path(__file__).resolve().parents[1]
    if not pasta.is_relative_to((base / 'outputs').resolve()):
        raise ValueError('Finalizacao so aceita experimento dentro de outputs/.')
    tdir = pasta / 'vencedor' / 'teste_final'
    if not tdir.is_dir():
        raise FileNotFoundError('Pasta teste_final do OOF ausente.')
    trava = tdir / '.finalizacao_tsne_3d.lock'
    try:
        with trava.open('x', encoding='utf-8') as f:
            f.write('Finalizacao t-SNE 3D em curso. Nao executar duas vezes em paralelo.\n')
    except FileExistsError as exc:
        raise RuntimeError('Finalizacao t-SNE ja em andamento; trava encontrada.') from exc
    try:
        return _concluir_ensaio_sem_lock(pasta, raiz=base)
    finally:
        trava.unlink(missing_ok=True)


def _concluir_ensaio_sem_lock(pasta_experimento: str | Path, *, raiz: str | Path | None = None) -> Path:
    """Avalia F10 depois do refit F1-F9 com hashes, sem renderizar galerias."""
    from src import orquestracao_oof as orq

    base = Path(raiz).resolve() if raiz is not None else Path(__file__).resolve().parents[1]
    pasta = Path(pasta_experimento).resolve()
    if not pasta.is_relative_to((base / 'outputs').resolve()):
        raise ValueError('Finalizacao so aceita experimento dentro de outputs/.')
    pre = verificar_oof_concluido_tsne(pasta)
    validar_automacao(pre.configuracao)
    manifest = _ler_json(pasta / 'manifesto.json')
    yaml_salvo = pasta / 'configuracao.yaml'
    if yaml.safe_load(yaml_salvo.read_text(encoding='utf-8-sig')) != pre.configuracao:
        raise RuntimeError('Configuracao OOF alterada.')
    pq = base / pre.configuracao['dados']['arquivo']
    folds, arq_folds, hash_parquet = orq._identificar_folds(base, pre.configuracao, pq)
    if hash_parquet != manifest['sha256_parquet'] or hash_arquivo(arq_folds) != manifest['sha256_folds']:
        raise RuntimeError('Parquet ou folds divergem do OOF.')
    with np.load(pasta/'vencedor/oof/predicoes.npz', allow_pickle=False) as arr:
        ids_dev = np.asarray(arr['linha_original'], dtype=np.int64)
        fold_dev = np.asarray(arr['fold'], dtype=np.uint8)
        y_dev = np.asarray(arr['y_true'], dtype=np.uint8)
    if not np.array_equal(folds[ids_dev], fold_dev) or np.any(fold_dev == 10):
        raise RuntimeError('Indices de desenvolvimento nao pertencem a F1-F9.')
    teste = pasta/'vencedor/teste_final'
    modelo_dir = teste/'modelo_final_f1_f9'
    marcador = teste/'manifesto_finalizacao_tsne.json'
    if marcador.exists():
        status = _ler_json(marcador)
        if (status.get('etapa') != 'F10_AVALIADO_GALERIA_PENDENTE' or
                status.get('dimensao') != 3 or status.get('metodo') != 'TSNE' or
                float(status.get('fracao_f10_dentro_do_fold', -1)) != fracao_amostra_f10(pre.configuracao)):
            raise RuntimeError('Manifesto de finalizacao TSNE inconsistente; nao repetir F10.')
        for nome, chave in (('metricas_f10.json', 'sha256_metricas'),
                            ('predicoes_f10.csv', 'sha256_predicoes'),
                            ('indices_f10_amostra.npy', 'sha256_indices_arquivo'),
                            ('coordenadas_TSNE1_TSNE2_TSNE3_f10.csv', 'sha256_coordenadas')):
            arq = teste/nome
            if not arq.is_file() or hash_arquivo(arq) != status.get(chave):
                raise RuntimeError(f'Saida F10 alterada ou incompleta: {nome}')
        if hash_arquivo(teste/'modelo_final_f1_f9/modelo_final_f1_f9.joblib') != status.get('sha256_modelo'):
            raise RuntimeError('Modelo F1-F9 divergente do manifesto de finalizacao.')
        print('[RETOMADA TSNE] F10 ja avaliado e hashes conferidos. Graficos F10 pendentes.', flush=True)
        return pasta

    if modelo_dir.is_dir():
        pacote = carregar_pacote_confiavel(modelo_dir)
        if (pacote.escolha_oof != pre.escolha_oof or pacote.configuracao != pre.configuracao or
                pacote.proveniencia['sha256_parquet'] != hash_parquet or
                pacote.proveniencia['sha256_folds'] != hash_arquivo(arq_folds)):
            raise RuntimeError('Modelo final existente diverge do OOF. Nao sobrescrever.')
        print('[RETOMADA TSNE] Modelo F1-F9 existente e verificado.', flush=True)
    else:
        df = orq._ler_desenvolvimento(pq, ids_dev)
        if len(df) != len(ids_dev) or not np.array_equal(df[pre.configuracao['dados']['coluna_alvo']].to_numpy(), y_dev):
            raise RuntimeError('Rotulos do desenvolvimento divergentes do OOF.')
        resultado = executar_ajuste_final_tsne_em_memoria(
            df, fold_dev, pre.configuracao, pre.escolha_oof,
            {**pre.proveniencia, 'sha256_predicoes_oof': hash_arquivo(pasta/'vencedor/oof/predicoes.npz')},
            registrar_historico_em=True,
        )
        persistir_finalizacao_controlada(resultado, modelo_dir)
        pacote = resultado.pacote
        print('[TSNE 3D] Modelo final F1-F9 salvo.', flush=True)

    # Nao acessar F10 antes de completar e congelar o modelo final.
    publicados = ('predicoes_f10.csv', 'metricas_f10.json', 'indices_f10_amostra.npy',
                  'manifesto_amostra_f10.json', 'coordenadas_TSNE1_TSNE2_TSNE3_f10.csv')
    if any((teste/nome).exists() for nome in publicados):
        raise RuntimeError('Saida F10 parcial existente: nao reavaliar nem sobrescrever.')
    y = orq._ler_alvo(pq, pre.configuracao['dados']['coluna_alvo'])
    if y.shape != folds.shape:
        raise RuntimeError('Tamanho da base difere dos folds.')
    fracao_f10 = fracao_amostra_f10(pre.configuracao)
    ids_f10 = _amostra_f10(folds, y, config=pre.configuracao)
    if not np.all(folds[ids_f10] == 10):
        raise RuntimeError('Indices de teste nao pertencem exclusivamente a F10.')
    from src.finalizacao_automatica_pca1d import ler_linhas_f10
    df_f10 = ler_linhas_f10(pq, ids_f10)  # Leitura posicional ja validada pelo PCA
    if not np.array_equal(df_f10[pre.configuracao['dados']['coluna_alvo']].to_numpy(), y[ids_f10]):
        raise RuntimeError('Rotulos F10 nao correspondem ao parquet/folds.')
    pred, met, coords = avaliar_f10_em_memoria(
        pacote, df_f10, folds[ids_f10], ids_f10, retornar_coordenadas=True)
    if (list(coords.columns) != ['linha_original', 'TSNE1', 'TSNE2', 'TSNE3'] or
            not np.array_equal(coords['linha_original'].to_numpy(dtype=np.int64), ids_f10) or
            not np.array_equal(pred['linha_original'].to_numpy(dtype=np.int64), ids_f10) or
            not np.isfinite(coords[['TSNE1', 'TSNE2', 'TSNE3']].to_numpy(dtype=np.float64)).all()):
        raise RuntimeError('Coordenadas F10 inconsistentes com as predicoes; nao publicar resultados.')
    met['carater'] = 'F10_EXPLORATORIO_REPETIDO_SEM_NOVA_INDEPENDENCIA'
    met['fracao_f10_dentro_do_fold'] = fracao_f10
    met['fracao_desenvolvimento_f1_f9'] = float(pre.configuracao['dados']['fracao_base'])
    # Arquivos novos exclusivamente; nunca permitir substituir predicoes existentes.
    with (teste/'predicoes_f10.csv').open('x', encoding='utf-8') as f:
        pred.to_csv(f, index=False, float_format='%.17g')
    with (teste/'coordenadas_TSNE1_TSNE2_TSNE3_f10.csv').open('x',encoding='utf-8') as f:
        coords.to_csv(f,index=False,float_format='%.17g')
    _salvar_json_novo(teste/'metricas_f10.json', met)
    with (teste/'indices_f10_amostra.npy').open('xb') as f:
        np.save(f, ids_f10, allow_pickle=False)
    _salvar_json_novo(teste/'manifesto_amostra_f10.json', {
        'fracao_dentro_fold_f10': fracao_f10, 'semente': 42,
        'n': len(ids_f10), 'n_fraudes': int(y[ids_f10].sum()),
        'sha256_indices': sha256(ids_f10.tobytes()).hexdigest(),
        'metodo': 'TSNE', 'dimensao': 3, 'reutilizacao_f10': 'exploratorio',
        'limiar_congelado_em': 'OOF_F1_F9', 'sem_refit_no_f10': True,
    })
    _salvar_json_novo(marcador, {
        'etapa': 'F10_AVALIADO_GALERIA_PENDENTE', 'metodo': 'TSNE',
        'dimensao': 3, 'id_candidato_oof': pre.escolha_oof['id_candidato'],
        'fracao_desenvolvimento': float(pre.configuracao['dados']['fracao_base']),
        'fracao_f10_dentro_do_fold': fracao_f10,
        'galeria_f10_concluida': False,
        'sha256_modelo': hash_arquivo(modelo_dir/'modelo_final_f1_f9.joblib'),
        'sha256_predicoes': hash_arquivo(teste/'predicoes_f10.csv'),
        'sha256_coordenadas': hash_arquivo(teste/'coordenadas_TSNE1_TSNE2_TSNE3_f10.csv'),
        'sha256_metricas': hash_arquivo(teste/'metricas_f10.json'),
        'sha256_indices_arquivo': hash_arquivo(teste/'indices_f10_amostra.npy'),
    })
    print('[TSNE 3D] F10 avaliado; PNG/HTML F10 pendentes.', flush=True)
    return pasta
