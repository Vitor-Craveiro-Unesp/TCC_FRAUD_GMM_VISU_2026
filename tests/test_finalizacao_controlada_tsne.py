"""Parte 4 - integração isolada da finalização t-SNE, sem dados ou F10 reais.

O YAML oficial é somente lido; parâmetros são alterados apenas na memória.
O executor não é ligado ao CLI nem à finalização PCA existente.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.finalizacao_controlada_tsne import (
    ResultadoFinalizacaoControladaTSNE,
    executar_ajuste_final_tsne_em_memoria,
    prever_sem_rotulos_controlado,
    persistir_finalizacao_controlada,
)
from src.roteamento_final_tsne import planejar_finalizacao_tsne
from src.treinamento_final_tsne import PacoteFinalTSNE, carregar_pacote_confiavel
from src.validacao_oof import grade_candidatos_oof


def caso_sintetico(dim=2):
    raiz = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (raiz / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf-8-sig')
    )
    cfg = deepcopy(cfg)
    cfg['reducao']['metodo'] = 'TSNE'
    cfg['tsne']['n_components'] = dim
    cfg['tsne']['initialization'] = ['random']
    cfg['tsne']['perplexity'] = [10]
    cfg['dados']['fracao_base'] = 0.1
    cfg['execucao']['n_jobs'] = 1
    cfg['gmm']['n_components'] = [4]
    cfg['gmm']['covariance_type'] = 'full'
    cfg['gmm']['n_init'] = 5
    cfg['amostragem']['modo'] = 'RUS'
    cfg['amostragem']['razao_negativos_positivos'] = 16
    cfg['risco_historico']['suavizacao'] = 0.0

    rng = np.random.default_rng(42)
    folds = np.repeat(np.arange(1, 10, dtype=np.uint8), 50)
    n = len(folds)
    y = np.zeros(n, dtype=np.uint8)
    for k in range(9):
        y[50*k:50*k+2] = 1
    ruido = rng.normal(size=n)
    df = pd.DataFrame({
        'TARGET_OMEGA': y,
        'NID_ALPHA': np.arange(n, dtype=np.int64),
        'TRANS_VALUE': ruido + 0.45*y,
        'TRANS_NUM_CARD_FEWF': rng.choice(['R','S','T'], n),
        'SEND_GENDER_BE': rng.choice(['F','M'], n),
        'TRANS_MONTH_SIN': np.sin(np.arange(n)*2*np.pi/12),
        'TRANS_MONTH_COS': np.cos(np.arange(n)*2*np.pi/12),
        'VALOR_CONTINUO': rng.normal(size=n) + 0.5*y,
    })
    grade = list(grade_candidatos_oof(cfg))
    assert len(grade) == 1
    escolha = {
        'id_candidato': f'tsne_sintetico_{dim}d',
        'configuracao': grade[0],
        'limiar_mcc_oof': 0.3,
        'status': 'selecionado_em_ensaio',
        'teste_final_usado': False,
    }
    origem = {
        'sha256_parquet': 'SINTETICO_SEM_PARQUET',
        'sha256_folds': 'SINTETICO_F1_F9',
        'sha256_yaml': 'CONFIG_EM_MEMORIA',
        'experimento_oof': 'TESTE_SINTETICO',
    }
    return df, folds, cfg, escolha, origem


@pytest.fixture(scope='module', params=[1,2,3], ids=['1D','2D','3D'])
def resultado(request):
    df, folds, cfg, escolha, origem = caso_sintetico(request.param)
    original_df = df.copy(deep=True)
    original_folds = folds.copy()
    cfg_original, escolha_original, origem_original = map(deepcopy, (cfg, escolha, origem))
    item = executar_ajuste_final_tsne_em_memoria(
        df, folds, cfg, escolha, origem, registrar_historico_em=False,
    )
    pd.testing.assert_frame_equal(df, original_df)
    np.testing.assert_array_equal(folds, original_folds)
    assert cfg == cfg_original and escolha == escolha_original and origem == origem_original
    return df, folds, cfg, escolha, origem, item


def test_integracao_plano_e_pacote(resultado):
    df, folds, cfg, escolha, origem, item = resultado
    assert isinstance(item, ResultadoFinalizacaoControladaTSNE)
    assert isinstance(item.pacote, PacoteFinalTSNE)
    assert item.plano.dimensao_tsne == escolha['configuracao']['reducao']['n_components']
    assert item.plano.fold_teste_reservado == 10
    assert item.plano.executar_treinamento is False
    assert item.plano.acessar_f10 is False
    assert item.pacote.n_treino_original == len(df) == 450
    assert item.pacote.n_ajuste_gmm == 18 + 16*18
    assert item.pacote.risco.n_treino_ == len(df)
    assert item.pacote.limiar_oof == 0.3
    assert item.pacote.escolha_oof['teste_final_usado'] is False
    assert item.pacote.escolha_oof['id_candidato'] == escolha['id_candidato']


def test_inferencia_oos_referencial(resultado):
    df, folds, cfg, escolha, origem, item = resultado
    entrada = df.drop(columns=['TARGET_OMEGA']).iloc[:25].copy()
    entrada.index = pd.Index(range(9000, 9025))
    interno_antes = np.asarray(item.pacote.tsne.embedding_).copy()
    cache_antes = item.pacote.tsne.transform_treino().copy()
    scores = prever_sem_rotulos_controlado(item, entrada)
    assert scores.index.equals(entrada.index)
    assert np.isfinite(scores.to_numpy()).all()
    assert scores.between(0, 1).all()
    np.testing.assert_allclose(np.asarray(item.pacote.tsne.embedding_), interno_antes, rtol=0, atol=1e-7)
    pd.testing.assert_frame_equal(item.pacote.tsne.transform_treino(), cache_antes)


def test_sem_alvo_na_inferencia(resultado):
    df, folds, cfg, escolha, origem, item = resultado
    with pytest.raises(ValueError, match='Remova o alvo'):
        prever_sem_rotulos_controlado(item, df.iloc[:10])


def test_persistencia_e_scores(resultado, tmp_path):
    df, folds, cfg, escolha, origem, item = resultado
    entrada = df.drop(columns=['TARGET_OMEGA']).iloc[:20].copy()
    esperado = prever_sem_rotulos_controlado(item, entrada)
    destino = tmp_path / f'pacote_{item.plano.dimensao_tsne}d'
    modelo, manifesto = persistir_finalizacao_controlada(item, destino)
    assert modelo.is_file() and manifesto.is_file()
    metadados = json.loads(manifesto.read_text(encoding='utf-8'))
    assert metadados['metodo'] == 'TSNE'
    assert metadados['dimensao_tsne'] == item.plano.dimensao_tsne
    assert metadados['teste_f10_executado'] is False
    assert metadados['limiar_mcc_oof_congelado'] == 0.3
    recarregado = carregar_pacote_confiavel(destino)
    encontrado = recarregado.prever_sem_rotulos(entrada)
    np.testing.assert_allclose(encontrado.to_numpy(), esperado.to_numpy(), rtol=0, atol=1e-7)
    with pytest.raises(FileExistsError):
        persistir_finalizacao_controlada(item, destino)


def test_rejeita_fold10_antes_do_treino(monkeypatch):
    import src.finalizacao_controlada_tsne as modulo
    df, folds, cfg, escolha, origem = caso_sintetico()
    falso = folds.copy()
    falso[-1] = 10
    # O pacote de treinamento deve bloquear F10 antes de ajustar t-SNE/GMM.
    def proibido(*args, **kwargs):
        raise AssertionError('O ajuste t-SNE nao deveria ter comecado')
    monkeypatch.setattr('src.treinamento_final_tsne.RedutorTSNE.fit', proibido)
    with pytest.raises(ValueError, match='F10 proibido'):
        modulo.executar_ajuste_final_tsne_em_memoria(
            df, falso, cfg, escolha, origem, registrar_historico_em=False
        )


def test_rejeita_pca_sem_treinar(monkeypatch):
    import src.finalizacao_controlada_tsne as modulo
    df, folds, cfg, escolha, origem = caso_sintetico()
    cfg['reducao']['metodo'] = 'PCA'
    def proibido(*args, **kwargs):
        raise AssertionError('Treinamento nao pode iniciar com PCA')
    monkeypatch.setattr(modulo, 'ajustar_final_f1_f9', proibido)
    with pytest.raises(ValueError, match='exclusivo TSNE'):
        modulo.executar_ajuste_final_tsne_em_memoria(
            df, folds, cfg, escolha, origem, registrar_historico_em=False
        )


def test_rejeita_limiar_alterado_sem_treinar(monkeypatch):
    import src.finalizacao_controlada_tsne as modulo
    df, folds, cfg, escolha, origem = caso_sintetico()
    escolha['limiar_mcc_oof'] = float('nan')
    def proibido(*args, **kwargs):
        raise AssertionError('Treinamento nao pode iniciar com tau invalido')
    monkeypatch.setattr(modulo, 'ajustar_final_f1_f9', proibido)
    with pytest.raises(ValueError, match='Limiar'):
        modulo.executar_ajuste_final_tsne_em_memoria(
            df, folds, cfg, escolha, origem, registrar_historico_em=False
        )


def test_rejeita_candidato_fora_grade_sem_treinar(monkeypatch):
    import src.finalizacao_controlada_tsne as modulo
    df, folds, cfg, escolha, origem = caso_sintetico()
    escolha['configuracao']['reducao']['perplexity'] = 97
    def proibido(*args, **kwargs):
        raise AssertionError('Treinamento nao pode iniciar com candidato invalido')
    monkeypatch.setattr(modulo, 'ajustar_final_f1_f9', proibido)
    with pytest.raises(ValueError, match='grade TSNE'):
        modulo.executar_ajuste_final_tsne_em_memoria(
            df, folds, cfg, escolha, origem, registrar_historico_em=False
        )


def test_automacao_pca_nao_muda_rota():
    df, folds, cfg, escolha, origem = caso_sintetico()
    cfg['execucao']['finalizacao_pca1d_automatica'] = True
    plano = planejar_finalizacao_tsne(cfg, escolha)
    assert plano.modulo_treinamento == 'src.treinamento_final_tsne'
    assert plano.modulo_avaliacao == 'src.avaliacao_final_tsne'
    assert plano.acessar_f10 is False
