"""Testes SINTETICOS do executor F10, sem ler dataset nem Parquet real."""
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

CODIGO = Path(__file__).resolve().parents[1] / 'scripts' / 'execucao' / 'ETAPA8_AVALIAR_F10_PCA_10PCT.py'


def carregar_executor(monkeypatch):
    # Stubs isolados: permitem testar as travas antes de existir qualquer
    # chamada a dados/modelos reais. Os testes rodam sem importar src real.
    etapa7 = types.ModuleType('ETAPA7_CONGELAR_PRE_F10_PCA')
    def _selecionar(ids, y):
        r = np.random.default_rng(42)
        saida = []
        for classe in (0, 1):
            pos = np.asarray(ids)[np.asarray(y) == classe]
            saida.append(r.choice(pos, size=max(2, round(.1 * len(pos))), replace=False))
        return np.sort(np.concatenate(saida))
    etapa7.selecionar_ids_f10_estratificado = _selecionar
    etapa7.coletar_e_verificar = lambda: (_ for _ in ()).throw(AssertionError('preflight real proibido em teste'))
    etapa7.ler_json = lambda path: (_ for _ in ()).throw(AssertionError('arquivo real proibido'))
    etapa7.sha_arquivo = lambda path: (_ for _ in ()).throw(AssertionError('arquivo real proibido'))
    treino = types.ModuleType('TREINAR_FINAL_PCA_F1_F9_10PCT')
    treino.verificar_origens = lambda: (_ for _ in ()).throw(AssertionError('treino/folds reais proibidos'))
    src = types.ModuleType('src')
    src.__path__ = []
    folds = types.ModuleType('src.folds')
    folds.carregar_alvo = lambda *args: (_ for _ in ()).throw(AssertionError('TARGET real proibido'))
    modelos = types.ModuleType('src.treinamento_final_pca')
    modelos.carregar_pacote_confiavel = lambda *args: (_ for _ in ()).throw(AssertionError('modelo real proibido'))
    aval = types.ModuleType('src.avaliacao_final_pca')
    aval.avaliar_f10_em_memoria = lambda *args: (_ for _ in ()).throw(AssertionError('avaliacao real proibida'))
    for nome, mod in [
        ('ETAPA7_CONGELAR_PRE_F10_PCA', etapa7),
        ('TREINAR_FINAL_PCA_F1_F9_10PCT', treino),
        ('src', src), ('src.folds', folds),
        ('src.treinamento_final_pca', modelos), ('src.avaliacao_final_pca', aval),
    ]:
        monkeypatch.setitem(sys.modules, nome, mod)
    spec = importlib.util.spec_from_file_location('etapa8_isolado', CODIGO)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def exemplo_folds():
    # 100 F1, 100 F10 (80 legitimas, 20 fraudes)
    folds = np.r_[np.ones(100, dtype=np.uint16), np.full(100, 10, dtype=np.uint16)]
    y = np.r_[np.zeros(100, dtype=np.uint8),
              np.zeros(80, dtype=np.uint8), np.ones(20, dtype=np.uint8)]
    proto = {'dados': {'numero_total_linhas': 200, 'numero_linhas_f10_integral': 100},
             'protocolo_teste_pre_registrado': {'fracao_teste': .1}}
    return proto, folds, y


def test_amostragem_estratificada_exata_e_reprodutivel(monkeypatch):
    mod = carregar_executor(monkeypatch)
    proto, folds, y = exemplo_folds()
    sel1 = mod.selecionar_f10(proto, folds, y)
    sel2 = mod.selecionar_f10(proto, folds, y)
    assert np.array_equal(sel1, sel2)
    assert len(sel1) == 10
    assert np.count_nonzero(y[sel1] == 0) == 8
    assert np.count_nonzero(y[sel1] == 1) == 2
    assert np.all(folds[sel1] == 10)
    assert np.all(np.diff(sel1) > 0)


def test_falha_se_numero_f10_diverge(monkeypatch):
    mod = carregar_executor(monkeypatch)
    proto, folds, y = exemplo_folds()
    proto['dados']['numero_linhas_f10_integral'] += 1
    with pytest.raises(RuntimeError, match='Numero de linhas F10'):
        mod.selecionar_f10(proto, folds, y)


def test_falha_se_classe_f10_invalida(monkeypatch):
    mod = carregar_executor(monkeypatch)
    proto, folds, y = exemplo_folds()
    y[100:] = 0
    with pytest.raises(RuntimeError, match='ambas as classes'):
        mod.selecionar_f10(proto, folds, y)


def test_falha_em_fold_fora_do_intervalo(monkeypatch):
    mod = carregar_executor(monkeypatch)
    proto, folds, y = exemplo_folds()
    folds[120] = 8
    with pytest.raises(RuntimeError, match='Numero de linhas F10'):
        mod.selecionar_f10(proto, folds, y)


def test_token_exigido_antes_do_preflight(monkeypatch):
    mod = carregar_executor(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['programa', '--executar-f10'])
    with pytest.raises(RuntimeError, match='autorizacao explicita'):
        mod.main()


def test_verificar_proibe_token_de_execucao(monkeypatch):
    mod = carregar_executor(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['programa', '--verificar', '--confirmacao', mod.TOKEN_AUTORIZACAO])
    with pytest.raises(RuntimeError, match='so faz sentido'):
        mod.main()


def test_preflight_comparacao_detecta_manifesto_alterado(monkeypatch, tmp_path):
    mod = carregar_executor(monkeypatch)
    p = tmp_path / 'manifesto.json'
    p.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(mod, 'ARQ_PROTOCOLO', p)
    monkeypatch.setattr(mod, 'sha_arquivo', lambda path: 'hash-incorreto')
    with pytest.raises(RuntimeError, match='Hash do manifesto'):
        mod.preflight()


def test_regra_do_congelamento_nao_escolhe_vencedor(monkeypatch):
    mod = carregar_executor(monkeypatch)
    proto, folds, y = exemplo_folds()
    amostra = mod.selecionar_f10(proto, folds, y)
    assert len(amostra) == 10
    assert not hasattr(mod, 'otimizar_limiar_mcc')
    assert not hasattr(mod, 'selecionar_vencedor')
