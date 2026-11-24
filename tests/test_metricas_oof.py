"""Passo 06.02: AP, MCC exato, LogLoss, score e integridade dos 9 folds."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import average_precision_score, log_loss, matthews_corrcoef

from src.metricas_oof import (
    avaliar_probabilidades, otimizar_limiar_mcc, avaliar_oof_consolidado,
)


@pytest.fixture
def cfg():
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "config_TCC_EXPERIMENTAL_V3_atualizado.yaml").read_text(encoding="utf-8-sig"))


@pytest.fixture
def oof():
    folds = np.tile(np.arange(1, 11, dtype=np.uint8), 4)
    ids = np.flatnonzero(folds != 10)
    # Uma fraude/legitima em cada fold de desenvolvimento e grupos repetidos de scores.
    y = np.tile([0, 1, 0, 1], 10).astype(np.uint8)
    p = np.clip(0.1 + 0.65 * y + (np.arange(len(y)) % 4) * 0.04, 0, 1)
    df = pd.DataFrame({"linha_original": ids, "fold": folds[ids],
                       "y_true": y[ids], "prob_fraude": p[ids]})
    return df, folds, y


def test_ap_exatamente_average_precision_sklearn(cfg):
    y = [1, 0, 1, 0, 0]
    p = [0.9, 0.8, 0.55, 0.5, 0.1]
    assert avaliar_probabilidades(y, p, cfg)["ap"] == pytest.approx(average_precision_score(y, p))


def test_ap_com_scores_empatados(cfg):
    y = [1, 0, 1, 0]
    p = [0.5, 0.5, 0.5, 0.1]
    assert avaliar_probabilidades(y, p, cfg)["ap"] == pytest.approx(average_precision_score(y, p))


def test_logloss_exatamente_sklearn(cfg):
    y, p = [0, 1, 0, 1], [0.2, 0.7, 0.4, 0.6]
    assert avaliar_probabilidades(y, p, cfg)["logloss"] == pytest.approx(log_loss(y, p, labels=[0,1]))


def test_logloss_extremos_zero_um_sem_infinito(cfg):
    r = avaliar_probabilidades([0,1,0,1], [0,1,1,0], cfg)
    assert np.isfinite(r["logloss"])
    assert np.isfinite(r["score"])


def test_score_composto_formato_oficial(cfg):
    r = avaliar_probabilidades([0,1,0,1], [0.05,0.9,0.3,0.4], cfg)
    assert r["score"] == pytest.approx((r["ap"] + (r["mcc"]+1)/2 + np.exp(-r["logloss"]))/3)


def test_metricas_nomes_serializaveis(cfg):
    r = avaliar_probabilidades([0,1], [0.2,0.8], cfg)
    assert r["n"] == 2 and r["n_fraudes"] == 1 and r["n_legitimas"] == 1
    assert isinstance(r["limiar_mcc"], float)
    assert isinstance(r["mcc"], float)
    assert r["prevalencia"] == 0.5


@pytest.mark.parametrize("y,p", [
    ([], []), ([0,1], [0.1]), ([[0,1]], [0.1,0.2]),
    ([0,1], [[0.1],[0.2]]),
])
def test_formato_invalido_falha(cfg, y, p):
    with pytest.raises(ValueError, match="vetores 1D"):
        avaliar_probabilidades(y, p, cfg)


@pytest.mark.parametrize("y", [[1,1,1],[0,0,0]])
def test_ap_exige_duas_classes(cfg, y):
    with pytest.raises(ValueError, match="ambas as classes"):
        avaliar_probabilidades(y, [0.1,0.5,0.9], cfg)


@pytest.mark.parametrize("y", [[1,0,2], [1,0,np.nan], ["fraud","normal","normal"]])
def test_rotulos_invalidos(cfg, y):
    with pytest.raises(ValueError, match="rotulos|0/1"):
        avaliar_probabilidades(y, [0.1,0.2,0.3], cfg)


@pytest.mark.parametrize("p", [[0.1,-0.1,0.2], [0.1,1.2,0.2], [0.1,np.nan,0.2], [0.1,np.inf,0.2]])
def test_scores_invalidos(cfg, p):
    with pytest.raises(ValueError, match="finitas em"):
        avaliar_probabilidades([0,1,0], p, cfg)


def test_empates_nao_separam_mesmo_score(cfg):
    y = [1,0,1,0,1,0]
    p = [0.5]*4 + [0.3,0.2]
    r = otimizar_limiar_mcc(y,p)
    pred = np.array(p) >= r["limiar_mcc"]
    assert len(set(pred[:4])) == 1
    assert r["mcc"] == pytest.approx(matthews_corrcoef(y, pred))


def test_1_componente_constante_usa_limiar_05(cfg):
    r = avaliar_probabilidades([0]*90 + [1]*10, [0.1]*100, cfg)
    assert r["mcc"] == 0
    assert r["limiar_mcc"] == 0.5
    assert r["ap"] == pytest.approx(0.1)


def test_all_negative_possivel_quando_probabilidade_um():
    r = otimizar_limiar_mcc([0,1,0,1], [1.,1.,1.,1.])
    assert r["mcc"] == 0
    assert r["limiar_mcc"] == 0.5


def test_mcc_ate_0_5_preservado(cfg):
    r = avaliar_probabilidades([0,0,1,1], [.2,.45,.49,.8], cfg)
    assert r["mcc_limiar_05"] == pytest.approx(matthews_corrcoef([0,0,1,1],[0,0,0,1]))


def test_otimo_exaustivo_varias_amostras():
    rng = np.random.default_rng(998)
    for n in [12, 30, 100]:
        for _ in range(14):
            y = rng.binomial(1, .3, n)
            if y.min() == y.max():
                y[0]=0; y[1]=1
            p = rng.choice([0., .1, .15, .25, .5, .75, .8, 1.],size=n)
            r = otimizar_limiar_mcc(y,p)
            limiares = [np.nextafter(float(p.max()), np.inf), .5, *np.unique(p)]
            mcc_ideal = max(matthews_corrcoef(y,(p>=t).astype(int)) for t in limiares)
            assert r["mcc"] == pytest.approx(mcc_ideal, abs=1e-12)


def test_permutacao_da_ordem_nao_altera_resultados(cfg):
    y = np.array([0,1,0,1,0,1,1,0])
    p = np.array([.5,.7,.1,.9,.8,.3,.7,.2])
    idx = np.array([5,0,7,4,1,2,6,3])
    a = avaliar_probabilidades(y,p,cfg)
    b = avaliar_probabilidades(y[idx],p[idx],cfg)
    for k in ("ap", "mcc", "logloss", "score", "limiar_mcc"):
        assert a[k] == pytest.approx(b[k])


def test_arrays_nao_modificados(cfg):
    y = np.array([0,1,0,1])
    p = np.array([.1,.3,.3,.6])
    antes = y.copy(), p.copy()
    avaliar_probabilidades(y,p,cfg)
    np.testing.assert_array_equal(y,antes[0])
    np.testing.assert_array_equal(p,antes[1])


@pytest.mark.parametrize("altera,msg", [
    (lambda c:c["avaliacao"].__setitem__("metrica_pr","auc_trapezoidal"), "average_precision"),
    (lambda c:c["avaliacao"].__setitem__("calcular_mcc",False), "calcular_mcc"),
    (lambda c:c["avaliacao"].__setitem__("calcular_logloss",False), "calcular_logloss"),
    (lambda c:c["avaliacao"].__setitem__("calcular_composite",False), "calcular_composite"),
    (lambda c:c["avaliacao"].__setitem__("selecao_limiar","mcc_teste"), "selecao_limiar"),
    (lambda c:c["avaliacao"].__setitem__("formula_composite","AP+MCC"), "formula_composite"),
    (lambda c:c["avaliacao"].__setitem__("congelar_modelo_antes_teste",False), "congelar_modelo_antes_teste"),
])
def test_nao_ignora_parametro_yaml(cfg, altera, msg):
    altera(cfg)
    with pytest.raises(ValueError, match=msg):
        avaliar_probabilidades([0,1], [.1,.9], cfg)


def test_oof_consolidado_aceita_nove_folds(cfg,oof):
    df, f, y = oof
    res = avaliar_oof_consolidado(df,f,cfg,y_referencia=y)
    assert res["tipo_avaliacao"] == "OOF_9_FOLDS"
    assert res["quantidade_folds_cv"] == 9
    assert res["n"] == 36


def test_oof_pode_chegar_desordenado(cfg,oof):
    df,f,y=oof
    x=avaliar_oof_consolidado(df,f,cfg,y_referencia=y)
    z=avaliar_oof_consolidado(df.sample(frac=1,random_state=42),f,cfg,y_referencia=y)
    assert x==z


def test_oof_nega_f10(cfg,oof):
    df,f,y=oof
    extra = pd.DataFrame({"linha_original":[9],"fold":[10],"y_true":[1],"prob_fraude":[0.9]})
    with pytest.raises(ValueError, match="F10"):
        avaliar_oof_consolidado(pd.concat([df,extra],ignore_index=True), f,cfg)


def test_oof_incompleto_falha(cfg,oof):
    df,f,y=oof
    with pytest.raises(ValueError, match="incompleto"):
        avaliar_oof_consolidado(df.iloc[:-1],f,cfg)


def test_oof_indice_duplicado_falha(cfg,oof):
    df,f,y=oof
    x=df.copy()
    x.iloc[-1,x.columns.get_loc("linha_original")]=int(x.iloc[0]["linha_original"])
    with pytest.raises(ValueError, match="duplicados"):
        avaliar_oof_consolidado(x,f,cfg)


def test_oof_fold_trocado_falha(cfg,oof):
    df,f,y=oof
    x=df.copy()
    x.loc[0,"fold"] = 9
    with pytest.raises(ValueError, match="folds"):
        avaliar_oof_consolidado(x,f,cfg)


def test_oof_rotulos_trocados_falha(cfg,oof):
    df,f,y=oof
    x=df.copy()
    x.loc[0,"y_true"] = 1 - x.loc[0,"y_true"]
    with pytest.raises(ValueError, match="rotulos originais"):
        avaliar_oof_consolidado(x,f,cfg,y_referencia=y)


def test_oof_coluna_ausente(cfg,oof):
    df,f,y=oof
    with pytest.raises(ValueError, match="colunas"):
        avaliar_oof_consolidado(df.drop(columns=["prob_fraude"]),f,cfg)


def test_oof_f10_rotulo_pode_mudar_sem_afetar_resultado(cfg,oof):
    df,f,y=oof
    antes=avaliar_oof_consolidado(df,f,cfg,y_referencia=y)
    outro=y.copy()
    outro[f==10]=1-outro[f==10]
    depois=avaliar_oof_consolidado(df,f,cfg,y_referencia=outro)
    assert antes == depois
