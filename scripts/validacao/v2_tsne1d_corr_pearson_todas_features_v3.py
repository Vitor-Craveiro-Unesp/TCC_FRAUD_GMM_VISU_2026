# -*- coding: utf-8 -*-
"""Prévia exploratória TSNE1: Pearson por feature, no estilo do grafico PCA 1D.
Amostra unicamente F1-F9 de um ensaio PCA existente. Nenhum F10/modelo oficial.
"""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src import orquestracao_oof as orq
from src import preprocessing as prep
from src import visualizacao_pca_biplot as biplot
from src.reducao_tsne import RedutorTSNE

RAIZ = Path.cwd()
EXP = RAIZ / 'outputs/ensaio_pca_gmm_20261030_114404'
SAIDA = RAIZ / 'outputs/validacao_tsne_v2_1d/TSNE_EXTRA_CORRELACAO_PEARSON_V3'
PNG = SAIDA / 'TSNE_CORRELACAO_PEARSON_TODAS_FEATURES.png'
CSV = SAIDA / 'TSNE_CORRELACAO_PEARSON_TODAS_FEATURES.csv'

cfg_path = EXP / 'configuracao.yaml'
man_path = EXP / 'manifesto.json'
npz_path = EXP / 'vencedor/oof/predicoes.npz'
protegidos = [
    cfg_path, man_path, npz_path,
    RAIZ / 'src/preprocessing.py',
    RAIZ / 'src/reducao_tsne.py',
    RAIZ / 'src/visualizacao_pca_biplot.py',
    RAIZ / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml',
]
for p in protegidos:
    if not p.is_file():
        raise FileNotFoundError(f'Arquivo de referencia ausente: {p}')
if SAIDA.exists():
    raise FileExistsError(f'Saida ja existente; nada foi sobrescrito: {SAIDA}')

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

hashes_antes = {str(p): sha(p) for p in protegidos}
cfg_pca = yaml.safe_load(cfg_path.read_text(encoding='utf-8-sig'))
man = json.loads(man_path.read_text(encoding='utf-8-sig'))
assert cfg_pca['reducao']['metodo'].upper() == 'PCA'
assert cfg_pca['divisao']['qual_fold_teste'] == 10
assert man['status'] == 'concluido_06_03'
assert man['tipo'] == 'ensaio'

with np.load(npz_path, allow_pickle=False) as arq:
    ids = np.asarray(arq['linha_original'], dtype=np.int64)
    folds_oof = np.asarray(arq['fold'], dtype=np.uint8)
    fold_esperado = int(arq['fold_esperado'].item())
assert fold_esperado == 0
assert len(ids) > 1200 and len(ids) == len(folds_oof)
assert np.all(np.diff(ids) > 0)
assert set(np.unique(folds_oof)) == set(range(1, 10))

parquet = RAIZ / cfg_pca['dados']['arquivo']
folds, caminho_folds, h_parquet = orq._identificar_folds(RAIZ, cfg_pca, parquet)
assert man['sha256_parquet'] == h_parquet
assert man['sha256_folds'] == orq.hash_arquivo(caminho_folds)
np.testing.assert_array_equal(folds[ids], folds_oof)
assert np.all(folds[ids] != 10)

print(f'REGISTROS_F1_F9_REFERENCIA={len(ids)}', flush=True)
print('LENDO_APENAS_LINHAS_SELECIONADAS_F1_F9=SIM', flush=True)
df = orq._ler_desenvolvimento(parquet, ids)
assert len(df) == len(ids)
X_raw = prep.selecionar_features(df, cfg_pca)
del df

# Fit de referencia F1-F9, somente descritivo. Preserva todas as features OHE/BE/FEWF.
enc = prep.EncodersPorSufixo(cfg_pca).fit(X_raw)
X_codificado = enc.transform(X_raw)
imp = prep.ImputadorPadronizador(cfg_pca, enc).fit(X_codificado)
del X_codificado

n_originais = len(X_raw.columns)
n_features = len(enc.saidas_)
assert n_features == len(set(enc.saidas_)) and n_features >= n_originais

# Amostragem apenas de observacoes. NENHUMA feature sera descartada.
rng = np.random.default_rng(42)
n_amostra = 1200
pos = np.sort(rng.choice(len(X_raw), size=n_amostra, replace=False))
X_sample = X_raw.iloc[pos].copy()
X_pre = imp.transform(enc.transform(X_sample))
del X_raw, X_sample
assert X_pre.shape == (n_amostra, n_features)
assert list(X_pre.columns) == enc.saidas_
assert np.isfinite(X_pre.to_numpy()).all()

cfg = deepcopy(cfg_pca)
cfg['reducao']['metodo'] = 'TSNE'
cfg['experimento']['semente'] = 42
cfg['execucao']['n_jobs'] = 2
params_tsne = {
    'perplexity': 10,
    'n_components': 1,
    'initialization': 'pca',
    'metric': 'euclidean',
    'learning_rate': 'auto',
    'early_exaggeration': 12.0,
    'n_iter': 300,
    'theta': 0.5,
}
red = RedutorTSNE(cfg, params_tsne).fit(X_pre)
Z = red.transform_treino()
assert Z.shape == (n_amostra, 1)
assert Z.index.equals(X_pre.index)

# Correlação de Pearson, na mesma metrica do grafico PCA, mas com TSNE1.
registros = []
for feature in X_pre.columns:
    s = X_pre[feature]
    constante = s.nunique(dropna=False) <= 1
    rho = np.nan if constante else float(s.corr(Z['TSNE1'], method='pearson'))
    if not constante:
        assert np.isfinite(rho) and -1.0000001 <= rho <= 1.0000001
    registros.append({
        'variavel_preprocessada': feature,
        'correlacao_pearson_TSNE1': rho,
        'modulo_correlacao': abs(rho) if np.isfinite(rho) else np.nan,
        'constante_na_amostra': bool(constante),
    })

tabela = pd.DataFrame(registros)
tabela.sort_values(
    ['modulo_correlacao', 'variavel_preprocessada'],
    ascending=[False, True], na_position='last', kind='stable', inplace=True,
)
tabela.reset_index(drop=True, inplace=True)
assert len(tabela) == n_features
assert tabela['variavel_preprocessada'].is_unique
assert set(tabela['variavel_preprocessada']) == set(X_pre.columns)

# REUTILIZACAO LITERAL DO RENDERIZADOR PCA 1D PARA MESMO PADRAO GRAFICO.
# Coluna temporaria corr_PC1 e somente um adaptador da interface do desenho.
# NENHUM PCA e ajustado; dados sao correlacoes com TSNE1.
selecionadas = pd.DataFrame({
    'variavel_preprocessada': tabela['variavel_preprocessada'],
    'corr_PC1': tabela['correlacao_pearson_TSNE1'].fillna(0).to_numpy(),
})
fig = biplot._fig_1d(
    Z['TSNE1'].to_numpy(),
    np.zeros(n_amostra, dtype=np.uint8),
    np.arange(n_amostra), cfg, selecionadas, None,
)
fig._suptitle.set_text('CORRELAÇÃO DAS VARIÁVEIS COM O t-SNE 1D')
fig.axes[0].set_xlabel('Correlação de Pearson com TSNE1')
# Colunas constantes NAO possuem correlacao definida; anotar explicitamente.
for i, constante in enumerate(tabela['constante_na_amostra'].to_numpy()):
    if constante:
        fig.axes[0].texts[i].set_text('n/d (constante)')

SAIDA.mkdir(parents=True, exist_ok=False)
try:
    fig.savefig(PNG, dpi=180, bbox_inches='tight')
finally:
    plt.close(fig)
tabela.to_csv(CSV, index=False, encoding='utf-8-sig', float_format='%.17g')

manifesto = {
    'tipo': 'PREVIA_EXPLORATORIA_NAO_OOF',
    'grafico': 'CORRELACAO_TODAS_FEATURES_COM_TSNE1',
    'origem_folds': 'F1-F9 do ensaio PCA 1D; F10 excluido',
    'n_referencia': len(ids),
    'n_amostra_embedding': n_amostra,
    'n_features_originais': n_originais,
    'n_features_preprocessadas': n_features,
    'n_features_no_grafico': len(tabela),
    'n_features_constantes': int(tabela['constante_na_amostra'].sum()),
    'metrica': 'correlacao_de_Pearson',
    'coeficientes_lineares_tsne': False,
    'orientacao_tsne_arbitraria': True,
    'ajuste_tsne_em_dados_reais_F1_F9': True,
    'oof_tsne_completo_executado': False,
    'f10_incluido_na_amostra': False,
    'renderizador': 'src.visualizacao_pca_biplot._fig_1d (somente estilo)',
}
(SAIDA / 'manifesto.json').write_text(
    json.dumps(manifesto, ensure_ascii=False, indent=2), encoding='utf-8',
)
assert PNG.is_file() and PNG.stat().st_size > 1000
assert CSV.is_file() and len(pd.read_csv(CSV)) == n_features
assert hashes_antes == {str(p): sha(p) for p in protegidos}

print('\n=== TSNE 1D - PEARSON - TODAS AS FEATURES ===')
print(f'FEATURES_ORIGINAIS={n_originais}')
print(f'FEATURES_PREPROCESSADAS={n_features}')
print(f'FEATURES_NO_GRAFICO={len(tabela)}')
print(f'FEATURES_CONSTANTES={int(tabela["constante_na_amostra"].sum())}')
print('METRICA=PEARSON')
print('PADRAO_VISUAL_PCA_REUTILIZADO=SIM')
print('TOP_N_APLICADO=NAO')
print('F10_INCLUIDO_NA_ANALISE=NAO')
print('OOF_TSNE_COMPLETO=NAO')
print('ARQUIVOS_OFICIAIS_ALTERADOS=NAO')
print(f'PNG={PNG}')
print(f'CSV={CSV}')
print('TSNE_PEARSON_V3_VALIDACAO=AGUARDANDO_AVALIACAO_VISUAL')
