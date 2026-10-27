"""Previa isolada PCA12 3D com 3 gaussianas, sem alterar YAML ou F10."""
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from src import visualizacao_geometrica as geo

raiz = Path.cwd()
yaml_arquivo = raiz / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
assert yaml_arquivo.is_file(), f'YAML nao encontrado: {yaml_arquivo}'
config = yaml.safe_load(yaml_arquivo.read_text(encoding='utf-8-sig'))
pca_original = list(config['pca']['n_components'])

rng = np.random.default_rng(20261027)
centros = np.array([[-2.5, -1.4, -.5], [2.5, -1.0, 1.3], [-.3, 2.6, -1.6]])
covs = np.array([
    [[.6, .2, .05], [.2, .8, -.1], [.05, -.1, .65]],
    [[.8, -.2, .13], [-.2, .6, .02], [.13, .02, .70]],
    [[.75, .15, -.10], [.15, .9, .2], [-.10, .2, .5]],
])
latente = np.vstack([rng.multivariate_normal(centros[k], covs[k], size=420) for k in range(3)])
pesos = rng.normal(size=(3, 39))
x = latente @ pesos + rng.normal(scale=.53, size=(len(latente), 39))
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=3, random_state=20261027).fit(x)
z = pca.transform(x)
gmm = GaussianMixture(n_components=3, covariance_type='full', n_init=3,
                      max_iter=300, random_state=20261027).fit(z)
gamma = gmm.predict_proba(z)
riscos = np.array([.025, .15, .48])
scores = gamma @ riscos
y = rng.binomial(1, scores)
assert z.shape == (1260, 3)
assert gmm.means_.shape == (3, 3) and gmm.covariances_.shape == (3, 3, 3)
assert config['pca']['n_components'] == pca_original

op = dict(config['visualizacao']['graficos']['pca_projecoes'])
op['colorir_por'] = 'classe_real'
op['mostrar_contornos_gaussianas'] = True
config['visualizacao']['geometria']['pontos_maximos_amostra'] = 2000
saida = raiz / 'outputs' / (
    'validacao_pca3d_01_projecao_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
)
saida.mkdir(parents=True, exist_ok=False)
dados = {
    'coord': pd.DataFrame(z, columns=['PC1', 'PC2', 'PC3']),
    'redutor': SimpleNamespace(modelo_=pca),
    'gmm': SimpleNamespace(modelo_=gmm, saidas_=['G1', 'G2', 'G3']),
    'gamma': pd.DataFrame(gamma, columns=['G1', 'G2', 'G3']),
    'scores': scores,
    'y': y,
    'limiar_oof': .5,
}
geo._pca_projecao_2d3d(config, dados, saida, 'png', 170, True, op, origem='simulado')
png = saida / '12_pca_projecao_PC1_PC2_PC3.png'
csv = saida / '12_pca_projecao_PC1_PC2_PC3.csv'
assert png.is_file() and csv.is_file(), 'Arquivos PCA3D nao gerados'
t = pd.read_csv(csv)
assert {'PC1', 'PC2', 'PC3', 'classe_real', 'gaussiana_dominante'}.issubset(t.columns)
assert len(t) == len(z)
assert yaml.safe_load(yaml_arquivo.read_text(encoding='utf-8-sig'))['pca']['n_components'] == pca_original
print('PCA12 3D: PREVIA CONCLUIDA')
print('Observacoes sinteticas:', len(z), '| Gaussianas:', gmm.n_components)
print('F10 nao usado. YAML nao alterado.')
print(str(png.resolve()))
