"""Smoke test GMM18 com dados sintéticos; não usa dados do TCC."""
import numpy as np
from sklearn.mixture import GaussianMixture
from src.gmm_convergencia_em import GaussianMixtureComHistorico

rng = np.random.RandomState(422)
X = np.r_[rng.normal(-2.0, .7, (130, 1)),
          rng.normal(1.6, .5, (160, 1)),
          rng.normal(4.5, .9, (110, 1))]
param = dict(n_components=3, covariance_type='full', reg_covar=1e-6,
             max_iter=100, n_init=3, tol=1e-3,
             init_params='kmeans', random_state=42)
normal = GaussianMixture(**param).fit(X)
registrado = GaussianMixtureComHistorico(**param).fit(X)
melhor = registrado.validar_historicos_em()
assert len(registrado.historicos_em_) == 3
for atributo in ('means_', 'weights_', 'covariances_'):
    assert np.array_equal(getattr(normal, atributo), getattr(registrado, atributo)), atributo
assert np.array_equal(normal.predict_proba(X), registrado.predict_proba(X))
assert np.isclose(normal.lower_bound_, registrado.lower_bound_)
assert normal.n_iter_ == registrado.n_iter_
print('GMM18 SMOKE TEST OK')
print('Scikit-learn original e instrumentado: mesmo ajuste e mesmos scores.')
print('Inicializacoes:', len(registrado.historicos_em_), '| Escolhida:', melhor)
print('Iteracoes registradas:', [len(h['lower_bounds']) for h in registrado.historicos_em_])
print('Lower bound escolhido:', f'{registrado.lower_bound_:.7f}')
print('Nenhum dado real ou F10 foi acessado.')
