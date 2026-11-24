"""Instrumentacao observacional do GaussianMixture para as n_init reais.

Nao muda EM, RNG, criterios de parada ou escolha do melhor ajuste.
Depende de hooks privados do scikit-learn: conferir ao atualizar a versao.
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


class GaussianMixtureComHistorico(GaussianMixture):
    """Registra o lower bound medio de CADA iteracao de CADA init real."""

    def _print_verbose_msg_init_beg(self, n_init):
        if not hasattr(self, "historicos_em_"):
            self.historicos_em_ = []
        registro = {
            "init": len(self.historicos_em_) + 1,
            "lower_bounds": [],
            "convergiu": None,
        }
        self.historicos_em_.append(registro)
        self._registro_em_atual = registro
        return super()._print_verbose_msg_init_beg(n_init)

    def _compute_lower_bound(self, *args, **kwargs):
        valor = super()._compute_lower_bound(*args, **kwargs)
        registro = getattr(self, "_registro_em_atual", None)
        if registro is not None:
            registro["lower_bounds"].append(float(np.asarray(valor).item()))
        return valor

    def _print_verbose_msg_init_end(self, lower_bound, converged):
        registro = getattr(self, "_registro_em_atual", None)
        if registro is not None:
            registro["convergiu"] = bool(converged)
            serie = registro["lower_bounds"]
            if not serie or not np.isclose(
                serie[-1], float(np.asarray(lower_bound).item()),
                rtol=1e-10, atol=1e-10,
            ):
                raise RuntimeError("Historico EM divergente do lower bound final.")
        resultado = super()._print_verbose_msg_init_end(lower_bound, converged)
        self._registro_em_atual = None
        return resultado

    def validar_historicos_em(self):
        """Relaciona a melhor init ao mesmo lower_bound_ escolhido pelo sklearn."""
        historicos = getattr(self, "historicos_em_", None)
        if not historicos or len(historicos) != self.n_init:
            raise RuntimeError("Nem todas as inicializacoes do EM foram registradas.")
        if any(not x["lower_bounds"] or x["convergiu"] is None for x in historicos):
            raise RuntimeError("Existe inicializacao sem historico/estado de convergencia.")
        finais = [h["lower_bounds"][-1] for h in historicos]
        melhor = int(np.argmax(finais))
        if not np.isclose(finais[melhor], self.lower_bound_, rtol=1e-10, atol=1e-10):
            raise RuntimeError("Melhor inicializacao divergente do GMM selecionado.")
        if len(historicos[melhor]["lower_bounds"]) != int(self.n_iter_):
            raise RuntimeError("Numero de iteracoes vencedor divergente.")
        return melhor + 1
