# -*- coding: utf-8 -*-
"""Teste isolado: somente GMM07/GMM08 1D PCA e t-SNE, SEM treino/F10 real.

Executar na raiz do projeto apos instalar a V3:
    python .\\TESTAR_AJUSTE_GMM07_GMM08_1D_V3_ISOLADO.py

Cria uma pasta nova de validacao sob outputs, sem sobrescrever experimentos.
Checa automaticamente que:
- subtitulo F10 e rodape foram removidos no 1D;
- cabecalho de risco e limiares permanecem;
- no GMM08 1D cada componente possui intervalo central 95% desenhado;
- GMM07 1D nao possui essas bandas;
- o cruzamento F10 simulado continua certificado.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


def modelo_1d_sintetico():
    gm = GaussianMixture(n_components=3, covariance_type='full', random_state=2026)
    gm.weights_ = np.array([.35, .35, .30])
    gm.means_ = np.array([[-2.0], [0.0], [2.0]], dtype=float)
    gm.covariances_ = np.array([[[.60]], [[1.0]], [[.85]]], dtype=float)
    gm.precisions_cholesky_ = np.stack([
        np.linalg.cholesky(np.linalg.inv(m)).T for m in gm.covariances_])
    gm.n_features_in_ = 1
    gm.converged_ = True
    return gm


def main():
    from src.gmm_contornos_quatro_niveis import gerar
    from src.gmm_fronteira_f10 import diagnosticar
    raiz = Path(__file__).resolve().parent
    caminho = raiz / 'outputs' / ('validacao_GMM07_GMM08_1D_95_V3_' +
                                 datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    caminho.mkdir(parents=True, exist_ok=False)
    gm = modelo_1d_sintetico()
    riscos = np.array([.003, .006, .020], dtype=float)
    tau = .010
    rng = np.random.default_rng(20261106)
    # Nenhuma operação de .fit() é utilizada.
    X = np.sort(np.r_[rng.normal(-2.0, .85, 80), rng.normal(2.0, .8, 25)]).reshape(-1, 1)
    y = np.zeros(len(X), dtype=np.int8)
    y[::10] = 1
    scores = gm.predict_proba(X) @ riscos
    certificado = diagnosticar(gm, riscos, X, scores, tau)
    assert certificado['cruzamento_confirmado'], certificado
    observado = []
    original_savefig = Figure.savefig

    def conferir_figura(fig, destino, **kwargs):
        nome = Path(destino).name
        textos = [t.get_text() for t in fig.texts]
        assert not any('F10 | 1D FULL' in s for s in textos), (nome, textos)
        assert not any('Positivos na MALHA' in s for s in textos), (nome, textos)
        assert not any('Predições positivas F10' in s for s in textos), (nome, textos)
        assert any('Malha s(x)' in s for s in textos), (nome, textos)
        ax = fig.axes[0]
        legenda = ax.get_legend()
        if nome.startswith('GMM08'):
            assert legenda is not None, 'Intervalos gaussianos 95% sem legenda'
            legend_text = [t.get_text() for t in legenda.get_texts()]
            assert len(legend_text) == gm.n_components, legend_text
            assert all(f'G{k+1} 95%' in legend_text[k] for k in range(gm.n_components)), legend_text
            assert len(ax.patches) >= gm.n_components, 'Faixas gaussianas 95% ausentes'
            # Verifica limites matemáticos, não apenas o visual.
            z = float(norm.ppf(.975))
            for k in range(gm.n_components):
                mu, sig = float(gm.means_[k,0]), float(np.sqrt(gm.covariances_[k,0,0]))
                a, b = mu - z*sig, mu + z*sig
                assert f'[{a:.2f}, {b:.2f}]' in legend_text[k], (k, legend_text[k], a, b)
        else:
            assert legenda is None, 'GMM07 não deve receber faixas gaussianas 95%'
        observado.append(nome)
        return original_savefig(fig, destino, **kwargs)

    cfg = {'experimento': {'semente': 2026}, 'visualizacao': {'graficos': {
        'sensibilidade_fronteira': {'resolucao_1d': 210, 'n_pontos_scatter': 90},
        'mapa_risco_fronteira': {'resolucao_1d': 210, 'n_pontos_scatter': 90},
    }}}
    escolha = {'limiar_mcc_oof': tau}
    for metodo in ('PCA', 'TSNE'):
        dados = {'coord': pd.DataFrame(X, columns=['PC1' if metodo == 'PCA' else 'TSNE1']),
                 'y': y, 'scores': scores,
                 'gmm': SimpleNamespace(modelo_=gm),
                 'rh': SimpleNamespace(riscos_=pd.Series(riscos, index=['G1','G2','G3']))}
        destino = caminho / metodo / 'vencedor' / 'teste_final' / 'graficos_gerados'
        destino.mkdir(parents=True)
        for tipo in ('GMM07','GMM08'):
            with patch.object(Figure, 'savefig', conferir_figura):
                gerar(cfg, dados, escolha, destino, 'png', 95, False, tipo=tipo,
                      avaliacao={'origem': 'f10_amostral_congelado'})
            arq = destino / (f'{tipo}_' + ('sensibilidade' if tipo == 'GMM07' else 'mapa_risco') +
                              '_1d_comparativo.png')
            assert arq.exists() and arq.stat().st_size > 4_000, arq
            print(f'[APROVADO] {metodo} 1D {tipo}: PNG, limpeza e intervalo 95% conferidos.')
    assert len(observado) == 4
    plt.close('all')
    print('\n[TESTE V3 APROVADO] 4 PNGs 1D gerados, sem fit e sem ler F10 real.')
    print('[PASTA]', caminho)


if __name__ == '__main__':
    main()
