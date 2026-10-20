"""Graficos t-SNE do catalogo PDF: 02, 13 em 3D e 14.

Toda entrada e proveniente do DESENVOLVIMENTO F1..F9. Nenhuma funcao
consulta dados de teste F10, reescolhe o vencedor OOF ou altera o GMM.
"""
from __future__ import annotations

from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.manifold import trustworthiness


def _geometria():
    # Importacao tardia: visualizacao_geometrica delega para este modulo.
    from src import visualizacao_geometrica as geo
    return geo


def _coordenadas(data, n_dim):
    geo = _geometria()
    red = data['redutor']
    parametros = getattr(red, 'parametros', None)
    coord = data['coord']
    geo._exigir(isinstance(parametros, dict) and
                int(parametros.get('n_components', 0)) == n_dim,
                f'Grafico t-SNE exige redutor ajustado em {n_dim}D.')
    esperado = [f'TSNE{i}' for i in range(1, n_dim + 1)]
    geo._exigir(isinstance(coord, pd.DataFrame) and
                list(coord.columns) == esperado and coord.shape[1] == n_dim,
                'Colunas do embedding t-SNE nao conferem com a dimensionalidade.')
    X = coord.to_numpy(dtype=float)
    y = np.asarray(data['y']).ravel()
    geo._exigir(len(X) == len(y) and len(X) >= 4 and
                np.isfinite(X).all() and np.isin(y, [0, 1]).all(),
                'Embedding t-SNE e classes invalidos ou desalinhados.')
    return X, y, parametros


def projecao_3d(config, data, dest, ext, dpi, csv, op, *, origem='desenvolvimento'):
    """Uma imagem 3D REAL, e nao tres scatterplots em 2D."""
    geo = _geometria()
    geo._exigir(origem in ('desenvolvimento', 'simulado'), 'Origem invalida.')
    Z, y, par = _coordenadas(data, 3)
    n = len(y)
    limite = int(config['visualizacao']['geometria'].get('pontos_maximos_amostra', 20000))
    indices = geo._amostra_plot(n, y, config, limite)
    colorir = op.get('colorir_por', 'classe_real')
    geo._exigir(colorir in ('classe_real', 'classe_predita', 'risco', 'gaussiana_dominante'),
                't-SNE 3D: colorir_por invalido.')
    fig = plt.figure(figsize=(11.5, 8.2))
    ax = fig.add_subplot(111, projection='3d')
    estilos = config['visualizacao']['classes']
    scores = np.asarray(data['scores'], dtype=float).ravel()
    geo._exigir(len(scores) == n and np.isfinite(scores).all() and
                np.all((0 <= scores) & (scores <= 1)), 'Scores invalidos.')
    if colorir in ('classe_real', 'classe_predita'):
        if colorir == 'classe_real':
            grupos = y
        else:
            tau = float(data['limiar_oof'])
            geo._exigir(np.isfinite(tau) and 0 <= tau <= 1,
                        'Limiar MCC OOF invalido.')
            grupos = (scores >= tau).astype(np.uint8)
        for classe, chave in ((0, 'legitima'), (1, 'fraude')):
            ix = indices[grupos[indices] == classe]
            if not len(ix):
                continue
            estilo = estilos[chave]
            ax.scatter(Z[ix, 0], Z[ix, 1], Z[ix, 2],
                       c=estilo.get('cor', '#428bc5' if classe == 0 else '#de4040'),
                       s=estilo.get('tamanho', 18) * (2.2 if classe else 1),
                       alpha=estilo.get('alpha', .65),
                       marker=estilo.get('marcador', '*' if classe else 'o'),
                       depthshade=False, rasterized=True,
                       label=f"{estilo.get('nome', chave)} (n={len(ix)} exibidas)")
    elif colorir == 'risco':
        nuvem = ax.scatter(Z[indices, 0], Z[indices, 1], Z[indices, 2],
                          c=scores[indices], vmin=0, vmax=1, cmap='viridis',
                          s=16, alpha=.67, depthshade=False, rasterized=True)
        fig.colorbar(nuvem, ax=ax, shrink=.62, pad=.13,
                     label='Risco de fraude estimado (desenvolvimento)')
    else:
        geo._exigir('gamma' in data and 'gmm' in data,
                    'Coloracao por gaussiana exige responsabilidades do GMM.')
        grupos = np.argmax(data['gamma'].to_numpy(), axis=1)
        geo._exigir(len(grupos) == n, 'Responsabilidades desalinhadas.')
        for k in np.unique(grupos[indices]):
            ix = indices[grupos[indices] == k]
            ax.scatter(Z[ix, 0], Z[ix, 1], Z[ix, 2],
                       color=plt.get_cmap('tab10')(int(k) % 10),
                       s=16, alpha=.66, depthshade=False, rasterized=True,
                       label=f'G{k+1} — maior responsabilidade (n={len(ix)})')
    ax.set_xlabel('t-SNE 1', labelpad=11)
    ax.set_ylabel('t-SNE 2', labelpad=11)
    ax.set_zlabel('t-SNE 3', labelpad=11)
    ax.view_init(elev=23, azim=-59)
    if colorir != 'risco':
        ax.legend(loc='upper left', fontsize=8)
    titulo = '13 — t-SNE 3D | Projecao t-SNE 1 × t-SNE 2 × t-SNE 3'
    fig.suptitle(titulo, fontsize=14, y=.965)
    fig.text(.5, .927,
             f"n_components=3 | perplexity={par.get('perplexity', 'n/d')} | "
             f"métrica={par.get('metric', 'n/d')} | random_state="
             f"{config['experimento']['semente']}",
             fontsize=9, ha='center', color='#374151')
    origem_txt = ('PRÉVIA SINTÉTICA' if origem == 'simulado' else
                  'Desenvolvimento F1–F9')
    fig.text(.5, .044, f'{origem_txt}; F10 não utilizado. '
             'Leitura local: não superinterpretar distâncias globais do t-SNE. '
             f'{len(indices)} pontos exibidos de {n}.',
             fontsize=8, ha='center', color='#475569')
    fig.subplots_adjust(left=.02, right=.95, top=.88, bottom=.09)
    nome = '13_tsne_mapa_TSNE1_TSNE2_TSNE3'
    geo._guardar(fig, dest, nome, ext, dpi)
    if csv:
        # Nao mascarar amostragem como conjunto completo: indices originais salvos.
        campos = {'linha_indice_desenvolvimento': indices,
                  'classe_real': y[indices], 'risco_estimado_treino': scores[indices],
                  'TSNE1': Z[indices, 0], 'TSNE2': Z[indices, 1],
                  'TSNE3': Z[indices, 2]}
        if 'gamma' in data:
            campos['gaussiana_dominante'] = (np.argmax(
                data['gamma'].to_numpy()[indices], axis=1) + 1)
        geo._salvar_dados(pd.DataFrame(campos), dest, nome, True)
    return nome


def trustworthiness_perplexity(config, data, dest, ext, dpi, csv, *, origem='desenvolvimento'):
    """Gráfico 02 do catálogo t-SNE; *nao* o MCC OOF 02.

    Reajustes somente em amostra dos atributos preprocessados de F1..F9,
    usando mesmos hiperparametros do vencedor exceto perplexity.
    Nao altera objeto RedutorTSNE vencedor nem sua pontuacao.
    """
    geo = _geometria()
    geo._exigir(origem in ('desenvolvimento', 'simulado'), 'Origem invalida.')
    cfg = config['visualizacao']['graficos']['tsne_trustworthiness']
    red = data['redutor']
    params = getattr(red, 'parametros', None)
    geo._exigir(isinstance(params, dict), 'Redutor t-SNE sem parametros originais.')
    fonte = data.get('pre')
    geo._exigir(isinstance(fonte, pd.DataFrame) and fonte.shape[1] >= 1 and
                len(fonte) >= 4, 'Diagnostico 02 requer preprocessamento F1..F9.')
    x = fonte.to_numpy(dtype=float)
    geo._exigir(np.isfinite(x).all(), 'Preprocessamento t-SNE com nao finitos.')
    maximo = int(cfg.get('tamanho_amostra_maximo', 360))
    geo._exigir(20 <= maximo <= 2000, 'tamanho_amostra_maximo fora de 20..2000.')
    seed = int(config['experimento']['semente'])
    rng = np.random.default_rng(seed)
    idx = (np.sort(rng.choice(len(x), size=maximo, replace=False))
           if len(x) > maximo else np.arange(len(x)))
    amostra = fonte.iloc[idx].copy()
    ks = int(cfg.get('vizinhos_trustworthiness', 10))
    geo._exigir(1 <= ks < (len(idx) - 1) / 2,
                'k deve ser menor que (n_amostra-1)/2 no trustworthiness.')
    valores = cfg.get('perplexidades', [5, 10, 20, 30, 50, 80])
    geo._exigir(isinstance(valores, list) and valores,
                'Lista de perplexidades vazia/invalida.')
    itens = [float(p) for p in valores]
    escolhida = float(params['perplexity'])
    if cfg.get('incluir_perplexity_vencedora', True):
        itens.append(escolhida)
    geo._exigir(all(np.isfinite(p) and p > 0 for p in itens),
                'Perplexidades invalidas.')
    itens = sorted(set(itens))
    # Remover apenas valores inviaveis para o tamanho da amostra;
    # nunca inventar dados ausentes.
    puladas = [p for p in itens if p >= len(idx)]
    itens = [p for p in itens if p < len(idx)]
    geo._exigir(itens, 'Nenhuma perplexidade cabe na amostra.')
    rows = []
    from src.reducao_tsne import RedutorTSNE
    for p in itens:
        p_atual = dict(params, perplexity=p)
        t0 = perf_counter()
        modelo = RedutorTSNE(config, p_atual).fit(amostra)
        Z = modelo.transform_treino().to_numpy(dtype=float)
        t = float(trustworthiness(amostra.to_numpy(dtype=float), Z,
                                  n_neighbors=ks, metric=params['metric']))
        geo._exigir(np.isfinite(t) and 0 <= t <= 1,
                    'Trustworthiness fora da faixa 0 a 1.')
        rows.append({'perplexity': p, 'trustworthiness': t,
                     'k_vizinhos': ks, 'n_amostra': len(idx),
                     'n_dim_reducao': int(params['n_components']),
                     'perplexity_vencedora': escolhida,
                     'tempo_ajuste_s': perf_counter() - t0})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.plot(df['perplexity'], df['trustworthiness'], 'o-',
            lw=2.1, ms=6.5, color='#1565c0',
            label=f'Trustworthiness (k={ks})')
    for p, t in zip(df['perplexity'], df['trustworthiness']):
        ax.annotate(f'{t:.3f}', (p, t), xytext=(0, 9),
                    textcoords='offset points', ha='center', fontsize=8.5)
    if any(np.isclose(df['perplexity'], escolhida)):
        ax.axvline(escolhida, color='#d97706', ls='--', lw=1.4,
                   label=f'Perplexity do vencedor OOF = {escolhida:g}')
    else:
        ax.text(.02,.06,f'Perplexity do vencedor OOF ({escolhida:g}) fora da amostra',
                transform=ax.transAxes,fontsize=9)
    ax.set_xticks(df['perplexity'].to_list())
    minimo = max(0, float(df['trustworthiness'].min()) - .025)
    ax.set_ylim(minimo, 1.003)
    ax.set(xlabel='Perplexity', ylabel='Trustworthiness (0 a 1; eixo ampliado)')
    ax.set_title('t-SNE 02 — Preservação de vizinhanças | Trustworthiness × Perplexity')
    ax.grid(alpha=.2)
    ax.legend(loc='best', fontsize=9)
    aviso = 'Prévia sintética' if origem == 'simulado' else 'Desenvolvimento F1–F9'
    fig.text(.5,.014,
             f'{aviso}: amostra não supervisionada de {len(idx)} observações; '
             f'dimensão {params["n_components"]}D; k={ks}; F10 não utilizado. '
             'Diagnóstico auxiliar, não altera seleção OOF.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.052,1,1))
    nome = 'TSNE02_trustworthiness_vs_perplexity'
    geo._guardar(fig, dest, nome, ext, dpi)
    if csv:
        geo._salvar_dados(df,dest,nome,True)
    return {'nome': nome,'n_ajustes': len(rows),'perplexidades_puladas': puladas}


def diagnostico_kl(config, data, dest, ext, dpi, csv, *, origem='desenvolvimento'):
    """KL/tempo/parametros verdadeiros do embedding, sem historico inventado."""
    geo = _geometria()
    geo._exigir(origem in ('desenvolvimento', 'simulado'), 'Origem invalida.')
    red = data['redutor']
    p = getattr(red,'parametros',None)
    geo._exigir(isinstance(p, dict), 'Diagnostico KL requer redutor ajustado.')
    op = config['visualizacao']['graficos']['tsne_diagnostico']
    valor = getattr(red, 'kl_divergence_', None)
    try:
        kl = None if valor is None else float(valor)
    except (TypeError,ValueError):
        kl = None
    if kl is not None and (not np.isfinite(kl) or kl < 0):
        kl = None
    tempo = float(data['tempo_reducao'])
    geo._exigir(np.isfinite(tempo) and tempo >= 0, 'Tempo de reducao invalido.')
    n = int(len(data['coord']))
    d = int(p['n_components'])
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.7))
    fig.patch.set_facecolor('#f8fafc')
    for ax in axs:
        ax.set_facecolor('#f1f5f9')
        ax.set_xticks([])
        ax.set_yticks([])
        for edge in ax.spines.values():
            edge.set_color('#cbd5e1')
    axs[0].text(.06,.88,'DIVERGÊNCIA KL FINAL',transform=axs[0].transAxes,
                fontweight='bold', color='#334155', fontsize=12)
    mostrarkl = op.get('mostrar_kl_final',True)
    axs[0].text(.06,.52,
                f'{kl:.6g}' if kl is not None and mostrarkl else 'Indisponível',
                transform=axs[0].transAxes,color='#1d4ed8',fontsize=24,weight='bold')
    axs[0].text(.06,.22,'Métrica de otimização interna do t-SNE.\n'
                'Não mede desempenho de detecção de fraude.',
                transform=axs[0].transAxes,color='#475569',fontsize=9.5)
    axs[1].text(.06,.88,'TEMPO REAL DA REDUÇÃO',transform=axs[1].transAxes,
                fontweight='bold',color='#334155',fontsize=12)
    axs[1].text(.06,.52, f'{tempo:.2f} s' if op.get('mostrar_tempo_execucao',True)
                else 'Oculto pela configuração', transform=axs[1].transAxes,
                color='#047857',fontsize=24,weight='bold')
    axs[1].text(.06,.22,
                f'Embedding: {d}D | perplexity={p.get("perplexity")}\n'
                f'{n:,} observações | métrica={p.get("metric", "n/d")}',
                transform=axs[1].transAxes,color='#475569',fontsize=9.5)
    fig.suptitle('14 — t-SNE | Diagnóstico KL / otimização', fontsize=15,
                 weight='bold', y=.985)
    nota = ('Histórico de KL por iteração: indisponível neste redutor; '
            'não foi criada curva sintética.')
    sufixo = 'PRÉVIA SINTÉTICA' if origem == 'simulado' else 'Desenvolvimento F1–F9'
    fig.text(.5,.065,nota,fontsize=9,ha='center',color='#b45309')
    fig.text(.5,.03,f'{sufixo}; F10 não utilizado. KL final e tempo não validam AP/MCC.',
             ha='center',fontsize=8.5,color='#64748b')
    fig.subplots_adjust(left=.055,right=.945,top=.83,bottom=.17,wspace=.085)
    nome = '14_tsne_diagnostico'
    geo._guardar(fig,dest,nome,ext,dpi)
    if csv:
        geo._salvar_dados(pd.DataFrame([{
            'kl_final': kl,'tempo_reducao_s': tempo, 'historico_disponivel': False,
            'n_componentes':d,'perplexity':p.get('perplexity'),
            'metrica':p.get('metric'), 'iteracoes_configuradas':p.get('n_iter'),
            'n_desenvolvimento':n,'origem':sufixo
        }]),dest,nome,True)
    return {'nome':nome,'kl_disponivel':kl is not None,'historico_disponivel':False}
