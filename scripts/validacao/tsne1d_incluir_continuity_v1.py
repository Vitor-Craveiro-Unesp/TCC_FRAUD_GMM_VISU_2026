# -*- coding: utf-8 -*-
"""Inclui Continuity ao lado do TSNE02 e oculta o cartao de Pearson.

Somente diagnostico auxiliar real em amostra F1-F9; nao reexecuta OOF,
nao muda modelos, YAML ou F10. Preserva os arquivos Pearson em disco.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
import yaml

from src import orquestracao_oof as orq
from src import preprocessing as prep
from src.reducao_tsne import RedutorTSNE

ROOT = Path(__file__).resolve().parents[2]
NAME = 'TSNE_CONT01_continuity_vs_perplexity'
TRUST = 'TSNE02_trustworthiness_vs_perplexity'
PEARSON_MARKER = '<!-- TSNE_PEARSON_TODAS_FEATURES_V1 -->'
CONT_MARKER = '<!-- TSNE_CONTINUITY_V1 -->'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_hash(p):
    h = sha256()
    with Path(p).open('rb') as arq:
        for block in iter(lambda: arq.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_write(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.continuity_v1.tmp')
    require(not tmp.exists(), f'Arquivo temporario presente: {tmp}')
    try:
        with tmp.open('xb') as f:
            f.write(data)
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def locate(explicit):
    if explicit is not None:
        return Path(explicit).resolve()
    options = []
    for p in (ROOT / 'outputs').glob('ensaio_tsne_gmm_*'):
        try:
            cfg = yaml.safe_load((p / 'configuracao.yaml').read_text(encoding='utf-8-sig'))
            manifest = json.loads((p / 'manifesto.json').read_text(encoding='utf-8-sig'))
            if cfg['reducao']['metodo'].upper() == 'TSNE' and manifest['status'] == 'concluido_06_03':
                options.append(p)
        except (OSError, KeyError, ValueError, TypeError, yaml.YAMLError):
            continue
    require(options, 'Nenhum experimento t-SNE OOF concluido localizado em outputs.')
    return max(options, key=lambda p: p.stat().st_mtime).resolve()


def validate(exp):
    require(exp.is_relative_to((ROOT / 'outputs').resolve()), 'Experimento deve estar em outputs.')
    cfg = yaml.safe_load((exp / 'configuracao.yaml').read_text(encoding='utf-8-sig'))
    manifest = json.loads((exp / 'manifesto.json').read_text(encoding='utf-8-sig'))
    winner = json.loads((exp / 'vencedor/metricas/melhor_configuracao.json').read_text(encoding='utf-8-sig'))
    require(cfg['reducao']['metodo'].upper() == 'TSNE', 'Nao e um experimento t-SNE.')
    require(winner['configuracao']['metodo'].upper() == 'TSNE', 'Vencedor nao e t-SNE.')
    require(int(winner['configuracao']['reducao']['n_components']) == 1, 'Exige t-SNE 1D.')
    require(manifest['status'] == 'concluido_06_03' and manifest['visualizacoes'] == 'concluidas',
            'OOF/galeria nao concluidos.')
    require(int(cfg['divisao']['qual_fold_teste']) == 10 and winner['teste_final_usado'] is False,
            'O vencedor deve estar congelado apenas no OOF.')
    with np.load(exp / 'vencedor/oof/predicoes.npz', allow_pickle=False) as a:
        ids = np.asarray(a['linha_original'], dtype=np.int64)
        folds = np.asarray(a['fold'], dtype=np.uint8)
        assert int(a['fold_esperado'].item()) == 0
    require(len(ids) > 360 and np.all(np.diff(ids) > 0), 'IDs F1-F9 nao validos.')
    require(set(np.unique(folds)) == set(range(1, 10)), 'OOF nao contem os nove folds.')

    view = exp / 'vencedor' / 'visualizacao'
    index = view / 'galeria' / 'index.html'
    original = view / 'geometria_desenvolvimento' / (TRUST + '.csv')
    require(index.is_file() and original.is_file(), 'Galeria ou CSV TSNE02 ausente.')
    table = pd.read_csv(original)
    require({'perplexity', 'trustworthiness', 'k_vizinhos', 'n_amostra'}.issubset(table.columns),
            'CSV TSNE02 nao possui todas as colunas necessarias.')
    require(len(table) >= 2 and table['perplexity'].is_unique, 'Grade TSNE02 incorreta.')
    k = int(table['k_vizinhos'].iloc[0]); n = int(table['n_amostra'].iloc[0])
    require(np.all(table['k_vizinhos'] == k) and np.all(table['n_amostra'] == n),
            'TSNE02 usa definicoes de vizinhanca/amostra divergentes.')
    require(1 <= k < (n - 1) / 2 and 20 <= n <= 2000, 'Amostra ou k invalido.')
    require(np.isfinite(table[['trustworthiness','perplexity']].to_numpy()).all(),
            'Trustworthiness/perplexity invalidos.')
    require(table['trustworthiness'].between(0, 1).all(), 'Trustworthiness fora de [0,1].')
    html = index.read_text(encoding='utf-8-sig')
    require(CONT_MARKER not in html, 'Continuity ja esta na galeria. Nao duplicar.')
    require('geometria_desenvolvimento/' + TRUST + '.png' in html,
            'Cartao Trustworthiness nao encontrado na galeria.')
    return cfg, manifest, winner, ids, folds, table, index, html, n, k


def insert_html(source):
    """Remove apenas secao Pearson; insere novo card imediatamente depois de TSNE02."""
    result = source
    if PEARSON_MARKER in result:
        pattern = r'<!-- TSNE_PEARSON_TODAS_FEATURES_V1 -->\s*<section\b[^>]*>.*?</section>\s*'
        result, count = re.subn(pattern, '', result, count=1, flags=re.S)
        require(count == 1, 'Secao Pearson inesperada; indice preservado.')
    else:
        # A versao de galeria original pode nao incluir o painel Pearson.
        require('t-SNE 1D — Correlação de Pearson com todas as variáveis' not in result,
                'Painel Pearson sem marcador esperado; nao editar automaticamente.')

    card = ('\n' + CONT_MARKER + '\n<article>'
            '<a href="geometria_desenvolvimento/' + NAME + '.png" target="_blank" rel="noopener">'
            '<img src="geometria_desenvolvimento/' + NAME + '.png" '
            'loading="lazy" alt="Continuity versus Perplexity — t-SNE 1D">'
            '<div>Continuity × Perplexity — t-SNE 1D</div></a></article>')

    pattern = r'(<article>\s*<a\b[^>]*\bhref=["\']geometria_desenvolvimento/' + re.escape(TRUST) + r'\.png["\'][^>]*>.*?</article>)'
    result, found = re.subn(pattern, lambda m: m.group(1) + card,
                            result, count=1, flags=re.S)
    require(found == 1, 'Nao foi possivel posicionar Continuity ao lado de Trustworthiness.')
    require(result.count(CONT_MARKER) == 1, 'Marcador Continuity duplicado.')
    require('TSNE_CORRELACAO_PEARSON_TODAS_FEATURES.png' not in result,
            'Pearson ainda aparece no indice; nenhuma alteracao aplicada.')
    return result


def plot_continuity(table, winner_p, k, png):
    table = table.sort_values('perplexity').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.plot(table['perplexity'], table['continuity'], 'o-', lw=2.1, ms=6.5,
            color='#008577', label=f'Continuity (k={k})')
    for p, score in zip(table['perplexity'], table['continuity']):
        ax.annotate(f'{score:.3f}', (p, score), xytext=(0, 9),
                    textcoords='offset points', ha='center', fontsize=8.5)
    ax.axvline(winner_p, ls='--', color='#d97706', lw=1.4,
               label=f'Perplexity do vencedor OOF = {winner_p:g}')
    ax.set_xticks(table['perplexity'].to_list())
    ax.set_ylim(max(0, float(table['continuity'].min()) - .035), 1.003)
    ax.set_xlabel('Perplexity')
    ax.set_ylabel('Continuity (0 a 1; eixo ampliado)')
    ax.set_title('Preservação de vizinhanças | Continuity × Perplexity')
    ax.grid(alpha=.2)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def compute(cfg, manifest, winner, ids, folds, table, k, n):
    """Reproduz exatamente o protocolo diagnostico TSNE02 em uma amostra F1-F9.

    Continuity(X,Z,k) = trustworthiness(Z,X,k): a metrica de rank e invertida.
    Esta equivalencia representa o custo dos vizinhos que existiam no espaco X
    mas desapareceram da vizinhanca k de Z. Nao e validacao do GMM.
    """
    parquet = ROOT / cfg['dados']['arquivo']
    original_folds, path_folds, hash_parquet = orq._identificar_folds(ROOT, cfg, parquet)
    require(hash_parquet == manifest['sha256_parquet'] and
            orq.hash_arquivo(path_folds) == manifest['sha256_folds'],
            'Parquet/folds divergiram dos arquivos usados no OOF.')
    require(np.array_equal(original_folds[ids], folds) and not np.any(folds == 10),
            'Os IDs/folDS F1-F9 nao correspondem ao OOF congelado.')
    df = orq._ler_desenvolvimento(parquet, ids)
    require(len(df) == len(ids), 'Leitura de dados F1-F9 incompleta.')
    X_raw = prep.selecionar_features(df, cfg)
    del df
    encoder = prep.EncodersPorSufixo(cfg).fit(X_raw)
    coded = encoder.transform(X_raw)
    imputer = prep.ImputadorPadronizador(cfg, encoder).fit(coded)
    pre_all = imputer.transform(coded)
    del X_raw, coded, encoder, imputer
    require(pre_all.shape[0] == len(ids) and pre_all.shape[1] >= 1,
            'Preprocessamento tem dimensoes inesperadas.')
    require(np.isfinite(pre_all.to_numpy()).all(), 'Preprocessamento nao finito.')
    seed = int(cfg['experimento']['semente'])
    rng = np.random.default_rng(seed)
    ix = np.sort(rng.choice(len(pre_all), size=n, replace=False)) if len(pre_all)>n else np.arange(len(pre_all))
    X_subset = pre_all.iloc[ix].copy()
    del pre_all
    assert len(X_subset) == n
    X = X_subset.to_numpy(dtype=float)
    params = winner['configuracao']['reducao']
    winner_p = float(params['perplexity'])
    require(str(params.get('metric', '')).lower() == 'euclidean',
            'Esta versao calcula distancias euclidianas; outra metrica requer implementacao especifica.')
    require(any(np.isclose(table['perplexity'], winner_p)),
            'Perplexity vencedora ausente no CSV TSNE02.')
    rows = []
    for row in table.sort_values('perplexity').itertuples(index=False):
        p = float(row.perplexity)
        params_p = dict(params, perplexity=p)
        start = perf_counter()
        red = RedutorTSNE(cfg, params_p).fit(X_subset)
        z = red.transform_treino().to_numpy(dtype=float)
        require(z.shape == (n, 1) and np.isfinite(z).all(), 'Embedding de diagnostico invalido.')
        # Trocar os papeis X e Z na mesma formula de Trustworthiness
        # calcula Continuity com penalidade pelo rank da vizinhanca perdida.
        tw = float(trustworthiness(X, z, n_neighbors=k, metric=params['metric']))
        cont = float(trustworthiness(z, X, n_neighbors=k, metric='euclidean'))
        require(np.isfinite(tw) and np.isfinite(cont) and 0 <= tw <= 1 and 0 <= cont <= 1,
                'Trustworthiness/Continuity fora de [0,1].')
        rows.append({'perplexity': p, 'continuity': cont, 'trustworthiness_recalculada': tw,
                     'trustworthiness_tsne02_salva': float(row.trustworthiness),
                     'delta_trustworthiness': tw - float(row.trustworthiness),
                     'k_vizinhos':k, 'n_amostra':n, 'n_dim_reducao':1,
                     'perplexity_vencedora':winner_p,
                     'tempo_reajuste_diagnostico_s':float(perf_counter()-start)})
        print(f'[CONTINUITY] Perplexity={p:g} | C={cont:.7f} | T={tw:.7f} '
              f'| T_original={float(row.trustworthiness):.7f}',flush=True)
    dfc = pd.DataFrame(rows)
    return dfc, winner_p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--planejar', action='store_true')
    group.add_argument('--executar', action='store_true')
    parser.add_argument('--experimento', type=Path, help='Pasta de ensaio TSNE; opcional')
    args = parser.parse_args()
    os.chdir(ROOT)
    exp = locate(args.experimento)
    cfg, manifest, winner, ids, folds, original, index, old_html, n, k = validate(exp)
    revised = insert_html(old_html)  # valida antes de qualquer escrita
    geo_dir = exp / 'vencedor/visualizacao/geometria_desenvolvimento'
    gal_geo = exp / 'vencedor/visualizacao/galeria/geometria_desenvolvimento'
    for p in (geo_dir / (NAME+'.png'), geo_dir / (NAME+'.csv'), gal_geo / (NAME+'.png')):
        require(not p.exists(), 'Arquivo de Continuity ja existe: ' + str(p))
    print('EXPERIMENTO=' + str(exp),flush=True)
    print('REGISTROS_F1_F9=' + str(len(ids)),flush=True)
    print(f'PONTOS_AMOSTRA={n} | K_VIZINHOS={k} | PERPLEXIDADES={original.perplexity.to_list()}',flush=True)
    print('REMOVER_PEARSON_SOMENTE_DA_GALERIA=SIM',flush=True)
    print('ADICIONAR_CONTINUITY_AO_LADO_TSNE02=SIM',flush=True)
    print('F10_ACESSADO=NAO | OOF_COMPLETO_REEXECUTADO=NAO',flush=True)
    if args.planejar:
        print('CONTINUITY_PLANEJAMENTO_EXIT_CODE=0')
        return

    run = ROOT / 'outputs' / ('validacao_tsne_continuity_v1_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    stage = run / 'PREPARADOS'
    stage.mkdir(parents=True, exist_ok=False)
    table, p = compute(cfg, manifest, winner, ids, folds, original, k, n)
    png_data = stage / (NAME + '.png')
    csv_data = stage / (NAME + '.csv')
    plot_continuity(table, p, k, png_data)
    table.to_csv(csv_data, index=False, encoding='utf-8-sig', float_format='%.17g')
    require(png_data.stat().st_size > 1000 and len(pd.read_csv(csv_data)) == len(original),
            'Falha ao gerar grafico ou CSV Continuity.')
    require(revised.count(CONT_MARKER) == 1, 'Indice nao contem Continuity unica.')
    revised_bytes = revised.encode('utf-8')
    destinations = [
        (geo_dir / (NAME+'.png'),png_data.read_bytes()),
        (geo_dir / (NAME+'.csv'),csv_data.read_bytes()),
        (gal_geo / (NAME+'.png'),png_data.read_bytes()),
        (index,revised_bytes),
    ]
    originals = []
    for target, content in destinations:
        require(not target.is_symlink(), 'Destino symlink nao permitido: ' + str(target))
        backup = None
        if target.exists():
            backup = run / 'BACKUP' / target.relative_to(ROOT)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            require(file_hash(target) == file_hash(backup), 'Backup invalido para ' + str(target))
        originals.append((target, backup))
    info = {'tipo':'DIAGNOSTICO_COMPLEMENTAR_CONTINUITY_TSNE1D',
            'experimento':str(exp),'n_amostra':n,'n_total_f1_f9':len(ids),
            'k_vizinhos':k,'perplexity_vencedora':p,
            'max_diferenca_trustworthiness':float(table['delta_trustworthiness'].abs().max()),
            'amostra_pareada_com_tsne02':True,
            'refits_auxiliares_tsne':len(table),
            'reajustes_oof_ou_gmm':0,'f10_acessado':False,
            'pearson_arquivos_originais_preservados':True,
            'backup_index':str(run/'BACKUP'/index.relative_to(ROOT)),
            'novos_arquivos':[str(f.relative_to(ROOT)) for f,_ in destinations]}
    atomic_write(run/'manifesto.json',(json.dumps(info,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    done = []
    try:
        for target, raw in destinations:
            atomic_write(target, raw)
            done.append(target)
    except Exception:
        for target, backup in reversed(originals):
            if target not in done:
                continue
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                shutil.copy2(backup,target)
        raise
    require(file_hash(index) == sha256(revised_bytes).hexdigest(),
            'Falha na verificacao do HTML atualizado.')
    print('CONTINUITY_CSV=' + str(geo_dir/(NAME+'.csv')),flush=True)
    print('CONTINUITY_PNG=' + str(gal_geo/(NAME+'.png')),flush=True)
    print('CONTINUITY_TRUST_COMPARAVEL=SIM',flush=True)
    print('DELTA_MAX_TRUST_RECALCULADA=' + f"{info['max_diferenca_trustworthiness']:.7f}",flush=True)
    print('PARES_CONTINUITY_E_TRUSTWORTHINESS=' + str(len(table)),flush=True)
    print('PAREAMENTO_NA_MESMA_AMOSTRA=SIM',flush=True)
    print('PEARSON_OCULTO_GALERIA=SIM',flush=True)
    print('PEARSON_CSV_PNG_PRESERVADOS=SIM',flush=True)
    print('OOF_COMPLETO_REEXECUTADO=NAO',flush=True)
    print('GMM_REAJUSTADO=NAO',flush=True)
    print('F10_ACESSADO=NAO',flush=True)
    print('GALERIA_HTML=' + str(index),flush=True)
    print('TSNE_CONTINUITY_EXIT_CODE=0',flush=True)


if __name__ == '__main__':
    main()
