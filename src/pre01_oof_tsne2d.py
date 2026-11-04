# -*- coding: utf-8 -*-
"""Inclui PRE01 V2 na galeria OOF do t-SNE 2D, sem treinar o GMM nem ler F10.

Pode ser chamado automaticamente pelo orquestrador ou, para um experimento ja
concluido, executado isoladamente com --experimento. Usa a configuracao e os
folds congelados no proprio experimento, nunca o YAML editavel da raiz.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

from src.folds import hash_arquivo
from src import orquestracao_oof as orq
from src import pre01_v2_diagnostico as pre
from src import preprocessing as prep

ARQUIVOS = (pre.FILENAME, pre.CSV_SUM, pre.CSV_DETAIL,
            pre.CSV_FOLDS, pre.CSV_PCA, pre.META)


def exigir(condicao, mensagem):
    if not condicao:
        raise RuntimeError(mensagem)


def _carregar_origem(pasta: Path, raiz: Path):
    exigir(pasta.is_relative_to((raiz / 'outputs').resolve()),
           'PRE01 so aceita pastas dentro de outputs/.')
    conf = pasta / 'configuracao.yaml'
    manifest = pasta / 'manifesto.json'
    predicoes = pasta / 'vencedor/oof/predicoes.npz'
    indice = pasta / 'vencedor/visualizacao/galeria/index.html'
    for arq in (conf, manifest, predicoes, indice):
        exigir(arq.is_file(), f'PRE01 exige arquivo existente: {arq}')
    cfg = yaml.safe_load(conf.read_text(encoding='utf-8-sig'))
    m = json.loads(manifest.read_text(encoding='utf-8-sig'))
    exigir(isinstance(cfg, dict) and isinstance(m, dict), 'Configuracao ou manifesto invalido.')
    exigir(str(cfg['reducao']['metodo']).upper() == 'TSNE' and
           cfg['tsne']['n_components'] in (2, [2]),
           'PRE01 automatico nesta fase e somente t-SNE 2D.')
    exigir(m.get('status') == 'concluido_06_03' and m.get('visualizacoes') == 'concluidas',
           'OOF ou galeria OOF nao concluidos.')
    return cfg, m, predicoes, indice


def _verificar_presenca(vis: Path, indice: Path):
    """None: ausente; True: completo; estados parciais sao bloqueados."""
    dirs = (vis / 'oof', vis / 'galeria/oof')
    paths = [pasta / nome for pasta in dirs for nome in ARQUIVOS]
    texto = indice.read_text(encoding='utf-8')
    tem_marcador = pre.NEW_MARKER in texto
    presentes = [p.is_file() for p in paths]
    if not any(presentes) and not tem_marcador:
        return None
    exigir(all(presentes) and tem_marcador,
           'PRE01 t-SNE OOF parcialmente instalado; preservar arquivos e revisar manualmente.')
    for nome in ARQUIVOS:
        exigir(hash_arquivo(dirs[0] / nome) == hash_arquivo(dirs[1] / nome),
               f'PRE01 OOF divergente entre pastas: {nome}')
    exigir(f'oof/{pre.FILENAME}' in texto,
           'Link PRE01 t-SNE OOF nao aponta para o painel gerado.')
    return True


def integrar_pre01_oof(pasta_experimento: str | Path, *, raiz: str | Path | None = None) -> Path:
    """Novo PRE01 completo OOF F1..F9; sem alterar modelo, F10 nem graficos.

    O modo de reparo e idempotente e nunca sobrescreve resultados existentes.
    """
    raiz = Path(raiz).resolve() if raiz else Path(__file__).resolve().parents[1]
    pasta = Path(pasta_experimento).resolve()
    cfg, m, predicoes, indice = _carregar_origem(pasta, raiz)
    vis = pasta / 'vencedor/visualizacao'
    if _verificar_presenca(vis, indice):
        print('[OOF PRE01 TSNE] Painel e link ja existentes; preservados.', flush=True)
        return indice

    trava = vis / '.integracao_pre01_tsne2d.lock'
    try:
        with trava.open('x', encoding='utf-8') as f:
            f.write('Integracao de diagnostico PRE01 OOF, sem treinamento.\n')
    except FileExistsError as exc:
        raise RuntimeError('Integracao PRE01 ja em andamento; trava existente.') from exc
    try:
        # Revalidacao apos adquirir a trava evita duas gravacoes simultaneas.
        if _verificar_presenca(vis, indice):
            return indice
        with np.load(predicoes, allow_pickle=False) as z:
            ids = np.asarray(z['linha_original'], dtype=np.int64)
            validacoes = np.asarray(z['fold'], dtype=np.uint8)
            y = np.asarray(z['y_true'], dtype=np.uint8)
        exigir(len(ids) == len(validacoes) == len(y) == int(m['registros_desenvolvimento'])
               and len(ids) > 0 and np.all(np.diff(ids) > 0)
               and set(np.unique(validacoes).tolist()) == set(range(1, 10)),
               'Indices/predicoes OOF invalidos para F1..F9.')
        parquet = raiz / cfg['dados']['arquivo']
        folds, caminho_folds, hash_parquet = orq._identificar_folds(raiz, cfg, parquet)
        exigir(hash_parquet == m['sha256_parquet'] and
               hash_arquivo(caminho_folds) == m['sha256_folds'],
               'Parquet/folds divergentes do manifesto OOF.')
        exigir(int(ids[-1]) < len(folds) and np.array_equal(folds[ids], validacoes)
               and np.all(validacoes != 10), 'Amostra OOF acessaria F10 ou mudou de fold.')
        df = orq._ler_desenvolvimento(parquet, ids)
        exigir(len(df) == len(ids) and
               np.array_equal(df[cfg['dados']['coluna_alvo']].to_numpy(), y),
               'Linhas/rotulos divergentes do treino OOF.')
        X = prep.selecionar_features(df, cfg)
        print('[OOF PRE01 TSNE] Calculando diagnostico completo F1..F9; sem F10...', flush=True)
        resumo, detalhes, por_fold, pca_aux = pre.calc(X, validacoes, cfg)
        metadata = {
            'tipo': m.get('tipo'), 'k': int(cfg['gmm']['n_components'][0]) if isinstance(cfg['gmm']['n_components'], list) else int(cfg['gmm']['n_components']),
            'n_init': int(cfg['gmm']['n_init']), 'metodo_do_experimento': 'TSNE', 'dimensao_experimento': 2,
            'total_observacoes': len(ids), 'total_features': len(resumo),
            'f10_usado': False, 'mesmos_registros_e_folds': True,
            'experimentos': {'TSNE2D': pasta.relative_to(raiz).as_posix()},
            'sha256_parquet': m['sha256_parquet'], 'sha256_folds': m['sha256_folds'],
            'medida_de_desempenho': False,
            'pca_ajuste_qualificacao': 'PCA auxiliar somente diagnostica; modelo preditivo permanece t-SNE 2D + GMM',
        }
        blobs = {
            pre.FILENAME: pre.build_html(resumo, detalhes, por_fold, pca_aux, metadata).encode('utf-8'),
            pre.CSV_SUM: resumo.to_csv(index=False, float_format='%.17g').encode('utf-8-sig'),
            pre.CSV_DETAIL: detalhes.to_csv(index=False, float_format='%.17g').encode('utf-8-sig'),
            pre.CSV_FOLDS: por_fold.to_csv(index=False, float_format='%.17g').encode('utf-8-sig'),
            pre.CSV_PCA: pca_aux.to_csv(index=False, float_format='%.17g').encode('utf-8-sig'),
            pre.META: (json.dumps(metadata, indent=2, ensure_ascii=False) + '\n').encode('utf-8'),
        }
        anterior = indice.read_text(encoding='utf-8')
        novo = pre.new_gallery_index(anterior)
        exigir(pre.NEW_MARKER in novo and f'oof/{pre.FILENAME}' in novo and novo != anterior,
               'Nao foi possivel inserir link PRE01 na galeria OOF.')
        dirs = (vis / 'oof', vis / 'galeria/oof')
        for pasta_saida in dirs:
            pasta_saida.mkdir(parents=True, exist_ok=True)
        for pasta_saida in dirs:
            for nome, dados in blobs.items():
                # Criacao exclusiva: nao sobrescreve resultado de outro processo.
                with (pasta_saida / nome).open('xb') as f:
                    f.write(dados)
        temp = indice.with_name(indice.name + '.pre01_tsne2d_tmp')
        exigir(not temp.exists(), f'Arquivo temporario ja existe: {temp}')
        with temp.open('x', encoding='utf-8') as f:
            f.write(novo)
        os.replace(temp, indice)
        exigir(_verificar_presenca(vis, indice), 'Integracao PRE01 incompleta.')
        print('[OOF PRE01 TSNE] Link e HTML interativo incluidos na galeria OOF.', flush=True)
        return indice
    finally:
        trava.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experimento', required=True, type=Path)
    args = p.parse_args(argv)
    destino = integrar_pre01_oof(args.experimento)
    print('[OOF PRE01 TSNE] Abrir:', destino, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
