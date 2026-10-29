"""Auditoria SOMENTE LEITURA das caixas desmarcadas do layout 2 do TCC.
Execute na raiz de TCC_EXPERIMENTAL_V3 com a .venv ativa.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
from collections import Counter

import numpy as np
import yaml

ENSAIO = Path('outputs/ensaio_pca_gmm_20261021_184011')
APROVADOS = []
REPROVADOS = []
AVISOS = []
DADOS = {}


def sha256_arquivo(caminho):
    h = hashlib.sha256()
    with Path(caminho).open('rb') as arq:
        for bloco in iter(lambda: arq.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def exigir(condicao, mensagem):
    if not condicao:
        raise AssertionError(mensagem)


def testar(titulo, funcao):
    try:
        funcao()
        APROVADOS.append(titulo)
        print(f'[APROVADO] {titulo}')
    except Exception as exc:
        REPROVADOS.append((titulo, str(exc)))
        print(f'[REPROVADO] {titulo}: {exc}')


def avisar(texto):
    AVISOS.append(texto)
    print(f'[AVISO] {texto}')


def ler_json(caminho):
    return json.loads(Path(caminho).read_text(encoding='utf-8-sig'))


def conferir_manifesto():
    m = ler_json(ENSAIO/'manifesto.json')
    cfg = yaml.safe_load((ENSAIO/'configuracao.yaml').read_text(encoding='utf-8-sig'))
    exigir(bool(m.get('inicio_local')) and bool(m.get('fim_local')), 'Datas/horarios ausentes')
    exigir(m.get('quantidade_candidatos') == 3 and m.get('n_folds_cv') == 9,
           'Numero de candidatos/folds divergente')
    vers = m.get('versoes') or {}
    exigir({'python', 'pandas', 'numpy', 'sklearn', 'pyarrow', 'plataforma'} <= set(vers),
           'Versoes de bibliotecas incompletas')
    for chave in ('sha256_parquet', 'sha256_folds', 'sha256_yaml'):
        valor = m.get(chave, '')
        exigir(len(valor) == 64 and all(x in '0123456789abcdef' for x in valor.lower()),
               f'Hash ausente ou invalido: {chave}')
    exigir(m.get('versao_layout') == 2 and m.get('tipo') == 'ensaio', 'Tipo/layout incorreto')
    exigir(cfg['reducao']['metodo'].upper() == 'PCA', 'Nao e PCA')
    exigir(cfg['dados']['fracao_base'] == 0.1, 'Fracao base mudou no snapshot')
    exigir(m.get('visualizacoes') == 'concluidas', 'Visualizacoes ainda nao concluidas')
    DADOS.update(manifesto=m, configuracao=cfg)
    print('  Inicio:', m['inicio_local'], '| Fim:', m['fim_local'])
    print('  Versoes:', ', '.join(f'{k}={v}' for k, v in vers.items() if k != 'plataforma'))


def conferir_hashes():
    m = DADOS['manifesto']
    cfg = DADOS['configuracao']
    arquivo_parquet = Path(cfg['dados']['arquivo'])
    exigir(arquivo_parquet.is_file(), f'Parquet nao encontrado: {arquivo_parquet}')
    exigir(sha256_arquivo(arquivo_parquet) == m['sha256_parquet'], 'SHA-256 do parquet divergente')
    arquivos_folds = list(Path('data/splits').rglob('folds.npy'))
    encontrados = [a for a in arquivos_folds if sha256_arquivo(a) == m['sha256_folds']]
    exigir(bool(encontrados), 'Nenhum folds.npy corresponde ao hash salvo')
    caminho_yaml_original = Path('config_TCC_EXPERIMENTAL_V3_atualizado.yaml')
    exigir(caminho_yaml_original.is_file(), 'YAML original nao encontrado')
    if sha256_arquivo(caminho_yaml_original) != m['sha256_yaml']:
        avisar('O YAML principal mudou desde o ensaio. Compare com configuracao.yaml salva.')
    else:
        print('  SHA-256 do YAML original: confere')
    folds = np.load(encontrados[0], allow_pickle=False)
    DADOS.update(folds=folds)
    print('  SHA-256 parquet: confere | folds: confere')
    print('  Divisao usada:', encontrados[0])


def conferir_contagens():
    m = DADOS['manifesto']
    folds = DADOS['folds']
    exigir(m['registros_desenvolvimento'] == 166716,
           'Desenvolvimento diferente de 166716')
    exigir(m['registros_reservados_f10'] == 185239,
           'Reserva F10 diferente de 185239')
    exigir(int(np.count_nonzero(folds == 10)) == 185239,
           'Contagem F10 no arquivo folds.npy divergente')
    print('  Desenvolvimento amostrado:', m['registros_desenvolvimento'])
    print('  F10 no arquivo folds.npy:', int(np.count_nonzero(folds == 10)))


def conferir_candidatos():
    pastas = sorted((ENSAIO/'candidatos').glob('K*_*'))
    exigir(len(pastas) == 3, f'Esperadas 3 pastas, encontradas {len(pastas)}')
    mapa = {}
    referencia = None
    folds_orig = DADOS.get('folds')
    for pasta in pastas:
        config = ler_json(pasta/'configuracao.json')
        k = int(config['gmm']['n_components'])
        exigir(k in {2, 4, 6}, f'K nao esperado: {k}')
        exigir(pasta.name.startswith(f'K{k:02d}_'), 'Pasta e configuracao do candidato divergentes')
        cid = pasta.name.split('_', 1)[1]
        exigir(cid not in mapa, f'ID repetido: {cid}')
        metricas = ler_json(pasta/'metricas'/'metricas.json')
        exigir(metricas.get('id_candidato') == cid, 'ID diferente em metricas.json')
        exigir((pasta/'metricas'/'por_fold.csv').is_file(), 'por_fold.csv ausente')
        with (pasta/'metricas'/'por_fold.csv').open(newline='', encoding='utf-8-sig') as arq:
            linhas_csv = list(csv.DictReader(arq))
        exigir(len(linhas_csv) == 9, 'por_fold.csv nao tem 9 linhas')
        exigir([int(r['fold']) for r in linhas_csv] == list(range(1, 10)),
               'por_fold.csv tem folds incorretos')
        exigir(all(int(r['n']) == 18524 for r in linhas_csv),
               'por_fold.csv nao registra 18524 por fold')
        exigir(sum(int(r['n_fraudes']) for r in linhas_csv) == 870,
               'por_fold.csv nao totaliza 870 fraudes')
        arq_pred = pasta/'oof'/'predicoes.npz'
        with np.load(arq_pred, allow_pickle=False) as z:
            exigir({'linha_original', 'fold', 'y_true', 'prob_fraude'} <= set(z.files),
                   'Colunas essenciais ausentes no NPZ')
            ids = z['linha_original'].copy()
            f = z['fold'].copy()
            y = z['y_true'].copy()
            prob = z['prob_fraude'].copy()
            exigir(str(z['id_candidato'].item()) == cid, 'ID do NPZ divergente')
        exigir(len(ids) == 166716 and len(np.unique(ids)) == 166716,
               'Nao ha exatamente 166716 indices originais unicos')
        exigir(all(np.count_nonzero(f == x) == 18524 for x in range(1, 10)),
               'Folds com tamanhos incorretos')
        exigir(set(np.unique(f)) == set(range(1,10)), 'Fold incorreto, inclusive possivel F10')
        exigir(int(y.sum()) == 870 and set(np.unique(y)) == {0,1},
               'Classes/870 fraudes nao conferem')
        exigir(np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all(),
               'Probabilidades nao validas')
        exigir(np.array_equal(folds_orig[ids], f), 'Folds reais divergem do NPZ (possivel F10)')
        exigir(not np.any(folds_orig[ids] == 10), 'Ha F10 nas predicoes')
        if referencia is None:
            referencia = (ids, f, y)
        else:
            exigir(all(np.array_equal(a,b) for a,b in zip(referencia, (ids,f,y))),
                   'Identidade ou rotulos diferentes entre candidatos')
        caches = sorted((pasta/'oof'/'cache').glob('F??.npz'))
        exigir([a.name for a in caches] == [f'F{j:02d}.npz' for j in range(1,10)],
               'Esperados F01.npz ate F09.npz')
        for j, arq in enumerate(caches, start=1):
            with np.load(arq, allow_pickle=False) as z:
                fc, yc = z['fold'], z['y_true']
                exigir(len(fc) == 18524 and np.all(fc == j),
                       f'{arq.name} tem tamanho/fold errado')
                exigir(int(z['fold_esperado']) == j, f'{arq.name} metadado incorreto')
                exigir(len(yc) == 18524, f'{arq.name} rotulos ausentes')
        mapa[k] = (cid, pasta)
        print(f'  K={k}: {cid} | folds=9 | OOF=166716 | fraudes=870 | sem F10')
    exigir(set(mapa) == {2,4,6}, 'A grade nao corresponde a K=2,4,6')
    DADOS['mapa_candidatos'] = mapa


def conferir_vencedor():
    dest = ENSAIO/'vencedor'
    metricas = dest/'metricas'
    esperados = ('metricas.json','por_fold.csv','resultados_grade.csv','melhor_configuracao.json')
    exigir(all((metricas/n).is_file() for n in esperados), 'Metricas do vencedor incompletas')
    escolha = ler_json(metricas/'melhor_configuracao.json')
    cid = escolha['id_candidato']
    exigir(escolha.get('teste_final_usado') is False, 'Escolha declara uso do F10')
    with (metricas/'resultados_grade.csv').open(encoding='utf-8-sig', newline='') as arq:
        grade = list(csv.DictReader(arq))
    exigir(len(grade) == 3 and grade[0]['id_candidato'] == cid, 'Ranking ou vencedor divergentes')
    scores = [float(r['score']) for r in grade]
    exigir(scores == sorted(scores, reverse=True), 'Ranking nao ordenado pelo score composto')
    pasta_origem = DADOS['mapa_candidatos'][6][1]
    exigir(cid == DADOS['mapa_candidatos'][6][0], 'Vencedor nao e K=6')
    for src, dest_arq in (
        (pasta_origem/'metricas'/'metricas.json', metricas/'metricas.json'),
        (pasta_origem/'metricas'/'por_fold.csv', metricas/'por_fold.csv'),
        (pasta_origem/'oof'/'predicoes.npz', dest/'oof'/'predicoes.npz'),
    ):
        exigir(dest_arq.is_file(), f'Arquivo ausente: {dest_arq}')
        exigir(sha256_arquivo(src) == sha256_arquivo(dest_arq),
               f'Arquivo do vencedor difere da origem: {dest_arq.name}')
    print('  Vencedor K=6:', cid, '| score=', scores[0])
    print('  metricas, por_fold.csv e predicoes.npz: IDENTICOS ao candidato')


def conferir_visualizacoes():
    v = ENSAIO/'vencedor'/'visualizacao'
    pastas = {'oof':7, 'geometria_desenvolvimento':9}
    for subpasta, qtd in pastas.items():
        d = v/subpasta
        exigir(d.is_dir(), f'Pasta visual ausente: {subpasta}')
        exigir(len(list(d.glob('*.png'))) == qtd,
               f'Esperados {qtd} PNG em {subpasta}')
    gal = v/'galeria'
    exigir(gal.is_dir() and (gal/'index.html').is_file(), 'Galeria HTML ausente')
    exigir(len(list(gal.rglob('*.png'))) == 16, 'Galeria nao possui 16 PNG')
    geoman = ler_json(v/'geometria_desenvolvimento'/'manifesto_geometria_07_02.json')
    exigir(geoman.get('f10_utilizado') is False and geoman.get('origem') == 'desenvolvimento_F1_F9',
           'Manifesto geometrico nao declara origem exclusiva F1-F9')
    print('  Imagens: OOF 7, geometria 9, galeria 16')
    avisar('F10 na geometria: manifestado como nao utilizado; a prova completa depende tambem do codigo executado.')


def conferir_f10_vazio():
    d = ENSAIO/'vencedor'/'teste_final'
    exigir(d.is_dir(), 'Diretorio teste_final nao existe')
    itens = list(d.rglob('*'))
    exigir(not itens, f'Diretorio teste_final nao esta vazio: {len(itens)} item(ns)')
    print('  teste_final existe e esta vazio')


if not ENSAIO.is_dir():
    raise SystemExit(f'Ensaio nao encontrado: {ENSAIO.resolve()}')

print('AUDITORIA AUTOMATICA - ITENS DESMARCADOS - PCA 1D - LAYOUT 2')
print('Experimento:', ENSAIO.resolve())

testar('1. Manifesto: data/hora, hashes, versoes, quantidade candidatos/folds', conferir_manifesto)
if 'manifesto' in DADOS:
    testar('2. SHA-256 reais do parquet, folds e YAML original', conferir_hashes)
if 'folds' in DADOS:
    testar('3. Contagens e reserva F10', conferir_contagens)
    testar('4. Candidatos: 9 caches, 18524/fold, 870 positivos, OOF sem F10', conferir_candidatos)
if 'mapa_candidatos' in DADOS:
    testar('5. Metricas do vencedor e igualdade com o candidato K=6', conferir_vencedor)
testar('6. Diretorios visuais, 7+9 imagens, galeria e manifesto geometrico', conferir_visualizacoes)
testar('7. Diretorio teste_final existe e esta vazio', conferir_f10_vazio)

print('\n===== RESUMO =====')
print('APROVADOS:', len(APROVADOS), '| REPROVADOS:', len(REPROVADOS), '| AVISOS:', len(AVISOS))
for nome, motivo in REPROVADOS:
    print('FALHA:', nome, '|', motivo)
if REPROVADOS:
    sys.exit(1)
if AVISOS:
    print('AUDITORIA AUTOMATICA APROVADA, COM AVISOS. Ainda falta inspecao visual humana.')
else:
    print('AUDITORIA AUTOMATICA APROVADA. Ainda falta inspecao visual humana.')
