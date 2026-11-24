"""Registro automático e auditável de folds para versões de datasets.

A integração ocorre antes da busca estrita existente em orquestracao_oof.py.
Não modifica folds nem manifestos previamente registrados.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _parametros(config: dict, sha: str, n: int) -> dict:
    dados, divisao, exp = config['dados'], config['divisao'], config['experimento']
    return {
        'caminho_relativo': str(dados['arquivo']).replace('\\', '/'),
        'coluna_alvo': dados['coluna_alvo'],
        'classe_positiva': int(dados['classe_positiva']),
        'total_folds': int(divisao['total_folds']),
        'fold_teste': int(divisao['qual_fold_teste']),
        'estratificada': bool(divisao['estratificada']),
        'embaralhar': bool(divisao['embaralhar']),
        'semente': int(exp['semente']),
        'sha256_parquet': sha,
        'total_linhas': n,
    }


def _mesma_estrategia(p: dict, esperado: dict) -> bool:
    return all(p.get(campo) == esperado[campo] for campo in (
        'coluna_alvo', 'classe_positiva', 'total_folds', 'fold_teste',
        'estratificada', 'embaralhar', 'semente'))


def _validar_folds(folds: np.ndarray, y: np.ndarray, nfolds: int) -> None:
    if folds.ndim != 1 or len(folds) != len(y):
        raise RuntimeError('Folds com dimensão ou comprimento incompatíveis.')
    if not np.issubdtype(folds.dtype, np.integer):
        raise RuntimeError('Os índices de folds devem ser inteiros.')
    if not np.all((folds >= 1) & (folds <= nfolds)):
        raise RuntimeError('Foram encontrados índices de folds fora da faixa permitida.')
    if len(np.unique(folds)) != nfolds:
        raise RuntimeError('Nem todos os folds foram representados.')
    for classe in (0, 1):
        quantidades = np.bincount(folds[y == classe].astype(int), minlength=nfolds + 1)[1:]
        if np.any(quantidades == 0) or int(quantidades.max() - quantidades.min()) > 1:
            raise RuntimeError(f'Divisão inválida ou não estratificada para a classe {classe}.')


def _contagens(folds: np.ndarray, y: np.ndarray, nfolds: int) -> list[dict]:
    return [
        {
            'fold': k,
            'total': int(np.count_nonzero(folds == k)),
            'classes': {
                '0': int(np.count_nonzero((folds == k) & (y == 0))),
                '1': int(np.count_nonzero((folds == k) & (y == 1))),
            },
        }
        for k in range(1, nfolds + 1)
    ]


def _ler_identidade(parquet: Path, col_id: str, col_alvo: str) -> pd.DataFrame:
    try:
        dados = pd.read_parquet(parquet, columns=[col_id, col_alvo])
    except Exception as exc:
        raise RuntimeError(
            f'Não foi possível ler ID e alvo do arquivo {parquet}. '
            f'Verifique as colunas {col_id!r} e {col_alvo!r}.'
        ) from exc
    if dados[col_id].isna().any() or dados[col_alvo].isna().any():
        raise RuntimeError(f'O dataset {parquet.name} contém ID ou alvo ausente.')
    if not dados[col_id].is_unique:
        raise RuntimeError(
            f'O identificador {col_id} deve ser único para preservar o isolamento entre folds.'
        )
    if not dados[col_alvo].isin([0, 1]).all():
        raise RuntimeError('A variável resposta precisa conter somente 0 e 1.')
    if not dados[col_alvo].isin([0]).any() or not dados[col_alvo].isin([1]).any():
        raise RuntimeError('Ambas as classes devem estar representadas.')
    return dados


def garantir_folds(raiz, config, arquivo_base) -> None:
    """Cria manifesto compatível automaticamente, ou usa o existente.

    A reutilização é permitida somente após conferir SHA do parquet anterior,
    SHA do folds.npy, mesma estratégia, ID/alvo linha a linha e ordem igual.
    """
    raiz = Path(raiz).resolve()
    arquivo = Path(arquivo_base).resolve()
    if not arquivo.is_file():
        raise FileNotFoundError(f'Parquet não encontrado: {arquivo}')

    divisao = config['divisao']
    nfolds = int(divisao['total_folds'])
    teste = int(divisao['qual_fold_teste'])
    if nfolds != 10 or teste != 10 or not divisao['estratificada'] or not divisao['embaralhar']:
        raise RuntimeError('A orquestração atual exige 10 folds, F10 como teste, estratificação e embaralhamento.')

    splits = raiz / 'data' / 'splits'
    splits.mkdir(parents=True, exist_ok=True)
    sha = _sha256(arquivo)
    caminhos = sorted(splits.glob('*/manifesto_folds.json'))
    cabecalhos = []
    for caminho_manifesto in caminhos:
        try:
            manifest = json.loads(caminho_manifesto.read_text(encoding='utf-8-sig'))
        except (ValueError, UnicodeError, OSError):
            print(f'[FOLDS AUTO] Aviso: manifesto ilegível não utilizado: {caminho_manifesto}')
            continue
        cabecalhos.append((caminho_manifesto, manifest))

    # Sem carregar o dataset quando uma divisão exata e íntegra já existe.
    contexto = _parametros(config, sha, 0)
    exatos = [(p, m) for p, m in cabecalhos
              if _mesma_estrategia(m.get('parametros', {}), contexto)
              and m.get('parametros', {}).get('sha256_parquet') == sha
              and m.get('parametros', {}).get('caminho_relativo') == contexto['caminho_relativo']]
    if len(exatos) > 1:
        raise RuntimeError('Mais de um manifesto exato encontrado; resolver a ambiguidade manualmente.')
    if len(exatos) == 1:
        p, m = exatos[0]
        folds_path = p.parent / 'folds.npy'
        if not folds_path.is_file() or _sha256(folds_path) != m.get('sha256_folds'):
            raise RuntimeError(f'Divisão exata existente, mas ausente ou corrompida: {folds_path}')
        print(f'[FOLDS AUTO] Reutilizando divisão exata: {folds_path}')
        return

    col_id = config['dados']['coluna_id']
    col_alvo = config['dados']['coluna_alvo']
    dados = _ler_identidade(arquivo, col_id, col_alvo)
    n = len(dados)
    contexto = _parametros(config, sha, n)
    y = dados[col_alvo].to_numpy(dtype=np.int8)
    if min(np.count_nonzero(y == 0), np.count_nonzero(y == 1)) < nfolds:
        raise RuntimeError('Cada classe precisa ter pelo menos 10 observações para a estratificação.')

    # Reaproveitar somente quando a fonte antiga existir, íntegra, e a
    # identidade ORDENADA (ID + resposta) das linhas for rigorosamente igual.
    reutilizacoes = []
    for p, m in cabecalhos:
        antigo = m.get('parametros', {})
        if not _mesma_estrategia(antigo, contexto):
            continue
        if int(antigo.get('total_linhas', -1)) != n:
            continue
        origem_rel = antigo.get('caminho_relativo')
        if not isinstance(origem_rel, str):
            continue
        origem = (raiz / origem_rel).resolve()
        if not origem.is_relative_to(raiz) or not origem.is_file() or origem == arquivo:
            continue
        if _sha256(origem) != antigo.get('sha256_parquet'):
            continue
        folds_antigos = p.parent / 'folds.npy'
        if not folds_antigos.is_file():
            raise RuntimeError(f'Arquivo de folds anterior ausente: {folds_antigos}')
        if _sha256(folds_antigos) != m.get('sha256_folds'):
            raise RuntimeError(f'Arquivo de folds anterior corrompido: {folds_antigos}')
        anterior = _ler_identidade(origem, col_id, col_alvo)
        if (np.array_equal(dados[col_id].to_numpy(), anterior[col_id].to_numpy())
                and np.array_equal(y, anterior[col_alvo].to_numpy())):
            folds = np.load(folds_antigos, allow_pickle=False)
            _validar_folds(folds, y, nfolds)
            reutilizacoes.append((p, folds_antigos, folds))

    if reutilizacoes:
        hashes = {_sha256(caminho) for _, caminho, _ in reutilizacoes}
        if len(hashes) != 1:
            raise RuntimeError('Há múltiplas divisões equivalentes com folds diferentes. Resolver manualmente.')
        origem_manifesto, origem_folds, folds = reutilizacoes[0]
        modo = 'reutilizado_identidade_ordenada'
        proveniencia = {'manifesto_origem': str(origem_manifesto.relative_to(raiz)).replace('\\', '/'),
                        'sha256_folds_origem': _sha256(origem_folds)}
    else:
        folds = np.empty(n, dtype=np.int16)
        skf = StratifiedKFold(n_splits=nfolds, shuffle=True,
                            random_state=int(config['experimento']['semente']))
        for indice, (_, indices_validacao) in enumerate(skf.split(np.zeros(n), y), start=1):
            folds[indices_validacao] = indice
        modo = 'gerado_estratificado'
        proveniencia = {'motivo': 'Sem divisão existente cuja identidade ordenada e classes coincidam.'}

    _validar_folds(folds, y, nfolds)
    nome_dataset = ''.join(c if c.isalnum() or c in '_-' else '_' for c in arquivo.stem)
    destino = splits / f'{nome_dataset}_{sha[:12]}'
    if destino.exists():
        raise RuntimeError(f'Destino de divisão já existe, mas não é reconhecido: {destino}. Não será sobrescrito.')

    manifesto = {
        'parametros': contexto,
        'sha256_folds': '',
        'contagem_por_fold': _contagens(folds, y, nfolds),
        'regra_cv': 'Para cada F1..F9: valida nele, treina nos outros oito; F10 nunca entra.',
        'geracao_automatica': {'modo': modo, **proveniencia},
    }
    tempdir = Path(tempfile.mkdtemp(prefix='.folds_temp_', dir=splits))
    try:
        temp_folds = tempdir / 'folds.npy'
        np.save(temp_folds, folds, allow_pickle=False)
        manifesto['sha256_folds'] = _sha256(temp_folds)
        (tempdir / 'manifesto_folds.json').write_text(
            json.dumps(manifesto, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        tempdir.rename(destino)
    finally:
        if tempdir.exists():
            shutil.rmtree(tempdir)

    print(f'[FOLDS AUTO] Modo: {modo}')
    print(f'[FOLDS AUTO] Divisão registrada: {destino / "folds.npy"}')
    print(f'[FOLDS AUTO] F10 preservado com {int(np.count_nonzero(folds == teste)):,} observações.')
