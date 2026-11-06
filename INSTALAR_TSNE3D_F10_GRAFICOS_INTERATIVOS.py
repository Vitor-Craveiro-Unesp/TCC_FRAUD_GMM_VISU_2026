# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import ast
from datetime import datetime
from pathlib import Path
import shutil

ARQ_NOVO = 'finalizacao_automatica_tsne3d.py'
ALVO_REL = Path('src/finalizacao_automatica_tsne3d.py')


def _raiz() -> Path:
    return Path.cwd().resolve()


def _local() -> Path:
    return Path(__file__).resolve().parent


def _validar_py(texto: str, nome: str) -> None:
    try:
        ast.parse(texto)
    except SyntaxError as e:
        raise SystemExit(f'[ERRO] AST inválido em {nome}: {e}')


def _checar(raiz: Path) -> tuple[Path, str]:
    alvo = raiz / ALVO_REL
    if not alvo.is_file():
        raise SystemExit(f'[ERRO] Arquivo alvo ausente: {alvo}')
    novo = (_local() / ARQ_NOVO).read_text(encoding='utf-8')
    _validar_py(alvo.read_text(encoding='utf-8-sig'), str(alvo))
    _validar_py(novo, ARQ_NOVO)
    print('[COMPATÍVEL] src/finalizacao_automatica_tsne3d.py (AST válido)')
    return alvo, novo


def _aplicar(raiz: Path) -> None:
    alvo, novo = _checar(raiz)
    backup_dir = raiz / 'outputs' / f'backup_tsne3d_graficos_f10_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(alvo, backup_dir / alvo.name)
    alvo.write_text(novo, encoding='utf-8')
    print(f'Backup do arquivo original: {backup_dir}')
    print('[ATUALIZADO] src/finalizacao_automatica_tsne3d.py')
    print('[INSTALADO] F10 t-SNE 3D agora gera:')
    print('            - TSNE_OBSERVACOES_TSNE1_TSNE2_TSNE3.png')
    print('            - GMM04_dominancia_3d.png')
    print('            - GMM07_sensibilidade_3d_comparativo.png')
    print('            - GMM07_sensibilidade_3d_comparativo_INTERATIVA.html')
    print('            - GMM08_mapa_risco_3d_comparativo.png')
    print('            - GMM08_mapa_risco_3d_comparativo_INTERATIVA.html')
    print('            - galeria/index.html em vencedor/teste_final')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Instala renderização 3D F10 interativa no finalizador t-SNE 3D.')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--checar', action='store_true')
    g.add_argument('--aplicar', action='store_true')
    args = ap.parse_args()
    raiz = _raiz()
    if args.checar:
        _checar(raiz)
        print('[OK] Verificação somente leitura. Nenhum arquivo modificado.')
    else:
        _aplicar(raiz)
