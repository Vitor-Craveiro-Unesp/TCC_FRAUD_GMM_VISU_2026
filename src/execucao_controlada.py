"""Lancador operacional do TCC: snapshot YAML e exclusao mutua por projeto.

Nao altera metodologia nem parametros; repassa a configuracao congelada
para src.orquestracao_oof. Submissoes SLURM usam --config com o snapshot
criado ANTES de sbatch; uso local sem --config congela o YAML no inicio.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

NOME_YAML = 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def criar_snapshot(raiz: Path, fonte: Path) -> Path:
    """Copies exactly the bytes read once; never rewrites the editable YAML."""
    data = fonte.read_bytes()
    base = raiz / 'outputs' / '.submissoes'
    base.mkdir(parents=True, exist_ok=True)
    ident = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:8]
    pasta = base / ident
    pasta.mkdir(mode=0o700)
    caminho = pasta / 'config.yaml'
    with caminho.open('xb') as arq:
        arq.write(data)
        arq.flush()
        os.fsync(arq.fileno())
    metadata = {
        'tipo': 'snapshot_entrada',
        'criado_em_utc': datetime.now(timezone.utc).isoformat(),
        'origem': str(fonte.resolve()),
        'sha256_yaml_original': sha256_bytes(data),
    }
    with (pasta / 'snapshot.json').open('x', encoding='utf-8') as arq:
        json.dump(metadata, arq, indent=2, ensure_ascii=False)
        arq.write('\n')
    return caminho


@contextmanager
def bloquear_execucoes(raiz: Path):
    """Advisory exclusive lock: released by OS even on process termination.

    All runs using this launcher on the SAME shared root are mutually
    exclusive. Requires filesystem support for advisory locks (check on HPC).
    """
    pasta = raiz / 'outputs'
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo = pasta / '.execucao_controlada.lock'
    with arquivo.open('a+b') as f:
        try:
            if os.name == 'nt':
                import msvcrt
                f.seek(0)
                if f.read(1) == b'':
                    f.seek(0)
                    f.write(b'0')
                    f.flush()
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            raise RuntimeError(
                'Outro processo controlado ja executa neste projeto. '
                'Nenhum treinamento foi iniciado; aguarde a liberacao da raiz ou use '
                'um workspace fisicamente independente.'
            ) from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Execucao TCC com YAML congelado e protecao entre jobs.')
    modos = parser.add_mutually_exclusive_group(required=True)
    modos.add_argument('--planejar', action='store_true', help='Apenas planejar; sem treinamento e sem saidas.')
    modos.add_argument('--executar', action='store_true', help='SEMPRE nova pasta experimental.')
    modos.add_argument('--retomar', action='store_true', help='Retomar explicitamente uma pasta incompleta.')
    parser.add_argument('--config', type=Path, help='Snapshot YAML; sem argumento usa o YAML editavel atual.')
    parser.add_argument('--experimento', type=Path, help='Pasta em_andamento_... para --retomar.')
    args = parser.parse_args(argv)
    raiz = Path(__file__).resolve().parents[1]
    if args.retomar and (args.config is None or args.experimento is None):
        parser.error('--retomar exige --config (snapshot ORIGINAL) e --experimento.')
    if not args.retomar and args.experimento is not None:
        parser.error('--experimento e exclusivo de --retomar.')

    origem = (args.config if args.config is not None else raiz / NOME_YAML)
    if not origem.is_absolute():
        origem = raiz / origem
    if not origem.is_file():
        parser.error(f'YAML nao encontrado: {origem}')
    if args.planejar:
        from src.orquestracao_oof import main as executar_orq
        executar_orq(['--planejar', '--config', str(origem)], raiz=raiz)
        return 0

    with bloquear_execucoes(raiz):
        snapshot = origem if args.config is not None else criar_snapshot(raiz, origem)
        yaml_bytes = snapshot.read_bytes()
        h = sha256_bytes(yaml_bytes)
        import yaml
        cfg = yaml.safe_load(yaml_bytes.decode('utf-8-sig'))
        if not isinstance(cfg, dict):
            raise ValueError('YAML deve conter um mapeamento de configuracoes.')
        from src.recursos_cluster import n_jobs_efetivo
        efetivos = n_jobs_efetivo(cfg['execucao']['n_jobs'])
        # Validation before loading the dataset / running any training.
        from src.orquestracao_oof import validar_operacao, main as executar_orq
        validar_operacao(cfg)
        if args.executar and cfg['execucao'].get('diretorio_retomada') is not None:
            raise RuntimeError('Nova execucao exige execucao.diretorio_retomada: null.')
        if args.retomar and not (cfg['execucao']['retomar'] and cfg['execucao']['usar_cache']):
            raise RuntimeError('Retomada exige retomar=true e usar_cache=true no YAML original congelado.')
        params = (['--executar', '--nova-execucao'] if args.executar
                  else ['--executar', '--retomar-execucao', str(args.experimento)])
        print(f'[CONTROLE] modo={"NOVA" if args.executar else "RETOMADA_EXPLICITA"}', flush=True)
        print(f'[CONTROLE] yaml_snapshot={snapshot}', flush=True)
        print(f'[CONTROLE] sha256_yaml={h}', flush=True)
        print(f'[CONTROLE] threads_efetivas={efetivos}', flush=True)
        executar_orq(params + ['--config', str(snapshot)], raiz=raiz)
        # The pipeline reads YAML once. This check detects later mutation of the
        # snapshot, even though it cannot prevent malicious external editing.
        if sha256_bytes(snapshot.read_bytes()) != h:
            raise RuntimeError('Snapshot YAML foi alterado durante a execucao!')
    return 0


if __name__ == '__main__':
    sys.exit(main())
