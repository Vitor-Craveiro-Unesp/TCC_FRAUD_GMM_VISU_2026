"""Operational CPU ceiling for SLURM; no changes to YAML or statistics."""
from __future__ import annotations

import os


def n_jobs_efetivo(n_jobs: int) -> int:
    """Return effective numerical threads without exceeding SLURM_CPUS_PER_TASK.

    A positive YAML value exceeding the allocation is rejected, not silently
    rewritten.  Outside SLURM, retain the legacy local behavior.
    """
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or (n_jobs != -1 and n_jobs < 1):
        raise ValueError('execucao.n_jobs deve ser -1 ou inteiro positivo.')
    granted = os.environ.get('SLURM_CPUS_PER_TASK')
    if granted is None:
        return (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    try:
        maximum = int(granted)
        if maximum < 1:
            raise ValueError()
    except ValueError as exc:
        raise RuntimeError('SLURM_CPUS_PER_TASK precisa ser inteiro positivo.') from exc
    if n_jobs != -1 and n_jobs > maximum:
        raise RuntimeError(f'YAML n_jobs={n_jobs} excede SLURM_CPUS_PER_TASK={maximum}.')
    return maximum if n_jobs == -1 else n_jobs
