#!/usr/bin/env bash
# Diagnostico somente de leitura para verificar a configuracao atual do hpcdmc.
set -u
printf '=== IDENTIFICACAO ===\n'
date -u '+Data UTC: %Y-%m-%d %H:%M:%S' || true
uname -srm || true
printf '\n=== SISTEMA OPERACIONAL ===\n'
if [ -r /etc/os-release ]; then cat /etc/os-release; elif [ -r /etc/centos-release ]; then cat /etc/centos-release; else uname -a; fi
printf '\n=== GLIBC ===\n'
if command -v ldd >/dev/null 2>&1; then ldd --version 2>&1 | head -2; else echo 'ldd indisponivel'; fi
printf '\n=== PYTHON DISPONIVEL ===\n'
for p in python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
  if command -v "$p" >/dev/null 2>&1; then
    printf '%s => ' "$p"; "$p" --version 2>&1 || true
  fi
done
printf '\n=== FERRAMENTAS ===\n'
for p in sbatch srun sinfo squeue scontrol sha256sum mktemp gcc; do
  if command -v "$p" >/dev/null 2>&1; then command -v "$p"; else printf '%s: indisponivel\n' "$p"; fi
done
printf '\n=== PARTICOES SLURM (somente consulta) ===\n'
if command -v sinfo >/dev/null 2>&1; then sinfo -o '%P %a %D %c %m %l' 2>&1 || true; else echo 'sinfo indisponivel'; fi
printf '\n=== RECURSOS VISIVEIS (nao equivalem a alocacao do job) ===\n'
if command -v nproc >/dev/null 2>&1; then nproc || true; fi
if command -v free >/dev/null 2>&1; then free -h || true; fi
printf '\n=== HOME E ARMAZENAMENTO ===\n'
printf 'HOME: %s\n' "$HOME"
if command -v df >/dev/null 2>&1; then df -h "$HOME" || true; fi
printf '\n=== FIM - SEM SUBMISSAO OU TREINAMENTO ===\n'
