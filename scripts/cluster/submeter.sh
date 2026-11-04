#!/usr/bin/env bash
# Captura o YAML ANTES da fila SLURM. Uso: bash scripts/cluster/submeter.sh [opcoes sbatch]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT"
YAML="$ROOT/config_TCC_EXPERIMENTAL_V3_atualizado.yaml"
if [[ ! -f "$YAML" ]]; then echo "YAML ausente: $YAML" >&2; exit 2; fi
if ! command -v sbatch >/dev/null 2>&1; then echo 'sbatch nao esta disponivel neste terminal.' >&2; exit 2; fi
mkdir -p outputs/.submissoes outputs/logs
RETOMADA=""
if [[ "${1:-}" == "--retomar" ]]; then
  if [[ $# -lt 3 ]]; then
    echo 'Uso retomada: submeter.sh --retomar <pasta_em_andamento> <snapshot_original> [opcoes_sbatch]' >&2
    exit 2
  fi
  RETOMADA="$2"
  SNAPSHOT="$3"
  shift 3
  if [[ ! -d "$RETOMADA" || ! -f "$SNAPSHOT" ]]; then
    echo 'Pasta de retomada ou snapshot nao encontrado.' >&2; exit 2
  fi
  SNAPSHOT="$(cd "$(dirname "$SNAPSHOT")" && pwd -P)/$(basename "$SNAPSHOT")"
  # Reaproveita exatamente os bytes do YAML original; nunca usa o YAML editado.
  if [[ ! -f "$(dirname "$SNAPSHOT")/sha256.txt" ]] ||
     ! sha256sum --check --status "$(dirname "$SNAPSHOT")/sha256.txt"; then
    echo 'Snapshot original sem hash valido para retomada.' >&2; exit 2
  fi
  RETOMADA="$(cd "$RETOMADA" && pwd -P)"
else
  DIR="$(mktemp -d "$ROOT/outputs/.submissoes/job_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXXXX")"
  chmod 700 "$DIR"
  cp -- "$YAML" "$DIR/config.yaml"
  chmod 400 "$DIR/config.yaml"
  sha256sum "$DIR/config.yaml" > "$DIR/sha256.txt"
  SNAPSHOT="$DIR/config.yaml"
fi
# Sem parametros cientificos no sbatch. Recursos passados conforme politica do cluster.
if ! saida="$(sbatch --parsable "$@" \
    "--export=ALL,TCC_CONFIG_SNAPSHOT=$SNAPSHOT,TCC_RETOMAR_EXPERIMENTO=$RETOMADA" \
    scripts/cluster/tcc_generic.batch)"; then
  echo "Submissao falhou. Snapshot preservado para auditoria: $SNAPSHOT" >&2
  exit 1
fi
printf 'SLURM_JOB_ID=%s\nSNAPSHOT=%s\n' "$saida" "$SNAPSHOT"
cat "$(dirname "$SNAPSHOT")/sha256.txt"
