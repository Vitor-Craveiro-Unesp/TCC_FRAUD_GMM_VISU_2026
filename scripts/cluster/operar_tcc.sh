#!/usr/bin/env bash
# Interface operacional: NAO altera o pipeline cientifico, YAML ou dados.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT"

uso() {
  cat <<'HELP'
Uso: bash scripts/cluster/operar_tcc.sh COMANDO [OPCOES]
  diagnostico                   Informacoes de Linux/SLURM, sem treinamento
  planejar                      Planejamento YAML, sem treinamento
  executar OPCOES_SBATCH        Novo experimento (snapshot antes da fila)
  retomar PASTA SNAPSHOT OPCOES Retomada explicita do experimento
  fila                          Jobs do usuario no SLURM

Exemplo de parametros operacionais, fornecidos pelo administrador:
  executar --cpus-per-task="$CPUS" --mem="$MEMORIA" --time="$TEMPO"
  Adicione --partition="$PARTICAO" e --account="$CONTA" SOMENTE quando exigidos.
A configuracao cientifica vem APENAS do YAML.
HELP
}

[[ $# -ge 1 ]] || { uso; exit 2; }
cmd="$1"
shift

case "$cmd" in
  diagnostico)
    [[ $# -eq 0 ]] || { uso; exit 2; }
    bash scripts/cluster/diagnostico_hpcdmc.sh
    ;;
  planejar)
    [[ $# -eq 0 ]] || { uso; exit 2; }
    if [[ -n "${TCC_VENV_DIR:-}" ]]; then
      VENV="$TCC_VENV_DIR"
      [[ "$VENV" == /* ]] || VENV="$ROOT/$VENV"
    elif [[ -f "$ROOT/meu_env/bin/activate" ]]; then
      VENV="$ROOT/meu_env"
    else
      VENV="$ROOT/.venv"
    fi
    [[ -x "$VENV/bin/python" ]] || {
      echo "Python do ambiente nao encontrado: $VENV/bin/python" >&2
      echo 'Prepare primeiro um ambiente Linux compativel, sem substituir o ambiente Windows.' >&2
      exit 2
    }
    "$VENV/bin/python" -m src.execucao_controlada --planejar
    ;;
  executar)
    [[ $# -ge 1 ]] || { echo 'Informe os recursos SBATCH validados no cluster.' >&2; uso; exit 2; }
    cpus_ok=0
    prev=''
    for arg in "$@"; do
      [[ "$arg" == --cpus-per-task=* ]] && cpus_ok=1
      [[ "$prev" == --cpus-per-task ]] && cpus_ok=1
      prev="$arg"
    done
    [[ "$cpus_ok" -eq 1 ]] || { echo 'Informe --cpus-per-task=VALOR (exigido pelo batch atual).' >&2; exit 2; }
    if [[ -z "${TCC_VENV_DIR:-}" ]]; then
      if [[ -f "$ROOT/meu_env/bin/activate" ]]; then export TCC_VENV_DIR="$ROOT/meu_env"
      else export TCC_VENV_DIR="$ROOT/.venv"; fi
    fi
    bash scripts/cluster/submeter.sh "$@"
    ;;
  retomar)
    [[ $# -ge 3 ]] || { uso; exit 2; }
    if [[ -z "${TCC_VENV_DIR:-}" ]]; then
      if [[ -f "$ROOT/meu_env/bin/activate" ]]; then export TCC_VENV_DIR="$ROOT/meu_env"
      else export TCC_VENV_DIR="$ROOT/.venv"; fi
    fi
    bash scripts/cluster/submeter.sh --retomar "$@"
    ;;
  fila)
    [[ $# -eq 0 ]] || { uso; exit 2; }
    command -v squeue >/dev/null || { echo 'squeue indisponivel.' >&2; exit 2; }
    squeue -u "$(id -un)"
    ;;
  -h|--help|ajuda) uso ;;
  *) echo "Comando desconhecido: $cmd" >&2; uso; exit 2 ;;
esac
