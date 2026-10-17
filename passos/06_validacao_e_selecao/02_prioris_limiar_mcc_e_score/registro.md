# PASSO 06.02 - METRICAS OOF

## Status
CONCLUIDO

## Data
2026-10-17 18:37:29

## Arquivos
- src/metricas_oof.py
- tests/test_metricas_oof.py
- tests/verificar_metricas_base_real.py

## Metodologia
- Average Precision calculada por average_precision_score.
- MCC calculado para classificacoes por limiar.
- Limiar selecionado pela maximizacao do MCC.
- LogLoss calculada sobre probabilidades.
- Score composto calculado conforme o YAML.
- Verificacao de integridade das predicoes OOF.
- F10 excluido da selecao de modelos e limiares.

## Score composto
S = [AP + (MCC+1)/2 + exp(-LogLoss)]/3

## Otimizacao do MCC
- Busca eficiente por ordenacao dos scores.
- Tratamento de probabilidades repetidas.
- Limiar definitivo sera selecionado no OOF
  consolidado dos nove folds.

## Validacoes
- Suite de 250 testes automatizados aprovada.
- Verificacao com base real aprovada.
- AP, MCC, LogLoss e score composto verificados.
- F10 preservado.

## Limitacoes
- A verificacao real foi realizada com amostra.
- Nove rodadas OOF completas ainda pendentes.
- Selecao definitiva do limiar MCC pendente.
- Grade e persistencia ainda nao implementadas.
- A avaliacao independente em F10 sera posterior.

## Proxima tarefa
06.03 - Grade, persistencia, cache e retomada.
