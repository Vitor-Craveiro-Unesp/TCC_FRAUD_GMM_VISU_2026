# PASSO 05 - GAUSSIAN MIXTURE MODEL

## Status
CONCLUIDO

## 05.01 - Ajuste do GMM
- GMM unico e nao supervisionado.
- Numero de gaussianas definido pelo YAML.
- Responsabilidades probabilisticas calculadas.
- NORMAL e RUS implementados.

## 05.02 - Risco por gaussiana
- Risco calculado por responsabilidades e rotulos.
- Estimativa realizada no treino original.
- RUS restrito ao ajuste do GMM.
- Para K=1, risco igual a prevalencia do treino.

## 05.03 - Probabilidade final
- Combinacao dos riscos por soft assignment.
- Scores entre 0 e 1.
- Sem classificacao rigida por gaussiana.
- Sem otimizacao antecipada de limiar.

## Validacoes
- 188 testes automatizados aprovados.
- Verificacoes com amostras reais aprovadas.
- F10 preservado para teste final.

## Proxima etapa
Passo 06 - Validacao cruzada OOF.
