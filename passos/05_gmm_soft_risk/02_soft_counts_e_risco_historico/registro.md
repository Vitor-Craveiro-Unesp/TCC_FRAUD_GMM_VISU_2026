# PASSO 05.02 - RISCO HISTORICO POR GAUSSIANA

## Status
CONCLUIDO

## Data
2026-10-17 10:01:57

## Arquivos
- src/risco.py
- tests/test_risco.py
- tests/verificar_risco_base_real.py

## Metodologia
- Estimacao do risco por componente gaussiano.
- Utilizacao de responsabilidades probabilisticas.
- Rotulos utilizados somente na estimacao de risco.
- Soft assignment, sem classificacao rigida.
- Estimacao realizada no treinamento original.

## Formula
RH_k = soma(gamma_ik * y_i) / soma(gamma_ik)

## NORMAL E RUS
- NORMAL: GMM ajustado com todo o treinamento.
- RUS: GMM ajustado com amostra reduzida.
- Nos dois modos, RH usa o treino original completo.
- A prevalencia da amostra RUS nao substitui
  a prevalencia do treinamento original.

## Verificacoes
- Suite total de 148 testes automatizados aprovada.
- Verificacao com amostra real aprovada.
- PCA, NORMAL/RUS e K=1/K=2 verificados.
- Para K=1, RH corresponde a prevalencia do treino.
- F9 nao utilizado para estimar RH.
- F10 preservado para o teste final.

## Limitacoes
- Testes com amostras reais.
- Execucao completa ainda pendente.
- Probabilidade final sera implementada no 05.03.

## Proxima tarefa
05.03 - Probabilidade final de fraude.
