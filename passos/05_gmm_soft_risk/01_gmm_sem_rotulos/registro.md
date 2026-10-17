# PASSO 05.01 - AJUSTE DO GMM

## Status
CONCLUIDO

## Data
2026-10-17 09:51:08

## Arquivos
- src/gmm.py
- tests/test_gmm.py
- tests/verificar_gmm_base_real.py

## Metodologia
- Um unico Gaussian Mixture Model nao supervisionado.
- Ajuste realizado exclusivamente com features.
- Nenhum rotulo utilizado no ajuste.
- Numero de componentes configuravel pelo YAML.
- Responsabilidades probabilisticas calculadas.
- Validacao realizada sem reajustar o GMM.
- F10 reservado para teste final.

## NORMAL
- Utiliza todas as observacoes do treinamento.

## RUS
- Reduz somente as observacoes legitimas.
- Preserva todas as fraudes.
- Afeta somente o ajuste do GMM.
- Treinamento original preservado para RH.

## Validacoes
- Suite total: 106 testes automatizados.
- Testes confirmados como aprovados pelo usuario.
- Verificacao com base real confirmada pelo usuario.
- Combinacoes de PCA, NORMAL/RUS e K verificadas.

## Limitacoes
- Validacao realizada em amostras reais.
- Execucao em escala completa ainda pendente.
- Risco por gaussiana ainda nao implementado.

## Proxima tarefa
05.02 - Estimativa de risco por gaussiana.
