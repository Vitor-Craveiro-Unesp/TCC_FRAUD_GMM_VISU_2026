# PASSO 03.03 - UNDERSAMPLING NORMAL/RUS

## Status
CONCLUIDO

## Data
2026-10-16 00:14:13

## Arquivos
- src/amostragem.py
- tests/test_amostragem.py
- tests/verificar_amostragem_base_real.py

## Metodologia
NORMAL:
- Utiliza todo o treinamento no ajuste do GMM.

RUS:
- Reduz apenas a classe legitima.
- Preserva todas as fraudes do treinamento.
- Utiliza semente para reprodutibilidade.
- Seleciona indices sem modificar a base original.

## Regras estatisticas
- Amostragem restrita ao treinamento.
- Validacao nao sofre undersampling.
- F10 permanece reservado.
- Risco por gaussiana sera estimado com
  todos os registros originais do treino.
- RUS afeta somente o ajuste do GMM.

## Validacoes
- 40 testes automatizados aprovados no total.
- Verificacao com base real aprovada.
- NORMAL e RUS verificados.

## Proxima etapa
Passo 04 - Reducao dimensional PCA/t-SNE.
