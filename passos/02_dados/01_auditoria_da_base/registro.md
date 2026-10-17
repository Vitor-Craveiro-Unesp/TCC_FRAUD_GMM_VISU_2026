# PASSO 02.01 - AUDITORIA DA BASE

## Status
CONCLUIDO

## Data
2026-10-15 22:58:57

## Base
data/raw/dataset_raw.parquet

## Distribuicao
- Total: 1.852.394 transacoes
- Legitimas: 1.842.743 (99,479%)
- Fraudes: 9.651 (0,521%)

## Verificacoes
- Leitura Parquet executada com sucesso.
- Tipos das variaveis identificados.
- Valores ausentes auditados.
- Cardinalidades amostrais verificadas.
- NID_ALPHA: nenhuma repeticao na amostra de 50.000 IDs.
- Encodings identificados pelos sufixos.

## RECEIVE_LOC_FEWF
- Prefixo fraud_ presente em todas as transacoes.
- Prefixo presente em ambas as classes.
- Nao identifica exclusivamente fraudes.
- Manter Frequency Encoding.
- Frequencias aprendidas somente no treino.

## Metodologia de divisao
- Estratificada com embaralhamento.
- Semente definida no YAML.
- 10 folds totais.
- Fold 10 reservado para teste final.
- Folds 1 a 9 para validacao cruzada.

## Proxima etapa
02.02 - Divisao e persistencia dos folds.

## Observacao
Nenhum modelo foi treinado nesta etapa.
A base original nao foi modificada.
