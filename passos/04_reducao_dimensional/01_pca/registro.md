# PASSO 04.01 - PCA

## Status
CONCLUIDO

## Data
2026-10-16 15:24:14

## Arquivos
- src/reducao.py
- tests/test_reducao_pca.py
- tests/verificar_pca_base_real.py

## Metodologia
- PCA aplicado apos o pre-processamento.
- Ajuste exclusivamente no treinamento.
- Validacao transformada sem reajuste.
- Configuracoes de 2 e 3 componentes.
- Semente e parametros definidos no YAML.
- Nenhum rotulo utilizado no PCA.
- F10 reservado para o teste final.

## Validacoes
- Testes automatizados confirmados pelo usuario.
- Verificacao com base real confirmada pelo usuario.
- Estrutura de colunas preservada.
- Variancia explicada avaliada.

## Proximo subpasso
04.02 - t-SNE com openTSNE.

## Observacao
PCA e t-SNE sao alternativas independentes.
Nao executar PCA antes do t-SNE.
