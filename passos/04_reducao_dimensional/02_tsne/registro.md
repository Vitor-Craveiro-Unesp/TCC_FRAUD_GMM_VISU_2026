# PASSO 04.02 - t-SNE COM openTSNE

## Status
CONCLUIDO

## Data
2026-10-16 19:37:58

## Arquivos
- src/reducao_tsne.py
- tests/test_reducao_tsne.py
- tests/verificar_tsne_base_real.py

## Metodologia
- t-SNE implementado como alternativa ao PCA.
- Utiliza a biblioteca openTSNE.
- Ajuste realizado apenas no treinamento.
- Validacao inserida no embedding ja ajustado.
- Rotulos nao utilizados no ajuste.
- F10 reservado para teste final.
- PCA nao executado como etapa anterior ao t-SNE.
- Inicializacao interna por PCA permitida.

## Configuracoes verificadas
- Perplexity: 10.
- Dimensoes: 2.
- Inicializacoes: random e pca.
- Iteracoes principais: 500.
- Early exaggeration: 12.
- Theta: 0.5.

## Validacoes
- Testes automatizados confirmados pelo usuario.
- Suite total ate o Passo 04: 75 testes.
- Verificacao com base real aprovada.
- Amostra de treinamento: 1000 registros.
- Amostra de validacao: 200 registros.
- F10 nao utilizado.

## Limitacoes
- Verificacao realizada com amostra reduzida.
- Custo computacional em escala completa
  ainda precisa ser avaliado.

## Proxima etapa
Passo 05 - GMM e risco por componente.
