# PASSO 04 - REDUCAO DIMENSIONAL

## Status
CONCLUIDO

## 04.01 - PCA
- Implementado em src/reducao.py.
- Componentes 2 e 3 previstos no YAML.
- Ajuste exclusivo no treinamento.
- Transformacao da validacao sem reajuste.
- Variancia explicada avaliada.

## 04.02 - t-SNE
- Implementado em src/reducao_tsne.py.
- Biblioteca openTSNE.
- Inicializacoes random e pca.
- Transformacao de novos dados verificada.
- Testado com amostras reais.
- Escalabilidade completa ainda nao avaliada.

## Regras gerais
- PCA e t-SNE sao alternativas independentes.
- Nunca aplicar PCA seguido de t-SNE.
- Nenhum rotulo utilizado na reducao.
- F10 permanece reservado.
- Parametros aprendidos somente no treino.

## Validacoes
- Suite de 75 testes confirmada como aprovada.
- PCA e t-SNE verificados com base real.

## Proxima etapa
Passo 05 - GMM e estimativa de risco.
