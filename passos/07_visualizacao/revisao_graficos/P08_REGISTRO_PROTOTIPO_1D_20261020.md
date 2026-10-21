# TCC_EXPERIMENTAL_V3 — P08 — Prévia de Biplot PCA 1D

**Data:** 2026-10-20  
**Status:** PRÉVIA SINTÉTICA GERADA — **AGUARDANDO AVALIAÇÃO VISUAL E APROVAÇÃO DO USUÁRIO**.  
**Não integrado** à pipeline. **Nenhum dado real/F10 utilizado.**

## 1. Fundamentação e fontes verificadas

- Catálogo fornecido `PCA_CATALOGO_GRAFICOS_11_12_histograma1D(1).pdf`: contém o Scree Plot (11) e a projeção PCA 1D/2D/3D (12), **não contém Biplot**.
- `src/reducao.py` do `src.rar`: `RedutorPCA` usa `sklearn.decomposition.PCA`, retém `modelo_` e `colunas_`, preserva os eixos `PC1...`.
- `src/visualizacao_geometrica.py`: já implementa 12 PCA 1D como histograma, além das projeções 2D/3D. Esta prévia **não altera** o 12.
- Cores de classe conforme YAML fornecido: legítima `#1f77b4` e fraude `#d62728`.

## 2. Decisão metodológica proposta para o Biplot 1D

Uma PCA com `n_components=1` só possui PC1; um Biplot 2D convencional não é possível sem inventar PC2. A solução proposta é **Biplot 1D adaptado em dois painéis**:

1. **Scores:** projeções individuais em PC1, com *jitter* vertical puramente visual; a altura não é PC2.
2. **Variáveis:** setas com correlação entre cada variável de entrada e PC1, de −1 a +1, indicando magnitude e sinal.

**As escalas são distintas e explícitas**; não há sobreposição incorreta de scores com *loadings*. Este gráfico não substitui o histograma 1D do gráfico 12.

Fórmula dos coeficientes de correlação com o score PC1, calculados no mesmo conjunto de treinamento:

`corr(X_j, PC1) = components_[0, j] × sqrt(explained_variance_[0]) / std_treino(X_j, ddof=1)`

No protótipo o valor analítico foi comparado com a correlação de Pearson calculada diretamente, com tolerância `1e-10` (aprovado). O sinal dos eixos PCA pode inverter sem alterar a decomposição.

## 3. Dados e execução

- Totalmente **sintéticos**, semente `20261020`, 600 observações (550 legítimas / 50 fraudes simuladas), seis características fictícias A–F; nomes não correspondem ao dataset real.
- PCA `n_components=1`, `svd_solver='full'`, **ajustado sem rótulos**. Os rótulos só foram usados para colorir as observações.
- Variância explicada no exemplo simulado: `0.569505` (56,95%), **não é métrica do TCC real**.
- Verificações locais: arrays finitos; apenas um componente; correlações dentro de [−1,+1]; fórmula da correlação igual ao cálculo direto.
- Não foi importado módulo do projeto, lido Parquet, usado F1–F10 nem treinado GMM. Nenhuma alteração de código/YAML/testes originais.

## 4. Arquivos criados nesta etapa

**Salvar todos em `TCC_EXPERIMENTAL_V3/transferencia_ias_1/`:**

- `P08_BIPLOT_PCA_1D_PREVIA_SIMULADA.png` — imagem para avaliação.
- `P08_BIPLOT_PCA_1D_PREVIA_SIMULADA.py` — geração numérica/visual reproduzível, sem entradas reais.
- `P08_BIPLOT_PCA_1D_VARIAVEIS_SIMULADAS.csv` — correlações e componentes da simulação.
- `P08_REGISTRO_PROTOTIPO_1D_20261020.md` — este registro.
- `PENDENCIAS_ACUMULADAS_TCC_20261020.md` — status geral atualizado: P08 iniciou, ainda aberta.

Para reproduzir após copiar para a pasta local, com `.venv` ativada na raiz:

`python .\transferencia_ias_1\P08_BIPLOT_PCA_1D_PREVIA_SIMULADA.py`

A execução reescreve **somente os dois arquivos derivados PNG/CSV do protótipo**, na mesma pasta, não acessa dados reais nem a pipeline.

## 5. Próximo portão de aprovação

Solicitar avaliação visual da prévia 1D. Se rejeitada, ajustar a imagem sintética. **Somente após aprovação explícita**, avançar ao protótipo Biplot PCA 2D, em seguida 3D; a integração e os testes reais da P08 vêm depois das três aprovações.

**Não afirmar P08 concluída nesta fase.**
