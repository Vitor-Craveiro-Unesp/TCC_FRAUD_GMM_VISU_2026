# TCC_EXPERIMENTAL_V3 — Registro parcial de auditoria estática de `src.rar`

**Data:** 20/10/2026  
**Estado:** extração e inspeção do código-fonte realizadas; **testes automatizados e saídas ainda não auditados**.  
**Escopo:** reconstrução de contexto e identificação dos pontos de integração; nenhuma alteração nos scripts originais, YAML ou etapas da pipeline.

## 1. Material recebido e procedimento

- Fonte: arquivo `src.rar` fornecido na conversa.
- Extração: 52 arquivos regulares, dos quais 21 arquivos `.py` e 31 arquivos compilados em `src/__pycache__`.
- Código-fonte `.py`: 6.517 linhas somadas e 327.751 bytes; inclui `src/__init__.py` vazio.
- Os 21 `.py` puderam ser analisados por `ast.parse`, o que confirma somente **sintaxe Python analisável**, e **não funcionamento em execução**.
- Foi comparada a lista de chaves `visualizacao.graficos` do YAML já recebido com os despachantes de gráficos nos módulos de `src/`.

## 2. Módulos fonte identificados

**Etapas de dados, modelo e OOF:**
- `src/data.py`, `src/folds.py`, `src/preprocessing.py`, `src/amostragem.py`;
- `src/reducao.py` (classe `RedutorPCA`), `src/reducao_tsne.py` (classe `RedutorTSNE`);
- `src/gmm.py` (`GMMNaoSupervisionado`), `src/risco.py` (`RiscoHistoricoGaussianas`), `src/probabilidade.py` (`PreditorProbabilisticoGMM`);
- `src/metricas_oof.py`, `src/validacao_oof.py`, `src/orquestracao_oof.py`, `src/preflight_oof.py`.

**Visualização:**
- `src/visualizacao_oof.py`: despacho da etapa 07.01;
- `src/visualizacao_geometrica.py`: despacho da etapa 07.02, reconstrução no desenvolvimento F1–F9 e geradores PCA/GMM/t-SNE;
- `src/visualizacao_gmm01.py`, `src/visualizacao_gmm04.py`, `src/visualizacao_gmm07.py`: geradores específicos de GMM;
- `src/visualizacao_tsne_pdf.py`: embedding t-SNE 3D, Trustworthiness × Perplexity e diagnóstico KL;
- `src/visualizacao_integrada.py`: etapa 07.03, conferência de artefatos, galeria e mapa complementar PCA2D condicional.

## 3. Correspondência estática YAML → despachantes de gráficos

O YAML recebido possui **17 chaves em `visualizacao.graficos`**. No código enviado foram encontrados os seguintes caminhos de despacho:

### 07.01 — OOF (`visualizacao_oof.gerar_graficos_oof`)

1. `painel_score_composto` → painel 00;
2. `curva_pr` → curva PR 01;
3. `curva_mcc` → MCC por limiar 02;
4. `matriz_confusao_05` → `03_matriz_confusao_05`;
5. `matriz_confusao_mcc` → `04_matriz_confusao_mcc`;
6. `distribuicao_probabilidades` → distribuição 05;
7. `comparacao_hiperparametros` → comparação 06 (visualização adicional no código; não aparece no catálogo OOF recebido).

### 07.02 — Geometria (`visualizacao_geometrica.executar`)

1. `densidades_marginais` → módulo GMM 01;
2. `regioes_dominancia` → módulo GMM 04;
3. `sensibilidade_fronteira` → módulo GMM 07;
4. `painel_parametros_gaussianos` → painel GMM 09;
5. `pca_cotovelo` → PCA 11;
6. `pca_projecoes` → PCA 12 com bifurcação para histograma 1D e projeções 2D/3D;
7. `tsne_trustworthiness` → catálogo t-SNE 02;
8. `tsne_mapa` → t-SNE 13 (histograma 1D, projeção convencional 2D, módulo específico 3D);
9. `tsne_diagnostico` → gráfico adicional t-SNE 14, diagnóstico KL/tempo;
10. `mapa_risco_fronteira` → GMM 08.

As 17 chaves têm caminhos de chamada identificáveis nos despachantes. **Isso não valida que todas as combinações e dimensionalidades sejam executáveis**: requer os testes, dependências, dados/artefatos e, quando possível, execução controlada.

### 07.03 — Consolidação

`visualizacao_integrada.integrar` lista/copía artefatos previstos e gera a galeria HTML. Há lógica de **mapa PCA2D complementar de candidato não vencedor** sob condições específicas (`_config_candidata_2d`, `_complementar` e `_desenhar_complementar`), com nome de saída iniciado em `16_mapa_complementar_PCA2_GMM`. Esta é uma capacidade adicional identificada no código, **não uma saída cuja existência tenha sido verificada**.

## 4. Fontes já disponíveis para o futuro Biplot PCA

- `RedutorPCA.fit` conserva `modelo_` (`sklearn.decomposition.PCA`) e a lista `colunas_` com nomes das features processadas; `modelo_` disponibiliza `components_` e `explained_variance_ratio_`.
- `visualizacao_geometrica._treinar_vencedor` mantém os dados pré-processados em `data['pre']`, projeções em `data['coord']`, o redutor ajustado em `data['redutor']` e o alvo para **coloração interpretativa** em `data['y']`.
- A configuração YAML enviada usa `reducao.metodo: PCA` e `pca.n_components: [2, 3]`; a versão **1D** do futuro Biplot deve ser avaliada primeiro em prévia sintética ou preparação separada, **sem alterar silenciosamente a configuração oficial**.
- A busca textual por `biplot` nos `.py` extraídos não retornou ocorrências. Também não há chave `biplot` na configuração recebida. Não há integração explícita identificada para o novo gráfico nesta versão.
- **Não foi desenvolvido, aprovado nem integrado nenhum Biplot nesta etapa.** A forma específica em 1D ainda precisa ser deliberada/testada visualmente, pois o biplot convencional é bidimensional.

## 5. Observações para verificação futura

- `visualizacao_geometrica.executar` verifica o manifesto, usa predições OOF para avaliar limiares, reabre dados e faz o ajuste geométrico do vencedor no desenvolvimento; no código há verificações explícitas para não incluir F10. **Ainda não foi testada a execução real**.
- Apesar do destino configurado sob `teste_final/graficos_gerados`, os comentários e a lógica do gerador identificam os diagnósticos geométricos como ajuste do desenvolvimento F1–F9, não avaliação em F10. Conferir nos testes e nas saídas.
- A configuração `visualizacao.geometria.apenas_1d_2d: true` não apareceu como referência textual nos módulos Python extraídos; não tratá-la como proibição efetiva de 3D sem testes. Os despachantes incluem casos explícitos 3D em PCA/t-SNE/GMM.
- Os itens adicionais a comparar na auditoria **pipeline → PDF** são ao menos `comparacao_hiperparametros`, `tsne_diagnostico` e o mapa PCA2D complementar condicional. Status neste momento: **identificados no código, não verificados em resultados**.
- Não confundir numeração interna de arquivos com numeração dos catálogos; a manutenção de `03_matriz_confusao_05` e `04_matriz_confusao_mcc` está explícita no código.

## 6. Limitações deste registro

- `tests.rar` **ainda não recebido/inspecionado**.
- `outputs.rar` **ainda não recebido/inspecionado**.
- Não foram executados `pytest`, OOF, ajustes de GMM, experimentos PCA/t-SNE nem geradores de gráficos.
- A aprovação visual reportada anteriormente pelo usuário não foi repetida neste passo.
- Não foram verificados o ambiente local, a versão efetiva do commit no GitHub ou a sincronização de todas as cópias.

## 7. Próxima ação

1. Receber e inventariar `tests.rar` para comparar sua cobertura com as 17 entradas de gráficos e os fluxos 1D/2D/3D.
2. Consultar `outputs.rar` para comprovar quais artefatos foram realmente gerados, **se necessário**.
3. Com a auditoria de contexto estabilizada, prototipar **Biplot PCA 1D → 2D → 3D**; avaliar visualmente e obter aprovação antes de alterar `src/`/YAML.
4. Após integração aprovada, fazer auditoria bidirecional PDF ↔ código e produzir um documento consolidado de retomada.

**Regra do projeto:** toda afirmação sobre execução ou aprovação exige evidência correspondente; diante de lacunas, pedir os arquivos em vez de inventar.
