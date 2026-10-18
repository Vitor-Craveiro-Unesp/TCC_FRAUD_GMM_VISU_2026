# Passo 07 — Visualização e diagnóstico do modelo

**07.01:** gráficos OOF 01–06 a partir do vencedor e ranking; no gráfico 06, comparar candidatos. Sem reajuste e sem F10.

**07.02:** gráficos geométricos 07–15 aplicáveis ao método do vencedor (PCA vs t-SNE). Refazer exatamente um ajuste do vencedor no desenvolvimento, sem F10. A pasta de destino original é `teste_final/graficos_gerados`, mas seu conteúdo nesta etapa é de **desenvolvimento**, não de teste F10.

**07.03:** integrar resultados em uma galeria HTML `visualizacao_consolidada/index.html`; gerar um mapa bidimensional suplementar quando o vencedor for PCA 1D e houver um candidato PCA 2D previamente avaliado no OOF. O segundo modelo **não se torna vencedor**, sua fronteira usa seu próprio limiar OOF. Gráficos incompatíveis são omitidos e registrados.

**Estado:** 07.01 e 07.02 executados pelo usuário no ensaio de 10%, conforme resultados informados. 07.03 implementado e testado sinteticamente; execução com o ensaio real pendente.

**Próxima fase:** Passo 08 — avaliar F10 apenas quando seleção, limiar e pipeline vencedor estiverem congelados; não derivar métricas finais dos ensaios de 5% ou 10%.
