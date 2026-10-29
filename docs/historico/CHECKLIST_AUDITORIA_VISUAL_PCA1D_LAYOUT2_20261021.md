# TCC_EXPERIMENTAL_V3 — Checklist de auditoria visual e documental

**Ensaio:** `outputs/ensaio_pca_gmm_20261021_184011/`  
**Escopo:** PCA 1D + GMM K=[2,4,6] + RUS 1:16, 10% do desenvolvimento, F1–F9 OOF. **F10 ainda não foi avaliado.**  
**Critério de escolha:** `score = [AP + (MCC+1)/2 + exp(-LogLoss)]/3`, maior primeiro; AP apenas desempate.

## A. Identificação, reprodutibilidade e estrutura

- [x ] `configuracao.yaml` e `manifesto.json` estão diretamente na raiz do ensaio, com YAML correspondente à execução.
- [ x] O manifesto informa `tipo=ensaio`, layout 2, `status=concluido_06_03` e `visualizacoes=concluidas`, com etapas 07.01, 07.02 e 07.03 registradas.
- [ ] Manifesto contém data/hora, SHA-256 do parquet/folds/YAML, versões Python/bibliotecas, quantidade de candidatos e folds.
- [ ] `registros_desenvolvimento=166716`, `registros_reservados_f10=185239`, sem usar o F10 na seleção/OOF/geométricos.
- [ x] O experimento registra `PCA n_components=1`, `RUS razao=16`, `GMM n_components=[2,4,6]`, `covariance_type=full`, `n_init=3`, `max_iter=300`, random seed e fração base 0.1.
- [x ] `candidatos/` possui 3 pastas (`K02_*`, `K04_*`, `K06_*`), cada uma com `configuracao.json`, `metricas/metricas.json`, `metricas/por_fold.csv`, `oof/predicoes.npz`, `oof/cache/F01.npz` ... `F09.npz`.
- [ ] Cada candidato tem 9 folds, 18.524 registros por fold, 166.716 predições OOF, mesmos 870 positivos reais e ausência de F10.
- [ ] `vencedor/metricas` contém `metricas.json`, `por_fold.csv`, `resultados_grade.csv` e `melhor_configuracao.json`.
- [ ] `vencedor/oof/predicoes.npz` corresponde ao arquivo do candidato vencedor.
- [ ] `vencedor/visualizacao/oof`, `geometria_desenvolvimento` e `galeria` estão presentes.
- [ ] `vencedor/teste_final/` existe, mas permanece reservado/vazio; não confundir com geometria de desenvolvimento.

## B. Métricas, ranking e método

- [ ] A tabela da grade contém os três candidatos, ordenados pelo **score composto**: K=6 (0.5031703503), K=4 (0.5013973292), K=2 (0.5009194092).
- [ ] AP, MCC, LogLoss, score, limiar MCC, N, prevalência e configuração estão registrados por candidato.
- [ ] `melhor_configuracao.json` identifica K=6, `id_candidato=aafef1fb941afd`, limiar 0.0144200998 e `teste_final_usado=false`.
- [ ] Vencedor: AP=0.0151322957, MCC=0.0510627410, LogLoss=0.0316481771, Score=0.5031703503, N=166716, prevalência=0.0052184553.
- [ ] Há 870 fraudes e 165846 legítimas no OOF amostral, prevalência 0.5218455%.
- [ ] As métricas por fold indicam AP, LogLoss, MCC no **limiar global OOF congelado**; não confundir com MCC máximo separado por fold.
- [ ] A amostragem RUS se aplica ao **ajuste** GMM em cada treinamento; validações OOF e estimativa de risco histórico seguem as regras do YAML, sem ajustar o modelo com F10.
- [ ] O score composto é uma função de seleção interna, não uma métrica externa padronizada; AP deve ser interpretada em relação à prevalência.

## C. Os sete gráficos OOF — `vencedor/visualizacao/oof/`

- [ ] **00 — `00_painel_score_composto.png`:** quatro indicadores AP, (MCC+1)/2, exp(-LogLoss), score; valores brutos de MCC/LogLoss, baseline de prevalência, fórmula, OOF/F10 ausente, boa legibilidade.
- [ ] **01 — `01_curva_precision_recall.png`:** Precision × Recall, AP correta (não AUC trapezoidal), linha de prevalência, ponto operacional τ* e ponto 0,50 ou sua omissão justificada, zoom indicado, CSV com curva completa.
- [ ] **02 — `02_mcc_por_limiar.png`:** limiar τ*≈0.0144201 destacado, MCC(τ*)≈0.0510627, referência 0,50 e MCC(0,50), curva com MCC bruto, eventual 0,50 fora do zoom explicada, CSV de curva completa.
- [ ] **03 — `03_matriz_confusao_05.png`:** TN/FP/FN/TP sob τ=0,50, totais da matriz somam 166716, orientação linhas reais/colunas preditas, percentuais por classe, métricas de rodapé, sem zero-as-confiabilidade artificial.
- [ ] **04 — `04_matriz_confusao_mcc.png`:** mesmas checagens sob τ*≈0.0144201; comparação com 03; contagens, proporções e rodapé legíveis.
- [ ] **05 — `05_distribuicao_probabilidades.png`:** scores de fraude por classe real, 50 bins, normalização por classe, legenda, limiares 0,50 e τ*, zoom indicado e sem confundir scores com rótulos.
- [ ] **06 — `06_comparacao_hiperparametros.png`:** K=2/4/6, AP, MCC, LogLoss, score composto, vencedor K=6 destacado, ordenação por score; CSV reúne toda a grade.
- [ ] Existem 7 PNGs, 9 CSVs e 2 JSONs na pasta OOF; os PNGs abrem sem cortes, sem sobreposições e com fonte legível.

## D. Nove imagens geométricas — `vencedor/visualizacao/geometria_desenvolvimento/`

- [ ] **GMM 09 parte 01 — `09_painel_parametros_gaussianos_parte_01.png`:** quatro primeiras gaussianas, π_k, risco histórico r_k, média µ_k e covariância Σ_k; título e regra descritiva legíveis.
- [ ] **GMM 09 parte 02 — `09_painel_parametros_gaussianos_parte_02.png`:** gaussianas 5 e 6, mesmas informações; rótulos alto/baixo risco puramente descritivos.
- [ ] **PCA 11 — `11_pca_cotovelo.png`:** variância explicada individual, cumulativa, dois eixos corretamente rotulados, linhas-guia e seleção PC1 visível; outras PCs no gráfico são diagnóstico, não reconfiguração do vencedor.
- [ ] **PCA 12 — `12_pca_projecao_PC1.png`:** distribuição/histograma de PC1, classes reais distinguíveis, médias e curvas de suavização conforme YAML, nada que simule um eixo PC2.
- [ ] **Biplot 1D — `PCA_BIPLOT_1D.png`:** scores ao longo de PC1, jitter vertical explicitamente visual, variância explicada, vetores/correlações Pearson das variáveis com PC1; nomes e valores legíveis; CSV de todas as variáveis.
- [ ] **GMM 08 — `GMM08_mapa_risco_1d_comparativo.png`:** dois painéis no mesmo campo de risco FULL (τ=0,50 vs τ*), faixas/regiões que mudam classe, intervalos gaussianos de 95% e valores MCC OOF corretos.
- [ ] **GMM 07 — `GMM07_sensibilidade_1d_comparativo.png`:** dois cortes de decisão no mesmo risco (0,50 vs τ*), região de mudança legível, MCC de cada corte calculado em OOF, sem inventar MCC da malha; sem exigir elipses (reservadas ao GMM 08).
- [ ] **GMM 01 — `GMM01_densidade_marginal_1d_PC1.png`:** seis gaussianas ponderadas por π_k, curva da mistura total, PC1 e densidade marginal rotulados; **densidade não é probabilidade de fraude**.
- [ ] **GMM 04 — `GMM04_dominancia_1d.png`:** regiões onde cada componente maximiza a responsabilidade γ_k, cores e legenda coerentes com componentes; **dominância não é classificação de fraude**.
- [ ] Há 9 PNGs de 8 tipos geométricos, CSVs auxiliares, `manifesto_geometria_07_02.json` e `relatorio_gaussianas_desenvolvimento.{csv,json,md}`.
- [ ] Relatório por gaussiana contém 6 componentes com π_k, massa de responsabilidades, massas soft por classe, r_k, razão risco/prevalência, vetor µ_k e matriz Σ_k.
- [ ] Regra alto risco gaussiana: `r_k >= 2 × prevalência do treino`, unicamente descritiva; não troca soft assignment nem limiar de transações.
- [ ] Origem dos dados: geometria reajustada **somente em F1–F9**, sem observar F10. Não confundir performance dentro do treino com avaliação independente.

## E. Galeria e apresentação humana

- [ ] `vencedor/visualizacao/galeria/index.html` abre localmente, sem internet, e mostra **16 imagens** (7 OOF + 9 geométricas).
- [ ] As seções da galeria identificam corretamente OOF e geometria do desenvolvimento; legenda informa ensaio amostral e ausência de F10.
- [ ] Todas as 16 prévias carregam e podem ser abertas/ampliadas; nomes, links e referências estão corretos.
- [ ] `manifesto_integracao_07_03.json` registra tipos presentes e omitidos, referências de arquivos e nenhuma avaliação F10.
- [ ] Nenhum eixo cortado, legenda sobre dados, marcador invisível, porcentagem incoerente, texto reduzido ou PNG corrompido.
- [ ] `manifesto.json` registra `visualizacoes=concluidas`, com as três etapas concluídas; terminal terminou com código 0.

## F. Limites e pendências que **não** são falhas deste ensaio

- [ ] Confirmado que o F10 ainda não possui avaliação independente. A pasta `teste_final/` deve ficar reservada.
- [ ] A serialização do modelo final completo está declarada como pendente; os outputs geométricos não substituem um modelo serializado.
- [ ] Não exigir gráficos t-SNE no PCA 1D, nem mapas PCA 2D/3D, nem diagramas removidos (AIC/BIC, calibração/Brier visual, Lift/Ganhos).
- [ ] Registrar separadamente problemas encontrados: arquivo/gráfico, descrição objetiva, gravidade (bloqueante/visual/documental), correção sugerida, resultado após revisão.

## Conferência de números OOF anteriores (mesma configuração)

| Candidato | AP | MCC OOF | LogLoss | Score composto | Limiar MCC |
|---|---:|---:|---:|---:|---:|
| K=6 | 0.0151322957 | 0.0510627410 | 0.0316481771 | 0.5031703503 | 0.0144200998 |
| K=4 | 0.0112546596 | 0.0485071462 | 0.0318170826 | 0.5013973292 | 0.0105237325 |
| K=2 | 0.0117036751 | 0.0454287328 | 0.0321718215 | 0.5009194092 | 0.0068882886 |

**Observação geral:** score 0,503 não equivale a 50,3% de acerto. O score é composto por três métricas com transformações específicas.

## Registro de inspeção

| Item / arquivo | Situação (OK / corrigir / N/A) | Problema observado | Prioridade |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
