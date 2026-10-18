## Etapa 07 — Visualização e diagnóstico do modelo (CONCLUÍDA)

As subetapas 07.01 (gráficos preditivos OOF), 07.02 (diagnósticos
geométricos do vencedor) e 07.03 (galeria consolidada e mapa
complementar PCA 2D) foram implementadas e tiveram funcionamento
confirmado pelo usuário no ensaio local de 10% do desenvolvimento.

**Ensaio de referência:** `outputs/ensaio_pca_gmm_20261017_195252`;
166.716 transações F1–F9; PCA `[1,2,3]`, GMM K `[2,4]`,
54 ajustes OOF; duração 132,57 s. Vencedor amostral:
PCA=1, K=4, ID `0d74604010afa3`, score 0,501178.
O mapa complementar usa outro candidato avaliado no OOF:
PCA=2, K=2, ID `535cfea05d0b69`, com seu próprio
limiar OOF, sem substituir o vencedor.

**Gráficos:** 15 tipos cadastrados no YAML; 13 aplicáveis ao
vencedor PCA do ensaio (6 OOF e 7 geométricos), além de
1 figura complementar 2D. Os 2 tipos t-SNE não se aplicam.
O diretório `teste_final/graficos_gerados` contém aqui
diagnósticos do desenvolvimento, **não teste F10**.

**Estado:** visualização validada funcionalmente; resultados
amostrais não são conclusões científicas definitivas.
Não utilizar F10 até concluir a grade oficial no cluster,
selecionar vencedor/limiar com OOF e congelar o pipeline.
Documentação detalhada: `passos/07_visualizacao/resumo.md`.

---
