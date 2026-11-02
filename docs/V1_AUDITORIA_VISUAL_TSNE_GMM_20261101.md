# V1 — Auditoria visual e mapeamento PCA → t-SNE + GMM

**Projeto:** `TCC_EXPERIMENTAL_V3`
**Data-base:** 01/11/2026 (America/Sao_Paulo)
**Etapa:** V1 de V7 — revisão estática, **sem iniciar V2**
**Estado:** auditoria documental concluída; checkpoint Git local/remoto depende da execução do comando de salvamento pelo responsável.
**Origem dos arquivos revisados:** repositório `Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026`, branch `feature/tsne-gmm-oof-f10`; códigos de finalização t-SNE das Partes 1–7 foram entregues localmente em ZIPs e não se presume que já estejam na branch remota.

## 1. Escopo, evidência e limites

- Auditoria estática do catálogo de gráficos e dos pontos de integração no código da branch citada, com leitura dos módulos-fonte via GitHub.
- Evidência de execução **informada pelo terminal do usuário**: `174 passed in 35.22s`, `TSNE_VALIDACAO_CONSOLIDADA_EXIT_CODE=0`, após a correção de dois defeitos da própria suíte V7. Esse resultado homologa o conjunto de testes selecionados de finalização/integração/regressão, **não** prova que os 17 tipos de gráficos t-SNE já são gerados no ambiente local.
- Os códigos `src/reducao_tsne.py` e finalização t-SNE no computador podem estar mais atualizados que a branch consultada (o redutor foi corrigido localmente nas etapas anteriores). **Antes de qualquer alteração V2, confirmar o diff local com Git.**
- Não foram lidos Parquet, folds locais, outputs históricos, F10 real, nem renderizados PNGs/HTMLs nesta auditoria. Não houve treino.
- A documentação histórica indica PCA aprovado em 1D/2D/3D, com **12/15/18 PNGs no F10** e cinco HTMLs GMM 3D. Esses números são referência PCA, **não previsão automática** de quantidade t-SNE.

## 2. Inventário: 17 tipos por método (não 17 arquivos por execução)

**Comuns (14):** oito preditivos OOF e seis geométricos/diagnósticos GMM.
**Exclusivos (3 por método):** PCA11 cotovelo ↔ TSNE02 trustworthiness; PCA12 projeções ↔ TSNE13 mapa; Biplot PCA ↔ TSNE14 KL/tempo.
**Cuidado:** `02_mcc_por_limiar` e `TSNE02_trustworthiness_vs_perplexity` usam o mesmo número no catálogo, mas são figuras diferentes, em subpastas diferentes. Manter nomes inequívocos.

### 2.1 OOF preditivo — 8 tipos, compartilhados

| Nº | Chave de configuração | Nome esperado (sem extensão) | Fonte principal |
|---|---|---|---|
| 00 | `painel_score_composto` | `00_painel_score_composto` | `src/visualizacao_oof.py` |
| 01 | `curva_pr` | `01_curva_precision_recall` | mesmo |
| 02 | `curva_mcc` | `02_mcc_por_limiar` | mesmo |
| 03 | `matriz_confusao_05` | `03_matriz_confusao_05` | mesmo |
| 04 | `matriz_confusao_mcc` | `04_matriz_confusao_mcc` | mesmo |
| 05 | `distribuicao_probabilidades` | `05_distribuicao_probabilidades` | mesmo |
| 06 | `comparacao_hiperparametros` | `06_comparacao_hiperparametros` | mesmo |
| 07 | `metricas_por_fold` | `07_metricas_por_fold` | mesmo |

`src/visualizacao_oof.py` define `ARQUIVOS` (linhas 34–43 no snapshot) e `gerar_graficos_oof` (linhas 1566–1623). Os gráficos dependem dos dados preexistentes do OOF (`predicoes.npz`, resultados da grade etc.). A implementação é genericamente compartilhada; há inclusive ramo específico t-SNE nos rótulos da comparação de hiperparâmetros. **Reuso provável, não homologação visual t-SNE.**

### 2.2 GMM — 6 tipos compartilhados conceitualmente

| Identificação | Chave | Fonte principal | Observação |
|---|---|---|---|
| GMM01 | `densidades_marginais` | `src/visualizacao_gmm01.py` e `_3d.py` | Marginais, conjuntas e HTML 3D; complementos atuais presos a PC1/PC2/PC3 |
| GMM04 | `regioes_dominancia` | `src/visualizacao_gmm04.py` e `_3d.py` | Dominância da mistura; HTML 3D contém rótulos PCA |
| GMM07 | `sensibilidade_fronteira` | `src/visualizacao_gmm07.py` e `_3d.py` | Dois cortes e MCC; saída interativa 3D rotulada PCA |
| GMM08 | `mapa_risco_fronteira` | `src/visualizacao_geometrica.py` e `visualizacao_gmm08_3d.py` | 3D só chama renderer específico com colunas PC1–PC3; t-SNE requer revisão |
| 09 | `painel_parametros_gaussianos` | `src/visualizacao_geometrica.py` | Reutiliza médias, covariâncias e riscos no embedding, sem inventar loadings |
| GMM18 | `convergencia_em` | `src/visualizacao_gmm18.py` | Diagnóstico do ajuste GMM; não é convergência do t-SNE |

### 2.3 Específicos de método — 3 em cada lado

| PCA | t-SNE | Fonte t-SNE |
|---|---|---|
| `pca_cotovelo` / gráfico 11 | `tsne_trustworthiness` / TSNE02 | `src/visualizacao_tsne_pdf.py` |
| `pca_projecoes` / gráfico 12 | `tsne_mapa` / TSNE13 | `src/visualizacao_geometrica.py` + `visualizacao_tsne_pdf.py` (3D) |
| `pca_biplot` / sem nº geral único | `tsne_diagnostico` / TSNE14 | `src/visualizacao_tsne_pdf.py` |

A substituição é **editorial, não equivalência estatística**: o t-SNE não tem variância explicada nem coeficientes/loadings de PCA. TSNE14 exibe KL final e tempo real; o código declara o histórico por iteração indisponível e não cria curva sintética.

## 3. Mapa de etapas e proveniência

| Fase | Fonte de dados | Destino atual | Situação para t-SNE |
|---|---|---|---|
| `07.01` OOF preditivo | Scores OOF F1–F9 persistidos; vencedor e grade | `vencedor/visualizacao/oof/` | Fonte compartilhada existente; falta validar cada figura com dados sintéticos t-SNE |
| `07.02` geometria/diagnósticos | **Reajuste do vencedor sobre F1–F9**; não lê F10 | `vencedor/visualizacao/geometria_desenvolvimento/` | Ramo t-SNE existe; complementos 2D/3D parcialmente específicos PCA; **não é renderização sem fit** |
| `07.03` galeria OOF | Arquivos 07.01 e 07.02 | `vencedor/visualizacao/galeria/` | Filtragem, verificação e inclusão de extras 2D/3D têm ramos exclusivos PCA |
| F10 final | Pacote final F1–F9, scores OOS e `τ*` OOF congelado | `vencedor/teste_final/` | Avaliação em memória t-SNE aprovada em testes sintéticos; **executor e galeria visual F10 t-SNE equivalentes ao PCA ainda não demonstrados** |

**Atenção:** no YAML, alguns tipos de geometria têm `gerar_em: teste_final`, mas o `07.02` declara e verifica que sua origem é **desenvolvimento F1–F9**, não observações F10. Não mudar o significado científico ou registrar esses gráficos como métricas F10. Há ainda flags de visualização e automação PCA no YAML oficial; a auditoria não as modificou.

## 4. Achados e bloqueios antes da geração completa t-SNE

### P0 — bloquear execução automática indevida

1. **Despacho PCA na CLI OOF.** `src/orquestracao_oof.py` (aprox. linhas 540–560, 605–625 no snapshot) usa `execucao.finalizacao_pca1d_automatica` e despacha para finalizadores PCA conforme `pca.n_components`. Como a chave está `true` no YAML de referência, **não executar** o fluxo OOF t-SNE oficial antes de criar/validar proteção explícita por método e autorização. As Partes 1–7 aprovaram o orquestrador t-SNE isolado; não comprovam a integração automática no CLI oficial.
2. **Proteção da origem F10.** Não permitir fitting, seleção, reajuste de `τ*`, `trustworthiness`, ou construção do embedding a partir do F10. Em F10 só `transform`/inferência OOS com pacote final congelado, e métricas exploratórias calculadas depois. O F10 já foi acessado em PCA e não constitui teste virgem para seleção t-SNE.

### P1 — impedir falhas/rotulagens PCA ao implementar os gráficos t-SNE

3. **GMM01 2D:** `visualizacao_gmm01.py`, `calcular_conjunta_2d` contém mensagem "exige PCA 2D" e a extensão de superfície usa nomes `PC1`/`PC2` e sufixo `PCA2D`. Precisa nomes/validação genéricos e teste matemático da densidade conjunta sobre o modelo TSNE2D.
4. **GMM01 3D:** `visualizacao_gmm01_3d.py` tem `NOMES = ('PC1','PC2','PC3')`, rejeita outras colunas e gera PNGs/HTMLs rotulados PCA. Adaptar sem mudar a saída PCA existente.
5. **GMM04/GMM07/GMM08 3D:** `visualizacao_gmm04_3d.py`, `visualizacao_gmm07_3d.py`, `visualizacao_gmm08_3d.py` contêm eixos, textos e/ou validações PCA. No GMM08, `visualizacao_geometrica._fronteira()` só despacha ao renderer 3D especial se as colunas forem `['PC1','PC2','PC3']`; a versão 3D t-SNE requer despacho ou renderer próprio com `TSNE1`–`TSNE3`.
6. **Galeria 07.03:** `src/visualizacao_integrada.py` possui verificações de extras PCA 1D/2D/3D e inclusão condicional no HTML, sem paralelos para t-SNE, além do campo de manifesto `pca_dim_vencedor` mesmo para outros métodos. Criar contratos t-SNE específicos, manter o caminho PCA idêntico e não exigir números idênticos de arquivos por conveniência.
7. **Bateria visual:** os 174 testes consolidados foram de finalização/roteamento/regressão; **ainda falta ensaio de geração de cada gráfico** com miniobjetos sintéticos 1D/2D/3D (PNG/CSV/HTML, sem rede e sem F10 real).

### P2 — convenções científicas e nomenclatura

8. Eixos `TSNE1`/`TSNE2`/`TSNE3`; 1D histograma; 2D conjunta FULL; 3D cortes e representação interativa quando cabíveis. Interpretar gaussianas/áreas no **espaço do embedding**, nunca no espaço original das features.
9. Limiares `τ=0,50` e `τ*` OOF; onde o gráfico depende de corte, dois painéis, mesmos scores/malha/escala e MCC do conjunto corretamente identificado (OOF ou F10). Sete casas decimais nas quantidades requeridas; gaussianas com cobertura de 95%; nenhuma mudança de seleção ou scores por causa de gráfico.
10. **TSNE02** recalcula vários embeddings em amostra de desenvolvimento para perplexidades distintas (até 360 observações no YAML); não confundir esse diagnóstico auxiliar com uma avaliação de desempenho nem com resultado OOF comparável. **TSNE14** não inventa histórico de KL indisponível.
11. Equalizar **17 tipos conceituais** por método, mas não assumir paridade de PNGs/HTMLs com PCA: biplot PCA e extras dimensionais geram quantidades diferentes. O inventário final de arquivos deve ser conferido por dimensão após a implementação.

## 5. Arquivos consultados e destinos de alteração futuros

**Compartilhados (proteger regressão PCA):**

- `src/visualizacao_oof.py` — 8 tipos OOF.
- `src/visualizacao_geometrica.py` — orquestra 07.02, geometria, TSNE13 1D/2D.
- `src/visualizacao_integrada.py` — verificação/copias/galeria 07.03.
- `src/visualizacao_gmm01.py`, `visualizacao_gmm01_3d.py` — densidades.
- `src/visualizacao_gmm04.py`, `visualizacao_gmm04_3d.py` — dominância.
- `src/visualizacao_gmm07.py`, `visualizacao_gmm07_3d.py` — sensibilidade.
- `src/visualizacao_gmm08_3d.py` — mapa de risco 3D.
- `src/visualizacao_gmm18.py` — EM.
- `src/orquestracao_oof.py` — despacho de execução, **não alterar sem teste de regressão**.

**t-SNE exclusivos (já existem, melhorar/validar sem recriar):**

- `src/visualizacao_tsne_pdf.py` — TSNE02, TSNE13 3D, TSNE14.
- `src/reducao_tsne.py` — correções de backend e centragem feitas localmente; respeitar versão homologada nos testes.
- Módulos locais `src/treinamento_final_tsne.py`, `avaliacao_final_tsne.py`, `roteamento_final_tsne.py`, `finalizacao_controlada_tsne.py`, `ponte_oof_final_tsne.py`, `orquestracao_final_tsne.py` — criados nas Partes 1–6 e aprovados em testes; verificar versão local e rastreamento Git antes de alterações V2.

**Configuração de referência:** `config_TCC_EXPERIMENTAL_V3_atualizado.yaml` — não foi editado.

## 6. Plano de implementação visual posterior (fora da V1)

- **V2 — t-SNE 1D OOF:** revisar uma figura de cada vez, começando por `00_painel_score_composto`, usando somente dados sintéticos e uma pasta de validação isolada. Reaproveitar as 8 preditivas sem duplicar fonte; verificar TSNE02/13/14 e GMM 1D. Não executar 07.02 real.
- **V3 — t-SNE 2D OOF:** ampliar GMM01 para densidade conjunta TSNE1×TSNE2, GMM04/07/08, TSNE13 e a galeria. Testar normalização/densidade e consistência com FULL.
- **V4 — t-SNE 3D OOF:** adaptar extras e HTMLs 3D para TSNE1/2/3, validar eixos, cobertura 95%, dois cortes, MCC, consistência de nomes e inclusão na galeria.
- **V5 — integração OOF:** contratos de arquivos por dimensão, sem treino real, regressões PCA + testes de percurso; só então avaliar autorização de um piloto OOF real.
- **V6 — avaliação/visualização F10:** planejar e implementar uso exclusivo do pacote final F1–F9 e coordenadas OOS, com `τ*` congelado e fonte F10 declarada exploratória. Não acessar F10 real na implementação.
- **V7 — homologação final:** testar saídas F10 sintéticas 1D/2D/3D, controle de hashes/proveniência, galeria, regressões PCA; avaliação real em F10 será decisão separada.

## 7. Critérios de aceite da V1 e checkpoint

- [x] Identificados 8 preditivos, 6 GMM e 3 exclusivos por método (**17 tipos**).
- [x] Mapeadas fontes e saídas dos passos 07.01, 07.02 e 07.03.
- [x] Registrada a diferença entre visualização de desenvolvimento e avaliação F10.
- [x] Documentados os pontos PCA-específicos que exigirão adaptação t-SNE.
- [x] Documentados os bloqueios de execução automática e vazamento F10.
- [x] Definidos critérios visuais/científicos e sequência V2–V7.
- [x] Criado pacote documental V1 nesta conversa, sem modificar módulos/YAML/outputs.
- [ ] **Checkpoint Git ainda depende de executar `git add` apenas nos dois arquivos `docs/V1_*` abaixo, `git commit` e opcionalmente `git push`, no computador do responsável.**

Checkpoint recomendado: `docs/V1_AUDITORIA_VISUAL_TSNE_GMM_20261101.md` + `docs/V1_MATRIZ_GRAFICOS_TSNE_GMM_20261101.csv`. Registrar o SHA do commit e `git status -sb` após salvar. **Não usar `git add -A`, `git clean`, `git reset --hard`, não versionar `outputs/` nem datasets.**

## 8. Fontes principais (branch consultada)

- https://github.com/Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026/blob/feature/tsne-gmm-oof-f10/src/visualizacao_oof.py
- https://github.com/Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026/blob/feature/tsne-gmm-oof-f10/src/visualizacao_geometrica.py
- https://github.com/Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026/blob/feature/tsne-gmm-oof-f10/src/visualizacao_integrada.py
- https://github.com/Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026/blob/feature/tsne-gmm-oof-f10/src/visualizacao_tsne_pdf.py
- https://github.com/Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026/blob/feature/tsne-gmm-oof-f10/src/visualizacao_gmm01.py
- https://github.com/Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026/blob/feature/tsne-gmm-oof-f10/src/visualizacao_gmm01_3d.py
- https://github.com/Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026/blob/feature/tsne-gmm-oof-f10/src/visualizacao_gmm04_3d.py
- https://github.com/Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026/blob/feature/tsne-gmm-oof-f10/src/visualizacao_gmm07_3d.py
- https://github.com/Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026/blob/feature/tsne-gmm-oof-f10/src/visualizacao_gmm08_3d.py
- https://github.com/Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026/blob/feature/tsne-gmm-oof-f10/src/orquestracao_oof.py
- https://github.com/Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026/blob/feature/tsne-gmm-oof-f10/config_TCC_EXPERIMENTAL_V3_atualizado.yaml
