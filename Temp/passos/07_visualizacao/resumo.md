# ETAPA 07 — VISUALIZAÇÃO E DIAGNÓSTICO DO MODELO

**Projeto:** TCC_EXPERIMENTAL_V3  
**Situação:** CONCLUÍDA — implementação e execução funcional validadas no notebook  
**Subetapas:** 07.01, 07.02 e 07.03  
**Escopo validado:** ensaio exploratório OOF com 10% dos dados de desenvolvimento  
**Observação:** os resultados científicos definitivos dependem da execução integral no cluster.

## 1. Objetivo

Disponibilizar visualizações reprodutíveis do classificador semissupervisionado de
GMM único e risco histórico, respeitando a seleção do vencedor pelo score
composto calculado no OOF. Os gráficos preditivos, geométricos e complementares
devem manter sua origem identificada, sem vazamento do conjunto F10.

## 2. Subetapas concluídas

### 07.01 — Visualização preditiva OOF

**Módulo:** `src/visualizacao_oof.py`  
**Teste:** `tests/test_visualizacao_oof.py`

Leitura das predições e métricas OOF previamente persistidas, sem repetir
treinamentos nem recalcular o limiar selecionado. Foram implementados:

1. Curva Precision–Recall e AP;
2. MCC em função do limiar, com indicação do ótimo OOF;
3. Matriz de confusão com limiar 0,5;
4. Matriz de confusão com limiar ótimo MCC OOF;
5. Distribuição dos scores por classe;
6. Comparação de hiperparâmetros **entre todos os candidatos**.

Os gráficos individuais representam apenas o vencedor. O gráfico 06 é a
exceção necessária para documentar a escolha entre candidatos.

### 07.02 — Diagnósticos geométricos do vencedor

**Módulo:** `src/visualizacao_geometrica.py`  
**Teste:** `tests/test_visualizacao_geometrica.py`

Utiliza a configuração vencedora e realiza um ajuste diagnóstico no
desenvolvimento, sem usar F10 e sem modificar os scores OOF. Implementa os tipos:

7. Risco histórico por gaussiana;
8. Pesos e responsabilidades dos componentes;
9. Médias dos componentes;
10. Covariâncias dos componentes;
11. Diagnóstico de variância do PCA;
12. Projeções do PCA;
13. Projeção t-SNE — condicional ao método;
14. Diagnóstico de otimização t-SNE — condicional à disponibilidade;
15. Mapa de risco e fronteira de decisão, condicionado à dimensão.

Apenas gráficos compatíveis com o método/dimensionalidade são gerados.
A pasta `teste_final/graficos_gerados` é o destino histórico estabelecido
no YAML para os diagnósticos geométricos, mas **nesta etapa contém
exclusivamente desenvolvimento, não resultados de F10**.

### 07.03 — Integração, galeria e gráfico complementar 2D

**Módulo:** `src/visualizacao_integrada.py`  
**Teste:** `tests/test_visualizacao_integrada.py`

Integra os gráficos gerados em `visualizacao_consolidada/index.html`, com
separação por origem OOF, geometria do desenvolvimento e complemento 2D,
além de manifestos de rastreabilidade.

Foi acrescentado um gráfico **diagnóstico complementar** com nuvens/elipses
das gaussianas, campo de risco e contorno do limiar, quando o vencedor é PCA
unidimensional e há candidato PCA 2D avaliado no OOF.

Este gráfico adicional **não é o gráfico oficial do vencedor**, não substitui
a seleção OOF e utiliza a configuração e o limiar MCC OOF próprios do
candidato PCA 2D. Não se deve descrever a sua fronteira como projeção
exata da fronteira do vencedor unidimensional.

## 3. Ensaio utilizado na validação

- Conjunto de desenvolvimento: F1–F9, com F10 separado.
- Amostragem exploratória: aproximadamente 10% do desenvolvimento,
  totalizando **166.716 transações** (18.524 por fold).
- Grade: PCA em `[1, 2, 3]` e GMM com K em `[2, 4]`,
  modo NORMAL; **6 candidatos × 9 folds = 54 ajustes OOF**.
- Duração informada pelo usuário: **132,57 s (2,21 min)**.
- Vencedor exploratório:
  - ID: `0d74604010afa3`
  - PCA: **1 componente**; GMM: **4 gaussianas**
  - AP: **0,011126**; MCC: **0,047470**;
    LogLoss: **0,031829**; score composto: **0,501178**;
    limiar MCC OOF: **0,010396**.
- Melhor candidato PCA 2D da mesma grade (complementar):
  - ID: `535cfea05d0b69`
  - PCA: **2 componentes**; GMM: **2 gaussianas**
  - Score composto: **0,499287**;
    limiar MCC OOF: **0,006842**.

**Esses números são exploratórios**: não representam estimativas
definitivas do desempenho nem autorizam avaliação do F10.

## 4. Catálogo de figuras e origem

O YAML contém **15 tipos de gráficos**, mas não exige 15 imagens em toda
execução. Para o vencedor PCA 1D do ensaio:

- Tipos 01–06: OOF (6 tipos aplicáveis);
- Tipos 07–12 e 15: geometria do desenvolvimento (7 tipos aplicáveis);
- Tipos 13–14: específicos de t-SNE, portanto não aplicáveis;
- Mapa 2D complementar: imagem adicional, fora dos 15 tipos do YAML.

Assim, **14 imagens PNG são esperadas** com todos esses tipos habilitados:
13 tipos aplicáveis do catálogo, mais uma imagem suplementar. A contagem
concreta deve ser conferida nos arquivos gerados, pois depende das opções
visuais e das condições geométricas.

## 5. Arquivos e diretórios relevantes

- `src/visualizacao_oof.py`
- `src/visualizacao_geometrica.py`
- `src/visualizacao_integrada.py`
- `tests/test_visualizacao_oof.py`
- `tests/test_visualizacao_geometrica.py`
- `tests/test_visualizacao_integrada.py`
- `outputs/ensaio_pca_gmm_20261017_195252/oof/graficos_gerados/`
- `outputs/ensaio_pca_gmm_20261017_195252/teste_final/graficos_gerados/`
- `outputs/ensaio_pca_gmm_20261017_195252/visualizacao_consolidada/index.html`

`outputs/` é diretório operacional ignorado pelo Git e não deve ser
incluído nos commits por engano.

## 6. Evidências e validação

- Os módulos foram testados sinteticamente durante o desenvolvimento.
- A suíte integrada reportou **305 testes aprovados no ambiente de
  desenvolvimento** na preparação do 07.03.
- O usuário confirmou o funcionamento do 07.01, do 07.02 e do 07.03
  no notebook; não há logs finais completos anexados para auditoria
  independente de todas as figuras.
- O planejamento 07.02 identificou o vencedor correto e indicou
  **zero ajustes no modo `--planejar`**.
- O YAML oficial e a seleção OOF não foram alterados intencionalmente
  pelos módulos de visualização.
- **F10 permanece reservado, sem avaliação nesta etapa.**

## 7. Critérios de encerramento

- [x] Visualizações preditivas do vencedor a partir das previsões OOF.
- [x] Comparação gráfica entre os candidatos.
- [x] Diagnósticos geométricos do modelo vencedor.
- [x] Respeito às dimensões reais do vencedor (PCA 1D).
- [x] Mapa complementar PCA 2D explicitamente não oficial.
- [x] Galeria consolidada e rastreabilidade de origem.
- [x] Funcionamento informado pelo usuário no ensaio local.
- [x] F10 isolado da geração e seleção.
- [ ] Execução científica integral em cluster — **fora do escopo da Etapa 07**.
- [ ] Avaliação independente no F10 — **reservada à Etapa 08**.

## 8. Próximos passos

1. Salvar esta documentação e atualizar `passos/CONTEXTO_GERAL.md`
   **sem substituir seu histórico existente**.
2. Registrar a documentação no Git; conferir mudanças pendentes em
   `requirements.txt` e o diretório não rastreado `Temp/`.
3. Preparar ambiente Linux, dependências e submissão SLURM no cluster.
4. Executar busca de hiperparâmetros com desenvolvimento completo;
   selecionar o vencedor e o limiar apenas com o OOF oficial.
5. No Passo 08, congelar a configuração, ajustar no desenvolvimento
   integral e avaliar uma única vez o F10 independente.

---

**Encerramento:** Etapa 07 concluída no escopo de implementação e
validação funcional local. Resultados amostrais não equivalem à
validação científica final do modelo.
