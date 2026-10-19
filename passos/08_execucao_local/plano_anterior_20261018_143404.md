# ETAPA 08 - EXECUCAO LOCAL E AUDITORIA COMPLETA

Projeto: TCC_EXPERIMENTAL_V3

Data de planejamento: 2026-10-18 00:53:53

Status: PLANEJADA

Total: 22 subetapas.

## Objetivo geral

Validar integralmente o pipeline em ambiente local,
utilizando os dados completos de desenvolvimento
e um numero reduzido de hiperparametros.

Executar separadamente PCA e t-SNE.

Auditar todas as saidas, metricas, preditoes,
graficos e diagnosticos de cada metodo.

Identificar melhorias, implementar correcoes,
executar novamente quando necessario e
somente depois preparar o cluster universitario.

## Regras fundamentais

- F1 a F9: desenvolvimento completo.
- F10: reservado para o teste final independente.
- Base de desenvolvimento: 1667155 registros.
- Dados nunca reduzidos apenas para economizar
  tempo na execucao local considerada completa.
- Comecar com um candidato por metodo.
- Grades pequenas apenas se forem necessarias.
- PCA e t-SNE executados separadamente.
- Nenhum ajuste utilizando dados de validacao.
- Limiar MCC selecionado somente no OOF.
- Graficos preditivos usam predicoes OOF.
- Graficos geometricos usam desenvolvimento.
- Resultados exploratorios nao sao finais.
- Nenhuma utilizacao do F10 nesta etapa.

# BLOCO I - PREPARACAO

## 08.01 - Auditoria de recursos computacionais

Objetivo:
Verificar RAM, armazenamento, CPU, ambiente Python,
integridade do Parquet e folds persistidos.

Verificacoes:
- RAM disponivel e estimativa de pico.
- Espaco livre e tamanho previsto dos resultados.
- Versoes das bibliotecas.
- Integridade de F1 a F10.
- Cache, retomada e protecao de resultados.
- Paralelismo adequado ao notebook.

Conclusao:
Ambiente local preparado para os testes reais.

## 08.02 - Configuracoes minimas oficiais locais

Objetivo:
Preparar configuracoes independentes para PCA
e t-SNE, sem modificar desnecessariamente
o YAML oficial.

Parametros iniciais:
- PCA: n_components=2.
- t-SNE: n_components=2.
- GMM: K=2.
- Amostragem: NORMAL.
- Um candidato por metodo.
- Nove folds de desenvolvimento.
- F10 excluido.

Registrar configuracoes efetivamente utilizadas.

# BLOCO II - PCA

## 08.03 - Execucao PCA na base completa

Executar os nove folds OOF de desenvolvimento.

Verificar:
- Pre-processamento por fold.
- PCA ajustado somente no treino.
- GMM unico sem rotulos.
- Risco historico usando treino original.
- Predicoes nas validacoes.
- Tempo e consumo de recursos.

## 08.04 - Auditoria de arquivos PCA

Inspecionar individualmente:
- Pasta principal da execucao.
- YAML salvo.
- Manifestos.
- Arquivos de cache.
- Predicoes OOF.
- Resultados da grade.
- Metricas.
- Melhor configuracao.
- Logs e relatorios.
- Pastas de graficos.

Verificar ausencia de arquivos incompletos,
duplicados, inconsistentes ou inesperados.

## 08.05 - Auditoria estatistica PCA

Verificar:
- Quantidade de preditos por fold.
- Cobertura de todos os registros F1-F9.
- Ausencia de duplicacoes.
- Distribuicao das probabilidades.
- AP e prevalencia.
- MCC e limiar selecionado.
- LogLoss.
- Score composto.
- Valores extremos e probabilidades invalidas.
- Correspondencia entre CSV, JSON e logs.

## 08.06 - Graficos preditivos PCA

Avaliar individualmente:
01 - Precision-Recall.
02 - MCC por limiar.
03 - Matriz de confusao com 0.5.
04 - Matriz de confusao com MCC otimo.
05 - Distribuicao dos scores.
06 - Comparacao de hiperparametros.

Inspecionar valores, escalas, legendas,
cores, resolucao, titulos e interpretacao.

## 08.07 - Diagnosticos das gaussianas PCA

Avaliar individualmente:
07 - Risco historico por gaussiana.
08 - Pesos e responsabilidades.
09 - Medias dos componentes.
10 - Covariancias.

Conferir massas, riscos, responsabilidades,
normalizacao e consistencia numerica.

Verificar se os graficos representam
corretamente os parametros ajustados.

## 08.08 - Geometria e fronteira PCA

Avaliar individualmente:
11 - Variancia explicada pelo PCA.
12 - Projecoes dos dados.
15 - Mapa de risco e fronteira.

Verificar:
- Coordenadas e dimensoes corretas.
- Nuvens gaussianas e elipses.
- Classes legitimas e fraudulentas.
- Campo de risco.
- Contorno do limiar MCC.
- Coerencia geometrica.
- Tratamento de PCA 1D, 2D e 3D.

O mapa complementar 2D deve ser identificado
como adicional quando nao representar o vencedor.

## 08.09 - Melhorias e correcoes PCA

Revisar:
- Escalas lineares e logaritmicas.
- Transparencia e sobreposicao de pontos.
- Cores e contraste.
- Tamanho de fontes.
- Posicao e conteudo das legendas.
- Organizacao das figuras.
- Informacoes estatisticas exibidas.
- Clareza na interpretacao dos riscos.
- Exportacao PNG, CSV e manifestos.

Modificar codigo somente quando necessario.

## 08.10 - Revalidacao completa PCA

Apos ajustes:
- Executar testes automatizados.
- Gerar novamente os graficos afetados.
- Conferir resultados numericos.
- Validar persistencia e integridade.
- Documentar correcoes.
- Aprovar formalmente PCA.

# BLOCO III - t-SNE

## 08.11 - Viabilidade computacional t-SNE

Verificar o custo de ajustar t-SNE em todos
os folds do desenvolvimento completo.

Avaliar:
- RAM.
- Tempo de processamento.
- Armazenamento.
- Implementacao openTSNE.
- Transformacao de novos registros.
- Estabilidade das coordenadas.

Executar ensaio tecnico progressivo se necessario.

Nao iniciar a execucao integral se os recursos
forem insuficientes.

## 08.12 - Preparacao da configuracao t-SNE

Parametros iniciais:
- Metodo: TSNE.
- n_components=2.
- perplexity=10.
- initialization=random.
- Um GMM com K=2.
- Amostragem NORMAL.

Verificar compatibilidade dos hiperparametros,
sementes e divisao treino-validacao.

## 08.13 - Execucao t-SNE no desenvolvimento

Executar F1-F9 completos quando for viavel.

Conferir:
- t-SNE ajustado somente no treino.
- Transformacao dos dados de validacao.
- GMM unico sem rotulos.
- Risco por responsabilidades.
- Predicoes OOF.
- Tempo e consumo de recursos.

Registrar qualquer limite computacional real.

## 08.14 - Auditoria de arquivos t-SNE

Inspecionar individualmente:
- Configuracao.
- Manifestos.
- Cache.
- Embeddings e seus metadados.
- Predicoes.
- Metricas.
- Ranking.
- Melhor configuracao.
- Logs.
- Graficos.
- Relatorios.

Confirmar que nao houve mistura de
coordenadas provenientes de folds distintos.

## 08.15 - Metricas e graficos OOF t-SNE

Auditar:
- AP.
- MCC.
- Limiar otimo.
- LogLoss.
- Score composto.
- Distribuicao dos scores.

Avaliar graficos 01 a 06 individualmente.

Conferir valores, curvas, escalas e legendas.

## 08.16 - Gaussianas no espaco t-SNE

Avaliar graficos:
07 - Risco historico por gaussiana.
08 - Pesos e responsabilidades.
09 - Medias dos componentes.
10 - Covariancias.

Verificar se gaussianas e responsabilidades
correspondem ao espaco t-SNE efetivamente
utilizado no ajuste do GMM.

## 08.17 - Geometria e fronteira t-SNE

Avaliar:
13 - Projecao t-SNE.
14 - Diagnostico de otimizacao t-SNE.
15 - Mapa de risco e fronteira.

Verificar:
- Posicao e separacao dos grupos.
- Sobreposicao de classes.
- Perda de informacao visual.
- Diagnosticos de convergencia disponiveis.
- Elipses gaussianas.
- Regioes de risco.
- Fronteira MCC.
- Legendas e cores.

Evitar interpretar distancias globais do
t-SNE como distancias estatisticas originais.

## 08.18 - Melhorias e revalidacao t-SNE

Ajustar se necessario:
- Escalas.
- Cores.
- Transparencia.
- Numero de pontos visualizados.
- Legendas.
- Representacao das responsabilidades.
- Desenho das gaussianas.
- Fronteira de decisao.
- Performance computacional.

Reexecutar testes e visualizacoes afetadas.

Aprovar formalmente t-SNE quando validado.

# BLOCO IV - AUDITORIA FINAL

## 08.19 - Comparacao PCA versus t-SNE

Comparar:
- AP.
- MCC.
- LogLoss.
- Score composto.
- Tempo de execucao.
- Memoria.
- Persistencia.
- Qualidade dos diagnosticos.
- Facilidade de interpretacao.

As comparacoes so serao conclusivas quando
ambos usarem os mesmos registros de
desenvolvimento e protocolo de validacao.

## 08.20 - Cache, retomada e reproducibilidade

Verificar para ambos os metodos:
- Interrupcao controlada.
- Retomada por fold.
- Reutilizacao de cache compativel.
- Protecao contra sobrescrita.
- Identificacao das configuracoes.
- Determinismo quando aplicavel.
- Consistencia de preditos e metricas.
- Integridade dos arquivos persistidos.

## 08.21 - Auditoria final de todos os outputs

Examinar pasta por pasta e arquivo por arquivo.

Verificar:
- Nomes e organizacao.
- Integridade dos arquivos.
- Coerencia CSV/JSON/NPZ.
- Todos os graficos habilitados aplicaveis.
- Manifestos de geracao.
- Resolucao e legibilidade.
- Dados de origem.
- Metadados e rastreabilidade.
- Ausencia de uso de F10.

Registrar melhorias futuras separadamente
das correcoes obrigatorias.

## 08.22 - Encerramento e aprovacao local

Executar a suite final de testes.

Documentar:
- Recursos utilizados.
- Configuracoes.
- Tempos.
- Metricas.
- Graficos aprovados.
- Arquivos auditados.
- Correcoes efetuadas.
- Limitacoes observadas.
- Pendencias remanescentes.
- Recomendacoes para o cluster.

Atualizar:
- Registros de cada subetapa.
- Resumo da Etapa 08.
- CONTEXTO_GERAL.md.
- requirements.txt, quando necessario.
- Git, com checkpoint final.

Status final possivel:
- VALIDADO PARA CLUSTER.
- VALIDADO COM RESSALVAS.
- BLOQUEADO POR LIMITACAO OU ERRO.

A Etapa 09 somente sera iniciada apos a
decisao formal sobre a validacao local.

# PROXIMA TAREFA

08.01 - Auditoria de recursos computacionais.

ETAPA 08 - PLANEJADA.
