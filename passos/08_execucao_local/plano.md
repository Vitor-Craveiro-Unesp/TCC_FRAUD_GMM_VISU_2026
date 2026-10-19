# ETAPA 08 - VALIDACAO LOCAL PROGRESSIVA PCA E t-SNE

Data: 2026-10-18 14:34:04

Status: PLANEJADA - VERSAO REVISADA

Total: 30 subetapas.

## Objetivo

Validar PCA e t-SNE inicialmente com 10% dos dados
de desenvolvimento, auditando individualmente
arquivos, metricas, graficos e resultados.

Apos aprovar e corrigir os dois experimentos
amostrais, executar novamente com 100% dos
dados de desenvolvimento e repetir a auditoria.

Somente depois preparar o cluster universitario.

## Regras gerais

- F1 a F9 utilizados para desenvolvimento.
- F10 reservado para avaliacao independente futura.
- 10% do desenvolvimento na primeira fase.
- 100% do desenvolvimento na segunda fase.
- PCA e t-SNE executados separadamente.
- Mesma amostra e mesmos folds para comparacao justa.
- Amostragem estratificada e semente fixa.
- Comecar com um candidato por metodo.
- Aumentar hiperparametros somente se necessario.
- GMM unico nao supervisionado.
- Risco calculado com rotulos do treino original.
- Pre-processamento e reducao ajustados no treino.
- Limiar MCC obtido exclusivamente via OOF.
- Grafico individual associado ao modelo vencedor.
- Graficos comparativos podem envolver outros modelos.
- Nenhum resultado de F10 nesta etapa.
- Nenhuma execucao deve sobrescrever resultados validos.

# BLOCO I - PREPARACAO DOS ENSAIOS DE 10%

## 08.01 - Auditoria inicial

Conferir dataset, folds, ambiente Python,
dependencias, RAM, CPU, disco e Git.

Verificar a separacao de F10.

## 08.02 - Definicao da amostra comum de 10%

Selecionar aproximadamente 10% de F1-F9.

Usar amostragem estratificada e deterministica,
preservando os nove folds.

Garantir que PCA e t-SNE utilizem exatamente
os mesmos registros e identificadores.

Salvar manifesto de indices e prevalencia.

Nao utilizar F10.

## 08.03 - Configuracoes dos experimentos

Preparar dois experimentos independentes:

PCA:
- Metodo PCA.
- Inicialmente 2 componentes.
- GMM K=2.
- Modo NORMAL.
- Nove folds OOF.

t-SNE:
- Metodo TSNE.
- Inicialmente 2 componentes.
- Perplexity=10.
- Initialization=random.
- GMM K=2.
- Modo NORMAL.
- Nove folds OOF.

Usar poucos hiperparametros.

Registrar configuracoes e seeds.

# BLOCO II - PCA COM 10%

## 08.04 - Execucao PCA amostral

Executar o pipeline PCA com 10%.

Registrar tempo, memoria e logs.

Confirmar nove folds finalizados.

## 08.05 - Auditoria dos arquivos PCA

Examinar individualmente:

- Configuracao salva.
- Manifestos.
- Logs.
- Resultados da grade.
- Melhor configuracao.
- Predicoes OOF.
- Metricas.
- Limiar MCC.
- Cache.
- Relatorios.
- Pastas de graficos.
- Galeria HTML.

Conferir arquivos ausentes, duplicados,
corrompidos ou inconsistentes.

## 08.06 - Auditoria das predicoes PCA

Conferir quantidade de preditos.

Verificar correspondencia com cada fold.

Verificar probabilidades entre 0 e 1.

Recalcular e conferir AP, MCC, LogLoss,
prevalencia e score composto.

Confirmar que F10 nao foi utilizado.

## 08.07 - Graficos preditivos PCA

Inspecionar um por um:

01 - Precision-Recall.
02 - MCC por limiar.
03 - Matriz de confusao com limiar 0.5.
04 - Matriz de confusao com MCC otimo.
05 - Distribuicao dos scores.
06 - Comparacao de hiperparametros.

Conferir valores, escalas, cores,
legendas, titulos e interpretacao.

## 08.08 - Graficos gaussianos PCA

Inspecionar um por um:

07 - Risco historico por gaussiana.
08 - Pesos e responsabilidades.
09 - Medias das gaussianas.
10 - Covariancias das gaussianas.

Conferir parametros numericos e
correspondencia com o modelo ajustado.

## 08.09 - Geometria e mapas PCA

Inspecionar:

11 - Variancia explicada pelo PCA.
12 - Projecoes PCA.
15 - Mapa de risco e fronteira.

Examinar tambem o mapa complementar 2D,
quando aplicavel e disponivel.

Avaliar elipses, nuvens, classes,
campo de risco e corte MCC.

## 08.10 - Melhorias e aprovacao PCA 10%

Identificar erros e melhorias.

Modificar graficos ou codigo quando necessario.

Reexecutar somente o que for afetado.

Registrar correcoes e aprovar o ensaio PCA.

# BLOCO III - t-SNE COM 10%

## 08.11 - Preflight t-SNE

Avaliar memoria, tempo e viabilidade
do ajuste openTSNE na amostra de 10%.

Verificar transformacao da validacao.

Nao iniciar carga excessiva sem estimativa.

## 08.12 - Execucao t-SNE amostral

Executar nove folds com a mesma amostra de
10% utilizada no experimento PCA.

Registrar tempo, memoria e logs.

## 08.13 - Auditoria dos arquivos t-SNE

Inspecionar individualmente configuracao,
manifestos, logs, embeddings, predicoes OOF,
metricas, ranking, limiar, cache,
relatorios, graficos e galeria.

Verificar integridade e rastreabilidade.

## 08.14 - Auditoria estatistica t-SNE

Conferir AP, MCC, LogLoss, score composto,
limiar MCC e distribuicao dos scores.

Verificar cobertura e isolamento dos folds.

Conferir que os embeddings da validacao
foram transformados sem reajustar no fold.

## 08.15 - Graficos preditivos t-SNE

Inspecionar individualmente os tipos 01 a 06.

Conferir curvas, limiares, matrizes,
distribuicao e comparacao de candidatos.

## 08.16 - Gaussianas no espaco t-SNE

Inspecionar individualmente os tipos 07 a 10.

Avaliar risco, responsabilidades,
pesos, medias e covariancias.

## 08.17 - Geometria, melhorias e aprovacao t-SNE

Inspecionar:

13 - Projecao t-SNE.
14 - Diagnosticos de otimizacao, se disponiveis.
15 - Mapa de risco e fronteira.

Analisar nuvens gaussianas, elipses,
sobreposicao de classes e campo de risco.

Corrigir problemas e revalidar.

# BLOCO IV - COMPARACAO E APROVACAO DOS 10%

## 08.18 - Conferencia da comparabilidade

Verificar que PCA e t-SNE usaram:
- Mesmos identificadores.
- Mesmos registros.
- Mesmos nove folds.
- Mesma variavel alvo.
- Mesmo protocolo de metricas.

Registrar diferencas metodologicas.

## 08.19 - Comparacao quantitativa

Criar tabela comparativa de:
- AP.
- MCC.
- LogLoss.
- Score composto.
- Limiar selecionado.
- Tempo de execucao.
- Consumo de memoria.
- Dimensoes.
- Numero de gaussianas.

Nao confundir ensaio com selecao final.

## 08.20 - Graficos comparativos PCA x t-SNE

Gerar e avaliar:

- Precision-Recall sobrepostas.
- MCC por limiar comparado.
- Distribuicoes dos scores.
- Barras de AP e MCC.
- Comparacao de LogLoss.
- Comparacao de score composto.
- Tempo e custo computacional.
- Matrizes de confusao lado a lado.
- Sintese visual das configuracoes.

Os graficos comparativos sao adicionais
ao catalogo de 15 tipos do YAML.

## 08.21 - Revisao visual e interpretativa

Revisar todos os graficos dos dois metodos.

Avaliar:
- Titulo.
- Eixos e unidades.
- Escala linear ou logaritmica.
- Legendas.
- Cores.
- Contraste.
- Tamanho dos pontos.
- Transparencia.
- Sobreposicao.
- Resolucao.
- Formato.
- Clareza estatistica.
- Coerencia com os arquivos numericos.

## 08.22 - Revisao dos arquivos de saida

Conferir cada arquivo de cada experimento.

Criar checklist de:
- Nome.
- Localizacao.
- Formato.
- Finalidade.
- Conteudo esperado.
- Conteudo encontrado.
- Integridade.
- Resultado aprovado ou pendente.

## 08.23 - Correcoes e repeticoes

Corrigir problemas estatisticos,
computacionais, visuais e de persistencia.

Distinguir correcoes obrigatorias
de melhorias apenas esteticas.

Repetir testes ou execucoes quando necessario.

## 08.24 - Aprovacao dos experimentos de 10%

Executar suite automatizada.

Conferir manifestos e outputs.

Documentar PCA e t-SNE separadamente.

Somente aprovar quando todos os
criterios obrigatorios forem atendidos.

# BLOCO V - BASE COMPLETA DE DESENVOLVIMENTO

## 08.25 - Preparacao da execucao completa

Utilizar 100% do desenvolvimento F1-F9.

Total esperado: 1667155 transacoes.

F10 permanece com 185239 transacoes
reservadas para teste final.

Fazer novo preflight de RAM, CPU e disco.

Para t-SNE, avaliar especialmente
viabilidade em escala de milhoes
de registros e uso de memoria.

Manter configuracoes pequenas.

## 08.26 - Execucao PCA 100%

Executar um candidato PCA com nove folds.

Registrar tempo, memoria e outputs.

Validar funcionamento em escala completa.

## 08.27 - Auditoria completa PCA 100%

Inspecionar novamente:
- Cada arquivo de saida.
- Cada metrica.
- Cada grafico aplicavel.
- Gaussianas e responsabilidades.
- Fronteira e campo de risco.
- Cache e manifestos.
- Integridade dos nove folds.

## 08.28 - Execucao t-SNE 100%

Executar um candidato t-SNE com nove folds
somente apos aprovar a viabilidade computacional.

Monitorar memoria, tempo e estabilidade.

Se nao couber no notebook, documentar
o impedimento e nao marcar como concluido.

## 08.29 - Auditoria completa t-SNE 100%

Inspecionar novamente todos os outputs,
metricas e graficos aplicaveis.

Comparar PCA e t-SNE sobre os mesmos dados
de desenvolvimento completo.

Registrar limitações e correcoes.

## 08.30 - Encerramento da Etapa 08

Conferir:
- Execucoes 10% aprovadas.
- PCA completo aprovado.
- t-SNE completo aprovado ou impedimento
  computacional formalmente documentado.
- Arquivos auditados.
- Graficos auditados.
- Comparacoes consistentes.
- Cache e retomada validados.
- F10 preservado.
- Testes automatizados aprovados.
- Correcões documentadas.

Gerar:
- registro.md por subetapa.
- resumo.md da Etapa 08.
- Atualizacao de CONTEXTO_GERAL.md.
- Atualizacao de RETOMADA.md.
- Checkpoint Git.

Registrar decisao de prontidao
para o cluster universitario.

## CATALOGO DE 15 GRAFICOS

01 - Precision-Recall.
02 - MCC por limiar.
03 - Matriz de confusao 0.5.
04 - Matriz de confusao MCC.
05 - Distribuicao de scores.
06 - Comparacao da grade.
07 - Risco por gaussiana.
08 - Pesos e responsabilidades.
09 - Medias gaussianas.
10 - Covariancias gaussianas.
11 - Variancia PCA.
12 - Projecao PCA.
13 - Projecao t-SNE.
14 - Diagnosticos t-SNE.
15 - Mapa de risco e fronteira.

Graficos especificos de PCA nao sao
obrigatorios em uma execucao t-SNE.

Graficos especificos de t-SNE nao sao
obrigatorios em uma execucao PCA.

Gráficos adicionais:
- Mapa complementar PCA 2D.
- Comparacoes entre PCA e t-SNE.

## PROXIMA TAREFA

08.01 - Auditoria inicial.

ETAPA 08 - PLANEJAMENTO REVISADO.
