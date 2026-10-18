# ETAPA 07 - VISUALIZACAO E DIAGNOSTICO DO MODELO

Data de encerramento: 2026-10-18 00:47:22

Status: CONCLUIDA - IMPLEMENTACAO E VALIDACAO FUNCIONAL LOCAL

Projeto: TCC_EXPERIMENTAL_V3

## Objetivo

Implementar as visualizacoes estatisticas e os diagnosticos
do modelo vencedor selecionado pelo score composto OOF.

Os resultados devem ser reproduziveis, interpretaveis
e organizados para utilizacao no TCC.

## Subetapas concluidas

### 07.01 - Graficos preditivos OOF

Modulo: src/visualizacao_oof.py

Implementados:

1. Curva Precision-Recall e Average Precision.
2. MCC por limiar, destacando o limiar otimo OOF.
3. Matriz de confusao com limiar 0.5.
4. Matriz de confusao com limiar MCC otimo.
5. Distribuicao dos scores por classe.
6. Comparacao de hiperparametros da grade.

Os graficos individuais representam o vencedor.
A comparacao da grade utiliza todos os candidatos.

Nao foi necessario repetir os treinamentos OOF.

### 07.02 - Diagnosticos geometricos

Modulo: src/visualizacao_geometrica.py

Implementados os diagnosticos aplicaveis:

7. Risco historico por gaussiana.
8. Pesos e responsabilidades dos componentes.
9. Medias das gaussianas.
10. Covariancias das gaussianas.
11. Variancia explicada pelo PCA.
12. Projecoes PCA.
13. Projecoes t-SNE, quando aplicavel.
14. Diagnosticos t-SNE, quando disponiveis.
15. Mapa de risco e fronteira de decisao.

O modelo diagnostico e ajustado exclusivamente
com dados de desenvolvimento.

As visualizacoes respeitam o metodo de reducao
dimensional e o numero de componentes do vencedor.

Graficos incompativeis com a configuracao
sao omitidos, sem criar visualizacoes artificiais.

### 07.03 - Integracao e mapa complementar 2D

Modulo: src/visualizacao_integrada.py

Implementados:

- Integracao das visualizacoes.
- Galeria HTML local.
- Organizacao por origem dos resultados.
- Mapa complementar bidimensional.
- Nuvens e elipses das gaussianas.
- Campo de risco estimado.
- Fronteira de decisao pelo limiar MCC OOF.
- Identificacao do candidato utilizado.

O mapa complementar 2D nao substitui o mapa
oficial do vencedor.

## Ensaio utilizado na validacao

Base: dataset_raw.parquet

Desenvolvimento: F1 a F9.

Amostra: 10% do desenvolvimento.

Registros utilizados: 166716.

Registros por fold: 18524.

PCA: [1, 2, 3].

GMM: [2, 4].

Amostragem: NORMAL.

Configuracoes: 6.

Rodadas OOF: 54.

Tempo observado: 132.57 segundos.

## Vencedor exploratorio

ID: 0d74604010afa3

Metodo: PCA.

Componentes PCA: 1.

Gaussianas: 4.

AP: 0.011126.

MCC: 0.047470.

LogLoss: 0.031829.

Score composto: 0.501178.

Limiar MCC OOF: 0.010396.

Este vencedor pertence somente ao ensaio
amostral e nao representa a selecao definitiva.

## Mapa complementar 2D

Candidato: 535cfea05d0b69.

PCA: 2 componentes.

GMM: 2 gaussianas.

Score composto: 0.499287.

Limiar MCC OOF: 0.006842.

A visualizacao representa a geometria
desse candidato complementar.

Nao representa a fronteira exata do
modelo vencedor PCA unidimensional.

## Graficos gerados

Catalogo configurado no YAML: 15 tipos.

Aplicaveis ao vencedor PCA: 13 tipos.

Tipos exclusivos de t-SNE nao aplicaveis: 2.

Grafico complementar 2D: 1 adicional.

Total esperado no ensaio: 14 figuras,
considerando todos os tipos habilitados.

Pastas principais:

- oof/graficos_gerados
- teste_final/graficos_gerados
- visualizacao_consolidada

Observacao: nesta etapa, a pasta teste_final
contem diagnosticos geometricos produzidos
somente a partir do desenvolvimento.

Ela nao contem avaliacao independente do F10.

## Verificacoes

- Implementacao das tres subetapas concluida.
- Suite integrada: 305 testes aprovados
  no ambiente de desenvolvimento.
- Funcionamento dos modulos confirmado pelo usuario.
- Geracao dos graficos OOF confirmada.
- Diagnosticos geometricos gerados.
- Integracao e galeria executadas.
- Mapa complementar 2D implementado e executado.
- Configuracao vencedora identificada corretamente.
- Limiar MCC preservado da validacao OOF.
- F10 reservado.
- Nenhuma selecao baseada no F10.

## Limitacoes

- Validacao funcional realizada sobre ensaio de 10%.
- Resultados atuais sao exploratorios.
- Grade extensa de hiperparametros ainda pendente.
- Execucao integral no cluster nao realizada.
- Vencedor definitivo ainda nao selecionado.
- Desempenho independente no F10 nao avaliado.
- Compatibilidade do ambiente Linux/SLURM
  ainda deve ser verificada.

## Arquivos principais

- src/visualizacao_oof.py
- src/visualizacao_geometrica.py
- src/visualizacao_integrada.py
- tests/test_visualizacao_oof.py
- tests/test_visualizacao_geometrica.py
- tests/test_visualizacao_integrada.py

## Conclusao

A Etapa 07 foi concluida quanto a implementacao
e a validacao funcional das visualizacoes.

O pipeline permite representar os resultados
OOF do vencedor, diagnosticar as gaussianas
e visualizar fronteiras de decisao.

A implementacao tambem permite gerar um
mapa 2D complementar, distinguindo-o
do modelo vencedor oficial.

A avaliacao cientifica definitiva permanece
dependente da grade completa no cluster.

## Proxima tarefa

Preparacao do ambiente de execucao no cluster.

- Verificar dependencias Python.
- Preparar ambiente Linux.
- Preparar arquivo SLURM/SBATCH.
- Validar caminhos e persistencia.
- Executar a grade de hiperparametros.
- Selecionar o vencedor definitivo pelo OOF.
- Preservar F10 ate a etapa de teste independente.

ETAPA 07 - CONCLUIDA.
