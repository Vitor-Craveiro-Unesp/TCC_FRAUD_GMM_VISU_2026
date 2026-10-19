# TCC_EXPERIMENTAL_V3 - RETOMADA

Ultima atualizacao: 2026-10-18 00:56:20

## Situacao do projeto

- Etapas 01 a 06: pipeline implementado e validado funcionalmente.
- Etapa 07: visualizacao e diagnosticos concluidos.
- Etapa 08: planejamento de 22 subetapas elaborado.
- Etapa 08 ainda nao iniciada operacionalmente.
- Proxima subetapa: 08.01.

## Decisoes metodologicas

- Utilizar F1 a F9 para desenvolvimento.
- Manter F10 intocado ate a avaliacao final.
- Realizar execucoes locais completas.
- Testar PCA e t-SNE separadamente.
- Comecar com um candidato por metodo.
- Utilizar poucos hiperparametros no notebook.
- Reservar grades extensas para o cluster.
- Nao considerar ensaios amostrais como resultado final.

## Base

- Total: 1852394 transacoes.
- Desenvolvimento: 1667155 transacoes.
- F10 reservado: 185239 transacoes.

## Validacoes anteriores

- Ensaio local com 10% do desenvolvimento.
- PCA [1,2,3] e GMM [2,4].
- 54 ajustes OOF concluidos.
- Tempo observado: 132.57 segundos.
- Vencedor exploratorio: PCA=1 e GMM=4.
- Score exploratorio: 0.501178.
- Visualizacoes OOF e geometricas implementadas.
- Mapa complementar 2D implementado.
- Galeria HTML implementada.

## Etapa 08 - Objetivo

Executar separadamente PCA e t-SNE utilizando
100% dos dados de desenvolvimento.

Auditar arquivo por arquivo e grafico por grafico.

Verificar metricas, gaussianas, responsabilidades,
fronteiras de decisao, distribuicoes e embeddings.

Identificar melhorias, corrigir e validar novamente.

O plano completo esta em:
passos/08_execucao_local/plano.md

## Proxima tarefa - 08.01

Auditoria de recursos computacionais:

1. Conferir RAM e armazenamento.
2. Conferir dependencias e ambiente Python.
3. Conferir integridade dos folds.
4. Avaliar consumo estimado de PCA e t-SNE.
5. Preparar execucao local segura.
6. Nao iniciar ainda a busca extensa de parametros.

## Regra para avancar

A preparacao do cluster universitario so sera
iniciada depois da auditoria e aprovacao local,
ou apos documentarmos eventuais limitacoes
computacionais que impeçam a execucao integral.

F10 DEVE PERMANECER RESERVADO.
