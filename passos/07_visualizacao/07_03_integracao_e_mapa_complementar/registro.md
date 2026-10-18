# Passo 07.03 — Integração, galeria e mapa 2D complementar

**Status:** implementação criada e testada com dados sintéticos. Validação com a base real e a execução no notebook do usuário **pendentes**.

## Objetivo e regras

- Um comando gera/verifica gráficos 01–06 a partir das predições OOF e gráficos 07–15 aplicáveis a partir do modelo vencedor reajustado em F1–F9.
- Imagens oficiais são copiadas (sem alteração dos originais) para `visualizacao_consolidada/oof` e `visualizacao_consolidada/geometria_desenvolvimento`, com um `index.html` para visualizar tudo no navegador.
- **Somente se o vencedor for PCA 1D** e já existir candidato PCA 2D na grade: criar um **16º diagnóstico complementar**, explicitamente não oficial. Seleciona o melhor candidato PCA 2D da grade OOF; reaproveita o **limiar MCC OOF desse candidato**, ajusta uma vez em F1–F9 e mostra sua geometria genuína 2D (nuvens/elipses de covariância, pontos por classe, mapa de risco, contorno da fronteira se houver cruzamento).
- Não recriar combinações nem reotimizar limiares. Não modificar YAML, resultados OOF, seleção oficial, código 06.03 ou dados originais. Não ler F10 no fluxo do complemento.
- Se o vencedor for PCA 2D/3D, a geometria oficial do gráfico 15 é suficiente; não há modelo 2D extra. Se não houver candidato 2D na grade, não inventar modelo 2D.
- 15 tipos configurados **não equivalem** a 15 imagens por execução. Por exemplo, no vencedor PCA 1D: 6 tipos OOF + 7 tipos geométricos + 1 imagem extra complementar = normalmente 14 arquivos PNG. Os tipos 13 e 14 são exclusivos de t-SNE.

## Origem e nomenclatura

- `oof`: validação OOF, métricas do vencedor; gráfico 06 compara candidatos.
- `geometria_desenvolvimento`: modelo vencedor reajustado no desenvolvimento, **não contém métricas do teste final** (a pasta original se chama `teste_final` por configuração histórica).
- `complementar_2d`: candidato diferente do vencedor, mapa explicativo e **não** avaliação oficial.
- `manifesto_integracao_07_03.json`: identidade do vencedor e diagnósticos incluídos.
- `manifesto_complementar_2d.json`: id do candidato PCA 2D, score, seu próprio limiar MCC, hash do gráfico e confirmação de não utilização de F10.

## Segurança e reprodutibilidade

1. Validar OOF/seleção e manifestação de integridade dos módulos 07.01/07.02.
2. Para o mapa extra, auditar hash do Parquet/folds, identidade e ordem dos índices originais, labels e folds (todos F1..F9) para candidato 2D.
3. `--planejar` não grava arquivos, não faz fit e não lê Parquet.
4. Imagens já consolidadas só são reutilizadas se hashes forem iguais. Reexecuções não sobrescrevem fontes nem refazem o ajuste extra se o manifesto complementar for compatível.
5. O nível de confiança/limiar mostrado no mapa é de um **modelo 2D distinto**, e não deve ser interpretado como projeção da fronteira do vencedor PCA 1D.

## Testes sintéticos

`python -m pytest tests/test_visualizacao_integrada.py -q`

Valida: seleção de candidato 2D, limiar próprio, segurança dos folds, preservação de imagens, integração idempotente, planejamento sem treino, renderização geométrica 2D e manifesto inválido. 

## Validação real a completar

Rodar os comandos no arquivo `INSTRUCOES_PASSO_07_03.md`, registrar tempo de ajuste, contagem de imagens, identificação do candidato 2D e verificar visualmente nuvens e contorno da fronteira. Atualizar `passos/CONTEXTO_GERAL.md` somente após validação real.
