# TCC_EXPERIMENTAL_V3 — Registro parcial de leitura do pacote `zips.rar`

**Data:** 20/10/2026  
**Estado:** pacote raiz extraído e YAML analisado estaticamente; `src/`, testes e saídas ainda não auditados.  
**Natureza:** documentação de ingestão; não representa execução da pipeline ou aprovação de implementação.

## Arquivos efetivamente recebidos e examinados

Arquivo recebido: `zips.rar` (RAR v5). Extração verificada com 5 arquivos:

1. `.gitignore` — 281 bytes.
2. `comparacao_yaml.diff` — 17.935 bytes.
3. `config_TCC_EXPERIMENTAL_V3_atualizado.yaml` — 30.570 bytes, **712 linhas**.
4. `criar_passos_TCC_EXPERIMENTAL_V3.ps1` — 2.486 bytes.
5. `requirements.txt` — 1.872 bytes.

O YAML foi carregado com sucesso usando parser YAML, sem erro de sintaxe reportado nesta leitura. Seu SHA-256 é `2c7ad4585215208df77d3d999dcc24486e66390cb95dff8e2f88913348b4e870`.

## Constatações fundamentadas no YAML enviado

- Configuração central do experimento: `TCC_EXPERIMENTAL_V3`, `semente: 42`.
- Dados: `TARGET_OMEGA` como alvo, `NID_ALPHA` como identificador; `fracao_base: 1.0`.
- Divisão: 10 folds, com **F10 reservado** para teste independente; estratificação e embaralhamento habilitados. Desenvolvimento OOF nos outros nove folds.
- Redução exclusiva: `reducao.metodo: "PCA"`, com `pca.n_components: [2, 3]` nessa versão; a seção alternativa do t-SNE contém `n_components: 2`, mas está inativa enquanto o método for PCA.
- GMM único: `gmm.n_components: [1, 2]`, `covariance_type: "full"`, `max_iter: 300`, `n_init: 3`.
- Risco histórico: `soft_counts` usando as observações rotuladas de treino original, com `suavizacao: 0.0`, predição como soma de responsabilidades multiplicadas por riscos dos componentes.
- Avaliação configurada: Average Precision, MCC, LogLoss, score composto; seleção do limiar a partir dos scores OOF.
- Saídas previstas: diretórios `oof` e `teste_final`, com subpasta `graficos_gerados`, nome de experimento pela data/hora de conclusão.
- A seção `visualizacao.graficos` contém **17 chaves configuradas, todas com `habilitar: true`**. Isso não comprova que todos os 17 geradores funcionam e não equivale ao número de gráficos distintos nos PDFs.
- Não há ocorrência de `biplot` no YAML recebido, de modo que **a configuração explícita do novo Biplot PCA não foi identificada** nesta versão.

## Pontos de verificação, não erros confirmados

1. A chave global `visualizacao.geometria.apenas_1d_2d: true` aparece junto de instruções para gráficos 3D em seções individuais; conferir a lógica de precedência/execução em `src/` antes de afirmar conflito.
2. O YAML atual tem `pca.n_components: [2, 3]`; os testes visuais propostos para Biplot PCA **1D** precisarão de dados sintéticos ou configuração temporária adequada, sem presumir alteração definitiva dos hiperparâmetros.
3. Diversos gráficos geométricos estão associados à pasta `teste_final`, enquanto os comentários dizem que a geometria é ajustada **somente no desenvolvimento**. Auditar a origem efetiva das observações e parâmetros; não confundir destino de gravação com base usada no ajuste.
4. O arquivo `comparacao_yaml.diff` contém uma comparação textual com versão anterior, mas a versão-base indicada no diff não foi recebida separadamente; não tratar o diff isoladamente como prova da história completa do Git.

## O que não foi verificado

- Se cada opção do YAML é realmente lida ou utilizada pelo código.
- Se os módulos GMM, PCA, t-SNE e OOF estão implementados conforme as especificações.
- Se os testes automatizados passam no estado atual.
- Se os gráficos recentes aprovados estão presentes nos artefatos gerados.
- Se existem gráficos adicionais no código não descritos nos quatro PDFs.

## Próxima entrada necessária

1. **`src.rar`**: identificar geradores, integração, tratamento de dimensão, avaliação, risco e leitura do YAML.
2. **`tests.rar`**: auditar cobertura e, quando viável, executar testes.
3. **`outputs.rar`**, se for preciso verificar as imagens efetivamente produzidas e os manifestos.
4. Outros arquivos apenas se o código ou testes indicar necessidade.

## Continuidade

Este registro complementa `REGISTRO_PARCIAL_LEITURA_PASSOS_TCC_20261020.md`. Preservar documentos originais sem sobrescrita. Prioridade funcional após reconstruir o contexto: prévias do **Biplot PCA 1D, 2D e 3D**, aprovação visual e somente então integração; em seguida auditoria **PDF → pipeline** e **pipeline → PDF**.

**Princípio:** relatar separadamente o que foi informado pelo usuário, o que foi documentado, o que foi inspecionado e o que foi efetivamente testado; nunca inventar evidências.
