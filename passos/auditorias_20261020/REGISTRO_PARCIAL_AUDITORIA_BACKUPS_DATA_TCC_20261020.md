# TCC_EXPERIMENTAL_V3 — Auditoria documental de `backups.rar` e `data.rar`

**Data:** 2026-10-20  
**Escopo:** inventário, leitura seletiva e verificação de integridade dos pacotes recebidos; nenhuma modificação na pipeline.  
**Status:** concluída a conferência estrutural destes dois pacotes. Não significa execução de experimentos, treinamento ou avaliação de F10.

## 1. Entradas e arquivos efetivamente examinados

- `backups.rar`: RAR v5, um arquivo regular: `backups/yaml/config_legado_20261017_190437.yaml` (11.419 bytes). Foi extraído para pasta isolada de análise e interpretado com parser YAML.
- `data.rar`: RAR v5, três arquivos regulares:
  - `data/raw/dataset_raw.parquet` (67.470.223 bytes);
  - `data/splits/dataset_raw_74867500a846/folds.npy` (3.704.916 bytes);
  - `data/splits/dataset_raw_74867500a846/manifesto_folds.json` (1.857 bytes).
- Total descompactado de `data.rar`: 71.176.996 bytes.

## 2. Integridade da base e da divisão

O manifesto recebido identifica como base `data/raw/dataset_raw.parquet`, alvo `TARGET_OMEGA`, classe positiva `1`, 10 folds estratificados com embaralhamento e semente 42, reservando F10 ao teste final. Regra inscrita no manifesto: para cada F1–F9, valida nesse fold e treina nos outros oito; F10 fica fora da seleção.

**Verificações executadas sobre os arquivos extraídos:**

| Verificação | Resultado |
|---|---|
| SHA-256 do Parquet vs. `sha256_parquet` no manifesto | **Coincide**: `1fdf25917a1439a672a507df808df0a2287830ce2ce6afccc3f431d0337a39d6` |
| SHA-256 de `folds.npy` vs. `sha256_folds` no manifesto | **Coincide**: `ec35267b9085f92c4f458a34e0ce244f79b8213c6b9ca43c84e3c3789a9cb9c6` |
| Assinatura de arquivo Parquet | Primeiros bytes `PAR1` |
| Array de atribuição de folds | `shape=(1_852_394,)`, `dtype=uint16`, sem falhas de leitura |
| IDs de folds | Exatamente os inteiros de 1 até 10 |
| Contagem de atribuições F1–F10 | Coincide integralmente com o manifesto |

Distribuição de atribuições observada no `folds.npy`: F1, F2, F3 e F4 = 185.240 registros cada; F5, F6, F7, F8, F9 e F10 = 185.239 registros cada; soma = **1.852.394**. Esta inspeção apenas contou identificadores do vetor e **não utilizou F10 em treinamento, seleção ou avaliação**.

O manifesto também registra 1.842.743 transações de classe 0 e 9.651 de classe 1 (0,521% classe positiva, aproximadamente). Esses números constam da documentação anterior em `passos/02_dados/01_auditoria_da_base/registro.md`; a distribuição por classe **não foi recalculada independentemente a partir do conteúdo do Parquet** nesta auditoria, pois não havia leitor Parquet disponível no ambiente isolado.

**Limite:** hashes e contagem de folds comprovam identidade de bytes e consistência de registros com o manifesto fornecido. Não demonstram, por si, adequação metodológica da divisão por grupos/tempo, ausência de vazamento, nem isolamento efetivo de F10 em toda a lógica da pipeline.

## 3. Backup YAML vs. YAML atualizado

Foram interpretados:

- Backup: `backups/yaml/config_legado_20261017_190437.yaml`.
- Atual: `config_TCC_EXPERIMENTAL_V3_atualizado.yaml` (recebido anteriormente em `zips.rar`).

Os dois têm os mesmos **15 grupos superiores** (`experimento`, `dados`, `divisao`, `preprocessamento`, `amostragem`, `reducao`, `pca`, `tsne`, `gmm`, `risco_historico`, `validacao_cruzada`, `avaliacao`, `execucao`, `saidas`, `visualizacao`). Foram observadas **19 diferenças semânticas por caminhos de configuração** em comparação recursiva, incluindo grupos de chaves acrescentados/removidos em bloco. Principais mudanças observadas:

- `gmm.covariance_type`: **`diag` (legado) → `full` (atual)**. Coerente com ensaios históricos `diag` em `outputs.rar` e com a configuração FULL atual; **não é um defeito**.
- `gmm.max_iter`: **250 → 300**.
- Adições em `amostragem`: `risco_historico` e `base_calculo`.
- Adição de `saidas.pastas`.
- Reorganização ampla do bloco `visualizacao` (subgrupos de configuração diferentes); não é seguro restaurar o bloco antigo nem interpretar diferenças como erros sem auditar compatibilidade com `src/`.

**Ressalva de proveniência:** o `comparacao_yaml.diff` recebido em `zips.rar` não registra a troca `diag → full`, embora ela esteja presente entre este backup e o YAML atual. Logo, o arquivo `.diff` não deve ser considerado uma comparação exaustiva *entre estes dois arquivos específicos*; pode referir-se a outra versão-base não identificada.

## 4. O que ainda precisa ser conferido depois

- Quando necessário para a próxima etapa, validar esquema/tipos do Parquet e regras de leitura e colunas efetivas com o ambiente apropriado; nesta revisão não foram inspecionadas nem divulgadas linhas individuais de transações.
- Confirmar que a separação estratificada embaralhada é apropriada ao objetivo metodológico e às eventuais dependências temporais/de entidades. Este cuidado já aparece como ressalva no YAML e **não constitui indício confirmado de vazamento**.
- Se necessário, cruzar a atribuição de folds com os dados por leitura autorizada, preservando a política de F10 reservado; não rodar validação final prematuramente.
- Fazer a auditoria de execução dos gráficos novos se o usuário fornecer HTMLs/imagens posteriores ao ensaio de 17/10.

## 5. Registro de intervenção

- Arquivos originais e código do projeto: **não modificados**.
- Experimentos, treinamento, escolha de limiar e avaliações em F10: **não executados**.
- Criado somente este registro `.md`; atualizado inventário de pendências separado, sem reparos.

**Próximo passo:** continuar recebendo materiais que o usuário queira enviar. Após confirmação de término do envio, consolidar prioridades e solicitar aprovação antes de qualquer correção. O primeiro desenvolvimento visual planejado continua sendo o **Biplot PCA 1D → 2D → 3D**, com prévias e aprovação prévia à integração.
