# TCC_EXPERIMENTAL_V3 — Pendências acumuladas durante o recebimento dos arquivos

**Data de abertura:** 2026-10-20  
**Situação:** execução sequencial autorizada pelo usuário; verificar e propor cada ajuste individualmente antes de alterar código ou dependências. **P13, P03, P02, P01, P14, P04, P10 e P07 concluídas; P11 encerrada como limitação documental conhecida; P15 dispensada por decisão metodológica em 20/10/2026; a regressão global posterior à correção de P01 aprovou 416/416 testes**. Na P03, a biblioteca `category-encoders==2.11.1` foi instalada por solicitação expressa do usuário; na P02, `openTSNE==1.0.4` foi disponibilizada e validada. Nenhum código-fonte foi alterado nessas pendências.

## Critérios de registro

- **Divergência observada**: comportamento ou expectativa inconsistente encontrado na análise já relatada.
- **A verificar**: indício ou limitação que não autoriza concluir que existe defeito.
- **Trabalho planejado**: funcionalidade ainda a desenvolver, sem caracterizá-la como bug.
- Registrar evidências (arquivo, teste, saída e resultado) antes de alterar qualquer código.
- Não confundir o resultado de teste em ambiente isolado com execução no computador do autor.

## Estado da resolução

- **P13: concluída em 20/10/2026**, por confirmação do uso do Python 3.13.3 da `.venv` no terminal. As divergências de dependências não foram tratadas como correção automática.
- **P03: concluída em 20/10/2026**, após `pyarrow` e `pyarrow.parquet` importarem corretamente, confirmação dos encoders internos e instalação solicitada de **`category-encoders==2.11.1`** (`BinaryEncoder: True`), seguida de `pip check` sem conflitos e **17 testes locais passando em 2,57 s** (antes da instalação: 17 em 3,29 s).
- **P02: concluída em 20/10/2026.** Após confirmação da instalação `openTSNE 1.0.4` e `pip check` sem conflitos, o teste específico previamente pulado passou (`1 passed, 6 deselected em 3.95s`); o arquivo completo `tests/test_tsne_pipeline_02_13_14.py` passou (`7 passed em 6.10s`). Ver `P02_REGISTRO_CONCLUSAO_20261020.md`.
- **P01: concluída em 20/10/2026.** Após reprodução da falha na covariância FULL, correção de uma asserção confirmada pelo `git diff`; teste específico **1 passed in 2.05s**, módulo GMM **31 passed in 3.04s**, suíte geral **416 passed in 80.56s** (sem falhas nem skips). Ver `P01_REGISTRO_CONCLUSAO_20261020.md`. Sem confirmação de novo commit/push.
- **P14: concluída em 20/10/2026.** Quatro identificadores históricos ausentes nos testes atuais; cache limpo e coleta atual de **416 testes em 2,55 s**. Tentativa de leitura de `nodeids` gerou `FileNotFoundError` porque o arquivo não foi encontrado após `--collect-only`; isso não é falha de testes nem confirma reconstrução do arquivo. Ver `P14_REGISTRO_CONCLUSAO_20261020.md`.
- **P15: encerrada como dispensada por decisão metodológica.** O usuário ratificou a manutenção da estratégia estratificada e do isolamento de F10, sem reabrir debate temporal/por entidades. Não significa que essas hipóteses foram empiricamente descartadas; significa que não fazem parte das correções autorizadas. Ver `P15_REGISTRO_DISPENSA_20261020.md`.
- **P04: concluída em 20/10/2026.** O usuário aplicou a remoção de `visualizacao.geometria.apenas_1d_2d` no YAML ativo e atualizou somente os comentários previstos. `yaml.safe_load` validou a estrutura e preservou `pares_2d`; `git diff` confirmou as alterações autorizadas; a suíte local passou **416/416 testes em 77,43 s**. Ver `P04_REGISTRO_CONCLUSAO_20261020.md`. Commit/push não confirmado.
- **P10: concluída em 20/10/2026 no escopo de galeria histórica revisada.** Painel 00 localizado, script `--verificar` aprovado com 14 arquivos históricos e 15 na revisão; três capturas exibem 7 OOF (00–06), 7 imagens de geometria e o complementar 16. Teste de inclusão da galeria passou (1 teste/3,06 s). Galeria revisada em pasta separada; manifesto revisado ainda não lido diretamente. Ver `P10_REGISTRO_CONCLUSAO_20261020.md`.
- **P07: concluída como inventário de tudo que foi recebido**, sem extrapolar resultados históricos para a pipeline atual. Ver `P07_REGISTRO_CONCLUSAO_20261020.md`.
- **P11: encerrada a busca documental por evidências adicionais**, pois o usuário informou que enviou tudo o que possuía. Ausência das imagens recentes não é defeito comprovado; futuras verificações de execução caberão às etapas próprias. Ver `P11_REGISTRO_ENCERRAMENTO_DOCUMENTAL_20261020.md`.
- Demais pendências: mantidas sem reparos.

## Pendências conhecidas até o momento

| ID | Categoria | Item | Evidência/estado conhecido | Encaminhamento proposto (ainda não executado) |
|---|---|---|---|---|
| P01 | **Concluída (20/10/2026)** | Teste `test_parametros_diagnosticos` para GMM FULL | Correção de uma asserção `(2, 2)` → `(2, 2, 2)` confirmada no `git diff`; teste específico **1 aprovado**, módulo GMM **31 aprovados** e regressão global **416/416 aprovados em 80,56 s** | **Encerrada.** Ver `P01_REGISTRO_CONCLUSAO_20261020.md`; alteração ainda sem commit/push confirmado |
| P02 | **Concluída (20/10/2026)** | Teste de `openTSNE` anteriormente pulado | A biblioteca era ausente; no PC, `openTSNE` 1.0.4 agora importa, `pip check` sem conflitos; **1 teste específico aprovado** e **7/7 testes do arquivo aprovados** | **Encerrada.** Ver `P02_REGISTRO_CONCLUSAO_20261020.md`; testes não equivalem a experimento OOF completo |
| P03 | **Concluída (20/10/2026)** | Dependências `pyarrow` e `category_encoders` | No PC, `pyarrow` 26.0.0 e `pyarrow.parquet` importam; o código examinado usa encoders internos. Por escolha do usuário, `category-encoders==2.11.1` foi instalado, importou corretamente e `BinaryEncoder: True`; `pip check` sem conflitos; **17 passed in 2.57s** após instalação. | **Encerrada.** Preservar encoder interno da pipeline e manter, por ora, PyArrow 26.0.0; detalhes em `P03_REGISTRO_CONCLUSAO_20261020.md`. |
| P04 | **Concluída (20/10/2026)** | Configuração geométrica obsoleta `apenas_1d_2d: true` | Correção aplicada ao YAML ativo pelo usuário; `yaml.safe_load` confirmou a ausência da chave, `git diff` mostrou somente os comentários e a remoção aprovados; suíte geral **416 passed in 77.43s** | **Encerrada.** Ver `P04_REGISTRO_CONCLUSAO_20261020.md`; commit/push ainda não confirmado |
| P05 | A verificar | Auditoria catálogo PDF → pipeline | Quatro PDFs foram recebidos: OOF, GMM, PCA e t-SNE | Conferir para cada gráfico a implementação, acionamento, testes e saída; listar ausentes e divergências |
| P06 | A verificar | Auditoria pipeline → catálogo PDF | Na análise estática anterior foram citadas visualizações adicionais: OOF `06` comparação de hiperparâmetros, t-SNE `14` diagnóstico KL e mapa complementar PCA2D `16` | Verificar se precisam ser documentadas nos PDFs e se correspondem a recursos efetivamente gerados |
| P07 | **Concluída (20/10/2026; inventário histórico)** | Artefatos produzidos nas execuções | `outputs.rar` recebido: 3 ensaios de 17/10, 220 arquivos, 47 PNGs estruturalmente íntegros; última configuração PCA 1D/diag; P10 revisou a galeria histórica para 15 imagens | **Encerrada no escopo do material fornecido**; resultados recentes não disponíveis; não extrapolar outputs históricos para a versão atual. Ver `P07_REGISTRO_CONCLUSAO_20261020.md` |
| P08 | Trabalho planejado | Biplot PCA 1D, 2D e 3D | Ainda não identificado como implementado no código/YAML recebidos; não consta dos PDFs de PCA fornecidos | Construir protótipos, avaliar visualmente um a um, integrar só após aprovação; criar testes e atualizar documentação |
| P09 | A verificar | Consistência dos registros históricos com o código atual | Alguns documentos de retomada de 19/10 relatavam pendências que o usuário informou já terem sido resolvidas e comitadas | Tratar documentação antiga como histórica; determinar estado atual por código, testes e saídas; atualizar arquivos `.md` quando aprovado |

| P10 | **Concluída no escopo histórico (20/10/2026)** | Painel 00 omitido da galeria antiga, embora PNG existisse. | Script `--verificar` aprovado: 14 imagens originais e 15 previstas; capturas da galeria revisada mostram **7 OOF (inclui 00), 7 geometria, 1 complementar**, sem imagens quebradas visíveis. Teste específico **1 passed in 3.06s**. | **Encerrada** para a galeria histórica; imagem adicionada em pasta separada, sem treino ou F10. Não inferir que os artefatos refletem GMM FULL atual. Manifesto revisado não inspecionado diretamente; ver `P10_REGISTRO_CONCLUSAO_20261020.md`. |
| P11 | **Encerrada como limitação documental (20/10/2026)** | Evidência recente de gráficos GMM FULL e t-SNE 1D/2D/3D | Outputs enviados são ensaios antigos PCA/diag; usuário confirmou que não possui outros arquivos para fornecer | **Não solicitar novos arquivos antigos**. Falta de output histórico não é bug; futuras validações práticas ocorrerão nas etapas previstas. Ver `P11_REGISTRO_ENCERRAMENTO_DOCUMENTAL_20261020.md` |
| P12 | Melhoria documental opcional | Nome da pasta `teste_final` é ambíguo | Manifesto interno declara `desenvolvimento_F1_F9`, `f10_utilizado: false` | Avaliar nota/renomeação futura apenas se necessário, preservando a compatibilidade e sem sugerir que F10 foi usado |

| P13 | **Concluída (20/10/2026)** | Ambiente Python da `.venv` no terminal | Antes da ativação, `python` apontava para `C:\Python314\python.exe`; após ativação, o terminal exibiu prompt `(.venv)`, `python` resolvendo para `.venv\Scripts\python.exe` e Python 3.13.3. `pip check` sem conflitos. Persistem 76 requisitos ausentes e 10 versões divergentes. | Nenhuma instalação; diferenças de dependências encaminhadas para P03/P02. Ver `P13_REGISTRO_CONCLUSAO_20261020.md`. Kernel de notebook deve ser conferido quando aplicável |
| P14 | **Concluída (20/10/2026)** | Cache pytest continha quatro identificadores antigos | Quatro nomes não foram encontrados nos testes atuais; `--cache-clear --collect-only` coletou **416 casos em 2,55 s**. Após limpeza, arquivo `nodeids` não encontrado; não afirmar que foi recriado. | **Encerrada sem alteração de código.** Ver `P14_REGISTRO_CONCLUSAO_20261020.md` |

| P15 | **Dispensada por decisão do usuário (20/10/2026)** | Revisão adicional da estratégia de partição | Mantida a divisão estratificada já aprovada com F10 reservado; não reabrir revisão de temporalidade/grupos, conforme decisão expressa ratificada na conversa | **Encerrada como dispensada**, sem afirmar que hipóteses foram testadas e sem alterar folds; ver `P15_REGISTRO_DISPENSA_20261020.md` |

## Verificações locais adicionais (20/10/2026)

- **P03:** a mesma bateria `python -m pytest -q tests/test_preprocessing.py tests/test_imputacao_normalizacao.py` passou **antes** da instalação (**17 passed in 3.29s**) e **depois** da instalação de `category-encoders==2.11.1` (**17 passed in 2.57s**) no PowerShell com `.venv` ativada.
- `pyarrow` e `pyarrow.parquet`: importação bem-sucedida; `pyarrow.__version__` **26.0.0**.
- `category_encoders` estava ausente inicialmente e foi **instalado a pedido do usuário**; confirmação: versão **2.11.1**, `BinaryEncoder: True`. `python -m pip check`: **No broken requirements found**. A codificação interna do projeto não foi substituída.
- **P02:** `openTSNE` 1.0.4 importável no ambiente ativo, `pip check` sem conflitos; teste específico `1 passed, 6 deselected in 3.95s`; suíte do módulo `7 passed in 6.10s`. Concluída; ainda sem experimentos reais completos.
- **P01:** `python -m pytest -q` executado no computador do usuário: **416 passed in 80.56s**; `git diff -- tests/test_gmm.py` confirmou uma alteração de asserção `(2, 2)` → `(2, 2, 2)`. Aviso de normalização LF/CRLF registrado sem efeito nos testes. **P01 concluída**; novo commit/push não informado.
- **P14:** busca dos quatro nomes históricos sem ocorrências; limpeza e coleta atuais com `416 tests collected in 2.55s`. A leitura de `nodeids` falhou por arquivo ausente; isso não invalida a coleta nem os 416 testes previamente aprovados. **P14 concluída sem mudanças em código ou testes.**
- **P04:** aplicação local confirmada pelo usuário: `YAML OK; chave removida; pares_2d = [[1, 2], [1, 3], [2, 3]]`. `python -m pytest -q`: **416 passed in 77.43s**. `git diff` confirma apenas correção de comentários e remoção da chave antiga. **P04 concluída; P10 revisada depois**.

## Resultado da bateria de testes já relatada nesta conversa

- **Execução isolada anterior:** 416 casos coletados; 414 passaram, 1 falhou (P01) e 1 foi pulado (P02).
- **Execução local posterior, em 20/10/2026, com correções/dependências verificadas:** `python -m pytest -q`: **416 passed in 80.56s**, sem skips nem falhas.
- **Nova execução local após a P04, em 20/10/2026:** `python -m pytest -q`: **416 passed in 77.43s**, sem skips nem falhas; `git diff` confirmou somente as alterações aprovadas no YAML ativo.
- Esses resultados não significam que os scripts que exigem a base real foram executados.

## Material recebido até esta atualização

- PDFs dos catálogos OOF, GMM, PCA e t-SNE.
- `passos.rar`.
- `zips.rar` (incluindo YAML de configuração, diff, requisitos, `.gitignore` e script de criação de passos).
- `src.rar`.
- `tests.rar`.
- `outputs.rar` e `outputs(1).rar` (SHA-256 idêntico, 220 arquivos de saída e 47 PNGs).
- `.pytest_cache.rar` e `.venv.rar` (inventário seletivo descrito em `REGISTRO_PARCIAL_AMBIENTE_CACHE_TCC_20261020.md`).
- `backups.rar` (YAML legado) e `data.rar` (Parquet, `folds.npy`, manifesto), com integridade SHA-256 confirmada; ver `REGISTRO_PARCIAL_AUDITORIA_BACKUPS_DATA_TCC_20261020.md`.

## Encerramento do recebimento

Em 20/10/2026, o usuário **confirmou que já enviou todo o material de que dispunha**. A busca por imagens/HTMLs recentes (P11) foi encerrada como limitação documental conhecida, sem conclusão de defeito. Materiais futuros só serão solicitados se uma nova tarefa necessitar de uma fonte efetivamente ausente; não insistir nos arquivos visuais antigos.

## Política de continuidade

1. Receber os arquivos restantes e registrar novos achados neste inventário.
2. Separar **correções comprovadas**, **verificações pendentes** e **novas implementações**.
3. Apresentar a lista consolidada ao usuário, com prioridades e evidências.
4. Obter aprovação antes de modificar código, testes, YAML ou documentos existentes.
5. Após cada etapa concluída, preparar/atualizar os Markdown de registro, resumo e retomada segundo o padrão verificado em `passos/`.

**Status de execução:** P13, P03, P02, P01, P14, P04, P10 e P07 **concluídas (8)**; P15 **dispensada por decisão (1)**; P11 **encerrada como limitação documental conhecida (1)**. Total: **10/15 tratadas, 5 abertas**. P01 e P04 têm regressões gerais **416/416** aprovadas no computador do usuário (respectivamente 80,56 s e 77,43 s); P14 teve coleta de 416 casos após limpeza de cache. **Próxima: P08 — Biplot PCA 1D, 2D e 3D, com aprovação visual antes da integração.** Os ajustes P01/P04 ainda não tiveram novo commit/push confirmado.

## Atualização P10 — avaliação visual de 20/10/2026

O usuário exibiu três capturas da galeria revisada com 15 imagens (7 OOF, 7 geométricas, 1 complementar), após o verificador do script confirmar os arquivos de origem. O Painel 00 aparece como primeiro cartão e os demais continuam visíveis. A correção é **somente da galeria histórica**, sem execução de novos modelos. O script produz saída em `visualizacao_consolidada_P10_revisada`, em pasta separada da galeria original. Ver registro próprio `P10_REGISTRO_CONCLUSAO_20261020.md`. **P10 encerrada, próxima P07.**

Observação de caminhos: a pasta local do usuário é `transferencia_ias_1` **sem underscore no início**; usar esse nome nos próximos comandos de PowerShell.

## Atualização P07/P11 — encerramento da busca de arquivos em 20/10/2026

O usuário confirmou ter enviado todos os documentos e saídas que possuía. **P07** foi concluída como inventário histórico com integridade e proveniência registradas; **P11** foi encerrada somente no eixo de busca documental, reconhecendo que faltam artefatos de execuções visuais recentes, sem atribuir falha ao código. Nenhum experimento real adicional foi executado. Ver os arquivos `P07_REGISTRO_CONCLUSAO_20261020.md` e `P11_REGISTRO_ENCERRAMENTO_DOCUMENTAL_20261020.md`. **Próxima P08**, depois P05/P06/P09/P12.
