# TCC_EXPERIMENTAL_V3 — Pendências acumuladas durante o recebimento dos arquivos

**Data de abertura:** 2026-10-20  
**Situação:** execução sequencial autorizada pelo usuário; verificar e propor cada ajuste individualmente antes de alterar código ou dependências. **P13, P03, P02, P01 e P14 concluídas; P15 dispensada por decisão metodológica em 20/10/2026; a regressão global posterior à correção de P01 aprovou 416/416 testes**. Na P03, a biblioteca `category-encoders==2.11.1` foi instalada por solicitação expressa do usuário; na P02, `openTSNE==1.0.4` foi disponibilizada e validada. Nenhum código-fonte foi alterado nessas pendências.

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
- **P04: correção aprovada pelo usuário e preparada em 20/10/2026; AGUARDA APLICAÇÃO/TESTES LOCAIS.** Removida `visualizacao.geometria.apenas_1d_2d` de cópia do YAML e ajustados apenas os comentários; análise semântica identificou ausência de outras mudanças. Ver `P04_REGISTRO_CORRECAO_PREPARADA_20261020.md`, YAML corrigido e diff. Não declarar que o YAML ativo no Windows foi atualizado sem confirmação.
- Demais pendências: mantidas sem reparos.

## Pendências conhecidas até o momento

| ID | Categoria | Item | Evidência/estado conhecido | Encaminhamento proposto (ainda não executado) |
|---|---|---|---|---|
| P01 | **Concluída (20/10/2026)** | Teste `test_parametros_diagnosticos` para GMM FULL | Correção de uma asserção `(2, 2)` → `(2, 2, 2)` confirmada no `git diff`; teste específico **1 aprovado**, módulo GMM **31 aprovados** e regressão global **416/416 aprovados em 80,56 s** | **Encerrada.** Ver `P01_REGISTRO_CONCLUSAO_20261020.md`; alteração ainda sem commit/push confirmado |
| P02 | **Concluída (20/10/2026)** | Teste de `openTSNE` anteriormente pulado | A biblioteca era ausente; no PC, `openTSNE` 1.0.4 agora importa, `pip check` sem conflitos; **1 teste específico aprovado** e **7/7 testes do arquivo aprovados** | **Encerrada.** Ver `P02_REGISTRO_CONCLUSAO_20261020.md`; testes não equivalem a experimento OOF completo |
| P03 | **Concluída (20/10/2026)** | Dependências `pyarrow` e `category_encoders` | No PC, `pyarrow` 26.0.0 e `pyarrow.parquet` importam; o código examinado usa encoders internos. Por escolha do usuário, `category-encoders==2.11.1` foi instalado, importou corretamente e `BinaryEncoder: True`; `pip check` sem conflitos; **17 passed in 2.57s** após instalação. | **Encerrada.** Preservar encoder interno da pipeline e manter, por ora, PyArrow 26.0.0; detalhes em `P03_REGISTRO_CONCLUSAO_20261020.md`. |
| P04 | **Correção preparada; aguarda aplicação e testes locais** | `apenas_1d_2d: true` era chave obsoleta não referenciada nos `.py` examinados | Usuário aprovou a correção. Foi preparado YAML com comentários atualizados e chave removida; PyYAML válido e comparação semântica confirmou todos os demais campos iguais aos originais da cópia recebida | Aplicar diff no YAML ativo, executar testes específicos e conferir `git diff`; somente então encerrar. Ver `P04_REGISTRO_CORRECAO_PREPARADA_20261020.md` |
| P05 | A verificar | Auditoria catálogo PDF → pipeline | Quatro PDFs foram recebidos: OOF, GMM, PCA e t-SNE | Conferir para cada gráfico a implementação, acionamento, testes e saída; listar ausentes e divergências |
| P06 | A verificar | Auditoria pipeline → catálogo PDF | Na análise estática anterior foram citadas visualizações adicionais: OOF `06` comparação de hiperparâmetros, t-SNE `14` diagnóstico KL e mapa complementar PCA2D `16` | Verificar se precisam ser documentadas nos PDFs e se correspondem a recursos efetivamente gerados |
| P07 | Parcialmente verificado | Artefatos produzidos nas execuções | `outputs.rar` recebido: 3 ensaios históricos de 17/10, 220 arquivos, 47 PNGs íntegros; documentação de F10 reservado | Confirmar separadamente os artefatos mais recentes 1D/2D/3D, se existirem; não extrapolar estes ensaios PCA/diag para o código atual |
| P08 | Trabalho planejado | Biplot PCA 1D, 2D e 3D | Ainda não identificado como implementado no código/YAML recebidos; não consta dos PDFs de PCA fornecidos | Construir protótipos, avaliar visualmente um a um, integrar só após aprovação; criar testes e atualizar documentação |
| P09 | A verificar | Consistência dos registros históricos com o código atual | Alguns documentos de retomada de 19/10 relatavam pendências que o usuário informou já terem sido resolvidas e comitadas | Tratar documentação antiga como histórica; determinar estado atual por código, testes e saídas; atualizar arquivos `.md` quando aprovado |

| P10 | Divergência observada no artefato histórico | Painel 00 omitido da galeria integrada do ensaio `195252` | Existe `oof/graficos_gerados/00_painel_score_composto.png`, mas não aparece nas 14 entradas de `visualizacao_consolidada/index.html` nem em seu manifesto | Verificar se a galeria atual deveria incorporar o painel e se o código mais recente já corrigiu a omissão; não alterar sem aprovação |
| P11 | A verificar | Evidência recente de gráficos GMM FULL e t-SNE 1D/2D/3D | `outputs.rar` traz apenas ensaios PCA/diag de 17/10, anteriores aos patches posteriores; não possui PNGs dessas variantes atualizadas | Receber saídas atualizadas caso estejam salvas, ou testar futuramente de forma controlada; ausência no pacote antigo não significa defeito |
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
- **P04:** correção documental aprovada pelo usuário e preparada em cópia do YAML, com validade sintática e equivalência de configurações (exceto chave removida) confirmadas. Ainda faltam aplicação e testes no PC. **Próximo passo:** validar P04 localmente; depois P10.

## Resultado da bateria de testes já relatada nesta conversa

- **Execução isolada anterior:** 416 casos coletados; 414 passaram, 1 falhou (P01) e 1 foi pulado (P02).
- **Execução local posterior, em 20/10/2026, com correções/dependências verificadas:** `python -m pytest -q`: **416 passed in 80.56s**, sem skips nem falhas.
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

## Material seguinte esperado

- Confirmar se ainda existem **saídas visuais e HTML de revisões posteriores a 17/10**, fora do `outputs.rar` recebido. O pacote enviado contém ensaios históricos; outros materiais serão solicitados apenas se relevantes.
- Encerrar recebimento quando o usuário confirmar; só então aprovar a lista de ações.

## Política de continuidade

1. Receber os arquivos restantes e registrar novos achados neste inventário.
2. Separar **correções comprovadas**, **verificações pendentes** e **novas implementações**.
3. Apresentar a lista consolidada ao usuário, com prioridades e evidências.
4. Obter aprovação antes de modificar código, testes, YAML ou documentos existentes.
5. Após cada etapa concluída, preparar/atualizar os Markdown de registro, resumo e retomada segundo o padrão verificado em `passos/`.

**Status de execução:** P13, P03, P02, P01 e P14 concluídas (5); P15 dispensada/encerrada por decisão (1), total 6/15 tratadas, 9 abertas. P01 possui regressão 416/416 aprovada e diff verificado; P14 possui coleta posterior à limpeza de 416 casos. Não foi relatada alteração no código-fonte do GMM nem no YAML. **Atual: P04 (diagnóstico, aguardando decisão sobre YAML).**
