# TCC_EXPERIMENTAL_V3 — Pendências acumuladas durante o recebimento dos arquivos

**Data de abertura:** 2026-10-20  
**Situação:** registro provisório, em construção. **Não executar correções até concluir o recebimento/análise dos arquivos e obter aprovação.**

## Critérios de registro

- **Divergência observada**: comportamento ou expectativa inconsistente encontrado na análise já relatada.
- **A verificar**: indício ou limitação que não autoriza concluir que existe defeito.
- **Trabalho planejado**: funcionalidade ainda a desenvolver, sem caracterizá-la como bug.
- Registrar evidências (arquivo, teste, saída e resultado) antes de alterar qualquer código.
- Não confundir o resultado de teste em ambiente isolado com execução no computador do autor.

## Pendências conhecidas até o momento

| ID | Categoria | Item | Evidência/estado conhecido | Encaminhamento proposto (ainda não executado) |
|---|---|---|---|---|
| P01 | Divergência observada | `tests/test_gmm.py::test_parametros_diagnosticos` | Na execução isolada anteriormente relatada, a asserção esperava `covariances_.shape == (2, 2)`, mas com `covariance_type='full'` o retorno foi `(2, 2, 2)` | Conferir intenção original do teste e a configuração atual; ajustar a expectativa **somente se confirmado e aprovado** |
| P02 | A verificar | Teste de `openTSNE` pulado | Na execução isolada anteriormente relatada, faltava `openTSNE`; 1 teste foi marcado como pulado. A biblioteca também não foi identificada no `.venv.rar` recebido, embora conste em `requirements.txt` | Conferir instalação e execução no ambiente real/previsto; não atribuir falha à pipeline sem evidência |
| P03 | A verificar | Dependências `pyarrow` e `category_encoders` | Apontadas como ausentes no ambiente isolado de auditoria. No `.venv.rar` recebido, `pyarrow` foi identificado na versão 26.0.0 (contra 25.0.1 do `requirements.txt`), mas `category_encoders` não foi identificada | Comparar com `requirements.txt` e ambiente do projeto; não presumir ausência no computador do autor |
| P04 | A verificar | Configuração dimensional da geometria e gráficos 3D | Análise preliminar do YAML indicou configurações de geometria 1D/2D e geradores com representação 3D | Examinar chamadas e regras concretas no código, testes e artefatos; só classificar como problema se houver incompatibilidade real |
| P05 | A verificar | Auditoria catálogo PDF → pipeline | Quatro PDFs foram recebidos: OOF, GMM, PCA e t-SNE | Conferir para cada gráfico a implementação, acionamento, testes e saída; listar ausentes e divergências |
| P06 | A verificar | Auditoria pipeline → catálogo PDF | Na análise estática anterior foram citadas visualizações adicionais: OOF `06` comparação de hiperparâmetros, t-SNE `14` diagnóstico KL e mapa complementar PCA2D `16` | Verificar se precisam ser documentadas nos PDFs e se correspondem a recursos efetivamente gerados |
| P07 | Parcialmente verificado | Artefatos produzidos nas execuções | `outputs.rar` recebido: 3 ensaios históricos de 17/10, 220 arquivos, 47 PNGs íntegros; documentação de F10 reservado | Confirmar separadamente os artefatos mais recentes 1D/2D/3D, se existirem; não extrapolar estes ensaios PCA/diag para o código atual |
| P08 | Trabalho planejado | Biplot PCA 1D, 2D e 3D | Ainda não identificado como implementado no código/YAML recebidos; não consta dos PDFs de PCA fornecidos | Construir protótipos, avaliar visualmente um a um, integrar só após aprovação; criar testes e atualizar documentação |
| P09 | A verificar | Consistência dos registros históricos com o código atual | Alguns documentos de retomada de 19/10 relatavam pendências que o usuário informou já terem sido resolvidas e comitadas | Tratar documentação antiga como histórica; determinar estado atual por código, testes e saídas; atualizar arquivos `.md` quando aprovado |

| P10 | Divergência observada no artefato histórico | Painel 00 omitido da galeria integrada do ensaio `195252` | Existe `oof/graficos_gerados/00_painel_score_composto.png`, mas não aparece nas 14 entradas de `visualizacao_consolidada/index.html` nem em seu manifesto | Verificar se a galeria atual deveria incorporar o painel e se o código mais recente já corrigiu a omissão; não alterar sem aprovação |
| P11 | A verificar | Evidência recente de gráficos GMM FULL e t-SNE 1D/2D/3D | `outputs.rar` traz apenas ensaios PCA/diag de 17/10, anteriores aos patches posteriores; não possui PNGs dessas variantes atualizadas | Receber saídas atualizadas caso estejam salvas, ou testar futuramente de forma controlada; ausência no pacote antigo não significa defeito |
| P12 | Melhoria documental opcional | Nome da pasta `teste_final` é ambíguo | Manifesto interno declara `desenvolvimento_F1_F9`, `f10_utilizado: false` | Avaliar nota/renomeação futura apenas se necessário, preservando a compatibilidade e sem sugerir que F10 foi usado |

| P13 | A verificar | Pacotes da `.venv` diferem de `requirements.txt` | `.venv.rar` tem metadados de 27 pacotes; `requirements.txt` fixa 102 dependências; 76 não foram identificadas no pacote e 10 das 26 encontradas divergem em versão | Verificar se o arquivo compactado está completo e qual ambiente o VS Code usa efetivamente; não alterar dependências antes da confirmação |
| P14 | A verificar | Cache pytest possui quatro casos adicionais | `.pytest_cache.rar` registra 420 nodeids, enquanto o `tests.rar` atual coletou 416; 4 nodeids constam somente no cache, e `lastfailed` está vazio | Identificar se refletem testes antigos/renomeados; o cache não prova êxito da suíte atual nem exige correção automática |

| P15 | A verificar (metodologia; sem evidência de defeito) | Adequação da estratégia estratificada e embaralhada diante de possíveis dependências temporais/entre entidades | YAML atual e manifesto de folds usam divisão estratificada, embaralhamento e semente 42; o YAML já registra ressalva sobre dependências temporais e grupos. Integridade dos arquivos de folds foi confirmada em `data.rar` | Apenas avaliar o critério metodológico à luz das colunas e do objetivo de previsão, sem refazer a divisão nem mexer em F10 sem decisão explícita |

## Resultado da bateria de testes já relatada nesta conversa

- 416 casos coletados; **414 passaram, 1 falhou e 1 foi pulado** na execução isolada relatada.
- A falha identificada está descrita em **P01**; o teste pulado, em **P02**.
- Esse registro não significa que os scripts que exigem a base real foram executados.

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

**Nenhuma correção executada por este documento.**
