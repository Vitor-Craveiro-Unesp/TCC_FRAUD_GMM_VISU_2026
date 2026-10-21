# P09 — Estado consolidado e precedência dos registros (21/10/2026)

## Objetivo e alcance

Este registro **atualiza o estado da documentação** das 15 pendências após P05 e P06. Preserva os MD antigos como histórico e **não** implica revisão linha a linha de todos os documentos ou execução real da Etapa 08. Onde documentos antigos divergem, prevalecem as evidências locais mais recentes registradas a seguir.

## Evidências locais informadas pelo usuário

1. P08: `python -m pytest -q`: **432 passed in 84.66s** (20/10/2026).
2. P05: patch GMM08 aplicado uma vez; `git diff --check`: código de saída **0**; suíte completa `python -m pytest -q`: **434 passed in 95.17s** (21/10/2026). Sem prova de treino OOF real/F10.
3. P06: **10/10 arquivos instalados e conferidos via SHA-256**, incluindo quatro PDFs atualizados em `passos/07_visualizacao/catalogos_atualizados/`, três PNGs simulados em `outputs/prototipos_catalogos_p06/` e três MDs de registro em `passos/07_visualizacao/revisao_graficos/` (21/10/2026).
4. Migração da pasta temporária: inventário **20 arquivos da pasta + dois da raiz**, cópias SHA-256 idênticas no momento da conferência. Imagens/status posteriores mostraram `transferencia_ias_1/` ainda existente; confirmar limpeza final no computador, sem supor que foi excluída.
5. O último `git status --short` enviado mostrava alterações não commitadas em src, tests e YAML, além de arquivos `passos/` não rastreados; não há prova de commit ou push.

## Estados atualizados das 15 pendências

| ID | Estado em 21/10/2026 | Limitação/observação |
|---|---|---|
| P01 | Correção de teste de GMM FULL integrada e validada na suíte | Regra matemática do GMM não alterada |
| P02 | Ambiente openTSNE 1.0.4 validado | Sem rodada t-SNE com dados reais da Etapa 08 |
| P03 | Dependências `pyarrow` e `category-encoders` conferidas | Sem troca silenciosa dos encoders usados |
| P04 | Removida configuração YAML obsoleta e testes aprovados | Manter histórico do patch |
| P05 | **Patch GMM08 aplicado, 434/434 testes da suíte aprovados** | Comparativos em dados reais ainda não executados |
| P06 | **Quatro catálogos PDF atualizados e 10/10 arquivos conferidos** | Suplementos gráficos simulados, não resultados reais |
| P07 | Inventário das saídas históricas encerrado | Saídas de 17/10 não comprovam corrida FULL atual |
| P08 | Biplots PCA integrados; P05 não causou regressão automatizada | Saídas reais de GMM/PCA pendentes da Etapa 08 |
| P09 | **Este índice consolidado atualizado** | Históricos mantidos, sem reescrita linha a linha |
| P10 | Galeria histórica revisada com painel OOF 00 | Não constitui execução F10 |
| P11 | Pendência de imagens inexistentes encerrada documentalmente | Não voltar a exigir imagens que não existem |
| P12 | Análise prévia: `teste_final` contém geometria de F1–F9 | Resolver/verificar requisitos antes da avaliação efetiva F10 |
| P13 | `.venv` Python 3.13.3 e pacotes conferidos | Diagnóstico preservado em `passos/auditorias_20261020/` |
| P14 | Cache pytest histórico reconciliado | Referência atual 434 testes da suíte |
| P15 | Dispensada por decisão do autor | Manter divisão estratificada estabelecida no YAML |

**Interpretação:** o conjunto das 15 pendências está tratado ou esclarecido até o escopo descrito; **isso não significa** que a Etapa 08 ou teste final F10 foi executado. P12 e os portões da Etapa 08 permanecem críticos.

## Arquivos de referência vigente

- `passos/07_visualizacao/revisao_graficos/P05_RESULTADO_VALIDACAO_LOCAL_20261021.md` — 434 testes e patch P05.
- `passos/07_visualizacao/revisao_graficos/P06_AUDITORIA_PIPELINE_PARA_PDFS_ATUALIZADA_20261021.md` — recursos adicionais do código.
- `passos/07_visualizacao/revisao_graficos/P06_REGISTRO_ATUALIZACAO_DOS_QUATRO_CATALOGOS_20261021.md` — PDFs e suplementos simulados.
- `passos/07_visualizacao/catalogos_atualizados/` — quatro PDFs atualizados.
- `passos/07_visualizacao/revisao_graficos/P12_ESCLARECIMENTO_TESTE_FINAL_VERSUS_DESENVOLVIMENTO_20261020.md` — se tiver sido extraído do pacote anterior.
- `passos/08_execucao_local/RETOMADA_ETAPA_08_20261021.md` — se tiver sido extraído do pacote anterior, **atualizar suas referências históricas 432→434 e P05/P06** por leitura deste índice, sem sobrescrever o documento original.
- `passos/auditorias_20261020/versoes_pendencias/` — versões antigas de listas preservadas apenas por rastreabilidade.

## Divergências importantes dos documentos antigos

- O índice `P09_ESTADO_CONSOLIDADO_E_INDICE_20261020.md`, caso exista, descreve P05 e P06 como apenas *auditorias estáticas*. Está superado neste ponto pela aplicação e testes P05 e pela entrega dos quatro PDFs P06.
- A retomada antiga diz que a P05 ainda precisaria de correções GMM08: **isso foi feito, e a suíte passou com 434 testes**.
- O nome `teste_final/graficos_gerados` não prova avaliação F10: no snapshot auditado trata-se de geometria de desenvolvimento F1–F9; é preciso verificar/implementar rotina explícita F10 antes de usá-lo para decisões finais.
- A classificação dos três PNGs `SIMULADO` em `outputs/prototipos_catalogos_p06/` jamais deve ser confundida com figuras de resultados reais.
- Pode haver `P08_REGISTRO_INTEGRACAO_VALIDADA_20261020.md` diretamente sob `passos/07_visualizacao/` segundo `git status`; verificar sua localização antes de deduplicar, não mover automaticamente.

## Limites e critérios da próxima etapa

1. **P12:** tornar inequívoca a separação entre diagnósticos de desenvolvimento e inferência/teste independente F10, preservando compatibilidade dos caminhos enquanto necessário.
2. **Etapa 08:** usuário quer testar o fluxo ponta a ponta com `dados.fracao_base` reduzida no YAML, preservando demais configurações. OOF em F1–F9, seleção/limiar congelados e avaliação F10 completo, nunca usado para treinamento, seleção ou ajuste de limiar.
3. Não alterar `src/`, `tests/`, YAML, outputs de experimentos reais nem acionar F10 pela P09. Suíte automatizada aprovada é evidência de regressão, não de validade experimental.
4. Antes de commit/push: inspecionar `git status --short`, `git diff --check`, e decidir inclusão seletiva de arquivos; evitar `git add -A` com ZIPs/protótipos não rastreados.

## Situação P09

**Documento de precedência atualizado e disponibilizado para instalação.** Sua instalação local deve ser confirmada via SHA-256. Não foi feita modificação automática de outros MDs existentes no computador do usuário.
