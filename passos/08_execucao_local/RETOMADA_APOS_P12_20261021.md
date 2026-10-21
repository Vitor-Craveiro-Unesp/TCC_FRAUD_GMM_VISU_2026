# Etapa 08 — Ponto de parada após conclusão documental da P12

**Data:** 21/10/2026. **Nenhum experimento real OOF/F10 foi iniciado nesta preparação.**

## Trabalho já confirmado pelo autor

- P05: patch comparativo do GMM 08 aplicado uma única vez; `git diff --check` retornou 0; suíte total **434 passed in 95.17s**.
- P06: pacote de **10 arquivos** conferido por SHA-256: quatro PDFs, três PNGs **simulados**, três registros Markdown.
- P09: `passos/P09_ESTADO_CONSOLIDADO_E_INDICE_20261021.md` instalado e verificado por SHA-256.
- P12: compatibilidade do nome `teste_final` esclarecida. **`teste_final/graficos_gerados` é geometria F1–F9**, não avaliação F10. Mantemos o caminho histórico até haver migração coordenada.
- Migração de `transferencia_ias_1`: 20 arquivos da pasta + dois extras raiz com cópias verificadas à época. Último `git status` informado ainda exibia `?? transferencia_ias_1/`; verificar no PC se já foi excluída.
- ZIPs P05/P06/P09 podem permanecer na raiz apenas enquanto úteis; não incluir cegamente em commit.

## Intenção do ensaio para a próxima sessão

O usuário quer reproduzir **todas as etapas experimentais** do TCC com **somente** `dados.fracao_base` reduzida no YAML e os demais parâmetros mantidos: 9 folds OOF de desenvolvimento F1–F9, seleção e congelamento do vencedor e τ* pelo OOF, diagnósticos, galeria e **uma avaliação independente no F10 completo**. Com fração inferior a 1, o experimento é **ensaio amostral**, não resultado oficial da base total. F10 não é reduzido nem usado para escolher o modelo.

## Sequência de retomada — sem executar automaticamente

1. Confirmar conteúdo atual de `src/`, `tests/`, YAML, `.venv`, alterações locais e **estado Git**; registrar hash dos inputs, arquivo de folds e configuração. `git status --short` e `git diff --check` são comandos de inspeção, não executar experimentos.
2. Confirmar valor desejado de `dados.fracao_base`, a preservação dos demais parâmetros e o método ativo. No snapshot prévio, PCA `[2,3]` × GMM `[1,2]` = 4 candidatos, **36 avaliações de fold-candidato**; `python -m src.orquestracao_oof --planejar` deve conferir o número real no PC.
3. Fazer preflight de RAM, CPU, disco, tempo e contagens por classe e fold após amostragem; verificar todas as proteções de F10.
4. Antes de tentar avaliação F10, **implementar e homologar** o módulo independente que falta, com testes sintéticos negativos de vazamento e verificação dos artefatos. Isso exige autorização específica para mudança de código; não chamar os gráficos `teste_final` existentes de resultados F10.
5. Executar OOF real amostral com os 9 folds por candidato somente após planejamento e aprovação; validar completos os arquivos/manifests/métricas, ranking AP/MCC/LogLoss e limiar τ* exclusivamente OOF.
6. Executar visualizações OOF, ajuste geométrico sobre F1–F9 e galeria, conferir os PNGs GMM 08 2D/3D comparativos e gráficos PCA condicionais à dimensionalidade.
7. Com configuração e limiar congelados, aplicar o modelo treinado em F1–F9 a F10 inteiro **sem fit**, gerar predições e métricas de teste em arquivos exclusivos F10, auditar independência e a origem de cada saída.
8. Registrar resultados reais, possíveis problemas e hashes; diferenciar explicitamente ensaio amostral e teste final independente. Não concluir desempenho oficial antes de rodar toda a base de desenvolvimento.

## Não confundir

- **434 testes unitários/integrados passaram:** não são 434 execuções reais de modelos.
- **4 PDFs P06 atualizados:** seus exemplos adicionais são simulados, não evidência de resultados com dados do TCC.
- **P12 documentalmente encerrada:** não significa módulo F10 implementado.
- **OOF F1–F9:** não é a validação externa F10.

## Comandos iniciais sugeridos para a próxima sessão

```powershell
git status --short
python -m src.orquestracao_oof --planejar
```

**Ponto de parada:** sem alteração de produção nesta etapa documental; próximo bloco de trabalho é engenharia/teste de avaliação F10 e planejamento do ensaio com autorização, seguindo a configuração do usuário.
