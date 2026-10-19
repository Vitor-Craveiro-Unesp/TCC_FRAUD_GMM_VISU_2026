# TCC_EXPERIMENTAL_V3 — Registro de retomada (19/10/2026)

## Onde retomaremos

**Próxima tarefa: GMM 07 — Sensibilidade da fronteira ao limiar.**
Comparar a classificação pelo corte fixo **0,50** com o corte ótimo **MCC OOF**, em conformidade com o PDF de gráficos GMM. Primeiro inspecionar a especificação original do PDF, depois verificar diferenças em relação ao GMM 08 aprovado, evitando redundâncias e alterações sem confirmação. Trabalhar um gráfico por vez, com prévia, testes e aprovação visual antes de continuar.

**Não iniciar fase 08 nem rodadas reais com F10 na retomada.**

## Decisões metodológicas vigentes

- Pipeline: um único **GMM não supervisionado**; a estimativa de risco por componente usa os rótulos somente do treino e responsabilidades soft; probabilidade final: `p(fraude | x) = Σ_k γ_k(x) RH_k`.
- Risco histórico: `RH_k = Σ_i γ_ik y_i / Σ_i γ_ik` (a suavização configurável deve ser respeitada quando diferente de zero).
- O GMM não reserva uma gaussiana para fraude e as demais para legítimas; `n_components` é a quantidade **total** de gaussianas.
- **Covariância `full`** definida para os **próximos experimentos**, conforme catálogo PDF. **O ensaio antigo PCA 10% foi treinado com `diag`**, e não deve ser apresentado como FULL. Apenas as prévias sintéticas FULL representam a geometria revisada.
- Rotulagem **apenas descritiva** de cada gaussiana em todos os diagnósticos/relatórios: **alto risco** se `RH_k >= 2 × prevalência do treino usado para calcular RH`; caso contrário, **baixo risco**. Não usar `RH_k >= 0,50` para o rótulo de componente. Essa rotulagem não altera scores, responsabilidades soft, predição ou a seleção do limiar das transações.
- **Decisão de transações:** cortes 0,50 (fixo) e τ* (ótimo MCC OOF). Não confundir com a classificação descritiva de gaussianas.
- Avaliação OOF no desenvolvimento F1–F9; F10 continua reservado e intocado. Métricas principais AP, MCC e LogLoss; score composto `[AP + (MCC+1)/2 + exp(-LogLoss)]/3`.
- Métricas de prévias sintéticas devem ser identificadas expressamente como **simuladas**, e não como MCC OOF real.

## Gráficos aprovados / revisados

| Gráfico | Situação |
|---|---|
| 00 — Painel do score composto | Aprovado |
| 01 — Curva Precision–Recall | Aprovado |
| 02 — MCC por limiar | Aprovado |
| 03 — Matriz de confusão com corte 0,50 | Aprovado |
| 04 — Matriz de confusão com limiar MCC | Aprovado |
| 05 — Distribuição das probabilidades | Aprovado |
| 06 — Comparação de hiperparâmetros | Aprovado |
| GMM 09 — Painel de parâmetros gaussianos | Aprovado, 1D/2D/3D sintéticos; FULL e rodapé com prevalência/2× prevalência |
| GMM 08 — Risco e geometria 95% em 1D | Aprovado: dois cortes lado a lado; mesmas escalas; legenda; regiões alteradas em amarelo |
| GMM 08 — Risco e geometria 95% em 2D | Dois mapas sintéticos revisados (corte 0,50 e MCC); usuário pediu avançar |
| GMM 08 — Risco e geometria 95% em 3D, MCC | Aprovado, incluindo legenda G1–G4, três planos e dimensão fixada |
| GMM 08 — Risco e geometria 95% em 3D, corte 0,50 | Aprovado (último print) |
| **GMM 07 — Sensibilidade da fronteira ao limiar** | **PRÓXIMO — ainda não revisado/implementado** |

### Detalhes do GMM 08

- Versão 1D compara dois cortes na mesma figura. Linha de fronteira indica onde o **score cruza o limiar**; o limiar 0,50 pode não possuir fronteira visível.
- Em 2D, dois mapas exibem campo de risco e contornos gaussianos FULL de 95%; hachura amarela marca regiões cuja classificação mudou entre os cortes.
- Em 3D, apresentar planos **PC1×PC2, PC1×PC3, PC2×PC3**, com a dimensão omitida fixada na média do desenvolvimento (cortes 2D, não uma fronteira 3D global projetada), além da identificação de gaussianas. A barra de cores precisa ficar visível.
- Os pontos plotados são projeções de observações; o mapa de risco é calculado numa malha com a terceira dimensão fixada. Essa distinção deve constar nas legendas.
- O rodapé das prévias sintéticas foi corrigido para **MCC da simulação**, não MCC OOF real.
- **Fronteira para 0,50 ausente** quando `max score` no plano/corte é menor que 0,50: isso é correto, não erro.
- A regra `RH_k >= 2 × prevalência` pertence ao painel descritivo dos componentes; as fronteiras de transações seguem τ=0,50 e τ* MCC.

## Últimos testes relatados pelo usuário

- Após o painel GMM 09 e a migração de testes: `60 passed in 41.19s`.
- Após o patch comparativo 1D GMM 08: `60 passed in 43.19s`.
- Para as prévias GMM 08: testes selecionados passaram; o refinamento 3D foi avaliado visualmente por prints, mas **não foi apresentado nesta conversa um resultado da suíte completa de testes no Windows após esse último patch**. Reexecutar antes de commit, se desejado.
- Aviso `LF will be replaced by CRLF` do Git é não fatal e separado dos resultados do pytest.

## Arquivos principais / mudanças locais

- `src/visualizacao_oof.py` — gráficos 00–06.
- `src/visualizacao_geometrica.py` — GMM 09, GMM 08, outros diagnósticos.
- `src/visualizacao_integrada.py` — galeria e mapa complementar.
- `config_TCC_EXPERIMENTAL_V3_atualizado.yaml` — configuração central; atualmente `gmm.covariance_type: "full"`, regra descritiva 2× prevalência.
- `tests/test_visualizacao_oof.py`, `tests/test_visualizacao_geometrica.py`, `tests/test_visualizacao_integrada.py`, `tests/test_painel_gaussianas_2x.py`.
- `passos/08_execucao_local/plano.md`, `passos/08_execucao_local/RETOMADA.md`, `passos/CONTEXTO_GERAL.md` — planos/documentação, podem conter modificações ainda não commitadas.
- Último patch aplicado pelo usuário e confirmado visualmente: `GMM08_refinamento_3D_legenda_margens_simulacao_TCC.patch`.
- Ensaio histórico: `outputs/ensaio_pca_gmm_20261017_195252/`. Os arquivos OOF aprovados continuam nesse ensaio. **Não sobrescrever resultados ou metadados desse ensaio antigo com FULL.**

## Procedimento de retomada do GMM 07

1. Conferir `git status --short` e, opcionalmente, testes dirigidos. Garantir checkpoint antes de nova intervenção.
2. Abrir o **catálogo PDF GMM**, localizar a especificação **GMM 07 — Sensibilidade da fronteira ao limiar** e listar campos/figuras obrigatórios sem acrescentar conteúdo imaginado.
3. Comparar o GMM 07 com GMM 08 para definir o elemento adicional, sem reproduzir desnecessariamente o mapa já aprovado.
4. Inspecionar as versões **locais atuais** de `visualizacao_geometrica.py`, `visualizacao_integrada.py`, YAML e testes antes de preparar qualquer patch incremental. Se os arquivos atuais não estiverem presentes, pedi-los ao usuário.
5. Gerar prévia sintética sem F10; classificar resultados de simulação como tais, validar testes e solicitar print/aprovação.
6. Não executar novo experimento FULL na base real antes de finalizar toda a revisão visual.

## Histórico de ambiente

- Projeto Windows: `C:\Users\Vitor Craveiro\Desktop\ClusterPosMac\MEU PROJETO\TCC_EXPERIMENTAL_V3`.
- Usar sempre `& .\.venv\Scripts\python.exe` no PowerShell para acessar o ambiente correto.
- Não usar `-m src.visualizacao_geometrica --experimento ...` sem `--planejar` ou `--executar`; `--executar` reajusta modelo, não usar por engano para prévias.

## Como salvar no repositório Git

Após baixar este MD em `passos/RETOMADA_GRAFICOS_TCC_20261019.md`, revisar `git status --short`, executar a suíte de testes atual, então salvar os arquivos rastreados com `git add -u` e adicionar explicitamente os **novos arquivos** de documentação e testes. Conferir `git diff --cached --name-only` antes de `git commit`. O assistente **não tem acesso direto ao repositório local Windows**, portanto este registro não substitui o commit que o usuário executa no computador.
