# Passo 07.01 — Gráficos preditivos OOF do vencedor

- **Escopo:** 6 primeiros tipos da seção `visualizacao.graficos` do YAML oficial.
- **Status:** implementação criada; testes sintéticos locais aprovados (6 testes). Validação no ambiente do TCC ainda pendente.
- **Fonte:** `configuracao.yaml` da execução já finalizada, `oof/resultados_grade.csv`, `oof/melhor_configuracao.json` e `oof/predicoes/<vencedor>.npz`.
- **Candidato do ensaio 10% (relatado pelo usuário):** `0d74604010afa3`; PCA=1, GMM=4; score=0,501178. O código NÃO fixa esse vencedor; lê a seleção de cada execução.
- **Operação:** somente leitura de artefatos OOF; saída em `oof/graficos_gerados/`. Não treina nem lê F10.
- **Gráficos:** (01) curva PR/AP, (02) MCC por limiar com ponto congelado, (03) matriz limiar 0,5, (04) matriz limiar MCC, (05) distribuição dos scores, (06) comparação das configurações.
- **Pré-condições:** a execução 06.03 terminou; o experimento contém os arquivos e métricas consistentes; `matplotlib` disponível no `.venv`.
- **Tarefas seguintes:** 07.02 (nove tipos geométricos e gaussianas usando refit do vencedor no desenvolvimento); 07.03 (integração automatizada + teste de regressão); 08 (teste final F10 independente).
- **Critério para concluir:** os testes passam no projeto local, as figuras abrem, os resultados são preservados, `melhor_configuracao.json` aponta o mesmo candidato e nenhum dado F10 é usado.

## Comandos de verificação

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_visualizacao_oof.py -q
& .\.venv\Scripts\python.exe -m src.visualizacao_oof --experimento ".\outputs\ensaio_pca_gmm_20261017_195252"
```
