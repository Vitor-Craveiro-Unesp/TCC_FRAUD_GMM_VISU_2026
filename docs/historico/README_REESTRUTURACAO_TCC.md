# TCC_EXPERIMENTAL_V3 — Layout 2 e orquestrador mestre

**Pacote gerado em 21/10/2026** a partir do ZIP enviado `TCC_REESTRUTURACAO_20261021_181152.zip`.

## Mudancas

- `src/orquestracao_oof.py --executar` coordena toda a grade OOF e chama automaticamente:
  1. `visualizacao_oof.gerar_graficos_oof(experimento)`;
  2. `visualizacao_geometrica.executar(experimento)`;
  3. `visualizacao_integrada.integrar(experimento, complementar_2d=False)`.
- Os candidatos possuem seus proprios caches, predicoes e metricas. Os resultados do vencedor sao copiados com verificacao de integridade para pasta autonoma. A galeria fica dentro do vencedor.
- A configuracao YAML e o manifesto do ensaio permanecem **na raiz da pasta mae**.
- O GMM vencedor continua selecionado por **score composto**, com AP e ID como desempates.
- Os dados de F10 **nao** participam de OOF, da selecao nem dos graficos de desenvolvimento. `vencedor/teste_final/` existe, mas fica reservado para a avaliacao final independente ainda nao implementada neste passo.
- O ajuste geometrico do vencedor continua acontecendo **uma vez em F1–F9**, sem refazer os 27 modelos OOF. A geometria nao constitui teste F10.
- As figuras e arquivos anteriores nao sao sobrescritos silenciosamente. Falhas posteriores a 06.03 podem ser retomadas com comando dedicado.

## Estrutura resultante (exemplo: PCA 1D; K=2,4,6)

```
outputs/
  ensaio_pca_gmm_AAAAMMDD_HHMMSS/
    configuracao.yaml
    manifesto.json
    candidatos/
      K02_<id>/
        configuracao.json
        metricas/metricas.json
        metricas/por_fold.csv
        oof/cache/F01.npz ... F09.npz
        oof/predicoes.npz
      K04_<id>/
        ...
      K06_<id>/
        ...
    vencedor/
      metricas/metricas.json
      metricas/por_fold.csv
      metricas/resultados_grade.csv
      metricas/melhor_configuracao.json
      oof/predicoes.npz
      visualizacao/
        oof/                     # graficos 00–06, CSV e JSON
        geometria_desenvolvimento/ # GMM/PCA/Biplot, CSV e JSON
        galeria/
          index.html
          manifesto_integracao_07_03.json
          oof/                   # copias das imagens OOF
          geometria_desenvolvimento/ # copias das imagens GMM/PCA
      teste_final/                # reservado: F10 nao executado
```

Os IDs de candidato sao hashes de suas configuracoes, nao valores aleatorios nem necessariamente os mesmos do ensaio anterior.

## Instalacao segura no Windows / PowerShell

1. **Nao apague o ensaio anterior antes de verificar o novo.** Faça backup ou registre o projeto no Git antes de substituir arquivos de codigo. Este pacote nao inclui base de dados nem outputs.
2. Coloque o ZIP corrigido na raiz `TCC_EXPERIMENTAL_V3` (onde existem `src`, `tests` e o YAML).
3. Com backup concluido, execute (substitua o nome real do ZIP):

```powershell
Expand-Archive -LiteralPath ".\TCC_REESTRUTURADO_LAYOUT2_20261021.zip" -DestinationPath "." -Force
```

4. Com `.venv` ativa, confira antes de executar o ensaio:

```powershell
python -m pytest -q tests/test_orquestrador_mestre_layout2.py tests/test_orquestracao_oof.py tests/test_visualizacao_oof.py tests/test_visualizacao_geometrica.py tests/test_visualizacao_integrada.py
python -m src.orquestracao_oof --planejar
```

5. A configuracao enviada ja contem `dados.fracao_base: 0.1`, `pca.n_components: [1]`, `gmm.n_components: [2,4,6]` e `RUS` com razao `16`. Depois de confirmar que estao corretos:

```powershell
python -m src.orquestracao_oof --executar
```

**Esse comando agora executa tambem 07.01, 07.02 e 07.03**, sem comandos manuais separados.

Se a visualizacao falhar por aviso do Matplotlib interpretado como erro pelo PowerShell 5.1, pode ser necessario utilizar `$ErrorActionPreference = 'Continue'` para a invocacao externa, conferindo o valor de `$LASTEXITCODE`. A pipeline nao repete automaticamente a grade OOF nessa situacao.

## Retomada de visualizacoes sem retreinar

Se o OOF ja tiver terminado e o manifesto registrar interrupcao das visualizacoes:

```powershell
python -m src.orquestracao_oof --so-visualizacoes --experimento "outputs\ensaio_pca_gmm_AAAAMMDD_HHMMSS"
```

Nao e necessario reabrir a grade OOF. As etapas anteriores marcadas como concluidas sao puladas; arquivos existentes sao preservados segundo a protecao de cada modulo.

## Como abrir a galeria

```powershell
$experimento = "outputs\ensaio_pca_gmm_AAAAMMDD_HHMMSS"
Start-Process (Resolve-Path "$experimento\vencedor\visualizacao\galeria\index.html").Path
```

## Validacao realizada neste pacote

- 406 testes de regressao/sinteticos do conjunto nao t-SNE aprovados, divididos em dois grupos;
- testes adicionais da grade real K=[2,4,6] **simulada**: 27 caches, vencedor por score, pastas individuais, F10 ausente;
- testes de falha e retomada das visualizacoes, sem refazer OOF;
- verificacao estatica de sintaxe dos modulos.

**Limites da validacao:** o arquivo `data/raw/dataset_raw.parquet` e os folds reais nao foram fornecidos neste ZIP; nao foi possivel rodar o ensaio completo sobre 166.716 registros neste ambiente. Os testes de t-SNE nao foram executados aqui (openTSNE indisponivel). O desempenho visual dos 16 graficos no novo layout deve ser verificado no novo ensaio real no computador local.

## Seguranca e escopo

- A secao estatistica do YAML (PCA, GMM, RUS, score, folds) foi preservada. Apenas a declaracao das pastas de saida e a descricao da origem geometrica foram atualizadas.
- Nao executar F10 ate existir uma etapa propria de avaliacao independente. `teste_final/` vazio e esperado.
- Os arquivos de codigo antigos que leem `oof/` na raiz **nao funcionam** com ensaios no layout 2; atualizados os quatro geradores/orquestrador e os utilitarios incluidos.
- Os ensaios anteriores no layout 1 nao sao migrados por este pacote e podem continuar arquivados separadamente.
