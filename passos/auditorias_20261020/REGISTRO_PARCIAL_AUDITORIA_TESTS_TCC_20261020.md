# TCC_EXPERIMENTAL_V3 — Registro parcial de auditoria de `tests.rar`

**Data de análise:** 20/10/2026  
**Situação:** inventário, inspeção estática e execução de `pytest` em **cópia isolada**, usando os arquivos enviados nesta conversa.  
**Resultado desta sessão:** 416 casos coletados; **414 passaram, 1 falhou e 1 foi pulado**. Não houve modificação dos arquivos originais do projeto.

## 1. Fontes efetivamente examinadas

- `tests.rar`: **72 arquivos regulares** extraídos, além de 2 entradas de diretório do arquivo compactado; **36 arquivos Python**: 24 arquivos `test_*.py`, 11 scripts `verificar_*_base_real.py` e um utilitário `gerar_graficos_ensaio_oof.py`. Os demais arquivos são caches de bytecode; não foram usados como fonte de verdade.
- `src.rar`: 21 módulos Python enviados anteriormente e copiados para o ambiente isolado de verificação.
- `zips.rar`: `config_TCC_EXPERIMENTAL_V3_atualizado.yaml` e `requirements.txt` previamente recebidos; foram copiados para o ambiente isolado.
- PDFs recebidos anteriormente: catálogos OOF, GMM, PCA e t-SNE, usados como referências documentais, **não** como prova de execução.

**Local de teste:** `/mnt/data/tcc_auditoria_exec_20261020/` — cópia separada dos arquivos enviados. Não foi utilizado o computador do usuário nem executada a pipeline sobre a base real.

## 2. Coleta e execução reproduzidas nesta sessão

- A coleta sem executar os testes retornou **416 testes** nos 24 arquivos `test_*.py`.
- A execução da suíte `pytest` terminou em **43,24 segundos** com **414 aprovados, 1 falha, 1 pulado**.
- Arquivo da coleta: `/mnt/data/tcc_pytest_collection_20261020.log`.
- Arquivo da execução: `/mnt/data/tcc_pytest_exec_20261020.log`.
- Comando reproduzido na cópia, com caminhos e opções:

```bash
cd /mnt/data/tcc_auditoria_exec_20261020
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=/mnt/data/tcc_auditoria_exec_20261020 \
python -m pytest -q -p no:cacheprovider --disable-warnings --tb=short --maxfail=8 tests
```

**Limitação ambiental:** o ambiente de auditoria possui versões diferentes das fixadas em `requirements.txt`; portanto, este resultado **não equivale** à execução no `.venv` do computador do usuário. Neste ambiente, `openTSNE`, `pyarrow` e `category_encoders` não estavam instalados. Não se instalaram pacotes e não se executaram os onze scripts de verificação com a base real.

## 3. Falha confirmada e interpretação

**Caso:** `tests/test_gmm.py::test_parametros_diagnosticos`  
**Local:** linha 186 do arquivo recebido.

O teste exige:

```python
assert gmm.modelo_.covariances_.shape == (2, 2)
```

Entretanto, no YAML recebido está `gmm.covariance_type: "full"`; foram ajustados 2 componentes GMM sobre 2 features. O modelo produziu `covariances_.shape == (2, 2, 2)`; com `full`, isso corresponde a duas matrizes de covariância 2 × 2.

**Conclusão limitada:** a condição de teste recebida aparenta refletir uma expectativa do tipo de covariância `diag`, incompatível com a configuração corrente `full`. A falha **não comprova erro numérico nem erro de treinamento**. É preciso revisar a expectativa desse teste; não alterá-lo sem aprovação do usuário. Uma futura correção pode validar a forma de acordo com o `covariance_type` efetivo.

## 4. Caso pulado e interpretação

**Caso:** `tests/test_tsne_pipeline_02_13_14.py::test_02_openTSNE_real_se_disponivel`.

O script usa `pytest.importorskip('openTSNE')`. A biblioteca não está instalada no ambiente isolado, de modo que o caso foi pulado. **Isso não prova defeito na implementação t-SNE nem valida uma execução real do openTSNE.** Outros testes da visualização t-SNE usaram dados sintéticos e/ou substituições controladas.

## 5. Cobertura identificada na suíte — resultados de `pytest`

| Arquivo | Casos coletados | Escopo reconhecível pelo código |
|---|---:|---|
| `test_amostragem.py` | 23 | Amostragem NORMAL/RUS |
| `test_gmm.py` | 31 | GMM e responsabilidades; **1 falha de forma da covariância** |
| `test_imputacao_normalizacao.py` | 10 | Imputação e normalização |
| `test_metricas_oof.py` | 42 | Métricas OOF e seleção |
| `test_orquestracao_oof.py` | 29 | Orquestração e arquivos OOF |
| `test_painel_gaussianas_2x.py` | 7 | Rótulos e painel descritivo das gaussianas |
| `test_pca11_scree_pdf.py` | 3 | PCA 11, scree plot |
| `test_pca12_2d3d_pdf.py` | 10 | PCA 12, projeção em 2D/3D |
| `test_pca12_histograma_pdf.py` | 6 | PCA 12, histograma 1D |
| `test_preflight_oof.py` | 19 | Verificações prévias OOF |
| `test_preprocessing.py` | 7 | Pré-processamento |
| `test_probabilidade.py` | 40 | Probabilidade final e riscos |
| `test_reducao_pca.py` | 18 | Redutor PCA |
| `test_reducao_tsne.py` | 17 | Redutor t-SNE em testes controlados |
| `test_risco.py` | 42 | Risco histórico por gaussiana |
| `test_tsne13_histograma_pdf.py` | 7 | t-SNE 13, embedding 1D |
| `test_tsne_pipeline_02_13_14.py` | 7 | t-SNE 02/13-3D/14; **1 pulado** |
| `test_validacao_oof.py` | 20 | Validação OOF, isolamento de F10 |
| `test_visualizacao_geometrica.py` | 9 | Geometria PCA/GMM e controles de execução |
| `test_visualizacao_gmm01.py` | 7 | GMM 01 1D/2D/3D |
| `test_visualizacao_gmm04.py` | 8 | GMM 04 1D/2D/3D |
| `test_visualizacao_gmm07.py` | 10 | GMM 07 1D/2D/3D |
| `test_visualizacao_integrada.py` | 10 | Integração, galerias e mapas complementares |
| `test_visualizacao_oof.py` | 34 | Geradores OOF, incluindo painel, PR, MCC, scores e confusão |
| **Total** | **416** | **414 passaram, 1 falhou, 1 pulado** |

Nota: casos coletados incluem parametrizações de funções de teste; não confundir 416 com o número de funções Python `test_` (309 funções antes da expansão de parâmetros).

## 6. Estado do Biplot PCA

- Não se encontrou ocorrência textual de `biplot` nos arquivos Python de `tests.rar` nem nos 21 `.py` de `src.rar`.
- Também não havia chave `biplot` no YAML recebido, de acordo com a auditoria anterior.
- Portanto, nenhum teste do Biplot 1D, 2D e 3D foi executado; a criação, revisão visual, aprovação e integração **continuam pendentes**.
- Para o Biplot 1D, definir uma adaptação matematicamente defensável antes de estabelecer o teste de regressão e a representação visual.

## 7. Proibições e diferenciações de evidência

1. **414 testes passaram em cópia isolada**, e não no `.venv` do Windows.
2. **Um teste falhou por uma condição aparentemente desatualizada**, ainda não ajustada.
3. **Um teste real de `openTSNE` foi pulado**, não aprovado.
4. Não se executou nenhum `verificar_*_base_real.py`; nenhuma base real, F10 ou experimento final foi processado nesta sessão.
5. Testes automatizados bem-sucedidos não substituem a aprovação visual nem provam que todas as saídas de `outputs/` correspondem ao código enviado.
6. Nenhum arquivo original enviado foi corrigido ou sobrescrito durante esta auditoria.

## 8. Próximos passos acordados

1. Solicitar `outputs.rar` para verificar a **existência efetiva** e o tipo dos artefatos gerados nas execuções históricas. Não supor que um arquivo exista apenas por estar previsto nos geradores.
2. Conferir, quando necessário, se resultados e documentação se referem à mesma execução/versão.
3. Antes de integrar o Biplot, desenvolver e apresentar prévias **1D → 2D → 3D** para avaliação visual e aprovação do usuário.
4. Depois da integração e seus testes, realizar a auditoria **PDF → pipeline e pipeline → PDF**, incluindo visualizações adicionais identificadas no código.
5. Após cada passo aprovado, criar/atualizar registros Markdown conforme a arquitetura documental existente, permitindo continuidade em novos chats.

**Regra imutável da colaboração:** nunca inventar resultados, caminhos, status ou histórico. Pedir arquivos e evidências quando faltar contexto.
