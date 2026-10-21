# TCC_EXPERIMENTAL_V3 — Auditoria parcial do ambiente Python e cache de testes

**Data:** 20/10/2026  
**Arquivos recebidos:** `.venv.rar` e `.pytest_cache.rar`.  
**Procedimento:** inventário e leitura seletiva dos arquivos compactados, sem executar programas dentro do ambiente enviado e sem modificar `src`, `tests` ou YAML.

## 1. `.venv.rar` — inventário objetivo

- Tamanho compactado: **142.991.981 bytes** (~136,4 MiB).
- Conteúdo: **16.014 entradas**, das quais **14.594 são arquivos regulares**.
- Soma dos tamanhos não compactados dos arquivos: **490.084.618 bytes** (~467,4 MiB).
- O `pyvenv.cfg` do pacote declara **Python 3.13.3** e caminhos de criação/execução **Windows**; portanto, esta cópia não foi utilizada para executar testes no ambiente Linux de auditoria.
- Identificaram-se **27 pacotes com arquivo `*.dist-info/METADATA`** no conteúdo do pacote; isso é um inventário do `.rar` recebido, **não** uma declaração sobre tudo o que está instalado atualmente no computador do usuário.

### Comparação com `requirements.txt` recebido em `zips.rar`

- `requirements.txt`: **102 dependências com versão fixada**.
- Presentes entre os metadados da `.venv.rar`: **26** das 102 dependências; **76** não identificadas nesse arquivo compactado.
- Entre as 26 presentes, **10 versões não coincidem** com os requisitos fixados: `fonttools`, `iniconfig`, `narwhals`, `numpy`, `pandas`, `pillow`, `pluggy`, `pyarrow`, `pytest` e `tzdata`.
- A versão de `scikit-learn` coincide: **1.9.1**.
- Diferenças relevantes:

| Pacote | Em `requirements.txt` | Encontrado em `.venv.rar` |
|---|---|---|
| `openTSNE` | `1.0.4` | Não identificado nos metadados nem no módulo esperado |
| `category_encoders` | `2.11.1` | Não identificado nos metadados nem no módulo esperado |
| `pyarrow` | `25.0.1` | `26.0.0` |
| `pytest` | `8.4.2` | `9.1.1` |
| `numpy` | `2.5.3` | `2.5.4` |
| `pandas` | `3.0.5` | `3.0.6` |

**Interpretação cuidadosa:** o arquivo `.venv.rar` não representa uma instalação inteiramente compatível com o `requirements.txt` recebido, conforme seus metadados. Isso **não** prova que o ambiente ativo no VS Code do usuário tenha a mesma situação e **não** justifica instalar/atualizar bibliotecas sem verificar qual ambiente é efetivamente usado na pipeline.

## 2. `.pytest_cache.rar` — evidência limitada

- Conteúdo: **8 entradas**, incluindo **5 arquivos regulares**.
- `v/cache/lastfailed`: objeto JSON **`{}`**, isto é, nenhum teste indicado como falha pendente **naquele cache**. Não é relatório de execução e não comprova que todos os testes tenham passado.
- `v/cache/nodeids`: **420 IDs de casos de teste** cadastrados no cache.
- Na coleta independente de `tests.rar` feita anteriormente, foram **416 casos**.
- Todos os 416 IDs da coleta independente estão presentes no cache. Quatro IDs existem **somente no cache**:
  - `tests/test_visualizacao_geometrica.py::test_covar_diag_nao_cria_dependencias`
  - `tests/test_visualizacao_geometrica.py::test_sete_tipos_pca_salvos`
  - `tests/test_visualizacao_oof.py::test_gera_seis_tipos_arquivos_e_csv`
  - `tests/test_visualizacao_oof.py::test_pr_visualizacao_zoom_usa_inset_e_mantem_pontos_csv`

**Interpretação cuidadosa:** é compatível com um cache de versão anterior ou diferente da suíte. O cache não altera o resultado **reproduzido nesta conversa**, que foi **414 aprovados, 1 falha e 1 pulado** na execução isolada de 416 testes. Não é possível inferir, apenas a partir do cache, por que os quatro testes não aparecem na suíte atual.

## 3. Pendências e encaminhamento

1. **P02:** verificar `openTSNE` no ambiente realmente utilizado pelo usuário; ela não está identificada no `.venv.rar` recebido, embora figure no `requirements.txt`.
2. **P03:** verificar `category_encoders` e diferenças de `pyarrow` e outras dependências na instalação ativa; não reinstalar preventivamente.
3. **P13:** conferir divergência entre as 102 dependências fixadas e os 27 pacotes identificáveis no `.venv.rar`; verificar se o pacote é uma cópia parcial/antiga e qual ambiente VS Code realmente seleciona.
4. **P14:** conferir, apenas se necessário, os quatro testes cadastrados no cache que não constam do `tests.rar` atual; não restaurar/alterar testes sem contextualização.

## 4. Limites desta etapa

- Não se executou `pytest` dentro da `.venv` Windows.
- Não se executou `openTSNE`, `category_encoders`, bases reais, pipeline completa ou F10.
- Não foi modificada nenhuma configuração, dependência, código ou arquivo recebido.
- A inspeção de metadados não equivale a executar `pip list` no ambiente ativo do usuário.

**Estado:** diagnóstico documental concluído. As verificações permanecem na lista acumulada; correções somente após o recebimento dos materiais e a aprovação do usuário.
