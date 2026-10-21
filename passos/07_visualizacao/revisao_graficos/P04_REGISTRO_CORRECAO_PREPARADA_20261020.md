# TCC_EXPERIMENTAL_V3 — P04: correção documental do YAML preparada

**Data:** 20/10/2026  
**Status:** CORREÇÃO PREPARADA E VALIDADA ESTATICAMENTE; AGUARDA APLICAÇÃO E TESTES NO COMPUTADOR DO USUÁRIO.

## 1. Escopo e autorização

O usuário autorizou expressamente, nesta conversa, a correção mínima da seção `visualizacao.geometria` de `config_TCC_EXPERIMENTAL_V3_atualizado.yaml`.

A finalidade é remover `apenas_1d_2d: true`, aparentemente obsoleta, e atualizar os comentários que contradiziam o suporte a gráficos 3D. **Não foi autorizada alteração nos módulos Python, nos folds ou nos hiperparâmetros ativos.**

## 2. Evidências de origem

- YAML original recebido em `zips.rar`, de 712 linhas, extraído para inspeção isolada.
- Código `src/` e testes `tests/` recebidos separadamente: busca literal não encontrou uso de `apenas_1d_2d` em arquivos `.py`.
- `P04_REGISTRO_DIAGNOSTICO_INICIAL_20261020.md` contém a inspeção de despachantes/visualizações e a distinção entre testes sintéticos e execuções reais.
- A suíte geral local **antes desta alteração YAML** foi relatada pelo usuário como `416 passed in 80.56s`; não atribuir essa execução ao arquivo modificado.

## 3. Alteração exata proposta

```diff
-  # Regras de GEOMETRIA para PCA/t-SNE/GMM: desenhar apenas 1D e 2D.
-  # Quando o modelo tiver >=3 dimensoes, gerar pares 2D.
+  # Geometria PCA/t-SNE/GMM: respeitar as dimensoes do espaco usado (1D, 2D ou 3D).
+  # Em GMM 3D, mapas volumetricos sao representados por cortes 2D do campo conjunto.
+  # pares_2d define cortes/projecoes 2D, sem bloquear graficos em 3D.
   # Os indices comecam em 1 e devem existir no embedding do modelo.
   # Nao combinar coordenadas de folds diferentes em um unico scatter.
   geometria:
-    apenas_1d_2d: true
     pares_2d: [[1, 2], [1, 3], [2, 3]]  # NAO e grid.
```

Os parâmetros `pares_2d`, `dimensoes_1d`, `pontos_maximos_amostra` e `amostragem_estratificada` foram preservados, assim como todas as demais configurações.

## 4. Validação já executada na cópia isolada

- `yaml.safe_load` interpretou o arquivo original e o arquivo corrigido sem erro.
- Após remover apenas `visualizacao.geometria.apenas_1d_2d` da estrutura antiga, as duas estruturas YAML carregadas foram **idênticas** (comparação de objetos Python).
- Busca em `src/*.py` e `tests/*.py` da cópia recebida não encontrou o nome da chave obsoleta.
- A alteração produzida foi limitada ao comentário e à retirada dessa chave; o diff foi guardado em `P04_CORRECAO_YAML_20261020.diff`.
- O arquivo integral revisado está disponível como `config_TCC_EXPERIMENTAL_V3_P04_corrigido.yaml`, para consulta/aplicação **apenas se coincidir com a cópia local atual**.

**Limite:** os arquivos em `/mnt/data` são cópias dos anexos fornecidos; a configuração viva no computador do usuário **não foi modificada remotamente**.

## 5. Verificação obrigatória no VS Code antes do encerramento

Após aplicar exatamente o diff acima em `config_TCC_EXPERIMENTAL_V3_atualizado.yaml`, executar em `.venv`:

```powershell
python -c "import yaml; from pathlib import Path; c=yaml.safe_load(Path('config_TCC_EXPERIMENTAL_V3_atualizado.yaml').read_text(encoding='utf-8-sig')); g=c['visualizacao']['geometria']; assert 'apenas_1d_2d' not in g; print('YAML OK; chave removida; pares_2d =', g['pares_2d'])"
python -m pytest -q tests/test_visualizacao_geometrica.py tests/test_pca12_2d3d_pdf.py tests/test_tsne_pipeline_02_13_14.py tests/test_visualizacao_gmm01.py tests/test_visualizacao_gmm04.py tests/test_visualizacao_gmm07.py
git diff -- config_TCC_EXPERIMENTAL_V3_atualizado.yaml
```

Confirmar no diff a **ausência de alterações não previstas**. Enviar saídas para encerramento da P04. Não é necessário executar experimentos com a base real nem usar F10.

## 6. Critério para concluir

- YAML válido no computador do usuário;
- diff local restrito à mudança aprovada;
- testes de visualização selecionados aprovados;
- registro de conclusão e lista de pendências atualizados com os resultados efetivos.

**Próximo item planejado após P04:** P10 — conferir Painel 00 na galeria integrada, conforme plano de pendências.
