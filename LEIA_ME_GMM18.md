# GMM 18 — Convergência real do algoritmo EM (3 inicializações)

## Como instalar (sem treino)

1. Faça commit do projeto atual antes de instalar.
2. Extraia TODO o conteúdo do ZIP na raiz `TCC_EXPERIMENTAL_V3` (incluindo os dois módulos na subpasta `src`).
3. Na raiz, execute:

   `\.venv\Scripts\python.exe .\instalar_grafico_em.py`

   O instalador altera quatro arquivos de implementação (gmm.py, visualizacao_geometrica.py,
   visualizacao_integrada.py e YAML), opcionalmente atualiza a contagem fixa de um teste
   legado de geometria (de 8 para 9 tipos), e acrescenta dois módulos novos. Antes de alterar,
   confere as âncoras de inserção; se não localizar alguma, para sem editar arquivos.
   Cópias dos quatro originais são criadas fora do repositório.

4. Verifique em dados sintéticos que o mesmo EM foi ajustado:

   `\.venv\Scripts\python.exe .\verificar_grafico_em.py`

5. Valide sintaxe:

   `\.venv\Scripts\python.exe -m compileall -q .\src`

6. Inspecione o diff e faça commit (não inclua logs e ZIPs grandes):

   `git status --short`

   `git diff --stat`

   `git diff --check`

   `git add -- src/gmm.py src/gmm_convergencia_em.py src/visualizacao_gmm18.py src/visualizacao_geometrica.py src/visualizacao_integrada.py config_TCC_EXPERIMENTAL_V3_atualizado.yaml tests/test_visualizacao_geometrica.py`

   `git diff --cached --check`

   `git commit -m "Integra curvas reais de convergencia EM para tres inicializacoes"`

## Como funciona

- Usa `GaussianMixture` do scikit-learn, com `n_init=3` mantido.
- Observa o valor real do lower bound médio em cada iteração EM e reinicialização,
  usando hooks internos do scikit-learn, **sem inserir treinamento adicional**.
- A melhor inicialização é a que o próprio scikit-learn seleciona (maior lower bound).
- A captura só ocorre no ajuste final do vencedor no desenvolvimento F1–F9 da etapa 07.02;
  os ajustes da validação OOF não sofrem instrumentação.
- Exporta `GMM18_convergencia_em_inicializacoes.png` e CSV em
  `vencedor/visualizacao/geometria_desenvolvimento/`, e integra o PNG à galeria.
- O CSV informa o limite inferior **médio por observação**, que é o critério de comparação
  entre inicializações do scikit-learn, e também o valor total equivalente (média x n).
- Não incorpora rótulos, classificação ou F10 na seleção do GMM.

## Atenção

Os hooks `_print_verbose_msg_init_beg`, `_print_verbose_msg_init_end` e
`_compute_lower_bound` são métodos privados do scikit-learn. Ao atualizar
scikit-learn, execute o smoke test de compatibilidade **antes** de um ensaio longo.

Não se deve reconstruir trajetórias a partir de `lower_bound_` final de experimentos
antigos nem interpolar pontos intermediários: os históricos só existirão para o
novo treinamento do vencedor instrumentado.
