# TCC_EXPERIMENTAL_V3 — Retomada t-SNE, 19/10/2026

## Objetivo da entrega

Integração de três diagnósticos no Passo 07.02 e na galeria 07.03, preservando o método vencedor OOF, o GMM e a reserva do F10:

- **t-SNE 02 — Trustworthiness × Perplexity**: curva a partir de ajustes reais e separados do `RedutorTSNE`/openTSNE sobre amostra limitada dos atributos pré-processados do desenvolvimento. O parâmetro vencedor OOF é apenas uma referência; o diagnóstico não o substitui. Imagem `TSNE02_trustworthiness_vs_perplexity.png` e CSV homônimo.
- **t-SNE 13 — Embedding 3D**: dispersão com três coordenadas `TSNE1`, `TSNE2`, `TSNE3`, usando o embedding realmente ajustado com `n_components=3`. Imagem `13_tsne_mapa_TSNE1_TSNE2_TSNE3.png` e CSV com índice da amostra plotada. A versão 1D aprovada permanece um histograma e a 2D permanece dispersão.
- **t-SNE 14 — Diagnóstico KL/otimização**: KL final do embedding, tempo real medido em torno do ajuste do redutor, dimensionalidade, perplexidade e metadados. Imagem `14_tsne_diagnostico.png` e CSV. Não se inventa histórico por iteração se o openTSNE não o disponibiliza.

## Estado de aprovação

- Visualmente aprovados previamente: t-SNE 02 e t-SNE 13 em 1D/2D/3D (a versão 3D inicialmente foi gerada por script de prévia independente).
- **t-SNE 14: prévia sintética gerada e testada; ainda requer aprovação visual do usuário.**
- Código novo integrado em `_projecoes`, `TIPOS`, `NUM`, `executar()` e no reconhecimento da galeria `visualizacao_integrada.py`.
- Nova configuração `visualizacao.graficos.tsne_trustworthiness` no YAML.
- O novo t-SNE 02 NÃO substitui o OOF 02 de MCC: os nomes e diretórios são diferentes.

## Verificações efetuadas nesta preparação

- Reconstrução dos patches aprovados: GMM 07, GMM 01, GMM 04, PCA 11, PCA 12 1D e 2D/3D, t-SNE 13 1D.
- **92 testes passaram** na suíte de regressão reconstruída antes do teste opcional novo; um teste histórico do painel OOF 00 foi excluído por incompatibilidade da cópia antiga de origem, e **não** deve ser tratado como falha real do Windows.
- Testes t-SNE novos: **6 passaram e 1 foi pulado** pela ausência de `openTSNE` no ambiente de preparação; o teste pulado executará localmente quando essa dependência estiver instalada.
- Patch validado contra a cópia-base em fins de linha mistos, LF e CRLF.
- **A integração no computador Windows do usuário ainda depende da aplicação do patch e da execução local dos testes.** Não declarar sucesso do pipeline real antes disso.

## Regra metodológica

- Somente desenvolvimento F1–F9: não ler F10 durante a seleção, ajustes de diagnóstico ou testes sintéticos.
- t-SNE não usa rótulos no ajuste. `trustworthiness` exige mesmo subconjunto X pré-processado e embedding produzido a partir desse X.
- Trustworthiness avalia vizinhança local, NÃO é AP, MCC ou avaliação de fraude.
- Redução t-SNE diagnóstica é limitada por `tamanho_amostra_maximo` para evitar custos excessivos; F10 nunca faz parte da amostra.
- Não alterar a configuração vencedora OOF com base nas curvas diagnósticas e não atribuir prévias sintéticas aos resultados reais do TCC.
- Em 3D, usar exatamente 3 coordenadas do mesmo embedding; nunca fabricar dimensão ou substituir por pares 2D.
- KL da otimização não é métrica de fraude; não criar curva de KL por iteração sem dados reais.

## Próxima sessão

1. Aplicar o patch incremental `TSNE02_13_3D_14_INTEGRACAO_TCC.patch` **após** os patches PCA/GMM/t-SNE 13 1D previamente aceitos; executar `git apply --check` antes de instalar.
2. Rodar os testes novos em `tests/test_tsne_pipeline_02_13_14.py` e os testes de regressão. Corrigir quaisquer diferenças na árvore local sem `git reset`/`--3way` automático.
3. Abrir a prévia do gráfico 14, examinar e pedir aprovação visual.
4. Executar o pipeline de visualização em ensaio local pequeno com método TSNE, confirmando geração automática dos gráficos 02, 13 e 14, CSV, manifesto, galeria e ausência do F10.
5. Criar commit do código e da documentação após testes locais; não adicionar `outputs/`, dados brutos ou `.venv` ao git. O `git push` é opcional e só deve acontecer com remoto verificado.
6. Depois, realizar auditoria final dos catálogos, sem utilizar F10 antes de congelar a seleção.
