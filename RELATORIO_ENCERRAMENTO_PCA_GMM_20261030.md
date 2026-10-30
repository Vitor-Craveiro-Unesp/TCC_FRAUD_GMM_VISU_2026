# Relatório de encerramento técnico — PCA + GMM

**Projeto:** TCC_EXPERIMENTAL_V3
**Data:** 30/10/2026
**Escopo:** PCA 1D, PCA 2D e PCA 3D + GMM (FULL, K=4, `n_init=5`, RUS 16:1), pipeline OOF → ajuste final F1–F9 → F10 exploratório → gráficos, PRE01 e galerias.
**Status:** **IMPLEMENTAÇÃO PCA + GMM APROVADA PARA ENCERRAMENTO TÉCNICO**, com ressalvas metodológicas documentadas. Não equivale a validação independente de desempenho preditivo.

## 1. Controle de versão e experimentos

| Dimensão | Experimento aprovado | Checkpoint GitHub |
|---|---|---|
| PCA 1D | `outputs/ensaio_pca_gmm_20261030_114404` | `199faf3` |
| PCA 2D | `outputs/ensaio_pca_gmm_20261030_160458` | `ae42eeb` |
| PCA 3D | `outputs/ensaio_pca_gmm_20261030_163133` | `26bc845` |

Repositório: `Vitor-Craveiro-Unesp/TCC_FRAUD_GMM_VISU_2026`. Branch de encerramento: `feature/pca3d-automacao-oof-f10`. Backups locais aprovados para cada dimensionalidade, incluindo `backups/PCA3D_APROVADO_20261030.zip`.

## 2. Auditoria de artefatos — evidência do terminal do usuário

O programa `AUDITORIA_FINAL_PCA_GMM_1D_2D_3D.py --salvar` executou verificações sem retreinar, sem recalcular predições e sem reler o Parquet. Resultado reportado:

| Dimensão | Verificações | Erros |
|---|---:|---:|
| PCA 1D | 101/101 | 0 |
| PCA 2D | 110/110 | 0 |
| PCA 3D | 129/129 | 0 |
| **Total** | **340/340** | **0** |

O auditor reportou coincidência de hashes do Parquet e dos folds entre os três experimentos e mesmo conjunto de observações F10 na mesma ordem. Arquivo do relatório bruto informado pelo terminal: `backups/auditorias_pca/AUDITORIA_PCA_GMM_20261030_170618_746904.json`. **Este relatório bruto está no computador do usuário e não foi lido diretamente nesta revisão; as constatações acima decorrem da saída do auditor apresentada.**

## 3. Resultados F10 — estritamente exploratórios

Amostra estratificada fixa F10: **18.523 transações, 96 fraudes, seed 42**, conforme log da execução PCA 3D. F10 já havia sido consultado previamente, logo não é teste independente inédito.

| Medida | PCA 1D | PCA 2D | PCA 3D |
|---|---:|---:|---:|
| AP | 0,0179459 | 0,0158103 | 0,0059662 |
| MCC no limiar OOF congelado | 0,0529182 | 0,0506788 | 0,0000000 |
| LogLoss | 0,0315337 | 0,0317559 | 0,0324062 |
| Limiar tau* OOF | 0,010524627848033968 | 0,008285377944159511 | 0,0066453856689419367 |

A maior AP/MCC observada no F10 amostral é descritiva. **Não selecionar PCA 1D, 2D ou 3D como vencedor metodológico usando esta comparação F10.** Não reotimizar limiares ou hiperparâmetros nesse conjunto.

## 4. Revisão estática da implementação no checkpoint `26bc845`

Foram inspecionados diretamente no repositório os módulos: `src/orquestracao_oof.py`, `src/validacao_oof.py`, `src/reducao.py`, `src/treinamento_final_pca.py`, `src/avaliacao_final_pca.py`, `src/metricas_oof.py`, `src/finalizacao_automatica_pca1d.py`, `src/finalizacao_automatica_pca2d.py`, `src/finalizacao_automatica_pca3d.py` e `src/reducao_tsne.py`.

**Proteções de desenho confirmadas no código:**

1. `indices_do_fold` separa, para cada validação F1–F9, os oito folds de treino, sem incluir F10.
2. Em `prever_fold_oof`, encoder e imputador/padronizador são ajustados somente no treino; PCA é ajustado somente no treino e projetado no conjunto de validação sem refit.
3. O RUS seleciona apenas índices do conjunto de treino para ajustar o GMM; o modelo GMM não recebe os rótulos. O risco histórico por gaussiana usa rótulos apenas do treino original do fold (não os rótulos do fold de validação).
4. O ajuste final `ajustar_final_f1_f9` recusa conjuntos com F10 e refaz encoder → padronização → PCA → GMM → risco somente em F1–F9.
5. O avaliador F10 remove o alvo antes de prever, utiliza transformadores congelados e produz métricas com `tau*` escolhido previamente em OOF.
6. A avaliação F10 indica `limiar_otimizado_no_f10: false`, e o código não invoca a função de otimização de limiar nos registros F10.
7. O orquestrador em `26bc845` roteia a execução e a retomada para os módulos 1D/2D/3D conforme a dimensão salva no experimento.
8. As galerias F10 exigem arquivos esperados: 12 PNGs (1D), 15 PNGs (2D), 18 PNGs e cinco HTMLs interativos (3D), além de PRE01.

**Ajuste cosmético não bloqueante identificado:** o texto de `--planejar` em `src/orquestracao_oof.py` anuncia genericamente “12 gráficos”, inclusive nos casos 2D e 3D. O inventário validado pelas galerias tem as quantidades específicas acima. A chave `execucao.finalizacao_pca1d_automatica` mantém nome histórico embora ative as três dimensões. Recomenda-se renomeá-la em uma manutenção futura com compatibilidade, não alterar os resultados históricos.

## 5. Ressalvas metodológicas explícitas

- **Escopo amostral:** os experimentos usaram 10% dos folds F1–F9 no desenvolvimento e 10% estratificados do F10 na etapa exploratória. Não apresentar resultados como avaliação na base integral.
- **F10 reutilizado:** a inspeção repetida do F10 inviabiliza tratá-lo como conjunto final intocado. Qualquer comparação posterior entre dimensionalidades por esse F10 é exploratória.
- **Otimismo do MCC OOF:** o limiar que maximiza MCC é escolhido com os mesmos scores/rótulos OOF em que o MCC é apresentado. A métrica OOF com limiar otimizado pode ser otimista em relação a um limiar definido independentemente. Isso não caracteriza vazamento do F10, mas deve ser descrito na metodologia.
- **Desempenho limitado:** AP baixo e MCC pequeno ou zero nesta amostra não permitem afirmar alta capacidade de detecção de fraude. Registrar os resultados sem alterá-los.
- **Escopo da auditoria:** integridade de hashes/arquivos e revisão estática do pipeline não certificam, por si só, a procedência causal/temporal de cada feature no Parquet, nem estabelecem generalização externa. Essas limitações devem ser registradas sem refazer a etapa aprovada.

## 6. Decisão de encerramento

As seis partes da auditoria (integridade, OOF, F10, visualizações/PRE01, comparação entre dimensões e revisão metodológica) permitem **encerrar a construção computacional da etapa PCA + GMM**, sujeita a registrar e versionar este relatório. O encerramento é da implementação e dos resultados exploratórios produzidos, **não** de uma validação científica independente de desempenho.

Não refazer o OOF nem o F10 para fins de encerramento. Preservar os três experimentos, seus backups, o relatório bruto do auditor e os commits já aprovados. Criar um último commit contendo somente este documento; não adicionar `outputs/`, dados, modelos ou arquivos ZIP.

## 7. Próxima fase: t-SNE + GMM

**Não começar do zero.** Já existe `src/reducao_tsne.py` usando `openTSNE`, com `fit` no treino, `transform_treino` e `transform` para inserir observações novas sem refazer o embedding de treino. `src/validacao_oof.py` também já roteia candidatos de método `TSNE` para esse redutor. Antes de executar t-SNE real, conferir a compatibilidade de `openTSNE`/versões e testar extensões out-of-sample nos folds e no F10; a finalização automática F1–F9/F10, as geometrias e as galerias t-SNE ainda precisam de adaptação/validação específica. Reutilizar as convenções aprovadas: F1–F9 OOF, F10 exclusivamente exploratório, GMM FULL, RUS, risco histórico, AP/MCC/LogLoss/Score, limiar tau* OOF congelado, PRE01, galeria e 1D/2D/3D.
