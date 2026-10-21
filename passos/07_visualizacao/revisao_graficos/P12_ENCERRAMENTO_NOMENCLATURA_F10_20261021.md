# P12 — Encerramento da ambiguidade `teste_final` × avaliação independente F10

**Data:** 21/10/2026  
**Natureza:** fechamento documental/de compatibilidade da P12, **não** implementação da avaliação F10.

## Evidências do código inspecionado

1. `src/orquestracao_oof.py` seleciona observações dos folds **F1–F9** e as processa com validação OOF; `dados.fracao_base` controla a fração do **desenvolvimento**. Em frações inferiores a 1, o código caracteriza a execução como **ensaio amostral não oficial**.
2. A chave `saidas.pastas.teste_final` integra o contrato YAML validado em `src/orquestracao_oof.py`, que exige precisamente as três chaves `oof`, `teste_final` e `graficos`. A raiz `teste_final` já é criada no experimento antes de existir uma rotina independente de F10.
3. `src/visualizacao_geometrica.py` grava imagens em `<experimento>/teste_final/graficos_gerados/` após **reajustar o vencedor em F1–F9**, e grava `manifesto_geometria_07_02.json` com `origem=desenvolvimento_F1_F9`, `f10_utilizado=false` e aviso de que não são gráficos/métricas F10.
4. `src/visualizacao_integrada.py` identifica esses arquivos como **geometria de desenvolvimento** e avisa que não contêm métricas F10. Eventual candidato PCA 2D complementar também não é um resultado de teste F10.
5. **Não foi localizada no snapshot disponibilizado uma rotina independente completa para inferir, avaliar e registrar resultados reais do F10.** A presença da pasta não comprova sua existência. Verificar novamente os arquivos locais antes de implementar.
6. A P05 foi aplicada e a suíte inteira passou (**434/434 em 95,17 s**, informado pelo autor). P06 instalou e verificou 4 PDFs atualizados e 10 arquivos por SHA-256; P09 instalou o índice consolidado.

## Decisão P12: preservar compatibilidade sem ambiguidade semântica

**Não renomear automaticamente** o diretório `teste_final` nem a chave YAML `saidas.pastas.teste_final` nesta etapa. Isso exigiria revisão coordenada de orquestração, despachadores, galerias, testes e configuração e não resolve por si só a falta de avaliação F10.

Regra de nomenclatura oficial durante a transição:

| Artefato | Origem verdadeira | Interpretação |
| --- | --- | --- |
| `<experimento>/oof/...` | Predições fora do treino em F1–F9 | Desenvolvimento/seleção OOF |
| `<experimento>/teste_final/graficos_gerados/...` | Modelo reajustado **somente F1–F9** | **Diagnósticos geométricos de desenvolvimento**, apesar do caminho histórico |
| `manifesto_geometria_07_02.json` | `origem=desenvolvimento_F1_F9`, `f10_utilizado=false` | Documento de evidência da origem |
| **Futuro teste F10** | Predições de linhas do fold 10 por modelo congelado | **Avaliação independente real**, ainda não implementada/homologada |

Proibição: chamar gráficos geométricos do diretório `teste_final` de métricas de generalização em F10. Não renomear a saída antiga sem projeto de migração compatível.

## Portão de implementação necessário para Etapa 08 completa

Antes de ler F10 para predição, desenvolver e testar módulo explícito de avaliação final, com arquivos próprios e manifesto independente. O módulo deverá:

1. Receber **apenas** a configuração candidata vencedora, o limiar τ* e a escolha de prior definidos em F1–F9; impedir nova seleção após acesso ao F10.
2. Reajustar a pipeline vencedora **somente no desenvolvimento amostrado F1–F9**; aplicar no F10 completo exclusivamente operações aprendidas em treino (`transform`/`predict_proba`/responsabilidades), sem `fit` ou seleção de limiar sobre F10.
3. Reaproveitar com precisão pré-processamento, tratamento dos rótulos de risco por componente e demais transformações do experimento OOF. Testar compatibilidade PCA/t-SNE apenas para o método realmente configurado.
4. Produzir métricas **exclusivamente F10** (AP, MCC para τ* congelado e τ=0,50, LogLoss, matriz de confusão, score descritivo quando definido), nº de observações, prevalência, probabilidades, hashes e manifesto `origem=F10`. Registrar a distinção dos gráficos de geometria F1–F9.
5. Criar testes de não vazamento: F10 nunca entra no ajuste, normalização, seleção de hiperparâmetros, prior ou limiar; integridade de índices de fold e contagem integral de F10; rastreabilidade de artefatos e nomes não conflitantes.
6. Aprovar revisão e preflight computacional **antes de realizar o teste F10 real**.

## Estado e limites

**P12 encerrada como esclarecimento de caminhos e documentação.** A avaliação real F10 permanece requisito funcional **não concluído** da Etapa 08. Nenhum arquivo de `src/`, `tests/`, YAML ou dados foi modificado por este fechamento; não se executou OOF nem F10 neste registro. Os testes 434/434 comprovam regressão automatizada do snapshot anterior, não o desempenho científico nem a rotina F10.

Referências vigentes: `passos/P09_ESTADO_CONSOLIDADO_E_INDICE_20261021.md`, `passos/07_visualizacao/revisao_graficos/P05_RESULTADO_VALIDACAO_LOCAL_20261021.md`, `passos/07_visualizacao/revisao_graficos/P06_REGISTRO_ATUALIZACAO_DOS_QUATRO_CATALOGOS_20261021.md`.
