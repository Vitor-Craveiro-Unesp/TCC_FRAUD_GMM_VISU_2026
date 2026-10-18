# Passo 07.02 — Diagnósticos geométricos e gaussianas do vencedor

Status: **implementação entregue; validação na máquina do usuário pendente**.

## Objetivo

Após 07.01, reajustar **apenas o candidato vencedor pelo score OOF** nos registros do desenvolvimento F1–F9 correspondentes exatamente ao experimento/ensaio concluído, para gerar diagnósticos de geometria e componentes gaussianos. **F10 continua integralmente reservado**. O limiar MCC selecionado no OOF permanece congelado.

## Procedimento

1. Ler seleção, ranking, configuração original do experimento e previsões OOF do vencedor; validá-los utilizando o módulo 07.01.
2. Ler o manifesto e validar hash do Parquet, hash dos folds, índices únicos, rótulos e ausência de F10.
3. Recarregar as mesmas transações de desenvolvimento do OOF na mesma ordem, nunca os dados de teste.
4. Ajustar novamente encoders, imputação e padronização no desenvolvimento; PCA **ou** t-SNE do vencedor, GMM único sem rótulos (NORMAL/RUS conforme vencedor), responsabilidades de todo o desenvolvimento e RH soft supervisionado com todos os labels de desenvolvimento.
5. Desenhar somente os tipos habilitados e aplicáveis no YAML: 07 risco, 08 pesos/massas, 09 médias, 10 variâncias/covariâncias, 11 PCA scree, 12 PCA projeções, 13 mapa t-SNE, 14 diagnóstico t-SNE, 15 mapa de risco / fronteira. Tipos de PCA são mutuamente exclusivos dos tipos de t-SNE.
6. Salvar imagens e dados CSV na pasta `teste_final/graficos_gerados` por **convenção YAML**, identificando sempre que a **origem é o desenvolvimento**, e não F10. Salvar relatórios por gaussiana e manifesto separado 07.02.
7. Nunca alterar métricas OOF, seleção ou arquivos pré-existentes. Em geometria de 3+ componentes, exibir cortes bidimensionais com demais dimensões fixas na média do desenvolvimento, não chamar de superfície global do classificador.

## Caso do ensaio local atual

- Pasta: `outputs/ensaio_pca_gmm_20261017_195252`
- Vencedor amostral: PCA 1 dimensão, GMM 4 gaussianas, NORMAL.
- Origem: 166.716 transações (10% de F1–F9), F10 excluído.
- Gráficos aplicáveis: 07, 08, 09, 10, 11, 12 e 15. Os tipos 13 e 14 são exclusivos do t-SNE e não se aplicam.
- A execução diagnóstica é um **novo ajuste**, e não o reaproveitamento de parâmetros GMM de um fold OOF individual. Não recalcula o ranking nem o limiar.

## Validação

Testes sintéticos automatizados adicionados em `tests/test_visualizacao_geometrica.py` (9 testes, incluindo bloqueio de F10 e planejamento sem treinar).
A confirmação sobre a base real requer execução e validação no computador do usuário.
