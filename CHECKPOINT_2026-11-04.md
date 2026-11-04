# CHECKPOINT TCC - 04/11/2026

## Estado atual
Seis experimentos executados pelo orquestrador oficial:
- PCA 1D
- TSNE 1D
- PCA 2D
- TSNE 2D
- PCA 3D
- TSNE 3D

## Configuracao de validacao
- F1-F9: 1% do desenvolvimento
- F10: 10% do proprio fold
- GMM K4 FULL
- RUS 16:1
- Semente 42
- F10 exploratorio

## Experimentos 3D de referencia
PCA:
outputs/ensaio_pca_gmm_20261104_015614

TSNE:
outputs/ensaio_tsne_gmm_20261104_031259

## Resultado TSNE 3D
- Execucao oficial concluida
- Vencedor: initialization=pca
- AP OOF: 0.009192
- MCC OOF: 0.044287
- LogLoss OOF: 0.032974
- Score composto OOF: 0.499633
- Modelo final F1-F9 salvo
- F10 avaliado
- PRE01 OOF integrado
- Galerias OOF e F10 geradas

## Pendencias
1. Executar PCA 3D do zero em pasta nova.
2. Executar TSNE 3D do zero em pasta nova.
3. Abrir quatro galerias: PCA OOF/F10 e TSNE OOF/F10.
4. Verificar cinco HTMLs interativos TSNE 3D.
5. Verificar GMM07 3D, ausente nos logs de geracao.
6. Conferir metricas F10 e integridade dos artefatos.
7. Auditar instalacao limpa do professor.
8. Validar documentacao e reprodutibilidade.

Nao modificar experimentos anteriores sem necessidade.
Nao utilizar F10 para selecao de modelos ou limiares.
