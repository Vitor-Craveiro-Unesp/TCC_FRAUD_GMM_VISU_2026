# RETOMADA DO TCC EXPERIMENTAL V3

## Situacao atual
- Auditoria visual PCA 1D concluida: 18/18 graficos aprovados.
- Novo grafico OOF 07 aprovado visualmente.
- Secao OOF passa de 7 para 8 graficos.
- Ultimo experimento utilizado: outputs/ensaio_pca_gmm_20261024_201043

## Novo grafico aprovado
07_metricas_por_fold.png

Paineis:
A. Average Precision por fold F1-F9.
B. MCC por fold, comparando tau=0.50 e tau* global OOF.
C. exp(-LogLoss) por fold.

- Subtitulo superior removido.
- Dados originais OOF reutilizados.
- Nenhum PCA ou GMM reajustado.
- F10 nao utilizado.

## Arquivos principais
- src/visualizacao_oof.py
- src/visualizacao_integrada.py
- config_TCC_EXPERIMENTAL_V3_atualizado.yaml

## Preview aprovado
outputs/validacao_manual_oof/metricas_por_fold_v02/

## Proximas tarefas
1. Confirmar a integracao automatica do grafico 07.
2. Fazer um ensaio novo do zero.
3. Verificar todos os PNGs e CSVs automaticamente gerados.
4. Comparar a nova galeria com os graficos aprovados.
5. Preservar o F10 reservado.

## Observacao
O grafico 07 foi aprovado na geracao manual.
A integracao ponta a ponta da nova versao ainda precisa ser testada.
