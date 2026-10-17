# PASSO 06.01 - MOTOR DE VALIDACAO OOF

## Status
CONCLUIDO

## Data
2026-10-17 18:25:58

## Arquivos
- src/validacao_oof.py
- tests/test_validacao_oof.py
- tests/verificar_oof_base_real.py

## Metodologia
- Validacao cruzada estratificada.
- Nove folds de desenvolvimento: F1 a F9.
- Oito folds de treinamento por rodada.
- Um fold de validacao por rodada.
- F10 reservado para teste final.
- Pre-processamento ajustado somente no treino.
- Reducao dimensional ajustada somente no treino.
- GMM ajustado sem rotulos.
- Risco estimado com rotulos do treino original.
- Predicoes geradas sem utilizar rotulos da validacao.

## Verificacoes
- Suite total: 208 testes automatizados.
- Testes confirmados como aprovados pelo usuario.
- Verificacao com amostra real aprovada.
- Treino: 10000 registros de F1 a F8.
- Validacao: 2000 registros de F9.
- PCA, NORMAL/RUS e K=1/K=2 verificados.
- F10 preservado.

## Limitacoes
- Ainda nao executadas as nove rodadas completas.
- Metricas OOF ainda nao implementadas.
- Grade completa de hiperparametros pendente.
- Persistencia, cache e retomada pendentes.

## Proxima tarefa
06.02 - Metricas e selecao do limiar MCC.
