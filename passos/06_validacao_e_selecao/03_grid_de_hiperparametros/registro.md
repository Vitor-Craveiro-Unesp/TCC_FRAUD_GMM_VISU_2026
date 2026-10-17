# PASSO 06.03 - ORQUESTRACAO OOF

## Status
IMPLEMENTACAO VALIDADA

## Data
2026-10-17 19:09:12

## Arquivos
- src/orquestracao_oof.py
- tests/test_orquestracao_oof.py
- tests/verificar_orquestracao_base_real.py

## Metodologia
- Nove rodadas OOF por configuracao.
- Treino em oito folds e validacao no restante.
- F10 reservado para o teste final.
- Metricas calculadas no OOF consolidado.
- Selecao pela formula composta do YAML.
- Limiar MCC selecionado usando apenas OOF.

## Grade atual
- PCA com 2 ou 3 componentes.
- GMM com K=1 ou K=2.
- Amostragem NORMAL.
- Quatro configuracoes.
- Trinta e seis rodadas previstas.

## Recursos operacionais
- Cache de resultados intermediarios.
- Retomada de execucoes interrompidas.
- Protecao dos resultados existentes.
- Uma pasta por execucao.

## Validacoes
- Suite automatizada aprovada.
- Planejamento da grade executado.
- Ensaio com base real aprovado.
- YAML oficial preservado.

## Pendencias
- Execucao oficial sobre todos os dados.
- Confirmacao de consumo de memoria.
- Selecao definitiva da configuracao vencedora.
- Treinamento final e avaliacao independente F10.

## Proxima tarefa
Preparar e executar a grade OOF oficial.
