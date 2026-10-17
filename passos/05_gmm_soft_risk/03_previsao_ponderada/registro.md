# PASSO 05.03 - PROBABILIDADE FINAL

## Status
CONCLUIDO

## Data
2026-10-17 17:37:39

## Arquivos
- src/probabilidade.py
- tests/test_probabilidade.py
- tests/verificar_probabilidade_base_real.py

## Metodologia
- Predicao probabilistica por soft assignment.
- Combina responsabilidades com riscos historicos.
- Retorna um score entre 0 e 1 por observacao.
- Preserva a ordem das observacoes.
- Nao aplica limiar de classificacao nesta etapa.

## Formula
P(Y=1|x) = soma_k(gamma_k(x) * RH_k)

## Validacoes
- Suite de 188 testes automatizados aprovada.
- Verificacao com base real aprovada.
- NORMAL/RUS e K=1/K=2 verificados.
- PCA com 2 e 3 componentes verificado.
- F9 utilizado apenas para gerar predicoes.
- F10 preservado.

## Limitacoes
- Verificacao realizada com amostras reais.
- Validacao cruzada completa ainda pendente.
- Selecao de hiperparametros ainda pendente.
- Limiar MCC sera estimado posteriormente via OOF.

## Proxima etapa
Passo 06 - Validacao cruzada OOF e metricas.
