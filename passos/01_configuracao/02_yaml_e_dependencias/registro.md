# PASSO 01.02 - YAML E DEPENDENCIAS

## Status
CONCLUIDO

## Data
2026-10-15 12:23:25

## Ambiente
- Python: Python 3.13.3
- Ambiente virtual: .venv
- Dependencias: requirements.txt
- Verificacao: pip check aprovado

## Arquivo oficial
config_TCC_EXPERIMENTAL_V3_atualizado.yaml

## SHA256 do YAML
6135530F7BFB4726E6F2FF33E7A0E87B5067D72E3C2BDDF2F207F4971ABD7546

## Validacoes realizadas
- YAML carregado corretamente.
- 15 secoes reconhecidas.
- 18 graficos configurados.
- GMM com n_components: [1, 2].
- Metodo inicial: PCA.
- Dependencias sem conflitos reportados pelo pip.

## Decisoes metodologicas
- Um unico GMM nao supervisionado.
- Quantidade de gaussianas configuravel por YAML.
- Soft assignment com risco historico por componente.
- Calculo de risco usando somente rotulos do treino.
- PCA ou t-SNE, nunca ambos como etapas sucessivas.
- AP, MCC otimizado em OOF e LogLoss.
- Dez folds totais, com um reservado para teste.
- Amostragem NORMAL ou RUS.
- Sem grade de prioris.
- Visualizacoes configuradas individualmente.

## Observacoes
A estrutura YAML foi validada.
O pipeline ainda nao foi implementado.
A auditoria da base permanece pendente.

## Proxima etapa
PASSO 02 - DADOS
