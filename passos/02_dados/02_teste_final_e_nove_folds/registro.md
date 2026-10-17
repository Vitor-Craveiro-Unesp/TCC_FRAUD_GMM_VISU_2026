# PASSO 02.02 - DIVISAO DOS FOLDS

## Status
CONCLUIDO

## Data
2026-10-15 23:23:05

## Metodologia
- Divisao estratificada em 10 folds.
- Semente definida no YAML.
- Fold 10 reservado para teste final.
- Folds 1 a 9 para desenvolvimento.
- Nove rodadas de validacao cruzada.
- Oito folds de treino e um de validacao por rodada.

## Persistencia
- Identificacao dos folds salva em folds.npy.
- Informacoes da divisao registradas em manifesto.
- Mesmos folds serao reutilizados entre configuracoes.

## Arquivos de verificacao
- passos\02_dados\02_teste_final_e_nove_folds\saida_folds.txt
- data/splits/

## Regras
- Nenhum encoder foi ajustado nesta etapa.
- Nenhum GMM foi treinado.
- Nenhuma observacao sera alterada pelo RUS aqui.
- Fold 10 nao participara da selecao de modelos.

## Proximo subpasso
03.01 - Encoders sem vazamento.
