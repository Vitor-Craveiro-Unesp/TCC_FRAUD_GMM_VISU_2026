# Painel 00 — Indicadores de desempenho e score composto

**Situação:** APROVADO VISUALMENTE PELO USUÁRIO em 2026-10-18.

## Referência

Catálogo fornecido pelo usuário: `ANALISES_CATALOGO_CONFUSAO_MCC_E_050.pdf`, seção "Painel 00".

## Implementação aprovada

- Quatro cartões: AP; MCC transformado `(MCC+1)/2`; `exp(-LogLoss)`; score composto com peso igual para as três parcelas.
- Escala contínua vermelho → amarelo → verde, proporcional ao valor em `[0,1]`; cores não constituem faixas universais de qualidade.
- AP comparada à prevalência OOF; MCC e LogLoss brutos disponíveis para interpretação.
- Título destacado, textos auxiliares mais legíveis e nota sobre referências de avaliação.
- Métricas obtidas do OOF do vencedor, sem recalcular seleção e sem utilizar F10.

## Evidências

- O usuário confirmou a geração de PNG, CSV e manifesto com resultados do ensaio local.
- O usuário aprovou a apresentação visual do painel refinado.
- Testes automatizados do refinamento haviam sido aprovados no ambiente de preparação.

## Limites da aprovação

A aprovação refere-se à visualização e integração funcional do Painel 00. Não significa aprovação científica definitiva das métricas do ensaio de 10% ou dos demais gráficos. A execução e auditoria local completa pertencem à Etapa 08.

## Próxima revisão

Gráfico 01 — Curva Precision–Recall e dois pontos de operação (MCC OOF e 0,50).
