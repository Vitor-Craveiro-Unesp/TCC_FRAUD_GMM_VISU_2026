# TCC_EXPERIMENTAL_V3 — Registro parcial de leitura documental

**Data:** 20/10/2026  
**Estado:** leitura do arquivo `passos.rar` realizada; validação do repositório ainda pendente.  
**Natureza:** registro de ingestão documental, não aprovação de código nem encerramento de etapa científica.

## Fonte examinada

- Arquivo recebido: `passos.rar`.
- Conteúdo verificado na extração: **46 arquivos**, dos quais **43 são `.md`**; há **9 arquivos `.md` vazios** (registros planejados, não evidência de implementação).
- Organização presente: `01_configuracao`, `02_dados`, `03_preprocessamento`, `04_reducao_dimensional`, `05_gmm_soft_risk`, `06_validacao_e_selecao`, `07_resultados`, `07_visualizacao`, `08_execucao_local` e documentos gerais em `passos/`.

## Decisões metodológicas registradas nos documentos

- Desenvolvimento por **F1–F9**, com **F10 reservado** para avaliação independente posterior.
- Pipeline GMM: ajuste não supervisionado, responsabilidades soft, risco histórico supervisionado por componente a partir do treino, risco final ponderado pelas responsabilidades.
- Alternativas de redução: **PCA ou t-SNE**, não ambos no mesmo candidato.
- Métricas citadas: **Average Precision, MCC e LogLoss**; score composto definido no projeto.
- Documento de retomada de 19/10 destaca que a rotulagem descritiva de **alto/baixo risco das gaussianas** usa `RH_k >= 2 × prevalência do treino`, diferente dos cortes de classificação de transações **0,50** e **τ* por MCC OOF**.
- A retomada de 19/10 diferencia ensaio histórico PCA 10% com covariância **`diag`** de prévias sintéticas e próximas configurações **`full`**. Não reclassificar retroativamente os resultados históricos como FULL.

## Histórico documental, sujeito a confirmação no código atual

- `passos/CONTEXTO_GERAL.md` registra implementação e validações progressivas dos passos 01 a 07; a etapa 08 está planejada/replanejada, não demonstrada como concluída nesse material.
- `passos/07_visualizacao/resumo.md` registra conclusão funcional da etapa 07 em 18/10 com **305 testes aprovados**, ensaio de 10% do desenvolvimento e F10 preservado. Esses são **relatos documentais históricos**, não testes reproduzidos durante esta leitura.
- `passos/RETOMADA_GRAFICOS_TCC_20261019.md` ainda descreve **GMM 07 como próximo**, mas o usuário corrigiu essa informação nesta conversa: relatou que **todos os gráficos dos quatro catálogos já foram revisados/integrados**, restando o Biplot PCA 1D, 2D e 3D.
- `passos/07_visualizacao/RETOMADA_TCC_TSNE_20261019.md` descreve uma preparação em que partes t-SNE ainda requeriam integração ou testes no Windows. Essa nota é de um estado anterior e **não confirma o estado atual** informado pelo usuário.
- A estrutura distingue `07_resultados` (com registros vazios) e `07_visualizacao` (com registros preenchidos); preservar essa diferença até verificar o código e a estrutura real.

## Restrições de interpretação

- Nenhum script de `src/`, YAML oficial ou teste automatizado foi recebido neste passo de leitura.
- Portanto, **não foi feita auditoria de implementação**, nem foi verificado que os testes históricos continuam passando.
- Não sobrescrever relatórios existentes com este registro parcial; reconciliar estados pelo histórico de datas e pelas evidências técnicas atuais.

## Próxima entrada solicitada

1. `config_TCC_EXPERIMENTAL_V3_atualizado.yaml`.
2. Depois: `src.rar`, `tests.rar`, `requirements.txt` e, quando necessário, `outputs.rar`.
3. Com fontes completas, verificar o Biplot PCA 1D/2D/3D; obter aprovação visual **antes de integrar**; então auditar catálogos PDF ↔ pipeline.

## Regra de continuidade

Documentar cada passo executado e aprovado em Markdown (`registro.md`, `resumo.md`, contexto/retomada conforme a organização vigente); distinguir sempre **informado pelo usuário**, **registrado em documentação**, **inspecionado no código** e **testado novamente**. Pedir arquivos faltantes e nunca inventar resultados, nomes ou estados.
