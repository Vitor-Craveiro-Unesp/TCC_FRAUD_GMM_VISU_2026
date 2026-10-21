# P05 — Registro da validação local no Windows (21/10/2026)

## Escopo aprovado
O usuário autorizou a substituição das duas imagens separadas do mapa de risco GMM 08 por **uma imagem comparativa 1×2 em 2D** e **uma imagem comparativa 3×2 em 3D**. O modo 1D previamente aprovado foi preservado. Corrigida a numeração `NUM['mapa_risco_fronteira']` de 15 para 8 e a integração na galeria, sem alterar a regra de risco/limiar, a grade, os folds, a amostragem ou os dados do F10.

## Evidência informada pelo usuário (ambiente Windows `.venv`)
1. ZIP `P05_GMM08_COMPARATIVO_PATCH_E_TESTES_20261021.zip` extraído, quatro arquivos instalados e conferidos com SHA-256.
2. `git apply --check --ignore-space-change` **aprovado**; `git apply --ignore-space-change` **aplicado uma vez**.
3. `git -c core.safecrlf=false diff --check`: **exit code 0**. Aviso de LF/CRLF em `tests/test_gmm.py` não indicou falha de whitespace.
4. `python -m pytest -q`: **434 passed in 95.17s** em 21/10/2026, sem erros/skips relatados.
5. Arquivos-fonte afetados no `git status`: `src/visualizacao_geometrica.py`, `src/visualizacao_integrada.py`, `tests/test_visualizacao_geometrica.py`, `tests/test_visualizacao_integrada.py`, e comentários do YAML. Existiam também alterações acumuladas de P01/P04/P08 não commitadas; não atribuir todas as diferenças do Git à P05.

## Estado
**P05 — integração local e validação automatizada aprovadas; gráficos com base real pendentes da Etapa 08.** Não houve execução real F1–F9 ou F10 como resultado dessa validação. As duas imagens do pacote P05 são prévias **sintéticas**, não resultados do TCC. Não foi registrado commit Git. A pasta com nome `teste_final/graficos_gerados` ainda corresponde a geometria calculada no desenvolvimento F1–F9.

## Próxima sequência
P06 (auditoria código→catálogos) → P09 (sincronização da documentação) → P12 (distinguir saída de desenvolvimento de F10) → preparar Etapa 08. Antes de avaliar F10 é preciso implementar e testar a rotina independente de teste final, ainda não verificada.
