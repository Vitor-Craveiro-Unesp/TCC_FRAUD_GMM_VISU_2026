# P05 — Correção do GMM 08: comparação 2D e 3D

Data: 2026-10-21. Estado: **PATCH PREPARADO E VALIDADO EM CÓPIA; AINDA NÃO CONFIRMADO NO COMPUTADOR DO USUÁRIO**.

## Aprovação

O usuário aprovou uma única imagem comparativa por dimensionalidade:
- 1D: comparativo 1×2 que já existia, sem mudanças de lógica.
- 2D: dois painéis na mesma imagem (1×2), τ=0,50 versus τ* ótimo MCC OOF.
- 3D: três linhas com pares PC1–PC2, PC1–PC3, PC2–PC3, e duas colunas correspondendo aos dois limiares (3×2).

## Escopo exato da modificação

1. `src/visualizacao_geometrica.py`: `NUM['mapa_risco_fronteira']` alterado de 15 para 8; `GMM08_mapa_risco_2d_comparativo` e `GMM08_mapa_risco_3d_comparativo` substituem os dois PNGs independentes por dimensão. Computação de malhas, score por GMM, `_mudanca_classes`, `_avaliar_cortes_oof`, limites, elipses/seções gaussianas FULL 95%, CSV por plano e F10 intocados.
2. `src/visualizacao_integrada.py`: verifica somente o novo arquivo comparativo da dimensionalidade escolhida, e só esse será incorporado à galeria; arquivos legados separados não contam como resultados completos.
3. `tests/test_visualizacao_geometrica.py`: cenários 2D e 3D conferem dimensões dos painéis, rótulos dos MCCs, mesmos limites cromáticos, arquivos e CSVs, origem simulada explícita e exclusão do F10.
4. `tests/test_visualizacao_integrada.py`: teste de integração rejeita os dois arquivos antigos como substitutos do novo comparativo.
5. `config_TCC_EXPERIMENTAL_V3_atualizado.yaml`: **somente comentários** do GMM 08, sem alterar valores de configuração, prior, folds, amostragem ou grade.

## Evidência local do ambiente de preparação

Com fontes restauradas dos RARs enviados e patch P08 já aplicado em cópia isolada:

`python -m pytest -q tests/test_visualizacao_geometrica.py tests/test_visualizacao_integrada.py`

Resultado: **21 passed in 21.70s**. Verificações visuais sintéticas: `outputs/prototipos_gmm08_p05/GMM08_mapa_risco_2d_comparativo.png`, `...3d_comparativo.png`. O MCC impresso nas prévias é **MCC da simulação**, não MCC OOF real.

Os 432 testes anteriormente concluídos no PC do usuário **não** cobrem ainda este patch; é necessário rodar os testes locais novamente após aplicação.

## Como aplicar após extrair ZIP na raiz

1. `git apply --check --ignore-space-change -- passos/07_visualizacao/revisao_graficos/historico_patches/P05_GMM08_COMPARATIVO_PARA_APLICAR.patch`
2. Só se esse comando não relatar erros: `git apply --ignore-space-change -- passos/07_visualizacao/revisao_graficos/historico_patches/P05_GMM08_COMPARATIVO_PARA_APLICAR.patch`
3. Rodar `python -m pytest -q tests/test_visualizacao_geometrica.py tests/test_visualizacao_integrada.py` e, depois, `python -m pytest -q` em `.venv` do VS Code.
4. Verificar `git diff --check` (eventual aviso CRLF não implica erro de espaços), `git diff --stat` e `git status --short`.
5. Registrar resultados no Markdown existente, **não** marcar P05 integrada se não houver confirmação dos testes no PC do usuário.

**Não executado:** OOF real, imagens reais de tese, F10/teste final, Etapa 08. A pasta `teste_final` continua contendo geometria de desenvolvimento F1–F9 (assunto P12).
