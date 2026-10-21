# P06 — Auditoria inversa da pipeline para os quatro catálogos (21/10/2026)

## Referências efetivamente verificadas
- `ANALISES_CATALOGO_CONFUSAO_MCC_E_050(1).pdf` (16 páginas).
- `GMM_CATALOGO_GMM_ZONAS_MUDANCA_050_MCC(1).pdf` (33 páginas).
- `PCA_CATALOGO_GRAFICOS_11_12_histograma1D(1).pdf` (10 páginas).
- `TSNE_CATALOGO_GRAFICOS_2_13(1).pdf` (7 páginas).
- Código de trabalho P08+P05: `src/visualizacao_oof.py`, `src/visualizacao_geometrica.py`, `src/visualizacao_integrada.py`, `src/visualizacao_tsne_pdf.py`, `src/visualizacao_pca_biplot.py` e YAML. A suíte completa local do usuário passou com **434 testes**, mas a presente auditoria é estática.

## Recursos presentes no código e AUSENTES dos quatro PDFs

| Item existente | Evidência em `src/` e YAML | Condição / restrições | Decisão de documentação |
|---|---|---|---|
| **OOF 06 — comparação de hiperparâmetros** | `visualizacao_oof.py`: `ARQUIVOS['comparacao_hiperparametros']='06_comparacao_hiperparametros'`, função `_comparacao`, CSV `06_comparacao_hiperparametros_todos.csv`; chave YAML `comparacao_hiperparametros.habilitar: true`. | F1–F9, usa resultados da grade OOF. Imagem exibe até `maximo_configuracoes` (25 no YAML) e CSV lista todos. | Adicionar ao catálogo OOF numa futura revisão; o atual PDF atribui **06** à confusão por quadrante, enquanto o código implementa matrizes **03/04** e comparação **06**. **Não renumerar automaticamente** sem convenção aprovada. |
| **t-SNE 14 — diagnóstico KL/tempo** | `visualizacao_tsne_pdf.py::diagnostico_kl`, invocado em `visualizacao_geometrica.py::_tsne_diagnostico`; saída `14_tsne_diagnostico`; YAML `tsne_diagnostico`. | Apenas método TSNE, redutor ajustado em desenvolvimento. Exibir KL final se disponível, tempo e parâmetros; **não inventar histórico KL por iteração** indisponível. | Adicionar seção 14 no catálogo t-SNE na próxima revisão. |
| **PCA mapa 2D complementar 16** | `visualizacao_integrada.py::_config_candidata_2d`, `_complementar`; saída `16_mapa_complementar_PCA2_GMM*.png` e manifesto. | **Somente** quando o vencedor OOF usa PCA 1D e já existe um candidato PCA 2D na mesma grade. O gráfico é um diagnóstico **não oficial** com limiar OOF do próprio candidato; não substitui o vencedor. | Acrescentar seção condicional explicativa em catálogo PCA/galeria, sem confundir com fronteira oficial. |
| **PCA Biplot 1D/2D/3D** | `visualizacao_pca_biplot.py`, `visualizacao_geometrica.py` e `visualizacao_integrada.py`; YAML `pca_biplot.habilitar: true`; prefixo `PCA_BIPLOT_`, arquivos dimensionais `PCA_BIPLOT_1D/2D/3D` e dados associados. | Somente PCA e conforme dimensionalidade do vencedor, sem novo fit e sem F10. Cálculos de correlação/carregamentos baseiam-se em variáveis pré-processadas do mesmo ajuste PCA. | Adicionar seção com ilustrações 1D/2D/3D ao catálogo PCA; **numeração ainda provisória**, não alterar nomes existentes sem decisão. |

## Achado após a P05
O `GMM 08` foi alterado para comparativos lado a lado em 2D e 3D, compatíveis com o layout aprovado: `GMM08_mapa_risco_2d_comparativo` e `GMM08_mapa_risco_3d_comparativo`. O catálogo GMM original já contempla GMM 08, mas suas imagens ilustrativas não foram substituídas automaticamente. O usuário confirmou `434 passed in 95.17s` após aplicar o patch.

## Regras e pendências de escopo
- **P06 = auditar pipeline → PDFs**: as quatro lacunas estão identificadas, mas os PDFs originais **não foram reeditados**, nem houve implementação nova nesta pendência. Uma futura atualização dos catálogos exige aprovação específica e revisão visual.
- A numeração é específica de cada domínio: `02` ocorre em OOF e t-SNE; `06` tem colisão entre o PDF de confusão por quadrante e o gráfico de hiperparâmetros do código. Não renumerar precipitadamente.
- Os gráficos condicionais não são obrigatórios em cada execução. O `PCA Biplot` só é aplicável ao PCA; t-SNE KL só ao t-SNE; complementar 16 só à condição PCA 1D vencedor/PCA 2D candidato.
- O termo `teste_final` na pasta de geometria designa **artefatos de F1–F9** na implementação auditada; **não** certifica F10. Este problema de nomenclatura pertence à P12.
- Para o próximo teste integral é necessária uma rotina real e independente de F10, que **não foi comprovada** pela aprovação dos 434 testes.

## Status
**P06 — auditoria estática concluída; revisão gráfica dos PDFs registrada como trabalho futuro opcional e independente da execução da Etapa 08.** Nenhum código ou YAML experimental alterado por P06.
