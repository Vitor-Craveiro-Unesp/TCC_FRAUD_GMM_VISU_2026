# P06 — Atualização dos quatro catálogos PDF (21/10/2026)

## Escopo e natureza das figuras
Os quatro PDFs originais foram preservados integralmente nas respectivas versões ATUALIZADO_P06_20261021, acrescentando-se um suplemento claramente identificado. **Nenhuma figura suplementar contém resultados reais do TCC**: são exclusivamente modelos visuais sintéticos, seguindo a estrutura dos gráficos implementados. Não houve execução F10 nem retreinamento de modelos. O código, os YAMLs e os testes locais não foram modificados por esta edição dos PDFs.

## Relação dos PDFs
- **OOF:** original 16 páginas + 2 páginas (OOF 06 comparação de hiperparâmetros e correspondência da numeração de matriz de confusão 06 no catálogo vs 03/04 no código).
- **GMM:** original 33 páginas + 2 páginas (imagens aprovadas na P05 do GMM 08 em 2D 1x2 e em 3D 3x2).
- **PCA:** original 10 páginas + 4 páginas (Biplots 1D/2D/3D prévias aprovadas e mapa PCA 16 complementar condicional, exemplo sintético).
- **t-SNE:** original 7 páginas + 1 página (14 diagnóstico KL final, tempo e parâmetros; explicitamente sem histórico KL inventado).

## Referências da implementação
- `src/visualizacao_oof.py`: OOF 06 comparação candidatos; curva, MCC, LogLoss e score; vencedor OOF; CSV completo.
- `src/visualizacao_geometrica.py`: GMM08; comparativos 2D, 3D; fronteiras de τ=0,50 e τ* MCC OOF; risco GMM FULL.
- `src/visualizacao_integrada.py`: mapa PCA 16 de candidato 2D alternativo, somente se vencedor PCA 1D; galeria.
- `src/visualizacao_pca_biplot.py`: Biplots dimensionais em PCA com máximo de setas no YAML; sem número oficial atribuído.
- `src/visualizacao_tsne_pdf.py`: t-SNE14, KL final quando disponível, tempo e parâmetros; não sintetiza histórico da divergência.

## Situação e limitações
- A seleção de limiar e hiperparâmetros utiliza OOF F1–F9; F10 reservado e não avaliado pela edição de catálogos.
- As figuras dos suplementos são sintéticas; no teste experimental verdadeiro devem ser substituídas/confrontadas com gráficos gerados pela pipeline.
- Nenhum gráfico/numeração pré-existente dos PDFs foi apagado ou substituído; conflitos de nomenclatura foram **explicados**, não resolvidos por renumeração silenciosa.
- A P06 fica como **documentação visual atualizada**; não comprova execução da Etapa 08.
- Arquivos de ilustração suplementar ficam em `outputs/prototipos_catalogos_p06/` (não são resultados oficiais).

## Verificação técnica
- Contagem de páginas das quatro fontes e das atualizadas conferida automaticamente.
- Todas as páginas originais preservadas por cópia PDF não destrutiva; anexos apenas adicionados ao final.
- Suplementos renderizados para inspeção visual de legibilidade e sem sobreposição.
- Para instalar, extrair ZIP na raiz do projeto preservando `passos/` e `outputs/`; não sobrescrever os PDFs originais.
