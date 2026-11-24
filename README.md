# TCC - Auditoria Cientifica e Reprodutibilidade

## Objetivo

Pacote de auditoria do Trabalho de Conclusao de Curso
do Bacharelado em Estatistica da UNESP, FCT.

O projeto investiga escores individuais de risco de fraude
em transacoes com cartoes de credito, utilizando PCA ou
t-SNE em 1D, 2D e 3D, associados a modelos de misturas
gaussianas com matrizes de covariancia FULL.

## Conteudo

- src/: implementacao computacional do pipeline.
- scripts/: dependencias auxiliares selecionadas.
- configs/execucoes_oficiais/: 12 configuracoes congeladas.
- resultados_referencia/: metricas dos 72 candidatos,
  resultados das grades e evidencias da avaliacao em F10.
- tests/: testes cientificos selecionados.
- docs/manifesto_datasets.csv: hashes dos datasets.
- requirements.txt: dependencias declaradas do projeto.

## Metodologia

As configuracoes fornecidas utilizam o modo NORMAL,
sem Random Undersampling no ajuste do GMM.

Os riscos historicos sao estimados utilizando
responsabilidades probabilisticas e observacoes
do respectivo treinamento original.

A selecao utiliza predicoes out-of-fold (OOF).
A avaliacao final documentada refere-se ao
modelo PCA 3D HAV em F10.

## Dados

Origem publica:
https://www.kaggle.com/datasets/kartik2112/fraud-detection/data

Arquivos esperados pelo pipeline:

data/raw/dataset_raw.parquet
data/whav/dataset_whav.parquet

Os arquivos Parquet nao estao incluidos neste repositorio.

Os hashes SHA-256 das versoes utilizadas estao
documentados em docs/manifesto_datasets.csv.

As transformacoes dos arquivos publicos originais
ate RAW e HAV ainda exigem verificacao documental
para assegurar reconstrucao exata.

## Ambiente

A instalacao declarada esta em requirements.txt.

O projeto possui documentacao historica de Python 3.13.3,
enquanto o ambiente local examinado utiliza Python 3.14.7.

A compatibilidade integral e a reproducao independente
ainda nao foram certificadas.

## Conferencia inicial

Instalar as dependencias:

python -m pip install -r requirements.txt

Planejar um cenario sem iniciar o treinamento:

python -m src.orquestracao_oof --planejar --config configs/execucoes_oficiais/PCA_3D_RAW_NORMAL.yaml

O comando acima e fornecido para verificacao do fluxo,
mas ainda nao foi validado neste pacote isolado.

## Limitacoes

- Os datasets devem ser obtidos separadamente.
- Nem todos os testes foram executados neste pacote.
- Algumas fixtures podem exigir complementacao.
- Os artefatos binarios completos dos modelos nao
  estao incluidos.
- A reproducao integral ainda depende de verificacao.
- Os resultados de referencia nao constituem
  comprovacao independente de reproducibilidade.

O objetivo deste repositorio e permitir uma auditoria
externa da implementacao, das configuracoes e das
evidencias experimentais disponibilizadas.
