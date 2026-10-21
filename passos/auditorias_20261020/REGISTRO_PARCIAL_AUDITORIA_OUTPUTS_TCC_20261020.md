# TCC_EXPERIMENTAL_V3 — Registro parcial de auditoria dos artefatos `outputs/`

**Data:** 2026-10-20  
**Estado:** auditoria documental/estrutural dos artefatos enviados. **Não é validação de execução da pipeline atual.**  
**Intervenções no projeto:** nenhuma. Os arquivos enviados foram lidos em cópia isolada.

## 1. Arquivos recebidos e integridade

- Foram recebidos `outputs.rar` e `outputs(1).rar`.
- Os dois têm exatamente 29.635.952 bytes e SHA-256 idêntico: `a60f191de71d26a4846df88326fa4b94f69c87ce09e560a958b2ad56d32f4ea6`. **São cópias byte a byte do mesmo pacote; não há duas versões a conciliar.**
- Arquivo RAR v5; **258 entradas**: **220 arquivos regulares** e **38 diretórios**.
- Os 220 arquivos ocupam 42.607.004 bytes descompactados.
- Extensões: **110 `.npz`**, **47 `.png`**, **35 `.csv`**, **23 `.json`**, **3 `.yaml`**, **1 `.md`** e **1 `.html`**.
- Todos os **47 PNGs** passaram pela verificação de integridade estrutural feita com Pillow. Isso **não** equivale a aprovação visual nem valida exatidão de conteúdo.

## 2. Execuções historicamente registradas

| Pasta em `outputs/` | Arquivos | Imagens PNG | Estado no manifesto | Registros de desenvolvimento | Registros F10 reservados |
|---|---:|---:|---|---:|---:|
| `ensaio_pca_gmm_20261017_190802` | 52 | 0 | `concluido_06_03` | 2.700 | 185.239 |
| `ensaio_pca_gmm_20261017_193911` | 16 | 0 | `concluido_06_03` | 83.358 | 185.239 |
| `ensaio_pca_gmm_20261017_195252` | 152 | 47 | `concluido_06_03` | 166.716 | 185.239 |

As três execuções são do dia **17/10/2026** e seus manifestos identificam experimentos/ensaios; não são a execução integral da configuração atual recebida em `zips.rar`. O YAML guardado nas execuções é diferente (inclusive hash de arquivo) do `config_TCC_EXPERIMENTAL_V3_atualizado.yaml` enviado nesta conversa.

**Configuração escolhida no último ensaio (`195252`):** PCA 1D, GMM com quatro componentes, `covariance_type: diag`, modo NORMAL, limiar MCC OOF `0.010396020833223555`, `score_oof = 0.5011775390615202`. Esses são **valores históricos de ensaio**, não resultados finais de um modelo GMM FULL da versão atual.

## 3. Evidência de reserva de F10

Os manifestos inspecionados do ensaio `195252` registram:

- `oof/graficos_gerados/manifesto_graficos_oof_07_01.json`: `usa_f10: false`.
- `oof/graficos_gerados/manifesto_painel_00.json`: `origem: OOF_F1_F9`, `f10_utilizado: false`.
- `teste_final/graficos_gerados/manifesto_geometria_07_02.json`: `origem: desenvolvimento_F1_F9`, `f10_utilizado: false` e aviso de que os gráficos **não são métricas F10**.
- `visualizacao_consolidada/manifesto_integracao_07_03.json`: `f10_utilizado: false`.
- `visualizacao_consolidada/complementar_2d/manifesto_complementar_2d.json`: `origem: desenvolvimento_F1_F9`, `f10_utilizado: false`.

**Conclusão limitada:** os registros da execução afirmam que F10 não foi utilizado, apesar do nome da pasta `teste_final`. Não foi reproduzida a execução nem reprocessada a base real; o registro não é uma comprovação independente do isolamento de F10 além das evidências documentais e testes anteriores.

## 4. Saídas visuais observadas

Na execução `195252`:

- Em `oof/graficos_gerados/`: painel `00`, curva PR `01`, curva MCC `02`, matriz de confusão em 0,50 `03`, matriz em MCC ótimo `04`, distribuição de probabilidades `05`, comparação de hiperparâmetros `06`, além de ensaios adicionais e históricos de revisão.
- Em `teste_final/graficos_gerados/`: `07_risco_por_gaussiana.png`, `08_pesos_responsabilidades.png`, `09_medias_gaussianas.png`, `10_variancias_gaussianas.png`, `11_pca_cotovelo.png`, `12_pca_projecao_PC1.png`, `15_mapa_risco_1d_PC1.png`.
- Em `visualizacao_consolidada/`: galeria `index.html` com **14 imagens**, todas as referências HTML de imagem resolvendo para PNGs existentes (6 OOF, 7 geométricas e 1 mapa PCA2D complementar).
- Em `visualizacao_consolidada/complementar_2d/`: `16_mapa_complementar_PCA2_GMM2.png` com manifesto declarando **candidato não vencedor**, visualização somente complementar e **F10 não utilizado**.

### Evidências úteis para a auditoria posterior

1. **Painel 00 ausente da galeria deste ensaio:** existe `oof/graficos_gerados/00_painel_score_composto.png` e seu manifesto, porém `visualizacao_consolidada/index.html` e seu manifesto contêm somente imagens OOF 01–06, sem o painel 00. É uma **diferença comprovada nesse artefato histórico**. Verificar depois se deveria integrar a galeria e se a versão atual já resolve.
2. **Apenas histórico PCA/diag nas saídas recebidas:** não há PNGs dos gráficos GMM 01/04/07/08/09 nas versões atuais, nem t-SNE 02/13/14 em 1D–3D dentro deste pacote. **A ausência aqui não prova ausência no código**: o pacote é de ensaios anteriores às alterações subsequentes documentadas/conversadas.
3. **Mapa PCA2D complementar:** seu manifesto registra `fronteira_mcc_visivel: false`. O valor de limiar (`0.0068422638742083`) fica acima do máximo registrado da malha (`0.00529399488964602`), o que **explica geometricamente** não aparecer esse contorno na malha. Não concluir por isso que o cálculo do mapa esteja quebrado.
4. O diretório `teste_final` **pode sugerir erroneamente uso do F10** quando o leitor não abre os manifestos. Avaliar futuramente uma nomenclatura/nota documental mais explícita, sem renomear caminhos por conta própria.

## 5. O que continua sem comprovação por este pacote

- Que o **commit mais recente** produza esses mesmos arquivos, ou que já tenha executado o pipeline inteiro com GMM FULL e representações 1D/2D/3D.
- Que todos os gráficos aprovados visualmente em 18–19/10 estejam reproduzidos nesta pasta de saídas.
- Que a execução em bases reais dos testes `verificar_*_base_real.py` esteja aprovada.
- Que o Biplot PCA 1D/2D/3D já exista — ele continua **planejado**.

## 6. Próximos passos (sem executar reparos ainda)

1. Acrescentar estes achados ao documento acumulativo de pendências.
2. Se o usuário ainda tiver materiais relevantes (principalmente versões recentes de **galerias de teste, capturas, HTML ou artefatos produzidos após os ensaios de 17/10**), solicitá-los. Caso não existam, trabalhar com os scripts e testes recebidos, deixando explícito que os novos gráficos não possuem saídas históricas neste pacote.
3. Encerrar fase de recebimento quando o usuário confirmar; somente então consolidar as pendências com prioridade e aprovação.
4. Retomar Biplot PCA 1D → 2D → 3D: prévia sintética, avaliação visual e, após aprovação, integração e testes. Finalmente, auditoria bidirecional entre os quatro PDFs e a versão realmente implementada.

**Regra operacional:** não corrigir código, testes, YAML ou registros anteriores sem autorização, e não confundir evidência histórica, declaração de manifesto, teste automatizado e execução efetiva recente.
