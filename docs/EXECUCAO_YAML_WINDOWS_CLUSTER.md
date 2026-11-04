# Execucao TCC por YAML — Windows e Linux/SLURM

**Escopo do patch consolidado:** lancadores, congelamento do YAML, identificação de execução, diretores sem colisao, limites de CPU e contratos compartilhados de finalização para PCA/t-SNE 1D/2D/3D. Os algoritmos estatísticos não foram substituídos; alguns textos de metadados e paletas de visualização foram atualizados para refletir o K escolhido.

## Contrato

- `config_TCC_EXPERIMENTAL_V3_atualizado.yaml` continua sendo a fonte editavel. Salvar esse arquivo **nao** dispara treinamento.
- Cada **nova execucao** captura os bytes do YAML na partida (local) ou **antes de `sbatch`** (cluster); o pipeline trabalha com esse snapshot, armazenado em `outputs/.submissoes/`.
- Os manifestos e o `configuracao.yaml` dentro do experimento continuam sendo gerados pelo orquestrador. O SHA do YAML original e diferente do YAML normalizado salvo no experimento, por projeto.
- O nome da pasta experimental utiliza o formato historico; colisões de timestamp recebem um sufixo unico. Nenhuma pasta de resultados existentes e removida nem sobrescrita.
- O wrapper controlado impede **dois jobs simultaneos na mesma raiz**: o segundo falha antes de iniciar treinamento, preservando a alocacao. Nao executa tarefas em paralelo na mesma raiz; para paralelismo de jobs independentes sera necessario isolar workspaces e verificar contratos de paths.
- O snapshot original deve ser guardado para **retomada explicita** do mesmo experimento; nao use o YAML que voce editou posteriormente.
- `execucao.retomar=true` e `usar_cache=true` devem ter sido definidos **antes** da primeira execucao para habilitar a retomada OOF. Isso nao torna uma nova submissao automaticamente uma retomada.

## Windows — PowerShell (na raiz do projeto)

```powershell
.\.venv\Scripts\python.exe -m src.execucao_controlada --planejar
.\.venv\Scripts\python.exe -m src.execucao_controlada --executar
```

A linha `--executar` chama o orquestrador de sempre (`src.orquestracao_oof`). O comando historico `python -m src.orquestracao_oof --executar` continua valido, mas **nao** utiliza o bloqueio de concorrencia implementado no wrapper; para novas execucoes prefira a forma controlada.

## Cluster — roteiro inicial pendente da documentacao oficial

1. No diretorio de trabalho visivel para os nós, disponibilize codigo e arquivos `data/raw` e `data/splits`, verificando os hashes. O script nao baixa datasets nem recria folds.
2. Crie um ambiente Python Linux compatível com o projeto (a definicao exata depende da faculdade). O script espera por padrao `.venv/bin/activate`; opcionalmente defina `TCC_VENV_DIR` para o diretorio real do ambiente.
3. Confirme recursos e flags de seu cluster na documentacao antes de submeter jobs longos.
4. Planeje sem treinar:

```bash
source .venv/bin/activate
python -m src.execucao_controlada --planejar
```

5. Submeta com opcoes de recursos **obtidas do manual do cluster** (os exemplos abaixo usam variaveis que voce deve preencher; nenhuma particao foi presumida):

```bash
bash scripts/cluster/submeter.sh --cpus-per-task="$CPUS" --mem="$MEMORIA" --time="$TEMPO"
```

`submeter.sh` cria um snapshot antes de invocar `sbatch`. O script `tcc_generic.batch` so ativa o ambiente, define os limites numericos, verifica o SHA-256 e executa `python -m src.execucao_controlada --executar --config <snapshot>`. Nenhum metodo, dimensao, K, fracao ou hiperparametro aparece fixado no `.batch`.

Os logs SLURM ficam em `outputs/logs/tcc_<jobid>.out` e `.err`. O ID e o caminho do snapshot sao impressos ao submeter.

### Retomada autorizada, quando a fase OOF estiver `em_andamento`

Use **exatamente o snapshot YAML da submissao original**, não o YAML editavel atualizado:

```bash
bash scripts/cluster/submeter.sh --retomar "outputs/em_andamento_ensaio_tsne_gmm_..." "outputs/.submissoes/job_.../config.yaml" --cpus-per-task="$CPUS" --mem="$MEMORIA" --time="$TEMPO"
```

O orquestrador verificara `chave_execucao`, `status`, hash original do YAML, cache e pasta escolhida. Para OOF ja finalizado e apenas F10/galeria pendente, continuam existindo comandos especificos `--concluir-teste-final`/`--concluir-galeria-f10` com as configuracoes congeladas; nao transforme uma avaliacao ja concluida em novo treinamento.

## Grade dinamica e finalizacao automatica — patch consolidado

A finalizacao automatica continua sendo despachada pela chave historica
`execucao.finalizacao_pca1d_automatica`. O nome da chave nao limita mais a PCA 1D.
O finalizador ativo e escolhido pela combinacao de metodo e dimensao do
YAML congelado. Cada finalizador valida que o candidato selecionado pertence
a grade OOF, que o limiar MCC veio de OOF e que o F10 nao participou da escolha.

O patch remove as exigencias artificiais de `K=4`, `n_init=5`, RUS `16:1`,
sementes `42` e frações `1%`, `3%`, `10%`, mantendo `covariance_type=full` e
F1-F9/F10. Exemplo: `gmm.n_components: [3, 4, 5]` e `dados.fracao_base: 0.05`
produzem uma grade em OOF; a finalizacao consome **somente o candidato
vencedor** (K, inicializacao, RUS, n_init, etc), e o F10 continua sendo
exploratório com limiar congelado.

**Domínio validado:** os novos testes exercitam K=3/4/5, n_init=3/5,
RUS=10/16, fração de desenvolvimento=0.05/0.01, frações de F10 distintas,
sementes diferentes, manifestos, pré-validações, geometria e integração
sintética com seis modalidades. A execução do openTSNE real, o Parquet
real, todos os HTMLs e os recursos do SLURM ainda dependem da homologação
no ambiente do usuário/cluster. Um K maior pode falhar legitimamente
quando excede o tamanho da amostra RUS ou quando o EM nao converge.

**Duas verificações antigas do roteador t-SNE esperam rejeitar K diferente
de quatro ou RUS diferente de 16.** Elas nao representam a regra atual,
portanto continuam falhando de propósito no checkpoint de testes antigo.
Nao relaxar outros validadores nem alterar os testes historicos sem uma
revisao separada de seu significado.

O comando simples `python -m src.orquestracao_oof --executar` continua
funcionando, mas prefira `python -m src.execucao_controlada --executar` para
snapshot e lock. `--planejar` nao executa treinamento.

## Verificacoes pendentes de cluster

- Confirmar particao, conta, `--cpus-per-task`, `--mem`, `--time`, cgroups e filesystem com lock compartilhado.
- Instalar versoes suportadas das bibliotecas Linux; confirmar timezone IANA e disponibilidade de `sha256sum`/`mktemp`.
- Avaliar pequenos testes de inicializacao e saidas **sem repetir os seis experimentos**.
- Comparar SHA-256 de entradas e identidade dos folds; diferencas numéricas entre sistemas requerem diagnostico.
