# TCC_EXPERIMENTAL_V3 — Entrega operacional ao professor (HPC/SLURM)

**Objetivo:** o professor configura seu acesso autorizado ao cluster e executa o **mesmo** pipeline cientifico do Windows, controlado pelo `config_TCC_EXPERIMENTAL_V3_atualizado.yaml`. Nenhuma credencial e inserida no repositório. Os scripts `operar_tcc.sh` e `conectar_professor.ps1` sao apenas interfaces de operacao.

## 1. O que ja esta pronto

- `python -m src.orquestracao_oof --planejar` / `--executar`: comandos historicos do pipeline.
- `python -m src.execucao_controlada --planejar` / `--executar`: interface recomendada, com snapshot do YAML e bloqueio entre execucoes na mesma raiz.
- `scripts/cluster/submeter.sh`: copia e verifica YAML **antes** do `sbatch`, de modo que o job nao leia mudancas posteriores.
- `scripts/cluster/tcc_generic.batch`: ativa um ambiente virtual Linux, aplica `SLURM_CPUS_PER_TASK` como limite de threads e roda a interface controlada no job.
- `scripts/cluster/diagnostico_hpcdmc.sh`: consulta ambiente do cluster sem treinamento.
- `scripts/cluster/operar_tcc.sh`: atalho operacional para diagnostico, planejamento, submissao, retomada e fila.

## 2. Dados sigilosos e acesso (duas coisas distintas)

**Credenciais de acesso:** senha, chave privada SSH e quaisquer tokens nao devem ser copiados para este projeto, enviados ao GitHub ou enviados ao aluno. Somente o professor as mantem no proprio computador/ambiente autorizado. O helper opcional Windows usa a chave **por caminho local**, sem grava-la:

```powershell
.\scripts\cluster\conectar_professor.ps1 -Usuario 'USUARIO_DA_FACULDADE' -ChavePrivada "$HOME\.ssh\id_rsa"
```

O servidor e a porta predefinidos no helper (`200.145.184.171:40665`) sao retirados do **guia historico de 2014**; confirmar endereco, porta e fingerprint com o administrador antes de utilizar. Tambem pode fazer SSH manualmente.

**Dataset privado:** o arquivo de dados deve permanecer no armazenamento permitido do cluster; **nao** versionar o Parquet. O YAML aponta para `dados.arquivo` relativo a raiz do TCC (historicamente `data/raw/dataset_raw.parquet`). Os indices de folds e seus manifestos em `data/splits/` tambem sao necessarios, e devem corresponder **exatamente** ao mesmo dataset e a sua ordem de linhas. **Um dataset novo do professor nao e intercambiavel** com os folds e hashes do dataset historico: o pipeline deve recusar essa combinacao. A preparacao cientifica autorizada de folds para outra base e uma etapa separada, nunca uma substituicao silenciosa ao submeter jobs.

## 3. Antes de qualquer instalacao ou submissao no cluster

A documentacao `welcome - cluster` e de 2014: CentOS 6.5, particao `XE52690`; o exemplo R `script2.batch` cita `turtle`. **Nao presumir que nenhuma das duas particoes ainda exista.** O professor, ja conectado ao nó mestre, executa:

```bash
cd /CAMINHO/REAL/TCC_EXPERIMENTAL_V3
bash scripts/cluster/operar_tcc.sh diagnostico
```

O professor confirma com a administracao: distribuicao Linux, glibc, versao de Python, bibliotecas instalaveis, particao (`sinfo`), conta SLURM, quantidade de CPUs, memoria e tempo permitidos; compartilhamento `/home` e permissoes de escrita. Python 3.13 e dependencias cientificas atuais **podem nao funcionar** no CentOS 6.5 historico. Nao se deve instalar ou atualizar bibliotecas do sistema no nó mestre sem permissao. O diagnóstico nao submete jobs.

## 4. Preparar o Python (somente se compatibilidade confirmada)

Os comandos abaixo sao ilustrativos para um Python 3 compativel ja disponivel, e devem ser adaptados ao ambiente real:

```bash
cd /CAMINHO/REAL/TCC_EXPERIMENTAL_V3
python3 -m venv meu_env
source ./meu_env/bin/activate
python --version
python -m pip install -r requirements.txt
export TCC_VENV_DIR="$PWD/meu_env"
```

Se nao houver Python/bibliotecas compatíveis, interromper e falar com a administracao do cluster. A criacao de `meu_env` e instalacao de dependencias nao acontecem automaticamente por este kit. Nao copiar `.venv` do Windows para o Linux. `meu_env/` deve permanecer fora do Git.

## 5. Comandos diarios para o professor

**Planejar sem treinar**, usando o YAML editavel atual:

```bash
cd /CAMINHO/REAL/TCC_EXPERIMENTAL_V3
source ./meu_env/bin/activate
python -m src.execucao_controlada --planejar
# equivale ao planejamento cientifico existente:
python -m src.orquestracao_oof --planejar
```

**Submeter NOVO experimento:** informar os recursos reais fornecidos pela administracao, por exemplo por variaveis definidas no shell; estes valores nao sao fornecidos pelo kit:

```bash
CPUS='PREENCHER'
MEMORIA='PREENCHER'
TEMPO='PREENCHER'
export TCC_VENV_DIR="$PWD/meu_env"
bash scripts/cluster/operar_tcc.sh executar \
  --cpus-per-task="$CPUS" --mem="$MEMORIA" --time="$TEMPO"
# Se exigido pela administracao, adicionar --partition=... e/ou --account=...
```

O comando `operar_tcc.sh executar` chama `submeter.sh`, que congela o YAML, calcula seu SHA-256 e chama `sbatch scripts/cluster/tcc_generic.batch`. O job executa automaticamente `python -m src.execucao_controlada --executar` **no nó de calculo**. Nao executar outro `python --executar` no nó mestre depois do `sbatch`.

**Consultar jobs e logs:**

```bash
bash scripts/cluster/operar_tcc.sh fila
# substituir ID pelo numero retornado na submissao:
tail -n 50 outputs/logs/tcc_ID.out
tail -n 50 outputs/logs/tcc_ID.err
```

Para cancelar um job, o professor usa `scancel ID`, apos conferir o numero do job.

**Retomar:** apenas com o snapshot original e pasta em andamento, quando a configuracao original havia habilitado `usar_cache=true` e `retomar=true`. Consultar `docs/EXECUCAO_YAML_WINDOWS_CLUSTER.md`; nao basta repetir `executar`.

## 6. O que o YAML controla

Metodo PCA/t-SNE, dimensao 1D/2D/3D, frações, `n_components`, `n_init`, RUS, semente, parametros da reducao, graficos e finalizacao. Esses parametros **nao** sao fixados no `.batch`. Cada nova submissao captura o YAML corrente. Salvar o arquivo **nao inicia** automaticamente um experimento.

O isolamento F1–F9/F10 e a selecao do limiar por MCC OOF continuam no pipeline. Cada job cria novos resultados e mantem artefatos e manifestos; a mesma raiz evita dois jobs simultaneos por lock (a confiabilidade do lock em NFS deve ser validada no cluster).

## 7. Checklist de entrega

- [ ] Conta SSH autorizada, endereco e fingerprint confirmados com TI.
- [ ] Saida do `diagnostico_hpcdmc.sh` revisada.
- [ ] Python e dependencias instalados em ambiente proprio autorizado.
- [ ] Parquet privado e folds compatíveis no destino definido pelo YAML, com hashes verificados.
- [ ] `python -m src.execucao_controlada --planejar` conclui com o metodo esperado.
- [ ] Valores SLURM (particao/conta/CPUs/memoria/tempo) confirmados.
- [ ] Primeiro job pequeno autorizado, acompanhado de logs e manifestos.

**Estado deste kit:** arquivos preparados e validacao local de sintaxe/simulacao; nao instalado nem executado em cluster real, portanto nao e uma homologacao HPC.
