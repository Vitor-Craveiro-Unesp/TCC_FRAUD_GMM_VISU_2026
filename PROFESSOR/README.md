# Referencias do professor - cluster hpcdmc

Os arquivos originais foram copiados sem alteracoes de conteudo.

- `guia_hpcdmc_original_2014.pdf`: manual do cluster (datado de janeiro de 2014).
- `exemplo_R/code_pogamp10.R`: programa R fornecido como exemplo; **nao e parte da metodologia do TCC**.
- `exemplo_R/script2.batch`: exemplo SLURM associado ao programa R; **nao deve ser submetido para executar o TCC**.

## Pontos documentados, ainda a confirmar no servidor atual

- Acesso remoto: SSH com chave na porta 40665, host historico 200.145.184.171.
- Submissao: SLURM por `sbatch`, consulta `squeue`, `sinfo`, cancelamento por `scancel`.
- Manual de 2014: CentOS 6.5; 6 nos, cerca de 30 CPUs disponiveis e 32 GB RAM por no; `/home` compartilhado por NFS.
- Manual (pag. 5): particao `XE52690`, `-N 1`, `-n 1`, `-c 1`, `--no-requeue`, `--share`.
- Exemplo R: conta `guilherme`, pasta `/home/guilherme/code_pogamp10`, particao `turtle`.

**Nao copiar `-A guilherme`, `-D /home/guilherme/...`, `-p turtle` ou `-p XE52690` para um job TCC sem verificar no cluster atual.** O guia e antigo, e as duas referencias discordam sobre a particao.

## Relacao com o codigo do TCC

O projeto ja contem `scripts/cluster/submeter.sh` e `scripts/cluster/tcc_generic.batch`, que chamam `src.execucao_controlada`. Esses lancadores devem continuar a ler a configuracao congelada do YAML; **nao** podem chamar o programa em R. A configuracao dos recursos SLURM sera definida apos consulta de `sinfo`, `scontrol show partition`, `python`, GLIBC, armazenamento e regras informadas pelo professor/administrador.

Para diagnostico somente de leitura apos obter acesso legitimo ao cluster:

```bash
bash scripts/cluster/diagnostico_hpcdmc.sh
```

Esse comando nao instala programas, nao altera o YAML, nao submete jobs, nao acessa dados do TCC e nao precisa de `sudo`. Revise a saida antes de compartilhar porque pode conter o nome da conta, caminho da home e nomes de particoes.

**Atencao:** nunca incluir a chave privada SSH em Git, ZIP do projeto ou mensagens; o manual apenas descreve seu uso local.
