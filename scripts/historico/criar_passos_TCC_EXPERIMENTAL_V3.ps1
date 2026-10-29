# TCC_EXPERIMENTAL_V3 - Estrutura de acompanhamento
# Execute no PowerShell. Cria somente pastas e arquivos .md vazios.
# Nenhum arquivo existente sera sobrescrito.

$projeto = 'C:\Users\Vitor Craveiro\Desktop\ClusterPosMac\MEU PROJETO\TCC_EXPERIMENTAL_V3'

if (-not (Test-Path -LiteralPath $projeto -PathType Container)) {
    throw "Pasta do projeto nao encontrada: $projeto"
}

$passos = Join-Path $projeto 'passos'
New-Item -ItemType Directory -Path $passos -Force | Out-Null

$etapas = [ordered]@{
    '01_configuracao' = @(
        '01_estrutura_e_ambiente',
        '02_yaml_e_dependencias'
    )
    '02_dados' = @(
        '01_auditoria_da_base',
        '02_teste_final_e_nove_folds'
    )
    '03_preprocessamento' = @(
        '01_encoders_sem_vazamento',
        '02_imputacao_e_normalizacao',
        '03_undersampling_opcional'
    )
    '04_reducao_dimensional' = @(
        '01_pca',
        '02_tsne'
    )
    '05_gmm_soft_risk' = @(
        '01_gmm_sem_rotulos',
        '02_soft_counts_e_risco_historico',
        '03_previsao_ponderada'
    )
    '06_validacao_e_selecao' = @(
        '01_previsoes_oof',
        '02_prioris_limiar_mcc_e_score',
        '03_grid_de_hiperparametros',
        '04_teste_final'
    )
    '07_resultados' = @(
        '01_metricas_e_artefatos',
        '02_graficos',
        '03_pasta_datahora_fim'
    )
    '08_execucao_local' = @(
        '01_testes_anti_vazamento',
        '02_benchmark',
        '03_experimento_base_completa'
    )
}

$indice = Join-Path $passos 'CONTEXTO_GERAL.md'
if (-not (Test-Path -LiteralPath $indice)) {
    New-Item -ItemType File -Path $indice | Out-Null
}

foreach ($etapa in $etapas.Keys) {
    $dirEtapa = Join-Path $passos $etapa
    New-Item -ItemType Directory -Path $dirEtapa -Force | Out-Null

    $resumo = Join-Path $dirEtapa 'resumo.md'
    if (-not (Test-Path -LiteralPath $resumo)) {
        New-Item -ItemType File -Path $resumo | Out-Null
    }

    foreach ($subpasso in $etapas[$etapa]) {
        $dirSubpasso = Join-Path $dirEtapa $subpasso
        New-Item -ItemType Directory -Path $dirSubpasso -Force | Out-Null

        $registro = Join-Path $dirSubpasso 'registro.md'
        if (-not (Test-Path -LiteralPath $registro)) {
            New-Item -ItemType File -Path $registro | Out-Null
        }
    }
}

Write-Host 'Estrutura de passos criada/verificada com sucesso.'
Write-Host "Local: $passos"
Write-Host 'Arquivos Markdown ja existentes foram preservados.'
tree $passos /F
