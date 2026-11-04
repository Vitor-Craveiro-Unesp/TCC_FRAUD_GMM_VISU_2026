# -*- coding: utf-8 -*-
"""Dois ensaios INDEPENDENTES, PCA 1D e TSNE 1D, cada um 3% real.

Somente usa o Parquet real e folds mestres existentes. NAO mistura resultados
OOF/PCA/TSNE; nao altera o YAML original e nao reexecuta F10 antigo.

  python -B -m scripts.execucao.experimentos_reais_3pct_1d --planejar
  python -B -m scripts.execucao.experimentos_reais_3pct_1d --pca
  python -B -m scripts.execucao.experimentos_reais_3pct_1d --tsne
  python -B -m scripts.execucao.experimentos_reais_3pct_1d --validar

--pca e --tsne sao processos separados (isolamento de memoria). Repetir um
comando APROVADO apenas valida os artefatos, nao treina novamente.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import traceback
import uuid

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
YAML_ORIGINAL = ROOT / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
FRACAO = .03
SEMENTE = 42
SAIDA_MODELO = 'modelo_final_f1_f9'
SAIDA_F10 = 'avaliacao_f10_3pct'
SAIDA_GRAFICOS = 'graficos_f10_3pct'
SAIDA_GALERIA = 'galeria_3pct'
MARCADOR = 'TCC_EXPERIMENTAL_V3_3PCT_1D_REAL_V1'
IMAGENS_PREDITIVAS = (
    '00_painel_score_composto.png',
    '01_curva_precision_recall.png',
    '02_mcc_por_limiar.png',
    '03_matriz_confusao_05.png',
    '04_matriz_confusao_mcc.png',
    '05_distribuicao_probabilidades.png',
)
IMAGENS_GEOMETRICAS = (
    'GMM01_densidade_marginal_1d_{dim}.png',
    'GMM04_dominancia_1d.png',
    'GMM07_sensibilidade_1d_comparativo.png',
    'GMM08_mapa_risco_1d_comparativo.png',
)


def exigir(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def ler_json(p):
    ob = json.loads(Path(p).read_text(encoding='utf-8-sig'))
    exigir(isinstance(ob, dict), f'JSON invalido: {p}')
    return ob


def salvar_json(p, ob):
    with Path(p).open('x', encoding='utf-8') as arq:
        json.dump(ob, arq, ensure_ascii=False, indent=2, allow_nan=False)
        arq.write('\n')


def hash_arquivo(p):
    h = sha256()
    with Path(p).open('rb') as arq:
        for b in iter(lambda: arq.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def yaml_texto(cfg):
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)


def construir_config(metodo):
    exigir(metodo in ('pca', 'tsne'), 'Metodo desconhecido.')
    exigir(YAML_ORIGINAL.is_file(), f'YAML original nao encontrado: {YAML_ORIGINAL}')
    orig = yaml.safe_load(YAML_ORIGINAL.read_text(encoding='utf-8-sig'))
    exigir(isinstance(orig, dict), 'YAML original nao e um mapa.')
    cfg = deepcopy(orig)
    cfg['experimento']['nome'] = MARCADOR + '_' + metodo.upper()
    cfg['experimento']['semente'] = SEMENTE
    cfg['dados']['fracao_base'] = FRACAO
    cfg['divisao']['total_folds'] = 10
    cfg['divisao']['quantidade_folds_teste'] = 1
    cfg['divisao']['qual_fold_teste'] = 10
    cfg['divisao']['estratificada'] = True
    cfg['divisao']['embaralhar'] = True
    cfg['reducao']['metodo'] = metodo.upper()
    cfg['pca']['n_components'] = [1]
    cfg['tsne']['n_components'] = [1]
    # Preservar os demais hiperparametros e a grade de inicializacoes do YAML.
    cfg['gmm']['n_components'] = [4]
    cfg['gmm']['n_init'] = 5
    cfg['gmm']['covariance_type'] = 'full'
    cfg['amostragem']['modo'] = 'RUS'
    cfg['amostragem']['razao_negativos_positivos'] = 16
    cfg['amostragem']['aplicar_somente_treino'] = True
    # A finalizacao PCA embutida aceita somente 10%; aqui usamos o
    # modulo generico de treinamento final apos concluir o OOF de 3%.
    cfg['execucao']['finalizacao_pca1d_automatica'] = False
    cfg['execucao']['diretorio_retomada'] = None
    cfg['execucao']['retomar'] = True
    cfg['execucao']['usar_cache'] = True
    cfg['execucao']['proteger_resultados'] = True
    cfg['saidas']['diretorio_base'] = 'outputs'
    # Evitar que uma versao de YAML desligue as visualizacoes OOF.
    cfg['visualizacao']['habilitar'] = True
    cfg['saidas']['salvar_graficos'] = True
    from src.validacao_oof import grade_candidatos_oof
    from src import orquestracao_oof as orq
    orq.validar_operacao(cfg)
    grade = list(grade_candidatos_oof(cfg))
    exigir(bool(grade), 'Grade OOF vazia.')
    exigir(all(x['metodo'] == metodo.upper() and
               int(x['reducao']['n_components']) == 1 and
               x['gmm']['covariance_type'] == 'full' and
               int(x['gmm']['n_components']) == 4 for x in grade),
           'Grade contem candidato inesperado.')
    return cfg, grade


def localizar_experimento(cfg, metodo):
    """Procura SOMENTE pastas que contenham o YAML exato deste ensaio.

    Duas finalizadas com mesmo YAML constituem ambiguidade: nao escolhemos
    arbitrariamente o resultado mais novo nem fazemos overwrite.
    """
    base = ROOT / 'outputs'
    finalizados, parciais = [], []
    for pattern, dest in ((f'ensaio_{metodo}_gmm_*', finalizados),
                          (f'em_andamento_ensaio_{metodo}_gmm_*', parciais)):
        for p in sorted(base.glob(pattern)):
            arquivo = p / 'configuracao.yaml'
            if not arquivo.is_file():
                continue
            try:
                config_saved = yaml.safe_load(arquivo.read_text(encoding='utf-8-sig'))
                if config_saved == cfg:
                    dest.append(p)
            except (OSError, yaml.YAMLError):
                pass
    exigir(len(finalizados) <= 1 and len(parciais) <= 1,
           f'Ambiguidade: varias execucoes 3% {metodo}: {finalizados + parciais}')
    exigir(not(finalizados and parciais),
           'Pastas completa e em andamento coexistem para o mesmo config: revisar.')
    return finalizados[0] if finalizados else None, parciais[0] if parciais else None


def imprimir_plano(metodo):
    cfg, grade = construir_config(metodo)
    concluido, parcial = localizar_experimento(cfg, metodo)
    print(f'=== PLANO {metodo.upper()} 1D REAL 3% ===', flush=True)
    print('PARQUET_REAL=' + str(ROOT / cfg['dados']['arquivo']), flush=True)
    print('YAML_ORIGINAL_NAO_ALTERADO=SIM', flush=True)
    print('FRACAO_DESENVOLVIMENTO=0.03', flush=True)
    print('FRACAO_F10=0.03_ESTRATIFICADA', flush=True)
    print('OOF_FOLDS=F1..F9', flush=True)
    print('TESTE_RESERVADO=F10', flush=True)
    print(f'CANDIDATOS_OOF={len(grade)}', flush=True)
    print(f'AJUSTES_GMM_OOF_PLANEJADOS={len(grade)*9}', flush=True)
    print('SEMENTE=42', flush=True)
    print('GMM=FULL_K4_NINIT5_RUS16', flush=True)
    print('DESTINO=outputs/ensaio_' + metodo + '_gmm_<datahora_fim>/', flush=True)
    print('OOF_JA_CONCLUIDO=' + str(concluido or ''), flush=True)
    print('OOF_EM_ANDAMENTO=' + str(parcial or ''), flush=True)
    print('F10_JA_ACESSADO_NO_PROJETO=SIM_AVALIACAO_EXPLORATORIA', flush=True)
    print('STATUS=PLANEJAMENTO_SEM_DADOS_NEM_TREINAMENTO', flush=True)
    return cfg, grade


def executar_oof(metodo, cfg):
    from src import orquestracao_oof as orq
    concluido, parcial = localizar_experimento(cfg, metodo)
    if concluido is None:
        print(f'[OOF {metodo.upper()}] Inicio/retomada do OOF sobre Parquet real.', flush=True)
        # Nunca alterar config_TCC_EXPERIMENTAL_V3_atualizado.yaml no disco.
        with tempfile.TemporaryDirectory(prefix='tcc_3pct_yaml_') as tmp:
            config_temp = Path(tmp) / 'config_3pct.yaml'
            config_temp.write_text(yaml_texto(cfg), encoding='utf-8')
            original = orq.ARQUIVO_CONFIG
            try:
                orq.ARQUIVO_CONFIG = str(config_temp.resolve())
                orq.main(['--executar'], raiz=ROOT)  # --ensaio limitaria a 300/fold
            finally:
                orq.ARQUIVO_CONFIG = original
        concluido, parcial = localizar_experimento(cfg, metodo)
    exigir(concluido is not None and parcial is None,
           'OOF ainda nao foi finalizado; veja traceback/cache na pasta em_andamento.')
    man = ler_json(concluido / 'manifesto.json')
    # Windows pode serializar o YAML temporario com CRLF em vez de LF.
    # Nunca ignorar o hash: validar o conteudo E os dois hashes possiveis.
    config_registrada = yaml.safe_load(
        (concluido / 'configuracao.yaml').read_text(encoding='utf-8-sig')
    )
    exigir(config_registrada == cfg,
           'Configuracao do OOF diverge do planejamento atual.')
    texto_yaml = yaml_texto(cfg)
    sha_yaml_validos = {
        sha256(texto_yaml.encode('utf-8')).hexdigest(),
        sha256(texto_yaml.replace('\n', '\r\n').encode('utf-8')).hexdigest(),
    }
    exigir(man.get('status') == 'concluido_06_03' and
           man.get('tipo') == 'ensaio' and
           man.get('sha256_yaml') in sha_yaml_validos,
           'OOF nao tem status/tipo/hash YAML correto.')
    print('[OOF] YAML estrutural e SHA LF/CRLF conferidos; sem refazer OOF.', flush=True)
    if man.get('visualizacoes') != 'concluidas':
        print(f'[OOF {metodo.upper()}] Retomando somente graficos/galeria OOF.', flush=True)
        orq.executar_visualizacoes(concluido)
    exigir(ler_json(concluido / 'manifesto.json').get('visualizacoes') == 'concluidas',
           'OOF visual nao concluido.')
    return concluido


def validar_oof_f1f9(pasta, cfg, metodo):
    from src import orquestracao_oof as orq
    from src.ponte_oof_final_tsne import verificar_oof_concluido_tsne
    man = ler_json(pasta / 'manifesto.json')
    escolha = ler_json(pasta / 'vencedor' / 'metricas' / 'melhor_configuracao.json')
    with np.load(pasta / 'vencedor/oof/predicoes.npz', allow_pickle=False) as npz:
        ids = np.asarray(npz['linha_original'], dtype=np.int64)
        folds_oof = np.asarray(npz['fold'], dtype=np.int8)
        y_oof = np.asarray(npz['y_true'], dtype=np.uint8)
    exigir(len(ids) > 0 and np.all(np.diff(ids) > 0) and
           set(np.unique(folds_oof).tolist()) == set(range(1, 10)) and
           (folds_oof != 10).all(), 'Indices/folds OOF inconsistentes.')
    exigir(len(ids) == man['n_desenvolvimento_trabalhado'] and
           len(ids) == man['registros_desenvolvimento'], 'N OOF divergente.')
    exigir(escolha['teste_final_usado'] is False and
           escolha['configuracao']['metodo'] == metodo.upper() and
           escolha['configuracao']['reducao']['n_components'] == 1,
           'Escolha vencedora nao corresponde ao metodo 1D.')
    arquivo_base = ROOT / cfg['dados']['arquivo']
    print(f'[{metodo.upper()}] Conferindo hashes reais do Parquet e dos folds.', flush=True)
    folds_master, arquivo_folds, hash_parq = orq._identificar_folds(ROOT, cfg, arquivo_base)
    exigir(hash_parq == man['sha256_parquet'] and
           hash_arquivo(arquivo_folds) == man['sha256_folds'],
           'Parquet/folds foram alterados desde o OOF.')
    exigir(ids[-1] < len(folds_master) and
           np.array_equal(folds_master[ids], folds_oof), 'IDs OOF divergem do fold mestre.')
    if metodo == 'tsne':
        verificar_oof_concluido_tsne(pasta)
    return {'ids': ids, 'folds': folds_oof, 'y': y_oof,
            'folds_master': folds_master, 'arquivo_parquet': arquivo_base,
            'escolha': escolha, 'manifesto': man}


def modulo_modelo(metodo):
    if metodo == 'pca':
        from src import treinamento_final_pca as mod
    else:
        from src import treinamento_final_tsne as mod
    return mod


def validar_modelo(pasta_modelo, cfg, escolha, metodo, n_oof):
    mod = modulo_modelo(metodo)
    manifest = ler_json(pasta_modelo / 'manifesto_modelo_final.json')
    exigir(float(manifest['fracao_base']) == FRACAO and
           int(manifest['n_treino_original']) == n_oof and
           manifest['id_candidato_oof'] == escolha['id_candidato'] and
           float(manifest['limiar_mcc_oof_congelado']) == float(escolha['limiar_mcc_oof']) and
           hash_arquivo(pasta_modelo / 'modelo_final_f1_f9.joblib') == manifest['sha256_modelo_final'],
           'Modelo final existente possui proveniencia/hash diferentes.')
    return manifest


def executar_modelo(pasta, cfg, info, metodo):
    mod = modulo_modelo(metodo)
    out = pasta / 'vencedor' / 'teste_final' / SAIDA_MODELO
    if out.is_dir():
        validar_modelo(out, cfg, info['escolha'], metodo, len(info['ids']))
        print(f'[{metodo.upper()}] Modelo final existente: nao repetir treinamento.', flush=True)
        return out
    exigir(not out.exists(), f'Caminho do modelo existe e nao e pasta: {out}')
    print(f'[{metodo.upper()}] Lendo SOMENTE {len(info["ids"])} transacoes reais F1..F9.', flush=True)
    from src import orquestracao_oof as orq
    df = orq._ler_desenvolvimento(info['arquivo_parquet'], info['ids'])
    exigir(np.array_equal(df[cfg['dados']['coluna_alvo']].to_numpy(dtype=np.uint8), info['y']),
           'Rotulos lidos de F1-F9 divergem das previsoes OOF.')
    prov = {
        'sha256_parquet': info['manifesto']['sha256_parquet'],
        'sha256_folds': info['manifesto']['sha256_folds'],
        'sha256_yaml': info['manifesto']['sha256_yaml'],
        'experimento_oof': str(pasta.resolve()),
    }
    print(f'[{metodo.upper()}] Reajuste final exclusivo F1..F9, sem F10.', flush=True)
    pacote = mod.ajustar_final_f1_f9(
        df, info['folds'], cfg, info['escolha'], prov,
        registrar_historico_em=False,
    )
    del df
    aux = out.with_name('.modelo_3pct_incompleto_' + uuid.uuid4().hex[:12])
    print(f'[{metodo.upper()}] Salvando pacote final protegido; verificacao de leitura.', flush=True)
    mod.salvar_pacote(pacote, aux)
    recarregado = mod.carregar_pacote_confiavel(aux)
    exigir(recarregado.limiar_oof == pacote.limiar_oof and
           recarregado.n_treino_original == len(info['ids']),
           'Modelo recarregado diverge do pacote treinado.')
    exigir(not out.exists(), 'Modelo final apareceu durante a gravacao.')
    aux.rename(out)
    validar_modelo(out, cfg, info['escolha'], metodo, len(info['ids']))
    print(f'[{metodo.upper()}] MODELO_F1_F9=APROVADO', flush=True)
    return out


def escolher_indices_f10(folds_master, arquivo_parquet, alvo, *, fracao=FRACAO):
    """Amostra estratificada FIXA F10, sem escolher por score/metricas."""
    from src import orquestracao_oof as orq
    y = orq._ler_alvo(arquivo_parquet, alvo)
    exigir(len(y) == len(folds_master), 'Alvo completo e folds master divergentes.')
    pos_f10 = np.flatnonzero(folds_master == 10)
    exigir(len(pos_f10) > 0, 'Fold 10 inexistente.')
    ys = y[pos_f10]
    exigir(np.isin(ys, [0, 1]).all() and set(np.unique(ys)) == {0, 1},
           'O fold 10 nao possui ambas as classes.')
    rng = np.random.default_rng(SEMENTE)
    partes = []
    for c in (0, 1):
        pos = pos_f10[ys == c]
        n = max(2, round(len(pos) * fracao))
        exigir(n <= len(pos), 'Amostra F10 impossivel para a classe ' + str(c))
        partes.append(rng.choice(pos, size=n, replace=False))
    ids = np.sort(np.concatenate(partes)).astype(np.int64)
    exigir(np.all(np.diff(ids) > 0) and np.all(folds_master[ids] == 10),
           'Amostra F10 contem IDs invalidos ou fora do F10.')
    return ids, y[ids].astype(np.uint8)


def validar_f10(pasta, cfg, info, metodo, model_dir):
    dest = pasta / 'vencedor' / 'teste_final' / SAIDA_F10
    m = ler_json(dest / 'manifesto_f10_3pct.json')
    exigir(m['status'] == 'APROVADA' and m['metodo'] == metodo.upper() and
           m['fracao_desenvolvimento'] == FRACAO and m['fracao_f10'] == FRACAO and
           m['id_candidato_oof'] == info['escolha']['id_candidato'] and
           m['tau_oof_congelado'] == info['escolha']['limiar_mcc_oof'] and
           m['sha256_modelo'] == hash_arquivo(model_dir / 'modelo_final_f1_f9.joblib'),
           'Manifesto F10 nao corresponde ao modelo/OOF.')
    for nome, digest in m['sha256_arquivos'].items():
        f = dest / nome
        exigir(f.is_file() and hash_arquivo(f) == digest,
               'Arquivo F10 ausente ou alterado: ' + nome)
    return dest, m


def executar_f10(pasta, cfg, info, metodo, model_dir):
    dest = pasta / 'vencedor' / 'teste_final' / SAIDA_F10
    if dest.is_dir():
        validado, man = validar_f10(pasta, cfg, info, metodo, model_dir)
        print(f'[{metodo.upper()}] F10 anterior validado; NAO repetir inferencia.', flush=True)
        return validado
    exigir(not dest.exists(), 'Pasta F10 existe mas nao e diretorio; nao sobrescrever.')
    # Nao ler nenhum rotulo F10 ANTES do modelo final persistido.
    validar_modelo(model_dir, cfg, info['escolha'], metodo, len(info['ids']))
    ids, y_referencia = escolher_indices_f10(info['folds_master'], info['arquivo_parquet'], cfg['dados']['coluna_alvo'])
    print(f'[{metodo.upper()}] Avaliando {len(ids)} registros REAIS de F10; 3% por classe.', flush=True)
    from src import orquestracao_oof as orq
    df = orq._ler_desenvolvimento(info['arquivo_parquet'], ids)
    y = df[cfg['dados']['coluna_alvo']].to_numpy(dtype=np.uint8)
    exigir(np.array_equal(y, y_referencia), 'Rotulos F10 divergem do Parquet original.')
    mod = modulo_modelo(metodo)
    pacote = mod.carregar_pacote_confiavel(model_dir)
    exigir(pacote.escolha_oof['id_candidato'] == info['escolha']['id_candidato'],
           'Pacote final nao coincide com candidato OOF vencedor.')
    alvo = cfg['dados']['coluna_alvo']
    excl = [alvo, cfg['dados'].get('coluna_id'), *cfg['dados'].get('colunas_excluir', [])]
    x = df.drop(columns=[c for c in excl if c], errors='ignore').copy()
    exigir(list(x.columns) == pacote.encoder.colunas_, 'Features F10 divergentes do treino.')
    xenc = pacote.encoder.transform(x)
    xpre = pacote.imputador.transform(xenc)
    red = pacote.pca if metodo == 'pca' else pacote.tsne
    z = red.transform(xpre)
    exigir(list(z.columns) == (['PC1'] if metodo == 'pca' else ['TSNE1']),
           'Embedding F10 nao e 1D esperado.')
    from src.probabilidade import PreditorProbabilisticoGMM
    s = PreditorProbabilisticoGMM(pacote.gmm, pacote.risco).prever_probabilidade(z)
    scores = s.to_numpy(dtype=np.float64)
    exigir(len(scores) == len(ids) and np.isfinite(scores).all() and
           np.all((scores >= 0) & (scores <= 1)), 'Probabilidades F10 invalidas.')
    tau = float(pacote.limiar_oof)
    if metodo == 'pca':
        from src.avaliacao_final_pca import avaliar_scores_com_limiar_oof_congelado
    else:
        from src.avaliacao_final_tsne import avaliar_scores_com_limiar_oof_congelado
    met = avaliar_scores_com_limiar_oof_congelado(y, scores, limiar_mcc_oof=tau)
    met.update({
        'metodo': metodo.upper(),
        'dimensao': 1,
        'id_candidato_oof': info['escolha']['id_candidato'],
        'fracao_desenvolvimento': FRACAO,
        'fracao_f10': FRACAO,
        'amostra_estratificada_por_classe': True,
        'semente_amostragem_f10': SEMENTE,
        'avaliacao_exploratoria_f10_previamente_utilizado_no_projeto': True,
        'testes_anteriores_nao_reutilizados_nesta_execucao': True,
        'limiar_otimizado_no_f10': False,
    })
    pred = pd.DataFrame({
        'linha_original': ids, 'fold': np.full(len(ids), 10, dtype=np.uint8),
        'y_true': y, 'prob_fraude': scores,
        'predicao_tau_0_50': (scores >= .5).astype(np.uint8),
        'predicao_tau_mcc_oof': (scores >= tau).astype(np.uint8),
    })
    c = z.columns[0]
    coord = pd.DataFrame({'linha_original': ids, c: z[c].to_numpy(dtype=np.float64)})
    stage = dest.with_name('.avaliacao_f10_3pct_incompleta_' + uuid.uuid4().hex[:12])
    stage.mkdir(exist_ok=False)
    pred.to_csv(stage / 'predicoes_f10.csv', index=False, float_format='%.17g')
    coord.to_csv(stage / f'coordenadas_{c}_f10.csv', index=False, float_format='%.17g')
    np.save(stage / 'indices_f10_amostra.npy', ids, allow_pickle=False)
    salvar_json(stage / 'metricas_f10.json', met)
    arquivos = [p for p in stage.iterdir() if p.is_file()]
    manifesto = {
        'versao': 'DOIS_ENSAIOS_REAIS_3PCT_V1', 'status': 'APROVADA',
        'metodo': metodo.upper(), 'fracao_desenvolvimento': FRACAO,
        'fracao_f10': FRACAO, 'n_f10': len(ids), 'n_fraudes_f10': int(y.sum()),
        'id_candidato_oof': info['escolha']['id_candidato'],
        'tau_oof_congelado': tau,
        'sha256_indices_f10': sha256(ids.tobytes()).hexdigest(),
        'sha256_modelo': hash_arquivo(model_dir / 'modelo_final_f1_f9.joblib'),
        'sha256_parquet': info['manifesto']['sha256_parquet'],
        'sha256_folds': info['manifesto']['sha256_folds'],
        'avaliacao_exploratoria': True,
        'sem_fit_no_f10': True, 'sem_otimizacao_de_tau_no_f10': True,
        'sha256_arquivos': {p.name: hash_arquivo(p) for p in arquivos},
        'gerado_em_utc': datetime.now(timezone.utc).isoformat(),
    }
    salvar_json(stage / 'manifesto_f10_3pct.json', manifesto)
    exigir(not dest.exists(), 'Pasta final F10 surgiu durante a escrita; abortar.')
    stage.rename(dest)
    validar_f10(pasta, cfg, info, metodo, model_dir)
    print(f'[{metodo.upper()}] AP_F10={met["ap"]:.7f}', flush=True)
    print(f'[{metodo.upper()}] MCC_TAU_OOF_F10={met["mcc_limiar_oof"]:.7f}', flush=True)
    print(f'[{metodo.upper()}] LOGLOSS_F10={met["logloss"]:.7f}', flush=True)
    print(f'[{metodo.upper()}] SCORE_COMPOSTO_F10={met["score_composto_limiar_oof"]:.7f}', flush=True)
    return dest


@contextmanager
def rotulos_preditivos_f10():
    from matplotlib.text import Text
    from src import visualizacao_oof as vo
    original = vo._salvar
    substituicoes = (
        ('Curva Precision–Recall do modelo selecionado em OOF', 'Curva Precision–Recall no F10 real (3%)'),
        ('Resultados da validação OOF', 'Resultados do teste F10 (3%)'),
        ('AP calculada sobre as predições OOF.', 'AP calculada nas predições F10.'),
        ('MCC calculado no limiar ótimo das predições OOF.', 'MCC F10 com τ* congelado no OOF.'),
        ('LogLoss calculado sobre as predições OOF.', 'LogLoss nas predições F10.'),
        ('OOF F1–F9', 'F10 real (3%)'),
        ('F10 não utilizado', 'F10 real (3%)'),
        ('Limiar ótimo do MCC (τ*)', 'Limiar τ* congelado em OOF'),
        ('Limiar ótimo (τ*)', 'Limiar τ* congelado em OOF'),
        ('Limiar ótimo do MCC (τ*)', 'Limiar τ* congelado no OOF'),
        ('LIMIAR ÓTIMO DO MCC:', 'LIMIAR τ* CONGELADO EM OOF:'),
    )
    def salvar(fig, destino, *, dpi):
        for t in fig.findobj(match=Text):
            texto = t.get_text()
            novo = texto
            for a, b in substituicoes:
                novo = novo.replace(a, b)
            if novo != texto:
                t.set_text(novo)
        return original(fig, destino, dpi=dpi)
    vo._salvar = salvar
    try:
        yield
    finally:
        vo._salvar = original


@contextmanager
def rotulos_geometria_f10(tau, y, p, metodo):
    from matplotlib.text import Text
    from sklearn.metrics import matthews_corrcoef
    from src import visualizacao_geometrica as geo
    from src import visualizacao_gmm07 as g07
    original_calcular = geo._avaliar_cortes_oof
    original_guardar = geo._guardar
    original_rotulo = g07._rotulo_mcc
    def cortes(av, limiar):
        exigir(av.get('origem') == 'f10_real_3pct' and limiar == tau,
               'Origem ou limiar invalido na geometria F10.')
        return (float(matthews_corrcoef(y, p >= .5)),
                float(matthews_corrcoef(y, p >= tau)))
    def rotulo(av):
        exigir(av.get('origem') == 'f10_real_3pct', 'Rotulo geometrico invalido.')
        return 'MCC F10', 'MCC do F10 com limiar τ* selecionado em OOF.'
    def guardar(fig, dest, base, ext, dpi):
        for t in fig.findobj(match=Text):
            atual = t.get_text()
            novo = atual.replace('MCC OOF', 'MCC F10')
            if metodo == 'tsne':
                novo = novo.replace('PC1 |', 'TSNE1 |')
            novo = novo.replace('Corte ótimo MCC OOF', 'Limiar τ* congelado em OOF')
            novo = novo.replace('F10 não utilizado', 'F10 real utilizado apenas para avaliação')
            if novo != atual:
                t.set_text(novo)
        return original_guardar(fig, dest, base, ext, dpi)
    geo._avaliar_cortes_oof = cortes
    geo._guardar = guardar
    g07._rotulo_mcc = rotulo
    try:
        yield
    finally:
        geo._avaliar_cortes_oof = original_calcular
        geo._guardar = original_guardar
        g07._rotulo_mcc = original_rotulo


def validar_graficos(pasta, metodo, f10):
    destino = pasta / 'vencedor' / 'teste_final' / SAIDA_GRAFICOS
    m = ler_json(destino / 'manifesto_graficos_f10_3pct.json')
    exigir(m.get('metodo') == metodo.upper() and
           m.get('sha256_predicoes_f10') == hash_arquivo(f10 / 'predicoes_f10.csv'),
           'Grafico F10 anterior nao pertence a esta avaliacao.')
    for nome, digest in m['sha256_arquivos'].items():
        p = destino / nome
        exigir(p.is_file() and hash_arquivo(p) == digest,
               'Grafico/CSV alterado: ' + nome)
    return destino, m


def executar_graficos(pasta, cfg, info, metodo, model_dir, f10):
    destino = pasta / 'vencedor' / 'teste_final' / SAIDA_GRAFICOS
    if destino.is_dir():
        validar_graficos(pasta, metodo, f10)
        print(f'[{metodo.upper()}] Graficos F10 existentes validados.', flush=True)
        return destino
    exigir(not destino.exists(), 'Destino F10 graficos ja existe e nao e diretorio.')
    import matplotlib
    matplotlib.use('Agg')
    from src import visualizacao_oof as vo
    from src import visualizacao_geometrica as geo
    from src import visualizacao_gmm01 as g01
    from src import visualizacao_gmm04 as g04
    from src import visualizacao_gmm07 as g07
    from sklearn.metrics import matthews_corrcoef
    mod = modulo_modelo(metodo)
    pacote = mod.carregar_pacote_confiavel(model_dir)
    pred = pd.read_csv(f10 / 'predicoes_f10.csv')
    c = 'PC1' if metodo == 'pca' else 'TSNE1'
    coord = pd.read_csv(f10 / f'coordenadas_{c}_f10.csv')
    exigir(np.array_equal(pred['linha_original'].to_numpy(dtype=np.int64),
                          coord['linha_original'].to_numpy(dtype=np.int64)),
           'Coordenadas F10 desalinhadas das previsoes.')
    y = pred['y_true'].to_numpy(dtype=np.uint8)
    p = pred['prob_fraude'].to_numpy(dtype=np.float64)
    x = pd.DataFrame({c: coord[c].to_numpy(dtype=np.float64)})
    tau = pacote.limiar_oof
    me = ler_json(f10 / 'metricas_f10.json')
    v = pd.Series({'ap': me['ap'], 'mcc': me['mcc_limiar_oof'],
                   'logloss': me['logloss'], 'score': me['score_composto_limiar_oof'],
                   'prevalencia': me['prevalencia']})
    escolha = {'limiar_mcc_oof': tau}
    gamma = pacote.gmm.responsabilidades(x)
    from src.probabilidade import calcular_probabilidades
    conferido = calcular_probabilidades(gamma, pacote.risco.riscos_).to_numpy(dtype=np.float64)
    exigir(np.max(np.abs(conferido - p)) <= 1e-6 and
           np.array_equal(conferido >= tau, p >= tau) and
           np.array_equal(conferido >= .5, p >= .5),
           'Scores reconstituídos da geometria F10 nao coincidem.')
    data = {'coord': x, 'y': y, 'scores': p, 'gamma': gamma,
            'gmm': pacote.gmm, 'rh': pacote.risco,
            'redutor': pacote.pca if metodo == 'pca' else pacote.tsne,
            'limiar_oof': tau}
    stage = destino.with_name('.graficos_f10_3pct_incompletos_' + uuid.uuid4().hex[:12])
    stage.mkdir(exist_ok=False)
    dpi = int(cfg['visualizacao']['padrao'].get('dpi', 180))
    print(f'[{metodo.upper()}] Gerando 6 figuras preditivas F10.', flush=True)
    with rotulos_preditivos_f10():
        vo._plot_painel_00(cfg, y, p, v, escolha, stage, 'png', dpi, True)
        vo._plot_pr(cfg, y, p, v, escolha, stage, 'png', dpi, True)
        vo._plot_mcc(cfg, y, p, v, escolha, stage, 'png', dpi, True)
        vo._matriz(cfg, y, p, .5, 'matriz_confusao_05', stage, 'png', dpi, True)
        vo._matriz(cfg, y, p, tau, 'matriz_confusao_mcc', stage, 'png', dpi, True)
        vo._distribuicao(cfg, y, p, escolha, stage, 'png', dpi, True)
    print(f'[{metodo.upper()}] Gerando GMM01/GMM04/GMM07/GMM08 no F10.', flush=True)
    g01.gerar(cfg, data, stage, 'png', dpi, True)
    g04.gerar(cfg, data, stage, 'png', dpi, True)
    with rotulos_geometria_f10(tau, y, p, metodo):
        avaliacao = {'origem': 'f10_real_3pct'}
        g07.gerar(cfg, data, escolha, stage, 'png', dpi, True, avaliacao=avaliacao)
        geo._fronteira(cfg, data, escolha, stage, 'png', dpi, True, avaliacao=avaliacao)
    # Histograma do metodo (sem eixos ficticios); sem ajuste ou inferencia.
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    z = x[c].to_numpy(dtype=np.float64)
    edges = np.linspace(float(z.min()) if z.min() < z.max() else float(z.min())-.5,
                        float(z.max()) if z.min() < z.max() else float(z.max())+.5, 37)
    centers = .5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    for cl, name, color in ((0, 'Legitimas', '#1f77b4'), (1, 'Fraudes', '#d62728')):
        valores = z[y == cl]
        d, _ = np.histogram(valores, bins=edges, density=True)
        ax.bar(centers, d, width=.92*(edges[1]-edges[0]), color=color,
               alpha=.35, label=f'{name} (n={len(valores)})')
        if len(valores) >= 3:
            ax.plot(centers, gaussian_filter1d(d, 1.15), color=color, lw=1.5)
    ax.set(xlabel=f'{c} (projecao F10)', ylabel='Densidade por classe',
           title=f'{metodo.upper()} 1D — histograma F10 real (3%)')
    ax.legend(); ax.grid(alpha=.1)
    fig.tight_layout()
    hist = f'{"12_pca_projecao_PC1" if metodo == "pca" else "13_tsne_mapa_TSNE1"}.png'
    fig.savefig(stage / hist, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    pngs = sorted(x for x in stage.glob('*.png'))
    esperados = set(IMAGENS_PREDITIVAS) | {nome.format(dim=c) for nome in IMAGENS_GEOMETRICAS} | {hist}
    exigir({x.name for x in pngs} == esperados,
           'Quantidade/nomes PNG F10 inesperados: ' + str({x.name for x in pngs} ^ esperados))
    for img in pngs:
        exigir(img.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n', 'PNG corrompido: '+img.name)
    files = {x.name: hash_arquivo(x) for x in stage.iterdir() if x.is_file()}
    salvar_json(stage / 'manifesto_graficos_f10_3pct.json', {
        'status': 'APROVADA', 'metodo': metodo.upper(),
        'n_imagens': len(pngs), 'imagens': [x.name for x in pngs],
        'sha256_predicoes_f10': hash_arquivo(f10 / 'predicoes_f10.csv'),
        'sha256_arquivos': files,
        'sem_fit': True, 'limiar_f10_reotimizado': False,
    })
    exigir(not destino.exists(), 'Destino graficos apareceu durante a gravacao.')
    stage.rename(destino)
    validar_graficos(pasta, metodo, f10)
    print(f'[{metodo.upper()}] FIGURAS_F10={len(pngs)}/{len(esperados)}', flush=True)
    return destino


def executar_galeria(pasta, cfg, metodo, f10, graficos):
    dest = pasta / 'vencedor' / 'teste_final' / SAIDA_GALERIA
    if dest.is_dir():
        m = ler_json(dest / 'manifesto_galeria_3pct.json')
        exigir(m['metodo'] == metodo.upper() and
               m['sha256_predicoes_f10'] == hash_arquivo(f10 / 'predicoes_f10.csv'),
               'Galeria anterior nao corresponde ao F10 atual.')
        exigir((dest / 'index.html').is_file() and
               hash_arquivo(dest/'index.html') == m['sha256_html'],
               'Galeria HTML ausente ou alterada.')
        for nome, digest in m['hashes_png'].items():
            exigir((dest/nome).is_file() and hash_arquivo(dest/nome)==digest,
                   'PNG da galeria ausente/alterado: ' + nome)
        print(f'[{metodo.upper()}] Galeria existente validada.', flush=True)
        return dest
    exigir(not dest.exists(), 'Destino galeria ja existe mas nao e pasta.')
    stage = dest.with_name('.galeria_3pct_incompleta_' + uuid.uuid4().hex[:12])
    stage.mkdir(exist_ok=False)
    m = ler_json(f10 / 'metricas_f10.json')
    import shutil
    imgs = sorted(graficos.glob('*.png'))
    for src in imgs:
        shutil.copy2(src, stage / src.name)
        exigir(hash_arquivo(src) == hash_arquivo(stage / src.name), 'Hash da imagem copiada divergente.')
    n = m['n']; fraudes = m['n_fraudes']
    cards = ''.join(f'<div class="card"><strong>{escape(t)}</strong><div>{float(v):.7f}</div></div>'
                    for t, v in [('AP', m['ap']), ('MCC tau* OOF',m['mcc_limiar_oof']),
                                 ('LogLoss',m['logloss']), ('Score composto',m['score_composto_limiar_oof'])])
    groups = [('Indicadores preditivos', IMAGENS_PREDITIVAS),
              ('Geometria GMM 1D', tuple(n.format(dim=('PC1' if metodo=='pca' else 'TSNE1'))
                                          for n in IMAGENS_GEOMETRICAS)),
              ('Projecao 1D', ('12_pca_projecao_PC1.png' if metodo=='pca'
                              else '13_tsne_mapa_TSNE1.png',))]
    parts = []
    for heading, nomes in groups:
        figures = '\n'.join(f'<figure><img loading="lazy" src="{escape(file)}" alt="{escape(file)}"><figcaption>{escape(file)}</figcaption></figure>'
                            for file in nomes)
        parts.append(f'<section><h2>{escape(heading)}</h2><div class="figs">{figures}</div></section>')
    page = f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{metodo.upper()} GMM 1D — 3% real</title>
<style>body{{background:#f4f7fa;color:#1f3048;font:15px system-ui,Arial;margin:0}}
main{{max-width:1380px;margin:auto;padding:24px}}.header{{background:#102b50;color:white;padding:20px;border-radius:12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}}
.card,section{{background:white;border:1px solid #dce3ef;border-radius:12px;padding:16px;margin:12px 0}}
.card strong{{display:block;font-size:12px}}.card div{{font-size:24px;font-weight:700}}
.figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:12px}}
figure{{margin:0;border:1px solid #dce3ef;border-radius:8px;overflow:hidden;background:white}}
img{{width:100%;height:auto;cursor:zoom-in}}figcaption{{font-size:12px;padding:8px;word-break:break-all}}
a{{color:#245b91}}.note{{color:#566478}}</style></head><body><main>
<div class="header"><h1>Ensaio real {metodo.upper()} + GMM 1D — 3%</h1>
<p>OOF F1–F9 e teste F10 real, em pasta independente. Experimento exploratorio; F10 foi utilizado antes no projeto.</p></div>
<div class="cards">{cards}</div>
<p>F10: {n:,} transacoes · {fraudes} fraudes · limiar tau* OOF congelado = {float(m['limiar_mcc_oof_congelado']):.7f}</p>
<p><a href="../../visualizacao/galeria/index.html">Abrir galeria OOF (F1–F9)</a> ·
<a href="../{SAIDA_F10}/metricas_f10.json">Metricas F10 (JSON)</a> ·
<a href="../{SAIDA_F10}/predicoes_f10.csv">Predicoes F10 (CSV)</a></p>
<p class="note">As gaussianas foram ajustadas apenas em F1–F9; o F10 contribuiu apenas com coordenadas, rotulos e metricas. Nenhum limiar foi selecionado em F10.</p>
{''.join(parts)}</main></body></html>'''
    (stage/'index.html').write_text(page, encoding='utf-8')
    salvar_json(stage / 'manifesto_galeria_3pct.json', {
        'status':'APROVADA', 'metodo':metodo.upper(),
        'sha256_predicoes_f10':hash_arquivo(f10 / 'predicoes_f10.csv'),
        'sha256_html':hash_arquivo(stage/'index.html'),
        'hashes_png':{p.name:hash_arquivo(p) for p in stage.glob('*.png')},
    })
    exigir(not dest.exists(), 'Galeria apareceu durante escrita.')
    stage.rename(dest)
    print(f'[{metodo.upper()}] GALERIA_F10=' + str(dest/'index.html'), flush=True)
    return dest


def _executar_sem_lock(metodo):
    # GUARDA_LAYOUT_TSNE_YAML_V1_20261103 / executor 3%
    from src.garantia_layout_tsne import verificar_layout_tsne
    cfg, grade = imprimir_plano(metodo)
    verificar_layout_tsne(cfg, fase='executor_3pct')
    print(f'[{metodo.upper()}] Iniciando OOF/retomada.', flush=True)
    pasta = executar_oof(metodo, cfg)
    info = validar_oof_f1f9(pasta, cfg, metodo)
    modelo = executar_modelo(pasta, cfg, info, metodo)
    f10 = executar_f10(pasta, cfg, info, metodo, modelo)
    graficos = executar_graficos(pasta, cfg, info, metodo, modelo, f10)
    galeria = executar_galeria(pasta, cfg, metodo, f10, graficos)
    validar_f10(pasta, cfg, info, metodo, modelo)
    validar_graficos(pasta, metodo, f10)
    print('==================================================', flush=True)
    print(f'{metodo.upper()}_STATUS=APROVADA_OOF_F10_3PCT', flush=True)
    print(f'{metodo.upper()}_EXPERIMENTO=' + str(pasta), flush=True)
    print(f'{metodo.upper()}_GALERIA_OOF=' + str(pasta/'vencedor/visualizacao/galeria/index.html'), flush=True)
    print(f'{metodo.upper()}_GALERIA_F10=' + str(galeria/'index.html'), flush=True)
    print(f'{metodo.upper()}_N_OOF=' + str(len(info['ids'])), flush=True)
    print(f'{metodo.upper()}_N_F10=' + str(ler_json(f10/'metricas_f10.json')['n']), flush=True)
    print('OOF_F10_ORIGEM=PARQUET_REAL', flush=True)
    print('YAML_ORIGINAL_ALTERADO=NAO', flush=True)
    return 0


def executar(metodo):
    """Impossibilita duas chamadas concorrentes ao mesmo metodo/pasta."""
    base = ROOT / 'outputs'
    base.mkdir(exist_ok=True)
    lock = base / ('.ensaio_real_3pct_' + metodo + '.lock')
    exigir(not lock.exists(), f'Lock de {metodo} presente: confirmar ausencia de processo ativo.')
    with lock.open('x', encoding='utf-8') as f:
        f.write(f'Execucao {metodo.upper()} 3%, sem concorrencia.\n')
    try:
        return _executar_sem_lock(metodo)
    finally:
        lock.unlink(missing_ok=True)


def validar_concluido(metodo):
    cfg, _ = construir_config(metodo)
    pasta, parcial = localizar_experimento(cfg, metodo)
    exigir(pasta is not None and parcial is None, 'Execucao OOF ainda nao concluiu.')
    man = ler_json(pasta/'manifesto.json')
    exigir(man['status'] == 'concluido_06_03' and man['visualizacoes'] == 'concluidas',
           'OOF/visualizacoes nao concluidos.')
    escol = ler_json(pasta/'vencedor/metricas/melhor_configuracao.json')
    mod = modulo_modelo(metodo)
    model = pasta/'vencedor/teste_final'/SAIDA_MODELO
    m = validar_modelo(model, cfg, escol, metodo, man['n_desenvolvimento_trabalhado'])
    info = {'escolha': escol}
    f10, mf = validar_f10(pasta, cfg, info, metodo, model)
    graf, mg = validar_graficos(pasta, metodo, f10)
    gal = pasta/'vencedor/teste_final'/SAIDA_GALERIA
    exigir((gal/'index.html').is_file() and len(mg['imagens']) == 11,
           'Galeria ou 11 imagens F10 ausentes.')
    print(metodo.upper() + '_VALIDACAO=OOF_F10_GALERIAS_APROVADOS', flush=True)
    print(metodo.upper() + '_PASTA=' + str(pasta), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument('--planejar', action='store_true')
    mx.add_argument('--pca', action='store_true')
    mx.add_argument('--tsne', action='store_true')
    mx.add_argument('--validar', action='store_true')
    args = ap.parse_args(argv)
    try:
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        if args.planejar:
            for k in ('pca','tsne'):
                imprimir_plano(k)
        elif args.pca:
            executar('pca')
        elif args.tsne:
            executar('tsne')
        else:
            for k in ('pca', 'tsne'):
                validar_concluido(k)
        return 0
    except Exception as err:
        print('STATUS=BLOQUEADO_SEM_SOBRESCRITA', file=sys.stderr, flush=True)
        print('MOTIVO=' + str(err), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
