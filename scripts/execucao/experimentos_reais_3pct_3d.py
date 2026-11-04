# -*- coding: utf-8 -*-
"""Dois ensaios INDEPENDENTES, PCA 3D e TSNE 3D, cada um 3% real.

Somente usa o Parquet real e folds mestres existentes. NAO mistura resultados
OOF/PCA/TSNE; nao altera o YAML original e nao reexecuta F10 antigo.

  python -B -m scripts.execucao.experimentos_reais_3pct_3d --planejar
  python -B -m scripts.execucao.experimentos_reais_3pct_3d --pca
  python -B -m scripts.execucao.experimentos_reais_3pct_3d --tsne
  python -B -m scripts.execucao.experimentos_reais_3pct_3d --validar

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
MARCADOR = 'TCC_EXPERIMENTAL_V3_3PCT_3D_REAL_BASE_V1'
IMAGENS_PREDITIVAS = (
    '00_painel_score_composto.png',
    '01_curva_precision_recall.png',
    '02_mcc_por_limiar.png',
    '03_matriz_confusao_05.png',
    '04_matriz_confusao_mcc.png',
    '05_distribuicao_probabilidades.png',
)
IMAGENS_GEOMETRICAS = ('GMM01_densidade_marginal_3d_',
                       'GMM04_dominancia_3d.png',
                       'GMM07_sensibilidade_3d_comparativo.png',
                       'GMM08_mapa_risco_3d_comparativo.png')


@contextmanager
def proteger_visualizacoes_tsne_3d():
    """Evita rotular eixos t-SNE como PC no codigo interativo legado PCA3D.

    GMM01 marginais, GMM04 cortes, GMM07 cortes e GMM08 generico
    usam os dados t-SNE originais. Complementos exclusivamente PCA3D
    nao sao fabricados para TSNE3D.
    """
    from src import visualizacao_gmm01_3d as g01_3d
    from src import visualizacao_gmm04_3d as g04_3d
    from src import visualizacao_gmm07_3d as g07_3d
    from src import visualizacao_gmm07 as g07
    from src import visualizacao_geometrica as geo
    from matplotlib.text import Text
    save1=g01_3d.gerar_complementares
    save4=g04_3d.gerar_interativo
    save7=g07_3d.gerar_3d
    savefig=geo._guardar
    def guardar_3d_sem_rodape(fig, destino, stem, ext, dpi):
        if stem=='13_tsne_mapa_TSNE1_TSNE2_TSNE3':
            for t in fig.findobj(match=Text):
                old=t.get_text()
                if old.startswith('13 — t-SNE 3D'):
                    t.set_text('PROJECAO DO EMBEDDING t-SNE 3D')
                elif ('n_components=3 | perplexity=' in old or
                      ('Desenvolvimento F1' in old and 'F10' in old) or
                      ('PRÉVIA SINTÉTICA' in old and 'distâncias' in old)):
                    t.set_text('')
            fig.subplots_adjust(top=.92,bottom=.10)
        return savefig(fig,destino,stem,ext,dpi)
    def nao_aplicavel(*args, **kwargs):
        return None
    def g07_generico(config, data, escolha, planos, medias, destino, ext, dpi,
                     avaliacao, mcc_fixo, mcc_otimo):
        return g07._grafico(config, data, escolha, planos, medias,
                           destino=destino, ext=ext, dpi=dpi,
                           mcc_fixo=mcc_fixo, mcc_otimo=mcc_otimo,
                           avaliacao=avaliacao)
    g01_3d.gerar_complementares=nao_aplicavel
    g04_3d.gerar_interativo=nao_aplicavel
    g07_3d.gerar_3d=g07_generico
    geo._guardar=guardar_3d_sem_rodape
    try:
        yield
    finally:
        g01_3d.gerar_complementares=save1
        g04_3d.gerar_interativo=save4
        g07_3d.gerar_3d=save7
        geo._guardar=savefig




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
    cfg['pca']['n_components'] = [3]
    cfg['tsne']['n_components'] = [3]
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
               int(x['reducao']['n_components']) == 3 and
               x['gmm']['covariance_type'] == 'full' and
               int(x['gmm']['n_components']) == 4 for x in grade),
           'Grade 3D contem candidato inesperado.')
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
    print(f'=== PLANO {metodo.upper()} 3D REAL 3% ===', flush=True)
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
                if metodo == 'tsne':
                    with proteger_visualizacoes_tsne_3d():
                        orq.main(['--executar'], raiz=ROOT)
                else:
                    orq.main(['--executar'], raiz=ROOT)  # --ensaio limitaria a 300/fold
            finally:
                orq.ARQUIVO_CONFIG = original
        concluido, parcial = localizar_experimento(cfg, metodo)
    exigir(concluido is not None and parcial is None,
           'OOF ainda nao foi finalizado; veja traceback/cache na pasta em_andamento.')
    man = ler_json(concluido / 'manifesto.json')
    # Windows: Path.write_text() pode converter LF em CRLF no YAML temporario.
    # Exigir a mesma configuracao, mantendo SHA-256 obrigatorio (LF ou CRLF).
    configuracao_do_ensaio = yaml.safe_load(
        (concluido / 'configuracao.yaml').read_text(encoding='utf-8-sig')
    )
    exigir(configuracao_do_ensaio == cfg,
           'OOF possui configuracao diferente do lote 3D atual.')
    texto_yaml = yaml_texto(cfg)
    hashes_yaml_permitidos = {
        sha256(texto_yaml.encode('utf-8')).hexdigest(),
        sha256(texto_yaml.replace('\n', '\r\n').encode('utf-8')).hexdigest(),
    }
    exigir(man.get('status') == 'concluido_06_03' and
           man.get('tipo') == 'ensaio' and
           man.get('sha256_yaml') in hashes_yaml_permitidos,
           'OOF nao tem status/tipo/hash YAML LF/CRLF correto.')
    print('[OOF 3D] YAML estrutural e SHA LF/CRLF validos; sem refazer OOF.', flush=True)
    if man.get('visualizacoes') != 'concluidas':
        print(f'[OOF {metodo.upper()}] Retomando somente graficos/galeria OOF.', flush=True)
        if metodo == 'tsne':
            with proteger_visualizacoes_tsne_3d():
                orq.executar_visualizacoes(concluido)
        else:
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
           escolha['configuracao']['reducao']['n_components'] == 3,
           'Escolha vencedora nao corresponde ao metodo 3D.')
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
    colunas = ([f'PC{i}' for i in range(1, 4)] if metodo == 'pca'
               else [f'TSNE{i}' for i in range(1, 4)])
    exigir(list(z.columns) == colunas and z.shape[1] == 3,
           'Embedding F10 nao e 3D esperado.')
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
        'dimensao': 3,
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
    colunas = list(z.columns)
    coord = pd.DataFrame({'linha_original': ids})
    for nome in colunas:
        coord[nome] = z[nome].to_numpy(dtype=np.float64)
    nome_coordenadas = 'coordenadas_' + '_'.join(colunas) + '_f10.csv'
    stage = dest.with_name('.avaliacao_f10_3pct_incompleta_' + uuid.uuid4().hex[:12])
    stage.mkdir(exist_ok=False)
    pred.to_csv(stage / 'predicoes_f10.csv', index=False, float_format='%.17g')
    coord.to_csv(stage / nome_coordenadas, index=False, float_format='%.17g')
    np.save(stage / 'indices_f10_amostra.npy', ids, allow_pickle=False)
    salvar_json(stage / 'metricas_f10.json', met)
    arquivos = [p for p in stage.iterdir() if p.is_file()]
    manifesto = {
        'versao': 'DOIS_ENSAIOS_REAIS_3PCT_3D_V1', 'status': 'APROVADA',
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
    from scripts.execucao.reteste_real_3d_graficos import gerar_graficos_f10_3d
    return gerar_graficos_f10_3d(
        pasta, cfg, info, metodo, model_dir, f10, base=__import__(__name__, fromlist=['*']))


def executar_galeria(pasta, cfg, metodo, f10, graficos):
    from scripts.execucao.reteste_real_3d_graficos import gerar_galeria_f10_3d
    return gerar_galeria_f10_3d(
        pasta, cfg, metodo, f10, graficos, base=__import__(__name__, fromlist=['*']))


def _executar_sem_lock(metodo):
    cfg, grade = imprimir_plano(metodo)
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
    lock = base / ('.ensaio_real_3pct_3d_' + metodo + '.lock')
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
    exigir((gal/'index.html').is_file() and len(mg['imagens']) >= 13,
           'Galeria ou imagens F10 insuficientes.')
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
