"""Passo 06.03: grade OOF com cache por candidato/fold e retomada segura.

Sem ajuste no F10. Saidas completas somente apos os nove folds de CADA
candidato. Uma unica pasta por execucao; renomeada com a data/hora de FIM.
O ensaio reduzido e explicitamente AMOSTRAL, nunca resultado oficial.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from src import estrutura_saida as layout
from src.folds import hash_arquivo
from src.validacao_oof import grade_candidatos_oof, prever_fold_oof, validar_divisao
from src.metricas_oof import avaliar_probabilidades, avaliar_oof_consolidado
from sklearn.metrics import average_precision_score, matthews_corrcoef, log_loss

ARQUIVO_CONFIG = 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'


def _json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _chave(obj):
    return sha256(_json(obj).encode('utf-8')).hexdigest()


def _salvar_bytes_atomicamente(caminho: Path, conteudo: bytes):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_name(caminho.name + '.incompleto')
    if tmp.exists():
        tmp.unlink()  # restos de queda anterior, nunca um artefato confirmado
    try:
        with tmp.open('xb') as f:
            f.write(conteudo)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, caminho)
    finally:
        if tmp.exists():
            tmp.unlink()


def _salvar_json(caminho, obj):
    _salvar_bytes_atomicamente(Path(caminho), (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))


def _salvar_npz(caminho, tabela, *, chave_execucao, id_candidato, fold):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_name(caminho.name + '.incompleto')
    if tmp.exists():
        tmp.unlink()
    try:
        with tmp.open('xb') as f:
            np.savez_compressed(
                f, linha_original=tabela['linha_original'].to_numpy(dtype=np.int64),
                fold=tabela['fold'].to_numpy(dtype=np.uint8),
                y_true=tabela['y_true'].to_numpy(dtype=np.uint8),
                prob_fraude=tabela['prob_fraude'].to_numpy(dtype=np.float64),
                chave_execucao=np.array(chave_execucao), id_candidato=np.array(id_candidato),
                fold_esperado=np.array(fold, dtype=np.uint8),
            )
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, caminho)
    finally:
        if tmp.exists():
            tmp.unlink()


def _ler_npz(caminho, *, chave_execucao, id_candidato, fold):
    with np.load(caminho, allow_pickle=False) as arr:
        if (arr['chave_execucao'].item() != chave_execucao
                or arr['id_candidato'].item() != id_candidato
                or int(arr['fold_esperado']) != fold):
            raise RuntimeError(f'Cache incompatível: {caminho}')
        df = pd.DataFrame({nome: arr[nome] for nome in
                           ('linha_original', 'fold', 'y_true', 'prob_fraude')})
    return df


def _verificar_tabela(tabela, *, indices_esperados, folds_trabalho, y_trabalho, fold=None):
    if list(tabela.columns) != ['linha_original', 'fold', 'y_true', 'prob_fraude']:
        raise RuntimeError('Formato das predicoes OOF incorreto.')
    ids = tabela['linha_original'].to_numpy()
    if len(ids) != len(indices_esperados) or not np.array_equal(ids, indices_esperados):
        raise RuntimeError('Cache OOF incompleto, duplicado ou fora de ordem.')
    if not np.array_equal(tabela['fold'].to_numpy(), folds_trabalho[indices_esperados]):
        raise RuntimeError('Identificacao dos folds foi alterada.')
    if fold is not None and not np.all(tabela['fold'] == fold):
        raise RuntimeError('Cache pertence a outro fold.')
    if not np.array_equal(tabela['y_true'].to_numpy(), y_trabalho[indices_esperados]):
        raise RuntimeError('Rotulos de validacao inconsistentes com a base.')
    p = tabela['prob_fraude'].to_numpy()
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise RuntimeError('Predicoes contem probabilidades invalidas.')


def validar_operacao(config):
    ex = config['execucao']
    for k in ('usar_cache', 'retomar', 'proteger_resultados'):
        if not isinstance(ex.get(k), bool):
            raise ValueError(f'execucao.{k} deve ser booleano.')
    if not ex['proteger_resultados']:
        raise ValueError('Neste passo proteger_resultados=false nao e suportado.')
    if ex['retomar'] and not ex['usar_cache']:
        raise ValueError('retomar=true requer usar_cache=true.')
    if ex.get('diretorio_retomada') is not None and not ex['retomar']:
        raise ValueError('diretorio_retomada exige retomar=true.')
    if not (ex['n_jobs'] == -1 or isinstance(ex['n_jobs'], int)
            and not isinstance(ex['n_jobs'], bool) and ex['n_jobs'] >= 1):
        raise ValueError('execucao.n_jobs precisa ser -1 ou inteiro positivo.')
    f = config['dados']['fracao_base']
    if not isinstance(f, (int, float)) or isinstance(f, bool) or not 0 < f <= 1:
        raise ValueError('dados.fracao_base precisa estar em (0,1].')
    if config['dados']['classe_positiva'] != 1:
        raise ValueError('Este classificador utiliza TARGET_OMEGA: fraude=1.')
    if not config['divisao']['estratificada'] or not config['divisao']['embaralhar']:
        raise ValueError('Particao salva deve ser estratificada e embaralhada.')
    saidas = config['saidas']
    for k in ('salvar_configuracao','salvar_metricas','salvar_predicoes_oof',
              'salvar_manifesto','registrar_versoes_bibliotecas'):
        if saidas.get(k) is not True:
            raise ValueError(f'saidas.{k}=false nao suportado neste passo.')
    if saidas['padrao_pasta'] != 'experimento_{metodo}_gmm_{datahora_fim}':
        raise ValueError('Padrao de pasta nao suportado; deve utilizar horario de finalizacao.')
    esperado = {'candidatos':'candidatos', 'vencedor':'vencedor',
                'metricas':'metricas','oof':'oof', 'visualizacao':'visualizacao',
                'geometria':'geometria_desenvolvimento','galeria':'galeria',
                'teste_final':'teste_final'}
    if saidas.get('pastas') != esperado:
        raise ValueError('saidas.pastas deve seguir o layout 2 (candidatos/vencedor).')
    return ex


def _agora(config):
    try:
        return datetime.now(ZoneInfo(config['saidas']['timezone']))
    except Exception as exc:
        raise RuntimeError('Timezone IANA indisponivel. Instale tzdata no .venv.') from exc


def _identificar_folds(raiz, config, arquivo_base):
    """Localiza APENAS manifesto de folds compatível com base e estratégia."""
    import json as _json_lib
    raiz_splits = raiz / 'data' / 'splits'
    h = hash_arquivo(arquivo_base)
    candidatos = []
    for m in raiz_splits.glob('*/manifesto_folds.json'):
        manifest = _json_lib.loads(m.read_text(encoding='utf-8-sig'))
        p = manifest.get('parametros', {})
        if (p.get('sha256_parquet') == h and p.get('caminho_relativo') == config['dados']['arquivo']
                and p.get('semente') == config['experimento']['semente']
                and p.get('coluna_alvo') == config['dados']['coluna_alvo']
                and p.get('classe_positiva') == 1 and p.get('total_folds') == 10
                and p.get('fold_teste') == 10 and p.get('estratificada') is True
                and p.get('embaralhar') is True):
            caminho = m.parent / 'folds.npy'
            if not caminho.is_file() or hash_arquivo(caminho) != manifest['sha256_folds']:
                raise RuntimeError(f'Folds ausentes ou alterados: {caminho}')
            candidatos.append(caminho)
    if len(candidatos) != 1:
        raise RuntimeError(f'Esperada exatamente 1 divisao compativel em data/splits; encontradas {len(candidatos)}.')
    folds = np.load(candidatos[0], allow_pickle=False)
    validar_divisao(folds, config, len(folds))
    return folds, candidatos[0], h


def _amostrar_desenvolvimento(folds, config, *, ensaio=False, por_fold=300):
    """Somente indices F1..F9. Nunca inclui o F10 na base de trabalho."""
    rng = np.random.default_rng(int(config['experimento']['semente']))
    f = float(config['dados']['fracao_base'])
    escolha = []
    for k in range(1, 10):
        posicoes = np.flatnonzero(folds == k)
        n_fracao = len(posicoes) if f == 1 else max(4, round(len(posicoes) * f))
        n = min(por_fold,n_fracao) if ensaio else n_fracao
        # Amostragem global deve ser estratificada usando y, realizada depois da leitura do alvo.
        # Se n == len(posicoes), podemos decidir aqui sem labels.
        escolha.append((posicoes, n))
    return escolha, rng


def _selecionar_posicoes(folds, y, config, *, ensaio=False, por_fold=300):
    escolhas, rng = _amostrar_desenvolvimento(folds, config, ensaio=ensaio, por_fold=por_fold)
    selecionadas = []
    for posicoes, n in escolhas:
        if n == len(posicoes):
            selecionadas.append(posicoes)
            continue
        idx0, idx1 = posicoes[y[posicoes] == 0], posicoes[y[posicoes] == 1]
        if min(len(idx0),len(idx1)) == 0 or n < 4:
            raise ValueError('Amostra estratificada requer ambas as classes e ao menos quatro linhas por fold.')
        n1 = max(2, int(round(n * len(idx1)/len(posicoes))))
        n1 = min(n1, len(idx1), n-2)
        n0 = n-n1
        if n0 > len(idx0):
            n0 = len(idx0); n1 = n-n0
        selecionadas.append(np.sort(np.r_[rng.choice(idx0, n0, replace=False),
                                           rng.choice(idx1, n1, replace=False)]))
    return np.sort(np.concatenate(selecionadas))


def _ler_desenvolvimento(parquet, selecionadas):
    """Lê em lotes e mantém apenas as linhas selecionadas, sem guardar F10."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet)
    partes, inicio, encontrados = [], 0, 0
    for lote in pf.iter_batches(batch_size=20_000):
        fim = inicio + lote.num_rows
        i = np.searchsorted(selecionadas, inicio, side='left')
        j = np.searchsorted(selecionadas, fim, side='left')
        if i != j:
            locais = selecionadas[i:j] - inicio
            # manter indices apenas dessas linhas. Nunca acrescentar F10 ao DataFrame.
            partes.append(lote.to_pandas().iloc[locais].copy())
            encontrados += j-i
        inicio = fim
    if encontrados != len(selecionadas):
        raise RuntimeError('Leitura da base nao corresponde aos indices selecionados.')
    return pd.concat(partes, ignore_index=True)


def _chave_execucao(config, hash_dados, hash_folds, *, ensaio, por_fold):
    return _chave({'config':config, 'sha_parquet':hash_dados, 'sha_folds':hash_folds,
                   'ensaio':ensaio,'por_fold':por_fold if ensaio else None,
                   'versao_passo':'06.03.2-layout2'})


def _abrir_execucao(raiz, config, chave, *, ensaio):
    out = raiz / config['saidas']['diretorio_base']
    out.mkdir(parents=True,exist_ok=True)
    prefixo = 'ensaio' if ensaio or float(config['dados']['fracao_base']) < 1 else 'experimento'
    metodo = config['reducao']['metodo'].lower()
    ex = config['execucao']
    explicito = ex['diretorio_retomada']
    if explicito is not None:
        caminho = Path(explicito)
        if not caminho.is_absolute(): caminho = raiz/caminho
        if not caminho.is_dir(): raise FileNotFoundError(f'Diretorio_retomada nao encontrado: {caminho}')
        candidatos = [caminho]
    elif ex['retomar']:
        candidatos=[]
        for d in out.glob(f'em_andamento_{prefixo}_{metodo}_gmm_*'):
            m = d / 'manifesto.json'
            if m.exists() and json.loads(m.read_text(encoding='utf-8')).get('chave_execucao') == chave:
                candidatos.append(d)
    else:
        candidatos=[]
    if len(candidatos) > 1:
        raise RuntimeError('Varias execucoes compativeis inacabadas; indique execucao.diretorio_retomada.')
    if candidatos:
        pasta = candidatos[0]
        m = json.loads((pasta/'manifesto.json').read_text(encoding='utf-8'))
        if m.get('chave_execucao') != chave or m.get('status') != 'em_andamento' or m.get('versao_layout') != layout.VERSAO_LAYOUT:
            raise RuntimeError('Retomada incompativel ou experimento ja finalizado.')
        return pasta, m, True
    inicio = _agora(config).strftime('%Y%m%d_%H%M%S')
    pasta = out/f'em_andamento_{prefixo}_{metodo}_gmm_{inicio}'
    if pasta.exists():
        raise FileExistsError(f'Pasta ja existe (protegida): {pasta}')
    pasta.mkdir()
    (pasta/'candidatos').mkdir()
    layout.metricas_vencedor(pasta).mkdir(parents=True)
    layout.predicoes_vencedor(pasta).parent.mkdir(parents=True)
    layout.teste_final(pasta).mkdir(parents=True)
    _salvar_bytes_atomicamente(pasta/'configuracao.yaml',
                              yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode('utf-8'))
    m = {'chave_execucao':chave,'status':'em_andamento','tipo':'ensaio' if prefixo == 'ensaio' else 'oficial',
         'inicio_local':_agora(config).isoformat(),
         'pendente_em_outros_passos':['modelo_final_serializado','teste_F10'],
         'versao_layout':layout.VERSAO_LAYOUT,'visualizacoes':'pendentes',
         'n_jobs':ex['n_jobs'],
         'politica_paralelismo':'folds/candidatos sequenciais para limitar memoria; threads numericas sob n_jobs',
         'candidatos_concluidos':[]}
    _salvar_json(pasta/'manifesto.json',m)
    return pasta,m,False


def _finalizar(pasta, manifest, config):
    agora = _agora(config)
    tipo = 'ensaio' if manifest['tipo'] == 'ensaio' else 'experimento'
    metodo = config['reducao']['metodo'].lower()
    fim=agora.strftime('%Y%m%d_%H%M%S')
    base=f'{tipo}_{metodo}_gmm_{fim}'
    destino = pasta.parent/base
    if destino.exists():
        raise FileExistsError(f'Pasta final ja existe: {destino}')
    # Gravacao primeiro, rename apenas depois de tudo concluir.
    manifest.update(status='concluido_06_03',fim_local=agora.isoformat())
    _salvar_json(pasta/'manifesto.json',manifest)
    pasta.rename(destino)
    return destino




def _salvar_resultado_sem_sobrescrever(caminho, conteudo):
    caminho=Path(caminho)
    if caminho.exists():
        if caminho.read_bytes() != conteudo:
            raise RuntimeError(f'Resultado protegido seria sobrescrito: {caminho}')
        return
    _salvar_bytes_atomicamente(caminho,conteudo)


def _metricas_por_fold(oof, limiar):
    linhas=[]
    for k in range(1,10):
        parte=oof[oof['fold']==k]
        y=parte['y_true'].to_numpy(dtype=np.uint8)
        scores=parte['prob_fraude'].to_numpy(dtype=np.float64)
        linhas.append({'fold':k,'n':len(y),'n_fraudes':int(y.sum()),
          'ap':float(average_precision_score(y,scores)),
          'logloss':float(log_loss(y,scores,labels=[0,1])),
          'mcc_limiar_congelado_oof':float(matthews_corrcoef(y,(scores>=limiar).astype(np.uint8))),
          'limiar_congelado_oof':float(limiar)})
    return pd.DataFrame(linhas)

def executar_grade(df, folds_dev, linhas_origem, folds_orig, config, pasta, manifest, *, ensaio=False):
    """Motor persistente. df contem exclusivamente o desenvolvimento selecionado.

    Aceita dados fornecidos diretamente para testes; no uso oficial o CLI
    certifica hash do Parquet, folds e YAML previamente.
    """
    if not df.index.equals(pd.RangeIndex(len(df))) or len(df) != len(folds_dev):
        raise ValueError('DataFrame de trabalho deve ter RangeIndex e folds alinhados.')
    if len(linhas_origem) != len(df) or not np.array_equal(folds_orig[linhas_origem], folds_dev):
        raise ValueError('Indices originais e folds incompativeis.')
    if (folds_dev == 10).any() or set(np.unique(folds_dev)) != set(range(1,10)):
        raise ValueError('Somente F1..F9 podem entrar na grade OOF.')
    y_dev = df[config['dados']['coluna_alvo']].to_numpy()
    if not np.isin(y_dev, [0,1]).all():
        raise ValueError('Alvo do desenvolvimento deve ser 0/1.')
    grade = list(grade_candidatos_oof(config))
    if not grade:
        raise RuntimeError('Grade vazia.')
    ids = [_chave(c)[:14] for c in grade]
    if len(set(ids)) != len(ids):
        raise RuntimeError('IDs de candidatos colidiram; amplie a chave hash.')
    manifest['quantidade_candidatos'] = len(grade)
    manifest['n_desenvolvimento_trabalhado'] = len(df)
    manifest['n_folds_cv'] = 9
    _salvar_json(pasta/'manifesto.json',manifest)
    resultados=[]
    ids_completos=set()
    cache_ligado = config['execucao']['usar_cache']
    chave_execucao = manifest['chave_execucao']
    for i,(candidato,cid) in enumerate(zip(grade,ids), 1):
        print(f'[{i}/{len(grade)}] candidato={cid} | {candidato}',flush=True)
        pasta_candidato = layout.candidato(pasta,cid,candidato)
        pasta_candidato.mkdir(parents=True,exist_ok=True)
        _salvar_resultado_sem_sobrescrever(
            pasta_candidato/'configuracao.json',
            (json.dumps(candidato,indent=2,ensure_ascii=False)+'\n').encode('utf-8'))
        arquivo_pred = pasta_candidato/'oof'/'predicoes.npz'
        arquivo_metricas = pasta_candidato/'metricas'/'metricas.json'
        if arquivo_metricas.exists() and not arquivo_pred.exists():
            raise RuntimeError('Metricas sem arquivo de predicoes: possivel remocao indevida: ' + cid)
        if arquivo_pred.exists():
            oof_salvo = _ler_npz(arquivo_pred,chave_execucao=chave_execucao,id_candidato=cid,fold=0)
            if not np.array_equal(oof_salvo['linha_original'].to_numpy(),linhas_origem):
                raise RuntimeError('Predicoes consolidadas existentes nao correspondem a origem.')
            if not np.array_equal(oof_salvo['fold'].to_numpy(),folds_dev) or not np.array_equal(oof_salvo['y_true'].to_numpy(),y_dev):
                raise RuntimeError('Rotulos/folds de candidato salvo nao correspondem ao desenvolvimento.')
            probabilidades = oof_salvo['prob_fraude'].to_numpy()
            if not np.isfinite(probabilidades).all() or ((probabilidades < 0)|(probabilidades > 1)).any():
                raise RuntimeError('Probabilidades salvas invalidas.')
            if not arquivo_metricas.exists():
                # Queda entre salvar as predicoes completas e salvar as metricas.
                if ensaio or float(config['dados']['fracao_base']) < 1:
                    metricas=avaliar_probabilidades(oof_salvo['y_true'],oof_salvo['prob_fraude'],config)
                    metricas.update(tipo_avaliacao='OOF_AMOSTRAL_NAO_OFICIAL',quantidade_folds_cv=9)
                else:
                    metricas=avaliar_oof_consolidado(oof_salvo,folds_orig,config)
                metricas.update(id_candidato=cid,configuracao=candidato)
                _salvar_json(arquivo_metricas,metricas)
            else:
                metricas=json.loads(arquivo_metricas.read_text(encoding='utf-8'))
            if metricas.get('id_candidato') != cid or metricas.get('configuracao') != candidato:
                raise RuntimeError('Metricas em cache incompativeis com o candidato.')
            resultados.append({'id_candidato':cid, **{x:metricas[x] for x in
                 ('ap','mcc','logloss','score','limiar_mcc','n','prevalencia')},
                 'configuracao':_json(candidato)})
            ids_completos.add(cid)
            print(f'  candidato completo reutilizado (sem sobrescrever).',flush=True)
            continue
        partes=[]
        for k in range(1,10):
            indices = np.flatnonzero(folds_dev==k)
            caminho = pasta_candidato/'oof'/'cache'/f'F{k:02d}.npz'
            if caminho.is_file():
                if not cache_ligado:
                    raise RuntimeError('Cache presente com usar_cache=false.')
                tabela = _ler_npz(caminho,chave_execucao=chave_execucao,id_candidato=cid,fold=k)
                print(f'  F{k}: reutilizado',flush=True)
            else:
                tabela = prever_fold_oof(df,folds_dev,config,candidato,k)
                _verificar_tabela(tabela,indices_esperados=indices,folds_trabalho=folds_dev,
                                 y_trabalho=y_dev,fold=k)
                if cache_ligado:
                    _salvar_npz(caminho,tabela,chave_execucao=chave_execucao,id_candidato=cid,fold=k)
                print(f'  F{k}: calculado ({len(tabela):,} predições)',flush=True)
            _verificar_tabela(tabela,indices_esperados=indices,folds_trabalho=folds_dev,
                             y_trabalho=y_dev,fold=k)
            partes.append(tabela)
        oof = pd.concat(partes,ignore_index=True).sort_values('linha_original').reset_index(drop=True)
        _verificar_tabela(oof,indices_esperados=np.arange(len(df)),folds_trabalho=folds_dev,y_trabalho=y_dev)
        # Tracar origem sem permitir que F10 apareca no arquivo.
        oof['linha_original'] = linhas_origem[oof['linha_original'].to_numpy()]
        if ensaio or float(config['dados']['fracao_base']) < 1.0:
            metricas = avaliar_probabilidades(oof['y_true'],oof['prob_fraude'],config)
            metricas.update(tipo_avaliacao='OOF_AMOSTRAL_NAO_OFICIAL',quantidade_folds_cv=9)
        else:
            metricas = avaliar_oof_consolidado(oof,folds_orig,config)
        metricas.update(id_candidato=cid,configuracao=candidato)
        # MCC por fold no limiar global OOF, NUNCA otimizado em cada fold.
        por_fold = _metricas_por_fold(oof,metricas['limiar_mcc'])
        _salvar_resultado_sem_sobrescrever(pasta_candidato/'metricas'/'por_fold.csv',
                                         por_fold.to_csv(index=False).encode('utf-8'))
        _salvar_npz(arquivo_pred,oof,
                    chave_execucao=chave_execucao,id_candidato=cid,fold=0)
        _salvar_json(arquivo_metricas,metricas)
        resultados.append({'id_candidato':cid, **{x:metricas[x] for x in
             ('ap','mcc','logloss','score','limiar_mcc','n','prevalencia')},
             'configuracao':_json(candidato)})
        ids_completos.add(cid)
        manifest['candidatos_concluidos'] = sorted(ids_completos)
        _salvar_json(pasta/'manifesto.json',manifest)
        print(f'  OOF: AP={metricas["ap"]:.6f} MCC={metricas["mcc"]:.6f} '
              f'LogLoss={metricas["logloss"]:.6f} S={metricas["score"]:.6f}',flush=True)
    tabela = pd.DataFrame(resultados).sort_values(['score','ap','id_candidato'],ascending=[False,False,True])
    tabela = tabela.reset_index(drop=True)
    _salvar_resultado_sem_sobrescrever(layout.metricas_vencedor(pasta)/'resultados_grade.csv',tabela.to_csv(index=False).encode('utf-8'))
    vencedor = next(c for c,cid in zip(grade,ids) if cid == tabela.iloc[0]['id_candidato'])
    escolha = {'id_candidato':tabela.iloc[0]['id_candidato'], 'configuracao':vencedor,
               'limiar_mcc_oof':float(tabela.iloc[0]['limiar_mcc']),
               'score_oof':float(tabela.iloc[0]['score']),
               'status':'selecionado_em_ensaio' if ensaio or float(config['dados']['fracao_base'])<1 else 'selecionado_em_oof_integral',
               'teste_final_usado':False}
    _salvar_resultado_sem_sobrescrever(layout.metricas_vencedor(pasta)/'melhor_configuracao.json',
       (json.dumps(escolha,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))
    # A pasta 'vencedor' e autocontida; copiar sem alterar arquivos de origem.
    cid_vencedor = str(escolha['id_candidato'])
    origem = layout.candidato(pasta,cid_vencedor,vencedor)
    _copiar_protegido(origem/'oof'/'predicoes.npz',layout.predicoes_vencedor(pasta))
    _copiar_protegido(origem/'metricas'/'metricas.json',layout.metricas_vencedor(pasta)/'metricas.json')
    _copiar_protegido(origem/'metricas'/'por_fold.csv',layout.metricas_vencedor(pasta)/'por_fold.csv')
    return escolha


def _copiar_protegido(origem, destino):
    origem, destino = Path(origem), Path(destino)
    if not origem.is_file():
        raise FileNotFoundError(f'Arquivo de candidato ausente: {origem}')
    if destino.exists():
        if sha256(origem.read_bytes()).digest() != sha256(destino.read_bytes()).digest():
            raise RuntimeError(f'Resultado protegido divergente: {destino}')
        return
    destino.parent.mkdir(parents=True,exist_ok=True)
    temporario = destino.with_name(destino.name + '.incompleto')
    if temporario.exists():temporario.unlink()
    try:
        shutil.copyfile(origem,temporario)
        if sha256(origem.read_bytes()).digest() != sha256(temporario.read_bytes()).digest():
            raise RuntimeError(f'Falha de integridade na copia: {destino}')
        os.replace(temporario,destino)
    finally:
        if temporario.exists():temporario.unlink()


def _ler_alvo(parquet, coluna):
    # Apenas o alvo para amostragem estratificada e verificacao de alinhamento.
    from src.folds import carregar_alvo
    y = carregar_alvo(parquet,coluna)
    if not np.isin(y,[0,1]).all():
        raise ValueError('TARGET_OMEGA deve conter 0 e 1.')
    return y


def planejar(config):
    validar_operacao(config)
    grade=list(grade_candidatos_oof(config))
    metodo=config['reducao']['metodo'].upper()
    print(f'Metodo: {metodo}; candidatos: {len(grade)}; rodadas por candidato: 9; '
          f'total de ajustes GMM: {9*len(grade)}')
    for i,c in enumerate(grade,1):
        print(f'{i:02d}: {_json(c)}')
    print('F10: fora do ajuste, da otimizacao de limiar e do ranking.')
    print('Nenhum modelo foi treinado e nenhuma saida foi criada.')
    return len(grade)


def main(argv=None, *, raiz=None):
    parser=argparse.ArgumentParser(description='Passo 06.03 - grade OOF persistente')
    modos=parser.add_mutually_exclusive_group(required=True)
    modos.add_argument('--planejar',action='store_true',help='Somente listar grade, sem ler dados.')
    modos.add_argument('--ensaio',action='store_true',help='Amostra estratificada por fold; NAO e experimento final.')
    modos.add_argument('--executar',action='store_true',help='OOF e visualizacoes automaticas.')
    modos.add_argument('--so-visualizacoes',action='store_true',help='Retomar as visualizacoes de um ensaio OOF concluido.')
    modos.add_argument('--concluir-teste-final',action='store_true',help='Completar F10 de OOF ja concluido sem repetir treinamento OOF.')
    parser.add_argument('--por-fold',type=int,default=300,help='Linhas por fold em --ensaio (>=4).')
    parser.add_argument('--experimento',type=Path,help='Pasta concluida, exigida em --so-visualizacoes.')
    a=parser.parse_args(argv)
    if a.por_fold < 4: parser.error('--por-fold precisa ser >=4.')
    raiz=Path(raiz) if raiz else Path(__file__).resolve().parents[1]
    if a.concluir_teste_final:
        if a.experimento is None:parser.error('--concluir-teste-final exige --experimento.')
        alvo = raiz/a.experimento if not a.experimento.is_absolute() else a.experimento
        config_salva=alvo/'configuracao.yaml'
        if not config_salva.is_file():
            raise RuntimeError('Configuração do experimento de retomada ausente.')
        dim=yaml.safe_load(config_salva.read_text(encoding='utf-8-sig'))['pca']['n_components']
        if dim == [1]:
            from src.finalizacao_automatica_pca1d import concluir_ensaio
        elif dim == [2]:
            from src.finalizacao_automatica_pca2d import concluir_ensaio
        elif dim == [3]:
            from src.finalizacao_automatica_pca3d import concluir_ensaio
        else:
            raise RuntimeError('Dimensão PCA de retomada não suportada.')
        concluir_ensaio(alvo)
        return alvo
    if a.so_visualizacoes:
        if a.experimento is None:parser.error('--so-visualizacoes requer --experimento.')
        executar_visualizacoes(raiz/a.experimento if not a.experimento.is_absolute() else a.experimento)
        return None
    arquivo_yaml=raiz/ARQUIVO_CONFIG
    yaml_bytes=arquivo_yaml.read_bytes()
    config=yaml.safe_load(yaml_bytes.decode('utf-8-sig'))
    validar_operacao(config)
    automatizar_f10 = bool(config.get('execucao', {}).get('finalizacao_pca1d_automatica', False))
    if automatizar_f10:
        dim=config.get('pca', {}).get('n_components')
        if dim == [1]:
            from src.finalizacao_automatica_pca1d import validar_automacao
        elif dim == [2]:
            from src.finalizacao_automatica_pca2d import validar_automacao
        elif dim == [3]:
            from src.finalizacao_automatica_pca3d import validar_automacao
        else:
            raise RuntimeError('Finalizacao automatica disponivel apenas para PCA 1D, 2D ou 3D.')
        validar_automacao(config)
    if a.planejar:
        planejar(config)
        if automatizar_f10:
            print('ETAPA FINAL AUTOMATICA: F1-F9 final, F10 exploratorio 10%, 12 graficos e PRE01.')
        return None
    if a.experimento is not None:parser.error('--experimento apenas com --so-visualizacoes ou --concluir-teste-final.')
    arquivo_base=raiz/config['dados']['arquivo']
    folds_orig, arquivo_folds, hash_dados = _identificar_folds(raiz,config,arquivo_base)
    y_original=_ler_alvo(arquivo_base,config['dados']['coluna_alvo'])
    if len(y_original)!=len(folds_orig):
        raise RuntimeError('Alvo e folds tem tamanhos distintos.')
    linhas=_selecionar_posicoes(folds_orig,y_original,config,ensaio=a.ensaio,por_fold=a.por_fold)
    if np.any(folds_orig[linhas]==10): raise RuntimeError('F10 incluido por engano.')
    # Valida nomes de colunas lendo somente registros do desenvolvimento.
    df=_ler_desenvolvimento(arquivo_base,linhas)
    if not np.array_equal(df[config['dados']['coluna_alvo']].to_numpy(),y_original[linhas]):
        raise RuntimeError('Ordem de registros Parquet diferente da divisao de folds.')
    del y_original
    folds_dev=folds_orig[linhas]
    chave=_chave_execucao(config,hash_dados,hash_arquivo(arquivo_folds),ensaio=a.ensaio,por_fold=a.por_fold)
    pasta,manifest,retomada=_abrir_execucao(raiz,config,chave,ensaio=a.ensaio)
    manifest.update(sha256_parquet=hash_dados,sha256_folds=hash_arquivo(arquivo_folds),
                    sha256_yaml=sha256(yaml_bytes).hexdigest(),
                    registros_desenvolvimento=int(len(df)),registros_reservados_f10=int((folds_orig==10).sum()),
                    ram_dataframe_bytes=int(df.memory_usage(deep=True).sum()),
                    reaberta_em_retomada=bool(retomada))
    if config['saidas']['registrar_versoes_bibliotecas']:
        import sklearn
        import pyarrow
        manifest['versoes']={'python':sys.version.split()[0],'pandas':pd.__version__,
                             'numpy':np.__version__,'sklearn':sklearn.__version__,
                             'pyarrow':pyarrow.__version__,'plataforma':platform.platform()}
    _salvar_json(pasta/'manifesto.json',manifest)
    print(f'Execucao em: {pasta}; desenvolvimento={len(df):,} linhas; F10 excluido.',flush=True)
    # Executa UM fold/candidato por vez; controla threads numericas com n_jobs.
    from threadpoolctl import threadpool_limits
    jobs=config['execucao']['n_jobs']
    n_threads=(os.cpu_count() or 1) if jobs==-1 else jobs
    with threadpool_limits(limits=n_threads):
        executar_grade(df,folds_dev,linhas,folds_orig,config,pasta,manifest,ensaio=a.ensaio)
    final=_finalizar(pasta,manifest,config)
    print(f'PASSO 06.03 - EXECUCAO CONCLUIDA EM {final}')
    executar_visualizacoes(final)
    if automatizar_f10 and a.executar:
        print('[ORQUESTRADOR] Iniciando finalizacao automatica PCA (F1-F9 + F10 exploratorio)...',flush=True)
        dim=config.get('pca', {}).get('n_components')
        if dim == [1]:
            from src.finalizacao_automatica_pca1d import concluir_ensaio
        elif dim == [2]:
            from src.finalizacao_automatica_pca2d import concluir_ensaio
        elif dim == [3]:
            from src.finalizacao_automatica_pca3d import concluir_ensaio
        else:
            raise RuntimeError('Dimensionalidade PCA final invalida.')
        concluir_ensaio(final)
        print('[ORQUESTRADOR] COMPLETO: OOF, modelo final, F10 e galeria no mesmo experimento.',flush=True)
    return final


def executar_visualizacoes(experimento):
    """Orquestra 07.01, 07.02 e 07.03, com retomada sem refazer OOF."""
    # Avisos nao-fatais de substituicao de fonte nao devem interromper o PowerShell.
    import logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    from src import visualizacao_oof, visualizacao_geometrica, visualizacao_integrada
    experimento=Path(experimento).resolve()
    caminho_manifest=experimento/'manifesto.json'
    manifest=json.loads(caminho_manifest.read_text(encoding='utf-8-sig'))
    if manifest.get('status')!='concluido_06_03' or manifest.get('versao_layout')!=2:
        raise RuntimeError('Somente ensaios OOF concluidos no novo layout podem gerar imagens.')
    etapas=(
        ('07.01',lambda:visualizacao_oof.gerar_graficos_oof(experimento)),
        ('07.02',lambda:visualizacao_geometrica.executar(experimento)),
        ('07.03',lambda:visualizacao_integrada.integrar(experimento,complementar_2d=False)),
    )
    try:
        for nome,func in etapas:
            if nome in manifest.get('visualizacoes_concluidas',[]):
                print(f'[RETOMADA] {nome} ja executada.',flush=True)
                continue
            print(f'[ORQUESTRADOR] Iniciando {nome}',flush=True)
            func()
            manifest=json.loads(caminho_manifest.read_text(encoding='utf-8-sig'))
            concluidas=list(manifest.get('visualizacoes_concluidas',[]))
            if nome not in concluidas: concluidas.append(nome)
            manifest['visualizacoes_concluidas']=concluidas
            manifest['visualizacoes']='em_andamento'
            _salvar_json(caminho_manifest,manifest)
        manifest['visualizacoes']='concluidas'
        _salvar_json(caminho_manifest,manifest)
        print('[ORQUESTRADOR] Galeria final:', layout.galeria(experimento)/'index.html',flush=True)
    except Exception:
        manifest['visualizacoes']='interrompidas_retomaveis'
        _salvar_json(caminho_manifest,manifest)
        print('[ORQUESTRADOR] OOF preservado. Retome com --so-visualizacoes --experimento <pasta>.',flush=True)
        raise


if __name__ == '__main__':
    main()
