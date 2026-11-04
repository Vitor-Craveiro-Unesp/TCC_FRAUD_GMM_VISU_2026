# -*- coding: utf-8 -*-
"""OOF real PCA1D e TSNE1D, pastas originais, comparacao F1-F9 e F10 PCA opcional.

Nao altera YAML oficial; executa o orquestrador real com snapshots isolados.
A configuracao original utiliza 10% dos dados: OOF completo nos nove folds
DA AMOSTRA, e nao OOF integral em 100% da base.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import orquestracao_oof as orq
from src.validacao_oof import grade_candidatos_oof

OFICIAL = ROOT / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
PAREAMENTO = ROOT / 'outputs' / 'COMPARACAO_OOF_PCA1D_TSNE1D_V1'
ESTADO = PAREAMENTO / 'estado.json'
TAG = 'COMPARACAO_OOF_PCA1D_TSNE1D_V1'


def exigir(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def ler_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def salvar_json_novo_ou_igual(path, conteudo):
    raw = (json.dumps(conteudo, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    path = Path(path)
    if path.exists():
        exigir(path.read_bytes() == raw, f'Estado diferente do existente: {path}')
    else:
        path.write_bytes(raw)


def salvar_estado(estado):
    # So escreve nosso proprio estado; os experimentos do orquestrador ficam intactos.
    tmp = ESTADO.with_suffix('.json.incompleto')
    tmp.write_text(json.dumps(estado, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(ESTADO)


def configs():
    exigir(OFICIAL.is_file(), f'YAML oficial ausente: {OFICIAL}')
    base = yaml.safe_load(OFICIAL.read_text(encoding='utf-8-sig'))
    exigir(base['experimento']['semente'] == 42, 'Semente diferente de 42; revisao obrigatoria.')
    exigir(float(base['dados']['fracao_base']) == 0.1,
           'O procedimento V1 preserva 10% da base. YAML mudou: verificar antes de executar.')
    exigir(base['divisao']['total_folds'] == 10 and base['divisao']['qual_fold_teste'] == 10,
           'Particao F1-F10 diferente da homologada.')
    exigir(base['amostragem']['modo'] == 'RUS' and base['amostragem']['razao_negativos_positivos'] == 16,
           'RUS deve ser 16:1.')
    exigir(base['gmm']['n_components'] == [4] and base['gmm']['n_init'] == 5
           and base['gmm']['covariance_type'] == 'full',
           'GMM deve preservar K4 FULL n_init5.')
    planos = {}
    for metodo in ('pca', 'tsne'):
        cfg = deepcopy(base)
        cfg['experimento']['nome'] = f"{base['experimento']['nome']}_{TAG}_{metodo.upper()}"
        cfg['reducao']['metodo'] = metodo.upper()
        cfg['pca']['n_components'] = [1]
        cfg['tsne']['n_components'] = 1
        cfg['execucao']['finalizacao_pca1d_automatica'] = False
        cfg['execucao']['usar_cache'] = True
        cfg['execucao']['retomar'] = True
        cfg['execucao']['diretorio_retomada'] = None
        cfg['execucao']['proteger_resultados'] = True
        cfg['execucao']['n_jobs'] = 2  # Limitar uso de recursos (nao altera modelo).
        exigir(len(list(grade_candidatos_oof(cfg))) > 0, 'Grade vazia.')
        orq.validar_operacao(cfg)
        planos[metodo] = cfg
    # Guardar o arquivo oficial: orquestrador altera apenas a variavel do modulo em memoria.
    exigir(orq.ARQUIVO_CONFIG == OFICIAL.name,
           'Orquestrador local usa outro YAML de entrada; nao executar sem auditar.')
    return planos


def preparar():
    planos = configs()
    PAREAMENTO.mkdir(parents=True, exist_ok=True)
    for metodo, cfg in planos.items():
        snap = PAREAMENTO / f'config_{metodo}1d_oof.yaml'
        raw = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False).encode('utf-8')
        if snap.exists():
            exigir(snap.read_bytes() == raw, f'Snapshot {snap.name} divergente: parar.')
        else:
            snap.write_bytes(raw)
    if ESTADO.exists():
        estado = ler_json(ESTADO)
        exigir(estado.get('versao') == TAG, 'Estado de outra versao.')
    else:
        estado = {'versao': TAG, 'pca': None, 'tsne': None, 'f10_pca': False}
        salvar_estado(estado)
    return planos, estado


def confere_pasta(pasta, metodo, esperado_sha=None):
    pasta = Path(pasta).resolve()
    exigir(pasta.is_relative_to((ROOT / 'outputs').resolve()), 'Pasta fora de outputs.')
    cfg = yaml.safe_load((pasta / 'configuracao.yaml').read_text(encoding='utf-8-sig'))
    man = ler_json(pasta / 'manifesto.json')
    escolha = ler_json(pasta / 'vencedor/metricas/melhor_configuracao.json')
    exigir(cfg['reducao']['metodo'].lower() == metodo, f'Metodo incorreto em {pasta}')
    exigir(cfg['pca']['n_components'] == [1] if metodo == 'pca' else cfg['tsne']['n_components'] == 1,
           'Dimensao nao e 1D.')
    exigir(man.get('status') == 'concluido_06_03', 'OOF nao concluido.')
    exigir(escolha.get('teste_final_usado') is False, 'Escolha OOF contaminada por F10.')
    arquivo = pasta / 'vencedor/oof/predicoes.npz'
    exigir(arquivo.is_file(), 'Predicoes OOF ausentes.')
    if esperado_sha:
        exigir(man.get('sha256_yaml') == esperado_sha, 'YAML nao coincide com snapshot.')
    return pasta


def procurar_concluido(metodo, snapshot):
    # Recupera execucao concluida antes de gravar o estado (queda na retomada).
    h = sha256(snapshot.read_bytes()).hexdigest()
    candidatos = []
    for pasta in (ROOT / 'outputs').glob(f'ensaio_{metodo}_gmm_*'):
        m = pasta / 'manifesto.json'
        if not m.is_file():
            continue
        try:
            obj = ler_json(m)
            if obj.get('sha256_yaml') == h and obj.get('status') == 'concluido_06_03':
                candidatos.append(confere_pasta(pasta, metodo, h))
        except (KeyError, ValueError, FileNotFoundError, RuntimeError):
            continue
    exigir(len(candidatos) <= 1,
           f'Multiplas pastas {metodo} com mesmo snapshot: selecionar manualmente antes de rodar.')
    return candidatos[0] if candidatos else None


def executar_metodo(metodo, estado):
    snap = PAREAMENTO / f'config_{metodo}1d_oof.yaml'
    h = sha256(snap.read_bytes()).hexdigest()
    if estado[metodo]:
        pasta = confere_pasta(estado[metodo], metodo, h)
        print(f'[RETOMADA] OOF {metodo.upper()} ja concluido: {pasta}', flush=True)
    else:
        pasta = procurar_concluido(metodo, snap)
        if pasta is None:
            print(f'\n[INICIO] OOF REAL {metodo.upper()} 1D, F1-F9 (10% da base).', flush=True)
            orq.ARQUIVO_CONFIG = str(snap.relative_to(ROOT)).replace('\\', '/')
            try:
                pasta = orq.main(['--executar'], raiz=ROOT)
            finally:
                orq.ARQUIVO_CONFIG = OFICIAL.name
            exigir(pasta is not None, 'Orquestrador nao retornou pasta final.')
            pasta = confere_pasta(pasta, metodo, h)
        estado[metodo] = str(pasta)
        salvar_estado(estado)
    man = ler_json(pasta / 'manifesto.json')
    if man.get('visualizacoes') != 'concluidas':
        print(f'[RETOMADA] Finalizando visualizacoes {metodo.upper()}.', flush=True)
        orq.executar_visualizacoes(pasta)
    exigir((pasta / 'vencedor/visualizacao/galeria/index.html').is_file(),
           f'Galeria OOF {metodo} ausente.')
    print(f'[OK] OOF E GALERIA {metodo.upper()} 1D: {pasta}', flush=True)
    return pasta


def comparar(pca, tsne):
    obj = []
    arrays = []
    for metodo, pasta in (('PCA', pca), ('TSNE', tsne)):
        metricas = ler_json(pasta / 'vencedor/metricas/metricas.json')
        esc = ler_json(pasta / 'vencedor/metricas/melhor_configuracao.json')
        with np.load(pasta / 'vencedor/oof/predicoes.npz', allow_pickle=False) as z:
            ids = np.asarray(z['linha_original'], dtype=np.int64)
            folds = np.asarray(z['fold'], dtype=np.uint8)
            y = np.asarray(z['y_true'], dtype=np.uint8)
            prob = np.asarray(z['prob_fraude'], dtype=np.float64)
        exigir(set(np.unique(folds)) == set(range(1, 10)), 'OOF incompleto.')
        exigir(np.isfinite(prob).all() and np.all((prob >= 0) & (prob <= 1)), 'Scores invalidos.')
        arrays.append((ids, folds, y))
        obj.append({'metodo':metodo, 'dimensao':1, 'n':len(ids),
                    'ap':float(metricas['ap']), 'mcc':float(metricas['mcc']),
                    'logloss':float(metricas['logloss']), 'score':float(metricas['score']),
                    'tau_mcc_oof':float(esc['limiar_mcc_oof']),
                    'id_candidato':str(esc['id_candidato']),
                    'experimento':str(pasta)})
    for a,b in zip(arrays[0],arrays[1]):
        exigir(np.array_equal(a,b),
               'PCA e TSNE nao usaram os mesmos registros, labels e folds. Comparacao bloqueada.')
    tabela = pd.DataFrame(obj)
    csv = PAREAMENTO / 'COMPARACAO_AP_MCC_LOGLOSS_SCORE_OOF_F1_F9.csv'
    tabela.to_csv(csv, index=False, encoding='utf-8-sig', float_format='%.10g')
    manifesto = {'tipo':'COMPARACAO_OOF_AMOSTRAL_REAL_F1_F9',
                 'mesmos_ids_folds_rotulos':True,
                 'n':len(arrays[0][0]),
                 'f10_tsne_usado':False,
                 'f10_pca_usado_nesta_comparacao':False,
                 'origem_pca':str(pca), 'origem_tsne':str(tsne),
                 'advertencia':'OOF real sobre amostra estratificada de 10% da base, nao 100% do parquet.'}
    (PAREAMENTO/'manifesto_comparacao.json').write_text(json.dumps(manifesto,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    galp = pca / 'vencedor/visualizacao/galeria/index.html'
    galt = tsne / 'vencedor/visualizacao/galeria/index.html'
    galpath = PAREAMENTO / 'ABRIR_COMPARACAO_OOF.html'
    def rel(p):
        return html.escape(Path(__import__('os').path.relpath(p, PAREAMENTO)).as_posix(),quote=True)
    tr=''.join('<tr>'+''.join(f'<td>{html.escape(str(row[c]))}</td>' for c in ['metodo','n','ap','mcc','logloss','score','tau_mcc_oof'])+'</tr>' for _,row in tabela.iterrows())
    galpath.write_text('<!DOCTYPE html><html lang="pt-BR"><meta charset="utf-8">'
        '<title>OOF PCA 1D x t-SNE 1D</title><style>body{font:16px system-ui;max-width:1000px;margin:55px auto;color:#142944}'
        'a{color:#2057a3}td,th{padding:10px;border:1px solid #d0daea}table{border-collapse:collapse}li{margin:14px}</style>'
        '<h1>Comparacao OOF — PCA+GMM 1D x t-SNE+GMM 1D</h1>'
        '<p>F1–F9 identicos, 10% da base, F10 fora desta comparacao.</p>'
        '<table><thead><tr><th>Metodo</th><th>N</th><th>AP</th><th>MCC</th><th>LogLoss</th><th>Score</th><th>tau*</th></tr></thead>'
        '<tbody>'+tr+'</tbody></table><ul>'
        f'<li><a href="{rel(galp)}">Abrir galeria OOF PCA 1D</a></li>'
        f'<li><a href="{rel(galt)}">Abrir galeria OOF t-SNE 1D</a></li></ul>'
        '<p>A comparacao usa apenas resultados de desenvolvimento OOF; F10 PCA, quando executado, pertence a outra etapa.</p>'
        '</html>',encoding='utf-8')
    print(f'\n[COMPARACAO VALIDADA] IDs, folds e rotulos iguais; n={len(arrays[0][0])}', flush=True)
    print(tabela[['metodo','ap','mcc','logloss','score','tau_mcc_oof']].to_string(index=False),flush=True)
    print(f'[CSV] {csv}',flush=True)
    print(f'[ABRIR] {galpath}',flush=True)
    return galpath


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planejar', action='store_true')
    parser.add_argument('--executar', action='store_true')
    parser.add_argument('--pca-f10', action='store_true', help='Apos ambos OOF, avaliar PCA F10 exploratorio.')
    args = parser.parse_args(argv)
    exigir(args.planejar != args.executar, 'Escolha exatamente um: --planejar ou --executar.')
    planos = configs()
    print('[PLANO] YAML oficial intacto; amostra de 10%; F1-F9; pasta outputs/COMPARACAO_OOF_PCA1D_TSNE1D_V1')
    for metodo, cfg in planos.items():
        print(f"[PLANO] {metodo.upper()} 1D: {len(list(grade_candidatos_oof(cfg)))} candidatos x 9 folds; "
              f"n_jobs={cfg['execucao']['n_jobs']}; F10 automatico=NAO")
    if args.planejar:
        print('[PLANO] Sem leitura de dados e sem treinar; F10 PCA so com --pca-f10 depois dos dois OOF.')
        return 0
    exigir((ROOT/'data/raw/dataset_raw.parquet').is_file(), 'Parquet nao encontrado.')
    cfgs, estado = preparar()
    for metodo in ('pca','tsne'):
        executar_metodo(metodo, estado)
    pca = Path(estado['pca'])
    tsne = Path(estado['tsne'])
    comparar(pca, tsne)
    if args.pca_f10:
        # F10 PCA ja foi previamente acessado neste projeto: carater exploratorio.
        from src.finalizacao_automatica_pca1d import concluir_ensaio
        m = ler_json(pca/'manifesto.json')
        if m.get('teste_final',{}).get('status') == 'concluido':
            print('[RETOMADA] F10 PCA ja concluido.',flush=True)
        else:
            print('[PCA F10] INICIANDO APENAS APOS AMBOS OS OOF E A COMPARACAO F1-F9.',flush=True)
            concluir_ensaio(pca)
        exigir((pca/'vencedor/teste_final/galeria/index.html').is_file(),
               'Galeria F10 PCA ausente.')
        estado['f10_pca'] = True
        salvar_estado(estado)
        print('[PCA F10] Exploratorio concluido: '+str(pca/'vencedor/teste_final/galeria/index.html'),flush=True)
    print('\n[CONCLUIDO] TSNE F10 permanece reservado; comparacao OOF preservada.',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
