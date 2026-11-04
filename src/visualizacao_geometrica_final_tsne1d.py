# -*- coding: utf-8 -*-
"""Fase 3B.2A: GMM01/04/07/08 de F10 do t-SNE 1D, sem novo fit/OOS.

Consome EXCLUSIVAMENTE os resultados assinados da finalizacao 3B.1.
O HTML completo, TSNE13/14 e PRE01 continuam pendentes.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from src.folds import hash_arquivo
from src.amostragem_f10 import fracao_amostra_f10
from src.visualizacao_final_tsne1d import _entradas, exigir, ler_json

PNG_GEOMETRIA = (
    'GMM01_densidade_marginal_1d_TSNE1.png',
    'GMM04_dominancia_1d.png',
    'GMM07_sensibilidade_1d_comparativo.png',
    'GMM08_mapa_risco_1d_comparativo.png',
)


def _fontes(pasta):
    pasta = Path(pasta).resolve()
    teste, config, met, marco, y, p, tau = _entradas(pasta)
    exigir(marco.get('sha256_coordenadas'),
           'Fase 3B.1 ausente: nao ha hash das coordenadas F10.')
    cpath = teste/'coordenadas_TSNE1_f10.csv'
    exigir(cpath.is_file() and hash_arquivo(cpath)==marco['sha256_coordenadas'],
           'Coordenadas F10 alteradas ou ausentes.')
    ids = np.load(teste/'indices_f10_amostra.npy', allow_pickle=False)
    coord = pd.read_csv(cpath)
    exigir(list(coord.columns)==['linha_original','TSNE1'] and len(coord)==len(ids)
           and np.array_equal(coord['linha_original'].to_numpy(dtype=np.int64),ids),
           'Coordenadas e preditos F10 desalinhados.')
    z = coord['TSNE1'].to_numpy(dtype=np.float64)
    exigir(np.isfinite(z).all(), 'Coordenadas t-SNE nao finitas.')
    for nome in ('densidades_marginais','regioes_dominancia',
                 'sensibilidade_fronteira','mapa_risco_fronteira'):
        exigir(nome in config['visualizacao']['graficos'],f'Grafico ausente do YAML: {nome}')
    return teste, config, met, marco, pd.DataFrame({'TSNE1':z}), y, p, tau


@contextmanager
def _rotulos_f10(y,p,tau):
    """Adapta rótulos no PNG gerado; não modifica módulos nem arquivos OOF."""
    from matplotlib.text import Text
    from src import visualizacao_geometrica as geo
    from src import visualizacao_gmm07 as gmm07
    orig_av,orig_save,orig_rot = geo._avaliar_cortes_oof,geo._guardar,gmm07._rotulo_mcc
    mcc05=float(matthews_corrcoef(y,p>=.5))
    mccstar=float(matthews_corrcoef(y,p>=tau))

    def avaliar(av,limiar):
        exigir(isinstance(av,dict) and av.get('origem')=='f10_amostral_congelado'
               and np.isclose(limiar,tau), 'Origem ou limiar alterados na geometria F10.')
        return mcc05,mccstar

    def rotulo(av):
        exigir(av.get('origem')=='f10_amostral_congelado','GMM07 sem origem F10.')
        return 'MCC F10', 'MCC F10 exploratorio; limiar tau* escolhido exclusivamente no OOF.'

    pares=(('MCC OOF','MCC F10'),
           ('B — Corte ótimo MCC OOF','B — τ* congelado OOF'),
           ('B — Corte ótimo MCC','B — τ* congelado OOF'),
           ('PC1 |','TSNE1 |'))

    def guardar(fig,dest,base,ext,dpi):
        for obj in fig.findobj(match=Text):
            titulo=obj.get_text()
            novo=titulo
            for antigo,novo_texto in pares:
                novo=novo.replace(antigo,novo_texto)
            if titulo != novo:
                obj.set_text(novo)
        return orig_save(fig,dest,base,ext,dpi)

    geo._avaliar_cortes_oof,geo._guardar,gmm07._rotulo_mcc=avaliar,guardar,rotulo
    try:
        yield mcc05,mccstar
    finally:
        geo._avaliar_cortes_oof,geo._guardar,gmm07._rotulo_mcc=orig_av,orig_save,orig_rot


def _conferir_existente(dest,marco):
    man=ler_json(dest/'manifesto_geometria_f10.json')
    exigir(man.get('estado')=='GEOMETRIA_4_DE_12_SEM_GALERIA_COMPLETA' and
           man.get('sha256_modelo')==marco['sha256_modelo'] and
           man.get('sha256_predicoes')==marco['sha256_predicoes'] and
           man.get('sha256_coordenadas')==marco['sha256_coordenadas'] and
           man.get('sha256_metricas')==marco['sha256_metricas'] and
           set(man.get('sha256_arquivos',{}))==set(PNG_GEOMETRIA) and
           np.isclose(float(man.get('fracao_f10_dentro_do_fold', -1)),
                      float(marco['fracao_f10_dentro_do_fold'])),
           'Geometria F10 previa nao corresponde as fontes congeladas.')
    for nome,h in man['sha256_arquivos'].items():
        f=dest/nome
        exigir(f.is_file() and hash_arquivo(f)==h,'Geometria F10 alterada: '+nome)
    exigir({f.name for f in dest.glob('*.png')}==set(PNG_GEOMETRIA),
           'Conjunto de PNGs geometricos F10 inesperado.')
    return dest


def gerar(pasta):
    """Gera 4 PNGs congelados; nao abre Parquet nem recalcula t-SNE/F10."""
    teste,cfg,met,marco,z,y,p,tau=_fontes(pasta)
    dest=teste/'visualizacao_geometrica'
    if dest.exists():
        return _conferir_existente(dest,marco)
    stage=teste/'.visualizacao_geometrica_incompleta'
    exigir(not stage.exists(),'Staging geometricos anterior: preservar para auditoria.')
    lock=teste/'.visualizacao_geometrica.lock'
    try:
        with lock.open('x',encoding='utf-8') as fd:
            fd.write('F10 geometria em andamento; nao executar em paralelo.\n')
    except FileExistsError as exc:
        raise RuntimeError('Geometria F10 ja em andamento.') from exc
    try:
        from src.treinamento_final_tsne import carregar_pacote_confiavel
        pacote=carregar_pacote_confiavel(teste/'modelo_final_f1_f9')
        exigir(pacote.configuracao==cfg and
               pacote.escolha_oof['id_candidato']==marco['id_candidato_oof'] and
               np.isclose(pacote.limiar_oof,tau) and
               list(pacote.gmm.colunas_)==['TSNE1'] and
               pacote.gmm.modelo_.covariance_type=='full',
               'Modelo GMM final nao corresponde ao t-SNE 1D F10.')
        gamma=pacote.gmm.responsabilidades(z)
        from src.probabilidade import calcular_probabilidades
        conferidos=np.asarray(calcular_probabilidades(gamma,pacote.risco.riscos_),dtype=np.float64)
        exigir(len(conferidos)==len(p) and np.isfinite(conferidos).all()
               and np.max(np.abs(conferidos-p))<=1e-6
               and np.array_equal(conferidos>=tau,p>=tau)
               and np.array_equal(conferidos>=.5,p>=.5),
               'Scores do GMM congelado divergem das predicoes F10.')
        data={'coord':z,'y':y,'scores':p,'gamma':gamma,'gmm':pacote.gmm,
              'rh':pacote.risco,'redutor':pacote.tsne,'limiar_oof':tau}
        from src import visualizacao_geometrica as geo
        from src import visualizacao_gmm01 as gmm01
        from src import visualizacao_gmm04 as gmm04
        from src import visualizacao_gmm07 as gmm07
        stage.mkdir(exist_ok=False)
        dpi=int(cfg['visualizacao']['padrao'].get('dpi',180))
        gmm01.gerar(cfg,data,stage,'png',dpi,False)
        gmm04.gerar(cfg,data,stage,'png',dpi,False)
        with _rotulos_f10(y,p,tau) as (mcc05,mccstar):
            escolha={'limiar_mcc_oof':tau}
            av={'origem':'f10_amostral_congelado'}
            gmm07.gerar(cfg,data,escolha,stage,'png',dpi,False,avaliacao=av)
            geo._fronteira(cfg,data,escolha,stage,'png',dpi,False,avaliacao=av)
        exigir({f.name for f in stage.glob('*.png')}==set(PNG_GEOMETRIA),
               'Renderizacao F10 nao produziu 4/4 PNGs geometricos.')
        hashes={}
        for nome in PNG_GEOMETRIA:
            f=stage/nome
            exigir(f.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n', 'PNG geometrico invalido: '+nome)
            hashes[nome]=hash_arquivo(f)
        manifesto={'estado':'GEOMETRIA_4_DE_12_SEM_GALERIA_COMPLETA',
                   'n_f10':len(y),'fracao_f10_dentro_do_fold':fracao_amostra_f10(cfg),
                   'fracao_desenvolvimento':float(cfg['dados']['fracao_base']),
                   'id_candidato_oof':marco['id_candidato_oof'],
                   'limiar_mcc_oof_congelado':tau,
                   'mcc_f10_05':mcc05,'mcc_f10_tau_oof':mccstar,
                   'sha256_modelo':marco['sha256_modelo'],
                   'sha256_predicoes':marco['sha256_predicoes'],
                   'sha256_coordenadas':marco['sha256_coordenadas'],
                   'sha256_metricas':marco['sha256_metricas'],
                   'sha256_arquivos':hashes,
                   'sem_fit':True,'sem_novas_predicoes':True,
                   'sem_reotimizacao_f10':True,'galeria_f10_completa':False}
        with (stage/'manifesto_geometria_f10.json').open('x',encoding='utf-8') as f:
            json.dump(manifesto,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write('\n')
        exigir(not dest.exists(),'Destino geometrico apareceu durante geracao.')
        os.replace(stage,dest)
        return _conferir_existente(dest,marco)
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    import argparse
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--experimento',type=Path,required=True,
                     help='Pasta de experimento t-SNE 1D ja avaliado em F10.')
    args=cli.parse_args(argv)
    destino=gerar(args.experimento)
    print('[TSNE1D F10] Geometria 4/4 gerada/verificada:',destino)
    print('[TSNE1D F10] TSNE13/14, PRE01 e HTML FINAL PENDENTES.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
