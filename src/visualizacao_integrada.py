"""Passo 07.03 — galeria integrada e mapa 2D COMPLEMENTAR (nao oficial).

    python -m src.visualizacao_integrada --experimento outputs/ensaio_pca_gmm_... --planejar
    python -m src.visualizacao_integrada --experimento outputs/ensaio_pca_gmm_... --executar

Nenhum F10 e utilizado. Os 15 TIPOS sao uma lista condicional, nao 15
imagens obrigatorias em cada metodo. Para PCA 1D, o melhor candidato
PCA 2D ja comparado na MESMA grade pode gerar um mapa extra rotulado
"COMPLEMENTAR, NAO VENCEDOR" e usando SEU proprio limiar OOF.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import visualizacao_geometrica as geo
from src import visualizacao_oof as oof

FORMATS = ('.png', '.pdf', '.svg')
NOMES_GEO = dict(geo.NUM)
GEO_PREFIXOS = {
    'densidades_marginais': 'GMM01_densidade_marginal_',
    'regioes_dominancia': 'GMM04_dominancia_',
    'painel_parametros_gaussianos': '09_painel_parametros_gaussianos',
    'pca_cotovelo': '11_pca_cotovelo',
    'pca_projecoes': '12_pca_projecao_',
    'tsne_trustworthiness': 'TSNE02_trustworthiness_vs_perplexity',
    'tsne_mapa': '13_tsne_mapa_',
    'tsne_diagnostico': '14_tsne_diagnostico',
    'mapa_risco_fronteira': 'GMM08_mapa_risco_',
    'sensibilidade_fronteira': 'GMM07_sensibilidade_',
}


def _exigir(condicao, mensagem):
    if not condicao:
        raise ValueError(mensagem)


def _ler_json(caminho):
    return json.loads(Path(caminho).read_text(encoding='utf-8-sig'))


def _hash(caminho):
    sha = hashlib.sha256()
    with Path(caminho).open('rb') as arq:
        for bloco in iter(lambda: arq.read(1024*1024),b''):
            sha.update(bloco)
    return sha.hexdigest()


def _guardar_gerenciado(caminho, texto):
    caminho.parent.mkdir(parents=True,exist_ok=True)
    # Somente arquivos de nossa galeria: escrita atomica. Fontes OOF nunca alteradas.
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=caminho.parent,
                                     prefix='.temp07_',suffix='.tmp',delete=False) as arq:
        temporario = Path(arq.name)
        arq.write(texto)
    try:
        os.replace(temporario,caminho)
    finally:
        temporario.unlink(missing_ok=True)


def _arquivo_seguro_copiar(origem, destino):
    _exigir(origem.is_file(),f'Arquivo de grafico nao encontrado: {origem}')
    _exigir(origem.suffix.lower() in FORMATS,'Copiar somente arquivos de imagem/grafico.')
    destino.parent.mkdir(parents=True,exist_ok=True)
    if destino.exists():
        _exigir(_hash(origem)==_hash(destino),
                f'Conflito no consolidado; nao sobrescrever: {destino}')
        return
    shutil.copy2(origem,destino)


def _config_candidata_2d(grade, escolha):
    """Melhor candidato PCA=2 no MESMO OOF, sem alterar ranking oficial."""
    if escolha['configuracao']['metodo'].upper()!='PCA':
        return None  # Nao refazer TSNE artificialmente para mapa complementar.
    if int(escolha['configuracao']['reducao']['n_components']) != 1:
        return None  # PCA>=2 ja permite grafico oficial apropriado.
    possiveis=[]
    for _,row in grade.iterrows():
        cand=json.loads(row['configuracao'])
        if (cand['metodo'].upper()=='PCA' and
            int(cand['reducao']['n_components'])==2 and
            str(row['id_candidato'])!=str(escolha['id_candidato'])):
            possiveis.append((float(row['score']),float(row['ap']),str(row['id_candidato']),row,cand))
    if not possiveis:
        return None
    possiveis.sort(key=lambda t: (-t[0],-t[1],t[2]))
    _,_,_,row,cand=possiveis[0]
    return {'id_candidato':str(row['id_candidato']), 'configuracao':cand,
            'limiar_mcc_oof':float(row['limiar_mcc']), 'score_oof':float(row['score']),
            'ap_oof':float(row['ap']), 'mcc_oof':float(row['mcc']),
            'teste_final_usado':False,'status':'complementar_nao_vencedor'}


def _validar_npz_candidato(experimento,candidato,ids,y_oof,folds):
    caminho=experimento/'oof'/'predicoes'/f'{candidato["id_candidato"]}.npz'
    _exigir(caminho.is_file(),f'Predicoes do candidato 2D ausentes: {caminho}')
    with np.load(caminho,allow_pickle=False) as arq:
        _exigir({'linha_original','y_true','fold','id_candidato','fold_esperado'}.issubset(arq.files),
                'NPZ do candidato 2D incompleto.')
        _exigir(str(arq['id_candidato'].item())==candidato['id_candidato'],
                'Identidade NPZ 2D incompativel.')
        _exigir(int(arq['fold_esperado'].item())==0,'Predicao 2D nao e OOF consolidado.')
        linhas=np.asarray(arq['linha_original'],dtype=np.int64)
        ys=np.asarray(arq['y_true'],dtype=np.int8)
        fs=np.asarray(arq['fold'],dtype=np.uint8)
    _exigir(np.array_equal(linhas,ids) and np.array_equal(ys,y_oof),
            'Candidato 2D nao usou exatamente as mesmas observacoes OOF.')
    _exigir(np.array_equal(fs,folds[ids]) and np.all(fs!=10),
            'Fold F10 ou folds errados no candidato 2D.')


def _desenhar_complementar(config,data,escolha_complementar,arquivo):
    """Nuvens de GMM ajustado no desenvolvimento de candidato PCA2D, nao vencedor."""
    C=np.asarray(data['coord'],dtype=float)
    y=np.asarray(data['y'],dtype=np.int8)
    _exigir(C.ndim==2 and C.shape[1]==2,'Mapa complementar requer exatamente PCA 2D.')
    gmm=data['gmm']; rh=data['rh']
    op=config['visualizacao']['graficos']['mapa_risco_fronteira']
    n=int(op.get('resolucao_2d',180))
    _exigir(20<=n<=450,'Resolucao da malha 2D invalida.')
    margem=float(op.get('margem',.15))
    _exigir(0<=margem<=1,'Margem de malha invalida.')
    a=np.linspace(*geo._faixa(C[:,0],margem),n)
    b=np.linspace(*geo._faixa(C[:,1],margem),n)
    xx,yy=np.meshgrid(a,b)
    coordenadas=np.c_[xx.ravel(),yy.ravel()]
    risco=np.asarray(geo._pontuar_gmm_array(gmm,rh,coordenadas)).reshape(n,n)
    _exigir(np.isfinite(risco).all(), 'Mapa complementar gerou risco invalido.')
    limiar=float(escolha_complementar['limiar_mcc_oof'])
    _exigir(0<=limiar<=1,'Limiar OOF do candidato 2D invalido.')
    fig,ax=plt.subplots(figsize=(11,8))
    malha=ax.pcolormesh(xx,yy,risco,shading='auto',cmap='viridis')
    fig.colorbar(malha,ax=ax,label='Risco estimado de fraude (PCA 2D complementar)')
    idx=geo._amostra_plot(len(y),y,config,min(int(op.get('n_pontos_scatter',5000)),len(y)))
    geo._scatter(ax,C,y,idx,config,(0,1))
    branco='white'
    # Contorno somente se o risco realmente atravessa o limiar OOF especifico do candidato 2D.
    fronteira_desenhada=False
    if float(risco.min())<limiar<float(risco.max()):
        try:
            cont=ax.contour(xx,yy,risco,levels=[limiar],colors=[branco],linewidths=2.1)
            fronteira_desenhada=bool(cont.allsegs and any(len(v) for v in cont.allsegs[0]))
        except ValueError:
            fronteira_desenhada=False
    if fronteira_desenhada:
        ax.plot([],[],color='white',lw=2,label=f'Fronteira MCC OOF do candidato 2D: {limiar:.5g}')
    else:
        ax.text(.02,.02,f'Limiar MCC OOF={limiar:.5g} fora da faixa/sem contorno nesta malha',
                transform=ax.transAxes,color='white',fontsize=9,
                bbox={'facecolor':'black','alpha':.65,'edgecolor':'none'})
    regra=geo._regra_gaussianas(config,rh)
    for k in range(len(gmm.saidas_)):
        media=np.asarray(gmm.modelo_.means_[k],dtype=float)
        cov=geo._covariancia(gmm,k)
        vals,vect=np.linalg.eigh(cov)
        if np.any(vals<=0):continue
        # Eixo principal = autovetor do maior autovalor.
        ang=np.degrees(np.arctan2(vect[1,1],vect[0,1]))
        for sigma,alpha in [(1.0,.13),(2.0,.06)]:
            ax.add_patch(Ellipse(media,
                width=2*sigma*np.sqrt(vals[1]),height=2*sigma*np.sqrt(vals[0]),
                angle=ang,facecolor='white',edgecolor='white',lw=1.3 if sigma==2 else .6,
                linestyle='-' if sigma==2 else ':',alpha=alpha if sigma==1 else .45))
        ax.plot(media[0],media[1],marker='x',ms=8,color='white',mew=2)
        risco_k=float(rh.riscos_.iloc[k]); peso_k=float(gmm.modelo_.weights_[k])
        ax.annotate(f'G{k+1} | RH={risco_k:.4g} | peso={peso_k:.2f}\n'
                     f'{geo._rotulo_gaussiana(risco_k,regra)[1]}',
                    xy=media,xytext=(5,6),textcoords='offset points',
                    fontsize=8,color='white',weight='bold',
                    path_effects=[])
    fig.text(.5,.025,
             f"Rotulo gaussiano: alto risco se RH >= {regra['multiplicador_prevalencia']:.3g} x "
             f"prevalencia do treino ({regra['prevalencia_treino']:.4%}) = "
             f"{regra['limiar_alto_risco_gaussiana']:.4%}. Apenas descritivo.",
             ha='center',fontsize=9)
    ax.set(xlabel='PC1 — candidato 2D',ylabel='PC2 — candidato 2D')
    ax.set_title('COMPLEMENTAR — nuvens gaussianas + risco + fronteira (NAO E O VENCEDOR)\n'
                 f'PCA=2, K={len(gmm.saidas_)} | ajuste: desenvolvimento F1–F9 | '
                 f'Score OOF={escolha_complementar["score_oof"]:.6f}',fontsize=11)
    handles,labels=ax.get_legend_handles_labels()
    if handles:ax.legend(loc='best',fontsize=8,facecolor='#eee',framealpha=.85)
    ax.set_xlim(a[0],a[-1]);ax.set_ylim(b[0],b[-1])
    fig.tight_layout(rect=[0,0.04,1,1])
    if arquivo.exists():
        plt.close(fig)
        raise FileExistsError(f'Grafico complementar preexistente sem manifesto validado: {arquivo}')
    arquivo.parent.mkdir(parents=True,exist_ok=True)
    try:
        fig.savefig(arquivo,dpi=int(config['visualizacao']['padrao'].get('dpi',180)),bbox_inches='tight')
    finally:
        plt.close(fig)
    if config['visualizacao']['padrao'].get('salvar_dados_grafico',True):
        tabela=pd.DataFrame({'PC1':coordenadas[:,0],'PC2':coordenadas[:,1],
                             'risco_estimado_candidato_complementar':risco.ravel()})
        geo._salvar_dados(tabela,arquivo.parent,arquivo.stem,True)
    return {'fronteira_mcc_visivel':bool(fronteira_desenhada),
            'risco_minimo_malha':float(risco.min()),'risco_maximo_malha':float(risco.max()),
            'n_pontos_malha':int(n*n),'n_scatter':len(idx),
             'regra_rotulagem_descritiva':regra}


def _verificar_geracoes(experimento,config,escolha,*,executar):
    oof_dir=experimento/'oof'/'graficos_gerados'
    geo_dir=experimento/'teste_final'/'graficos_gerados'
    ext='.'+config['visualizacao']['padrao'].get('formato','png').lower()
    _exigir(ext in FORMATS,'Extensao de grafico desconhecida.')
    oof_ativos=[nome for nome in oof.ARQUIVOS if oof._ativado(config,nome)]
    esperado_geo=[nome for nome in geo.TIPOS if geo._ativo(config,nome) and
        (nome not in ('pca_cotovelo','pca_projecoes','tsne_trustworthiness','tsne_mapa','tsne_diagnostico')
         or (nome.startswith('pca_') and escolha['configuracao']['metodo']=='PCA')
         or (nome.startswith('tsne_') and escolha['configuracao']['metodo']=='TSNE'))]
    fonte_oof=oof_dir/'manifesto_graficos_oof_07_01.json'
    fonte_geo=geo_dir/'manifesto_geometria_07_02.json'

    def completos():
        imagens_oof=all((oof_dir/(oof.ARQUIVOS[n]+ext)).is_file() for n in oof_ativos)
        imagens_geo=all(
            (
                ((geo_dir/f'GMM08_mapa_risco_1d_comparativo{ext}').is_file()
                  if int(escolha['configuracao']['reducao']['n_components'])==1
                  else all((geo_dir/f'GMM08_mapa_risco_{int(escolha["configuracao"]["reducao"]["n_components"])}d_{corte}{ext}').is_file()
                           for corte in ('fixo','mcc')))
                if n=='mapa_risco_fronteira'
                else ((geo_dir/f'GMM07_sensibilidade_{int(escolha["configuracao"]["reducao"]["n_components"])}d_comparativo{ext}').is_file()
                      if n=='sensibilidade_fronteira'
                      else (len(list(geo_dir.glob(f'GMM01_densidade_marginal_{int(escolha["configuracao"]["reducao"]["n_components"])}d_*{ext}')))
                            == int(escolha['configuracao']['reducao']['n_components'])
                            if n=='densidades_marginais'
                            else (geo_dir/f'GMM04_dominancia_{int(escolha["configuracao"]["reducao"]["n_components"])}d{ext}').is_file()
                            if n=='regioes_dominancia'
                            else any(geo_dir.glob(f'{GEO_PREFIXOS[n]}*{ext}'))))
            ) for n in esperado_geo)
        return imagens_oof,imagens_geo

    tem_oof,tem_geo=completos()
    if executar and not tem_oof:
        oof.gerar_graficos_oof(experimento)
    if executar and not tem_geo:
        print('[07.03] Faltam graficos geometricos: executando 07.02 UMA vez.')
        geo.executar(experimento)
    tem_oof,tem_geo=completos()
    if executar:
        _exigir(tem_oof and tem_geo,'Nao foi possivel completar os graficos habilitados no YAML.')
        _exigir(fonte_oof.is_file() and fonte_geo.is_file(), 'Manifestações 07.01/07.02 ausentes.')
    if fonte_oof.is_file():
        mo=_ler_json(fonte_oof)
        _exigir(str(mo.get('id_vencedor'))==str(escolha['id_candidato']) and mo.get('usa_f10') is False,
                'Manifesto do 07.01 diverge ou utilizou F10.')
    if fonte_geo.is_file():
        mg=_ler_json(fonte_geo)
        _exigir(str(mg.get('id_vencedor'))==str(escolha['id_candidato']) and
                mg.get('f10_utilizado') is False and mg.get('origem')=='desenvolvimento_F1_F9',
                'Manifesto do 07.02 diverge ou nao corresponde ao desenvolvimento.')
    return oof_dir,geo_dir,oof_ativos,esperado_geo,tem_oof,tem_geo


def _complementar(experimento,config,grade,escolha,*,executar):
    candidato=_config_candidata_2d(grade,escolha)
    if candidato is None:
        return {'situacao':'nao_aplicavel','motivo':'Vencedor ja e 2D/3D, nao e PCA ou nao ha candidato PCA=2 no OOF.'}
    destino=experimento/'visualizacao_consolidada'/'complementar_2d'
    ext='.'+config['visualizacao']['padrao'].get('formato','png').lower()
    arquivo=destino/('16_mapa_complementar_PCA2_GMM'+str(candidato['configuracao']['gmm']['n_components'])+ext)
    arquivo_meta=destino/'manifesto_complementar_2d.json'
    if not executar:
        return {'situacao':'planejado','candidato':candidato['id_candidato'],
                'limiar_oof_proprio':candidato['limiar_mcc_oof'], 'arquivo':str(arquivo)}
    # Execucao idempotente: nao ajustar novamente se o artefato ja foi validado.
    if arquivo.exists() and arquivo_meta.exists():
        antigo=_ler_json(arquivo_meta)
        _exigir(antigo.get('id_candidato')==candidato['id_candidato'] and
                antigo.get('id_vencedor_oficial')==escolha['id_candidato'] and
                antigo.get('f10_utilizado') is False and
                np.isclose(float(antigo.get('limiar_mcc_oof_candidato')),
                           candidato['limiar_mcc_oof'],atol=1e-12) and
                antigo.get('sha256_grafico')==_hash(arquivo),
                'Grafico 2D anterior nao corresponde ao candidato ou foi alterado.')
        print('[REUTILIZADO] mapa 2D complementar validado.')
        return {'situacao':'reutilizado','arquivo':str(arquivo),'candidato':candidato['id_candidato']}
    _exigir(not arquivo.exists() and not arquivo_meta.exists(),
            'Resultado 2D parcial: nao sobrescrever. Inspecione antes de tentar novamente.')
    # Protecao dos dados de origem e fold do segundo candidato tambem.
    cfg,esc,ids,y_oof,arquivo_base,folds=geo._selecao(experimento,Path(__file__).resolve().parents[1])
    _exigir(esc['id_candidato']==escolha['id_candidato'],'Vencedor mudou entre auditorias.')
    _validar_npz_candidato(experimento,candidato,ids,y_oof,folds)
    df=geo.orq._ler_desenvolvimento(arquivo_base,ids)
    _exigir(np.array_equal(df[cfg['dados']['coluna_alvo']].to_numpy(dtype=np.int8),y_oof),
            'Dados alvo divergem do OOF; treino complementar bloqueado.')
    # O mesmo pipeline, modificando apenas o candidato selecionado dentro da grade OOF.
    jobs=int(cfg['execucao'].get('n_jobs',1))
    print('[07.03] Reajustando um candidato PCA2D complementar no desenvolvimento, SEM F10:',
          candidato['id_candidato'],flush=True)
    with threadpool_limits(limits=1 if jobs==-1 else max(1,jobs)):
        data=geo._treinar_vencedor(df,cfg,candidato)
    info=_desenhar_complementar(cfg,data,candidato,arquivo)
    manifesto={'etapa':'07.03','tipo':'diagnostico_complementar_2d_nao_oficial',
              'id_candidato':candidato['id_candidato'],
              'id_vencedor_oficial':escolha['id_candidato'],
              'score_oof_candidato':candidato['score_oof'],
              'limiar_mcc_oof_candidato':candidato['limiar_mcc_oof'],
              'pca_n_components':2,'gmm_n_components':candidato['configuracao']['gmm']['n_components'],
              'origem':'desenvolvimento_F1_F9','f10_utilizado':False,
              'nao_e_o_vencedor':True,'n_desenvolvimento':len(df),
              'sha256_grafico':_hash(arquivo),**info}
    _guardar_gerenciado(arquivo_meta,json.dumps(manifesto,ensure_ascii=False,indent=2)+'\n')
    print('[GERADO] mapa 2D complementar:',arquivo)
    return {'situacao':'gerado','arquivo':str(arquivo),'candidato':candidato['id_candidato']}


def _html_galeria(experimento,arquivos,escolha,complemento,metodo,tipo):
    blocos=[]
    for secao,titulo,aviso in (
        ('oof','00–06 — Painel e validacao OOF','Predicoes fora do treino; somente o grafico 06 compara os candidatos.'),
        ('geometria_desenvolvimento','GMM 01/07/08/09 e diagnosticos PCA/t-SNE do vencedor','Modelo reajustado em F1–F9. Esta pasta NAO contem medidas do F10.'),
        ('complementar_2d','16 — Diagnostico 2D complementar','Outro candidato da grade OOF; NAO representa a fronteira do vencedor.')
    ):
        fotos=[(p,n) for p,n in arquivos if p.parts[0]==secao]
        if not fotos:continue
        partes=[f'<section><h2>{html.escape(titulo)}</h2><p>{html.escape(aviso)}</p><div class="grade">']
        for caminho,rotulo in fotos:
            ref=html.escape(caminho.as_posix(),quote=True)
            nome=html.escape(rotulo)
            previa=(f'<img src="{ref}" loading="lazy" alt="{nome}">' if
                    caminho.suffix.lower() in ('.png','.svg') else '<div class="sem-previa">Visualização em PDF — clique para abrir</div>')
            partes.append(f'<article><a href="{ref}" target="_blank" rel="noopener">{previa}<div>{nome}</div></a></article>')
        partes.append('</div></section>')
        blocos.extend(partes)
    cabecalho=(f'Vencedor OOF: {escolha["id_candidato"]} | {metodo} | '
               f'Score = {escolha["score_oof"]:.6f}')
    return '''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TCC — Galeria do Passo 07</title>
<style>body{font:16px system-ui,sans-serif;background:#f5f7fa;color:#172236;max-width:1320px;margin:auto;padding:24px}
h1{margin-bottom:6px}.selo{padding:12px;background:#e8edf4;border-left:4px solid #355877;border-radius:4px}
section{margin-top:40px}.grade{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:20px}
article{background:white;border:1px solid #d7dee6;border-radius:10px;padding:12px;box-shadow:0 2px 8px #0001}
article img{width:100%;aspect-ratio:4/3;object-fit:contain}article a{color:#233a58;text-decoration:none}
.sem-previa{padding:90px 12px;text-align:center;color:#536579}</style></head><body>''' + \
      f'<h1>Passo 07 — Galeria consolidada</h1><p class="selo">{html.escape(cabecalho)}</p>' + \
      ('<p>Ensaio amostral: resultados exploratorios. Sem F10. ' if tipo=='ensaio' else
       '<p>Selecao OOF sobre desenvolvimento. Sem F10. ') + \
      'Mapa complementar 2D nao e o modelo vencedor nem altera o ranking OOF.</p>' + \
      ''.join(blocos)+'</body></html>\n'


def integrar(experimento, *, planejar=False, complementar_2d=True):
    experimento=Path(experimento).resolve()
    config,grade,escolha,y,p=oof._ler_resultados(experimento)
    manifesto=_ler_json(experimento/'manifesto.json')
    _exigir(manifesto.get('status')=='concluido_06_03' and
            manifesto.get('tipo') in ('ensaio','oficial') and
            escolha.get('teste_final_usado') is False,
            'Experimento incompleto ou F10 ja usado na selecao.')
    od,gd,ativos_oof,ativos_geo,pronto_oof,pronto_geo=_verificar_geracoes(
        experimento,config,escolha,executar=not planejar)
    comp=_config_candidata_2d(grade,escolha) if complementar_2d else None
    resumo={'passo':'07.03','experimento':str(experimento),'tipo_experimento':manifesto['tipo'],
            'id_vencedor':escolha['id_candidato'],'score_vencedor_oof':float(escolha['score_oof']),
            'pca_dim_vencedor':escolha['configuracao']['reducao']['n_components'],
            'tipos_oof':[oof.ARQUIVOS[n] for n in ativos_oof],
            'tipos_geometricos':ativos_geo,
            'tipos_geometricos_nao_aplicaveis':[n for n in geo.TIPOS if n not in ativos_geo],
            'oof_pronto':pronto_oof,'geometria_pronta':pronto_geo,
            'id_candidato_2d_complementar':comp['id_candidato'] if comp else None,
            'f10_utilizado':False}
    if planejar:
        print('PASSO 07.03 — PLANEJAMENTO SOMENTE LEITURA')
        print('Vencedor:',escolha['id_candidato'],'Score:',escolha['score_oof'])
        print('OOF existente:',pronto_oof,'| Tipos:',len(ativos_oof))
        print('Geometria existente:',pronto_geo,'| Tipos:',len(ativos_geo))
        print('Complementar PCA2D:', comp['id_candidato'] if comp else 'nao aplicavel')
        print('F10: nao lido; nenhum ajuste realizado.')
        return resumo
    adicional= _complementar(experimento,config,grade,escolha,executar=True) if comp else {'situacao':'nao_aplicavel'}
    base=experimento/'visualizacao_consolidada'
    salvos=[]
    for origem,secao in ((od,'oof'),(gd,'geometria_desenvolvimento')):
        if not origem.is_dir():continue
        for arq in sorted(origem.iterdir()):
            if not arq.is_file() or arq.suffix.lower() not in FORMATS:
                continue
            if secao=='oof':
                permitido=arq.stem in (oof.ARQUIVOS[n] for n in ativos_oof)
            else:
                # Ignorar imagens legadas 07/08/09-heatmap/10 ainda no disco.
                permitido=any(arq.stem.startswith(GEO_PREFIXOS[n]) for n in ativos_geo
                              if n not in ('mapa_risco_fronteira','sensibilidade_fronteira','densidades_marginais'))
                if 'regioes_dominancia' in ativos_geo:
                    dim=int(escolha['configuracao']['reducao']['n_components'])
                    permitido=permitido and (not arq.stem.startswith('GMM04_dominancia_') or
                        arq.stem==f'GMM04_dominancia_{dim}d')
                if 'mapa_risco_fronteira' in ativos_geo:
                    dim=int(escolha['configuracao']['reducao']['n_components'])
                    mapas={'GMM08_mapa_risco_1d_comparativo'} if dim==1 else {
                        f'GMM08_mapa_risco_{dim}d_fixo',f'GMM08_mapa_risco_{dim}d_mcc'}
                    permitido=permitido or arq.stem in mapas
                if 'sensibilidade_fronteira' in ativos_geo:
                    dim=int(escolha['configuracao']['reducao']['n_components'])
                    permitido=permitido or arq.stem==f'GMM07_sensibilidade_{dim}d_comparativo'
                if 'densidades_marginais' in ativos_geo:
                    dim=int(escolha['configuracao']['reducao']['n_components'])
                    permitido=permitido or arq.stem.startswith(f'GMM01_densidade_marginal_{dim}d_')
            if permitido:
                relativo=Path(secao)/arq.name
                _arquivo_seguro_copiar(arq,base/relativo)
                salvos.append((relativo,arq.stem.replace('_',' ')))
    if adicional.get('arquivo'):
        arq=Path(adicional['arquivo']);salvos.append((Path('complementar_2d')/arq.name,
                               'Mapa 2D complementar — nao vencedor'))
    indice=base/'index.html'
    _guardar_gerenciado(indice,_html_galeria(experimento,salvos,escolha,adicional,
                                            escolha['configuracao']['metodo'],manifesto['tipo']))
    resumo.update({'numero_imagens':len(salvos),'complementar_2d':adicional,
                   'galeria_html':str(indice),
                   'arquivos':[str(p.as_posix()) for p,_ in salvos]})
    _guardar_gerenciado(base/'manifesto_integracao_07_03.json',
                        json.dumps(resumo,ensure_ascii=False,indent=2)+'\n')
    print('\nPASSO 07.03 FINALIZADO —',len(salvos),'imagens')
    print('Galeria:',indice)
    print('OOF e dados de origem preservados; F10 nao utilizado.')
    return resumo


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experimento',required=True,type=Path)
    opt=parser.add_mutually_exclusive_group(required=True)
    opt.add_argument('--planejar',action='store_true')
    opt.add_argument('--executar',action='store_true')
    parser.add_argument('--sem-complementar-2d',action='store_true',
                        help='Gerar galeria sem ajuste adicional de candidato PCA2D.')
    a=parser.parse_args(argv)
    integrar(a.experimento,planejar=a.planejar,complementar_2d=not a.sem_complementar_2d)


if __name__=='__main__':
    main()
