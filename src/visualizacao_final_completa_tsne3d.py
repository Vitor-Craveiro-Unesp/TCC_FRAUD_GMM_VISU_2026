# -*- coding: utf-8 -*-
"""F10 t-SNE 3D: 12 PNG, PRE01 offline e galeria imutavel.

Nao realiza fit, nova inferencia, otimizacao de limiar ou selecao OOF.
Consome somente coordenadas/scores do passe OOS previamente congelado.
O PRE01 descreve F10 usando transformadores ja treinados em F1-F9.
"""
from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import matthews_corrcoef

from src.amostragem_f10 import fracao_amostra_f10
from src.avaliacao_final_tsne import avaliar_scores_com_limiar_oof_congelado
from src.folds import hash_arquivo
from src.treinamento_final_tsne import carregar_pacote_confiavel
from src.visualizacao_final_tsne1d import _rotulos_f10 as rotulos_preditivos
from src.visualizacao_geometrica_final_tsne1d import _rotulos_f10 as rotulos_geometricos

PRE01 = 'PRE01_F10/PRE01_F10_diagnostico_padronizacao_completo.html'
PREDITIVOS = (
    '00_painel_score_composto.png',
    '01_curva_precision_recall.png',
    '02_mcc_por_limiar.png',
    '03_matriz_confusao_05.png',
    '04_matriz_confusao_mcc.png',
    '05_distribuicao_probabilidades.png',
)
from src.visualizacao_geometria_tsne3d import NOMES as GMM_NOMES
from src.visualizacao_tsne3d_interativa import PNG as MAPA_PNG, HTML as MAPA_HTML
GMM = tuple(par[0] for par in GMM_NOMES.values())
TSNE = (MAPA_PNG, '14_tsne_diagnostico.png')
PNG_ESPERADOS = PREDITIVOS + GMM + TSNE
HTML_INTERATIVOS = tuple(par[1] for par in GMM_NOMES.values()) + (MAPA_HTML,)
assert len(PNG_ESPERADOS) == len(set(PNG_ESPERADOS)) == 12
assert len(HTML_INTERATIVOS) == len(set(HTML_INTERATIVOS)) == 5


def exigir(ok, mensagem):
    if not ok:
        raise RuntimeError(mensagem)


def ler_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _criar_json_novo(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def _fontes(pasta_experimento):
    pasta = Path(pasta_experimento).resolve()
    teste = pasta / 'vencedor' / 'teste_final'
    marcador = ler_json(teste / 'manifesto_finalizacao_tsne.json')
    config = yaml.safe_load((pasta / 'configuracao.yaml').read_text(encoding='utf-8-sig'))
    exigir(str(config['reducao']['metodo']).upper() == 'TSNE'
           and config['tsne']['n_components'] in (3, [3])
           and marcador.get('metodo') == 'TSNE' and marcador.get('dimensao') == 3
           and marcador.get('etapa') == 'F10_AVALIADO_GALERIA_PENDENTE',
           'Somente t-SNE 3D com F10 previamente avaliado e aceito.')
    exigir(np.isclose(float(marcador.get('fracao_f10_dentro_do_fold', -1)),
                         fracao_amostra_f10(config)),
           'F10 no manifesto nao corresponde a fracao do YAML congelado.')
    arquivos = {
        'predicoes_f10.csv': 'sha256_predicoes',
        'coordenadas_TSNE1_TSNE2_TSNE3_f10.csv': 'sha256_coordenadas',
        'metricas_f10.json': 'sha256_metricas',
        'indices_f10_amostra.npy': 'sha256_indices_arquivo',
        'modelo_final_f1_f9/modelo_final_f1_f9.joblib': 'sha256_modelo',
    }
    for nome, campo in arquivos.items():
        arq = teste / nome
        exigir(arq.is_file() and hash_arquivo(arq) == marcador.get(campo),
               'Artefato congelado F10 nao existe ou mudou: ' + nome)
    pred = pd.read_csv(teste / 'predicoes_f10.csv')
    coord = pd.read_csv(teste / 'coordenadas_TSNE1_TSNE2_TSNE3_f10.csv')
    indices = np.load(teste / 'indices_f10_amostra.npy', allow_pickle=False)
    exigir(list(coord.columns) == ['linha_original', 'TSNE1', 'TSNE2', 'TSNE3']
           and len(pred) == len(coord) == len(indices) > 0,
           'Dimensoes ou colunas de coordenadas F10 invalidas.')
    exigir({'linha_original', 'fold', 'y_true', 'prob_fraude',
            'predicao_tau_0_50', 'predicao_tau_mcc_oof'}.issubset(pred.columns),
           'Predicoes F10 nao contem campos obrigatorios.')
    ids = pred['linha_original'].to_numpy(dtype=np.int64)
    y = pred['y_true'].to_numpy(dtype=np.uint8)
    p = pred['prob_fraude'].to_numpy(dtype=np.float64)
    z = coord[['TSNE1', 'TSNE2', 'TSNE3']].copy()
    met = ler_json(teste / 'metricas_f10.json')
    tau = float(met['limiar_mcc_oof_congelado'])
    exigir(np.array_equal(ids, indices)
           and np.array_equal(coord['linha_original'].to_numpy(dtype=np.int64), indices)
           and np.all(np.diff(ids) > 0) and (pred['fold'] == 10).all()
           and np.isin(y, [0, 1]).all() and len(np.unique(y)) == 2
           and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
           and np.isfinite(z.to_numpy()).all() and 0 <= tau <= 1
           and np.array_equal(pred['predicao_tau_0_50'].to_numpy(dtype=np.uint8), (p >= .5).astype(np.uint8))
           and np.array_equal(pred['predicao_tau_mcc_oof'].to_numpy(dtype=np.uint8), (p >= tau).astype(np.uint8)),
           'Rotulos, ordem, probabilidades ou limiares F10 inconsistentes.')
    recalculadas = avaliar_scores_com_limiar_oof_congelado(y, p, limiar_mcc_oof=tau)
    for chave in ('ap', 'mcc_limiar_oof', 'logloss', 'score_composto_limiar_oof'):
        exigir(np.isclose(recalculadas[chave], met[chave], atol=1e-9, rtol=1e-9),
               'Metrica divergente do resultado congelado: ' + chave)
    exigir(met.get('limiar_otimizado_no_f10') is False
           and met.get('id_candidato_oof') == marcador['id_candidato_oof']
           and np.isclose(float(met.get('fracao_f10_dentro_do_fold', -1)), fracao_amostra_f10(config)),
           'F10 parece ter sido otimizado ou esta fora da fracao configurada.')
    pacote = carregar_pacote_confiavel(teste / 'modelo_final_f1_f9')
    exigir(pacote.configuracao == config
           and pacote.escolha_oof['id_candidato'] == marcador['id_candidato_oof']
           and np.isclose(pacote.limiar_oof, tau)
           and int(pacote.tsne.parametros['n_components']) == 3
           and list(pacote.gmm.colunas_) == ['TSNE1', 'TSNE2', 'TSNE3']
           and pacote.gmm.modelo_.covariance_type == 'full',
           'Pacote t-SNE/GMM nao e o vencedor 3D congelado.')
    gamma = pacote.gmm.responsabilidades(z)
    from src.probabilidade import calcular_probabilidades
    scores_conferidos = np.asarray(calcular_probabilidades(gamma, pacote.risco.riscos_), dtype=np.float64)
    exigir(isinstance(gamma, pd.DataFrame) and gamma.shape == (len(y), 4)
           and np.isfinite(gamma.to_numpy(dtype=np.float64)).all()
           and scores_conferidos.shape == p.shape
           and np.max(np.abs(scores_conferidos - p)) <= 1e-6
           and np.array_equal(scores_conferidos >= tau, p >= tau)
           and np.array_equal(scores_conferidos >= .5, p >= .5),
           'Scores do GMM FULL sobre o embedding nao coincidem com F10 salvo.')
    return teste, config, marcador, met, indices, y, p, z, tau, pacote, gamma


def _pre01_html(pasta, teste, cfg, indices, y, z, pacote):
    """Relatorio descritivo offline. Nao ajusta scaler nem executa t-SNE."""
    from plotly.offline import get_plotlyjs
    from src.visualizacao_final_completa_tsne1d import _resumo_pre01

    variaveis, n_padronizadas = _resumo_pre01(pasta, pacote, indices, y, cfg)
    exigir(len(variaveis) > 0 and n_padronizadas > 0, 'PRE01 F10 vazio.')
    numeros = ['media_transformada_f10', 'desvio_transformado_f10']
    dados = variaveis[['feature', 'tipo', *numeros,
                       'media_treino_f1_f9_codificada', 'escala_treino_f1_f9']].copy()
    dados = dados.astype(object).where(pd.notna(dados), None)
    def limpar(d):
        if isinstance(d, dict): return {str(k): limpar(v) for k, v in d.items()}
        if isinstance(d, list): return [limpar(x) for x in d]
        if isinstance(d, (np.floating, np.integer)): return limpar(d.item())
        if isinstance(d, float) and not np.isfinite(d): return None
        return d
    payload = limpar({
        'variaveis': dados.to_dict(orient='records'),
        'z1': z['TSNE1'].to_numpy(dtype=float).tolist(),
        'z2': z['TSNE2'].to_numpy(dtype=float).tolist(),
        'z3': z['TSNE3'].to_numpy(dtype=float).tolist(),
        'classes': y.tolist(),
    })
    json_data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    plotly_script = get_plotlyjs().replace('</', '<\\/')
    html = '''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<title>PRE01 F10 - t-SNE 3D</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{background:#f3f6fa;color:#203047;font:14px system-ui;margin:0}
main{max-width:1500px;margin:auto;padding:23px}h1{margin-bottom:6px}.note{color:#526982}
section{background:white;border:1px solid #dbe3ed;border-radius:10px;padding:17px;margin:15px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px}
#media,#desvio,#mapa{height:480px}input{padding:10px;width:min(430px,90%)}
.scroll{overflow:auto;max-height:530px}table{width:100%;border-collapse:collapse;font-size:12px}
th,td{border-bottom:1px solid #e2e8f0;padding:8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{position:sticky;top:0;background:#e8eef6}</style></head>
<body><main><h1>PRE01 - Diagnostico completo da padronizacao (F10 exploratorio)</h1>
<p class="note">t-SNE 3D + GMM FULL | F1-F9 congelado | F10 apenas descritivo. Nenhum fit ou ajuste de limiar em F10.</p>
<div class="grid"><section><h2>Media transformada F10</h2><div id="media"></div></section>
<section><h2>Desvio transformado F10</h2><div id="desvio"></div></section></div>
<section><h2>Coordenadas t-SNE 3D fora da amostra</h2><div id="mapa"></div>
<p class="note">A posicao e apenas descritiva; distancias globais do t-SNE nao devem ser superinterpretadas.</p></section>
<section><h2>Parametros congelados e estatisticas de F10</h2><input id="busca" placeholder="Filtrar variavel ou tipo..." type="search">
<div class="scroll"><table><thead><tr><th>Variavel</th><th>Tipo</th><th>Media F10 transformada</th><th>DP F10 transformado</th>
<th>Media treino</th><th>Escala treino</th></tr></thead><tbody id="linhas"></tbody></table></div></section></main>
<script>''' + plotly_script + '''</script><script>
const D=''' + json_data + ''';
const P=D.variaveis.filter(r=>r.tipo==='padronizada');
const layout={margin:{l:175,r:20,t:25,b:65},paper_bgcolor:'#fff',plot_bgcolor:'#fff'};
const cfg={responsive:true,displaylogo:false};
Plotly.newPlot('media',[{type:'bar',orientation:'h',y:P.map(x=>x.feature),x:P.map(x=>x.media_transformada_f10)}],
{...layout,yaxis:{automargin:true,autorange:'reversed'},xaxis:{title:'Media z em F10'}},cfg);
Plotly.newPlot('desvio',[{type:'bar',orientation:'h',y:P.map(x=>x.feature),x:P.map(x=>x.desvio_transformado_f10)}],
{...layout,yaxis:{automargin:true,autorange:'reversed'},xaxis:{title:'Desvio z em F10'}},cfg);
const C=[0,1].map(v=>({type:'scatter3d',mode:'markers',name:v?'Fraudes':'Legitimas',
x:D.z1.filter((_,i)=>D.classes[i]===v),y:D.z2.filter((_,i)=>D.classes[i]===v),z:D.z3.filter((_,i)=>D.classes[i]===v),
marker:{size:v?6:3,opacity:v?.8:.4,color:v?'#d62728':'#1f77b4'}}));
Plotly.newPlot('mapa',C,{...layout,margin:{l:70,r:20,t:20,b:75},scene:{xaxis:{title:'TSNE1'},yaxis:{title:'TSNE2'},zaxis:{title:'TSNE3'},aspectmode:'data'}},cfg);
const fmt=x=>x===null||x===undefined?'—':(typeof x==='number'?x.toFixed(7):String(x));
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function desenhar(){const q=document.getElementById('busca').value.toLowerCase();
const campos=['feature','tipo','media_transformada_f10','desvio_transformado_f10','media_treino_f1_f9_codificada','escala_treino_f1_f9'];
document.getElementById('linhas').innerHTML=D.variaveis.filter(r=>(r.feature+' '+r.tipo).toLowerCase().includes(q))
.map(r=>'<tr>'+campos.map(k=>'<td>'+esc(fmt(r[k]))+'</td>').join('')+'</tr>').join('');}
document.getElementById('busca').addEventListener('input',desenhar);desenhar();
</script></body></html>'''
    exigir('Plotly.newPlot' in html and 'PRE01' in html and len(html) > 100000,
           'PRE01 t-SNE 3D nao parece um HTML offline valido.')
    return html, variaveis




def _validar_visuais_existentes(dest, marco):
    man = ler_json(dest / 'manifesto_visualizacao_f10_tsne3d.json')
    exigir(man.get('estado') == 'VISUAIS_F10_TSNE3D_COMPLETOS'
           and man.get('sha256_modelo') == marco['sha256_modelo']
           and man.get('sha256_predicoes') == marco['sha256_predicoes']
           and man.get('sha256_coordenadas') == marco['sha256_coordenadas']
           and man.get('sha256_metricas') == marco['sha256_metricas']
           and set(man.get('pngs', [])) == set(PNG_ESPERADOS)
           and set(man.get('htmls', [])) == set(HTML_INTERATIVOS)
           and PRE01 in man.get('sha256_arquivos', {}),
           'Visuais anteriores nao correspondem ao t-SNE 3D congelado.')
    for nome, digest in man['sha256_arquivos'].items():
        exigir((dest / nome).is_file() and hash_arquivo(dest / nome) == digest,
               'Visualizacao t-SNE 3D ausente/alterada: ' + nome)
    exigir({q.name for q in dest.glob('*.png')} == set(PNG_ESPERADOS),
           'Conjunto PNG t-SNE 3D incompleto.')
    exigir({q.name for q in dest.glob('*.html')} == set(HTML_INTERATIVOS),
           'Conjunto HTML 3D incompleto.')
    return dest


def _oof_congelado(pasta, marco):
    caminho = pasta / 'vencedor' / 'oof' / 'predicoes.npz'
    exigir(caminho.is_file(), 'Predicoes OOF de F1-F9 ausentes.')
    with np.load(caminho, allow_pickle=False) as arr:
        ids = np.asarray(arr['linha_original'], dtype=np.int64)
        fold = np.asarray(arr['fold'], dtype=np.uint8)
        y = np.asarray(arr['y_true'], dtype=np.uint8)
        scores = np.asarray(arr['prob_fraude'], dtype=np.float64)
        id_candidato = str(arr['id_candidato'].item())
    exigir(len(ids) > 0 and len(ids) == len(fold) == len(y) == len(scores)
           and np.all(np.diff(ids) > 0) and set(np.unique(fold)) == set(range(1, 10))
           and np.isin(y, [0,1]).all() and len(np.unique(y)) == 2
           and np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))
           and id_candidato == marco['id_candidato_oof'],
           'OOF congelado nao corresponde ao candidato vencedor/F1-F9.')
    return y, scores


def gerar_visuais(pasta_experimento):
    """Somente desenhar a partir do modelo e F10 congelados; nao treinar."""
    pasta = Path(pasta_experimento).resolve()
    teste, cfg, marco, met, ids, y, p, z, tau, pacote, gamma = _fontes(pasta)
    dest = teste / 'visualizacao_f10_3d'
    if dest.exists():
        return _validar_visuais_existentes(dest, marco)
    stage = teste / '.visualizacao_f10_3d_incompleta'
    exigir(not stage.exists(), 'Visualizacao 3D parcial anterior; preservar para auditoria.')
    trava = teste / '.visualizacao_f10_3d.lock'
    try:
        with trava.open('x', encoding='utf-8') as f:
            f.write('Renderizacao somente; sem acesso novo a F10 nem ajuste.\n')
    except FileExistsError as exc:
        raise RuntimeError('Visualizacao F10 t-SNE 3D em andamento.') from exc
    try:
        stage.mkdir(exist_ok=False)
        from src import visualizacao_oof as vo
        from src.visualizacao_final_completa_tsne1d import desenhar_tsne14
        from src.visualizacao_tsne3d_interativa import gerar_tsne13_3d
        from src.visualizacao_geometria_tsne3d import gerar_geometria_tsne3d
        dpi = int(cfg['visualizacao']['padrao'].get('dpi', 180))
        metserie = pd.Series({'ap': met['ap'], 'mcc': met['mcc_limiar_oof'],
                              'logloss': met['logloss'], 'score': met['score_composto_limiar_oof'],
                              'prevalencia': met['prevalencia']})
        escolha = {'limiar_mcc_oof': tau}
        with rotulos_preditivos():
            vo._plot_painel_00(cfg, y, p, metserie, escolha, stage, 'png', dpi, True)
            vo._plot_pr(cfg, y, p, metserie, escolha, stage, 'png', dpi, True)
            vo._plot_mcc(cfg, y, p, metserie, escolha, stage, 'png', dpi, True)
            vo._matriz(cfg, y, p, .5, 'matriz_confusao_05', stage, 'png', dpi, True)
            vo._matriz(cfg, y, p, tau, 'matriz_confusao_mcc', stage, 'png', dpi, True)
            vo._distribuicao(cfg, y, p, escolha, stage, 'png', dpi, True)
        gerar_tsne13_3d(z.assign(linha_original=ids), y, stage,
                        scores=p, limiar_oof=tau,
                        responsabilidades=gamma.to_numpy(dtype=np.float64),
                        origem='f10', fracao_f10=fracao_amostra_f10(cfg),
                        max_pontos=int(cfg['visualizacao']['geometria'].get('pontos_maximos_amostra', 20000)),
                        dpi=dpi, semente=int(cfg['experimento']['semente']))
        y_oof, p_oof = _oof_congelado(pasta, marco)
        gerar_geometria_tsne3d(
            z, y, pacote.gmm.modelo_, pacote.risco.riscos_.to_numpy(dtype=np.float64),
            stage, limiar_oof=tau, y_oof=y_oof, scores_oof=p_oof,
            origem='f10', fracao_f10=fracao_amostra_f10(cfg),
            dpi=min(dpi, 180), resolucao_cortes=60, resolucao_volume=21,
            salvar_csv=True,
        )
        desenhar_tsne14(stage, pacote, len(y))
        nomes = {q.name for q in stage.glob('*.png')}
        exigir(nomes == set(PNG_ESPERADOS),
               'F10 t-SNE 3D com PNGs faltantes ou excedentes: ' + str(nomes ^ set(PNG_ESPERADOS)))
        exigir({q.name for q in stage.glob('*.html')} == set(HTML_INTERATIVOS),
               't-SNE 3D: HTMLs interativos nao correspondem ao catalogo.')
        pre_html, tabela = _pre01_html(pasta, teste, cfg, ids, y, z, pacote)
        (stage / 'PRE01_F10').mkdir()
        (stage / PRE01).write_text(pre_html, encoding='utf-8')
        tabela.to_csv(stage / 'PRE01_F10/PRE01_F10_todas_variaveis.csv',
                      index=False, float_format='%.17g')
        arquivos = {}
        for arquivo in stage.rglob('*'):
            if arquivo.is_file():
                rel = arquivo.relative_to(stage).as_posix()
                exigir(arquivo.stat().st_size > 0, 'Artefato visual vazio: ' + rel)
                if rel.endswith('.png'):
                    exigir(arquivo.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n', 'PNG invalido: ' + rel)
                arquivos[rel] = hash_arquivo(arquivo)
        _criar_json_novo(stage / 'manifesto_visualizacao_f10_tsne3d.json', {
            'estado': 'VISUAIS_F10_TSNE3D_COMPLETOS',
            'dimensao': 3, 'n_f10': len(y), 'n_fraudes': int(y.sum()),
            'fracao_f10_dentro_do_fold': fracao_amostra_f10(cfg),
            'sha256_modelo': marco['sha256_modelo'],
            'sha256_predicoes': marco['sha256_predicoes'],
            'sha256_coordenadas': marco['sha256_coordenadas'],
            'sha256_metricas': marco['sha256_metricas'],
            'limiar_f10_reotimizado': False, 'sem_fit_f10': True,
            'pngs': list(PNG_ESPERADOS), 'htmls': list(HTML_INTERATIVOS),
            'sha256_arquivos': arquivos,
        })
        exigir(not dest.exists(), 'Outro processo publicou visualizacao F10 3D.')
        os.replace(stage, dest)
        return _validar_visuais_existentes(dest, marco)
    finally:
        trava.unlink(missing_ok=True)


def _validar_galeria_existente(dest, marco, visuais):
    man = ler_json(dest / 'manifesto_galeria_f10.json')
    esperado = ({'index.html', PRE01}
                | {'figuras/' + n for n in PNG_ESPERADOS}
                | {'interativos/' + n for n in HTML_INTERATIVOS})
    exigir(man.get('estado') == 'GALERIA_F10_TSNE3D_COMPLETA'
           and man.get('sha256_modelo') == marco['sha256_modelo']
           and man.get('sha256_predicoes') == marco['sha256_predicoes']
           and man.get('sha256_coordenadas') == marco['sha256_coordenadas']
           and set(man.get('sha256_arquivos', {})) == esperado,
           'Galeria t-SNE 3D anterior tem fontes divergentes.')
    for nome, digest in man['sha256_arquivos'].items():
        exigir((dest / nome).is_file() and hash_arquivo(dest / nome) == digest,
               'Galeria t-SNE 3D alterada: ' + nome)
    exigir(hash_arquivo(dest / PRE01) == hash_arquivo(visuais / PRE01),
           'PRE01 F10 diverge dos visuais originais.')
    for nome in PNG_ESPERADOS:
        exigir(hash_arquivo(dest / 'figuras' / nome) == hash_arquivo(visuais / nome),
               'PNG da galeria difere da fonte: ' + nome)
    for nome in HTML_INTERATIVOS:
        exigir(hash_arquivo(dest / 'interativos' / nome) == hash_arquivo(visuais / nome),
               'HTML da galeria difere da fonte: ' + nome)
    return dest


def gerar_galeria(pasta_experimento):
    pasta = Path(pasta_experimento).resolve()
    teste, cfg, marco, met, ids, y, p, z, tau, pacote, gamma = _fontes(pasta)
    vis = gerar_visuais(pasta)
    dest = teste / 'galeria'
    if dest.exists():
        return _validar_galeria_existente(dest, marco, vis)
    stage = teste / '.galeria_tsne3d_incompleta'
    exigir(not stage.exists(), 'Galeria parcial t-SNE 3D: preservar para auditoria.')
    lock = teste / '.galeria_tsne3d.lock'
    try:
        with lock.open('x', encoding='utf-8') as f:
            f.write('Montagem de galeria F10 t-SNE 3D.\n')
    except FileExistsError as exc:
        raise RuntimeError('Galeria t-SNE 3D em andamento.') from exc
    try:
        stage.mkdir(exist_ok=False)
        (stage / 'figuras').mkdir()
        (stage / 'interativos').mkdir()
        (stage / 'PRE01_F10').mkdir()
        arquivos = {}
        from html import escape as escapar
        titulos = {
            '00_painel_score_composto.png': 'Painel inicial',
            '01_curva_precision_recall.png': 'Precision-Recall',
            '02_mcc_por_limiar.png': 'MCC por limiar',
            '03_matriz_confusao_05.png': 'Matriz de confusao tau=0,50',
            '04_matriz_confusao_mcc.png': 'Matriz de confusao tau* OOF',
            '05_distribuicao_probabilidades.png': 'Distribuicao de probabilidades',
            MAPA_PNG: 'TSNE13 - Projecao tridimensional',
            '14_tsne_diagnostico.png': 'TSNE14 - Diagnostico KL',
        }
        cards = []
        for nome in PNG_ESPERADOS:
            origem, alvo = vis / nome, stage / 'figuras' / nome
            shutil.copyfile(origem, alvo)
            exigir(hash_arquivo(origem) == hash_arquivo(alvo), 'PNG copiado diverge: ' + nome)
            arquivos['figuras/' + nome] = hash_arquivo(alvo)
            titulo = escapar(titulos.get(nome, nome.replace('_',' ').removesuffix('.png')))
            cards.append('<a class="card" href="figuras/' + escapar(nome, quote=True) + '">'
                         '<img loading="lazy" src="figuras/' + escapar(nome, quote=True) + '" alt="' + titulo + '">'
                         '<span>' + titulo + '</span></a>')
        links = []
        for nome in HTML_INTERATIVOS:
            origem, alvo = vis / nome, stage / 'interativos' / nome
            shutil.copyfile(origem, alvo)
            exigir(hash_arquivo(origem) == hash_arquivo(alvo), 'HTML copiado diverge: ' + nome)
            arquivos['interativos/' + nome] = hash_arquivo(alvo)
            links.append('<a class="link" href="interativos/' + escapar(nome, quote=True) + '">' +
                         escapar(nome.replace('_',' ').removesuffix('.html')) + '</a>')
        shutil.copyfile(vis / PRE01, stage / PRE01)
        arquivos[PRE01] = hash_arquivo(stage / PRE01)
        html = ('<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>t-SNE 3D + GMM - F10 exploratorio</title>'
                '<style>body{font:15px system-ui;background:#f4f7fb;color:#1b2b45;max-width:1450px;margin:auto;padding:22px}'
                '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}'
                '.card{display:block;background:#fff;padding:12px;border:1px solid #ccd6e1;border-radius:10px;color:#134e87;text-decoration:none}'
                'img{width:100%;height:auto}span{display:block;padding:6px}.link{display:inline-block;margin:9px;padding:12px;background:white;border:1px solid #bccbdc;border-radius:8px}'
                '</style></head><body><h1>t-SNE 3D + GMM FULL — F10 exploratorio</h1>'
                '<p>Modelo, pre-processamento e embedding congelados com F1-F9. F10 ja acessado; nenhum ajuste ou ranking no F10.</p>'
                f'<p>Desenvolvimento {float(cfg["dados"]["fracao_base"]):.2%}; F10 {fracao_amostra_f10(cfg):.2%} do fold; '
                f'n={len(ids)}; fraudes={int(y.sum())}; tau* OOF={tau:.7f}; '
                f'AP={met["ap"]:.7f}; MCC={met["mcc_limiar_oof"]:.7f}; LogLoss={met["logloss"]:.7f}</p>'
                '<p><a href="../../visualizacao/galeria/index.html">Galeria OOF</a> | '
                f'<a href="{PRE01}">Diagnostico completo da padronizacao (HTML interativo)</a></p>'
                '<h2>Visualizacoes tridimensionais interativas — offline</h2><div>' + ''.join(links) + '</div>'
                '<h2>Catalogo de imagens</h2><div class="grid">' + ''.join(cards) + '</div></body></html>')
        (stage / 'index.html').write_text(html, encoding='utf-8')
        arquivos['index.html'] = hash_arquivo(stage / 'index.html')
        _criar_json_novo(stage / 'manifesto_galeria_f10.json', {
            'estado': 'GALERIA_F10_TSNE3D_COMPLETA',
            'sha256_modelo': marco['sha256_modelo'],
            'sha256_predicoes': marco['sha256_predicoes'],
            'sha256_coordenadas': marco['sha256_coordenadas'],
            'fracao_f10_dentro_do_fold': fracao_amostra_f10(cfg),
            'sem_fit_no_f10': True, 'limiar_f10_otimizado': False,
            'sha256_arquivos': arquivos,
        })
        exigir(not dest.exists(), 'Outra galeria F10 3D surgiu durante a geracao.')
        os.replace(stage, dest)
        return _validar_galeria_existente(dest, marco, vis)
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experimento', type=Path, required=True)
    args = parser.parse_args(argv)
    resultado = gerar_galeria(args.experimento)
    print('[TSNE 3D] Galeria F10 aprovada:', resultado, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
