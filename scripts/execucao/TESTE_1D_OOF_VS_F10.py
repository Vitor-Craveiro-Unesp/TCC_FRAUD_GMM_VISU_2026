# -*- coding: utf-8 -*-
"""Novo ensaio OOF PCA 1D e painel comparativo com F10 JA AVALIADO.

    python scripts/execucao/TESTE_1D_OOF_VS_F10.py --verificar
    python scripts/execucao/TESTE_1D_OOF_VS_F10.py --novo-oof
    python scripts/execucao/TESTE_1D_OOF_VS_F10.py --comparar-existente

NUNCA executa avaliacao F10. Compara SOMENTE os PNGs existentes da Etapa 9.
--novo-oof treina os nove folds F1-F9 de PCA 1D em 10% do desenvolvimento,
com YAML temporario restaurado byte a byte no fim. Uma unica pasta nova.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
OUT = ROOT / 'outputs'
OOF_ANTIGO = OUT / 'ensaio_pca_gmm_20261029_173123'
F10_FIGURAS = OUT / 'galeria_final_f10_pca_10pct_20261029T233621357569Z'
F10_RESULTADOS = OUT / 'avaliacao_final_f10_pca_10pct_20261029T232726363532Z'
F10_NOMES = {
    'painel': '01_painel_final.png',
    'pr': '02_curva_precision_recall.png',
    'scores': '03_distribuicao_scores_2_limiares.png',
    'cm': '04_matrizes_confusao_050_vs_tau_oof.png',
}
NOMES = {
    'painel': 'Painel de métricas',
    'pr': 'Curva Precision–Recall',
    'scores': 'Distribuição dos scores',
    'cm': 'Matrizes de confusão (τ=0,50 × τ* OOF)',
}


def exigir(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def hash_file(p):
    h = sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def ler_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def ler_yaml():
    exigir(CFG.is_file(), f'YAML nao encontrado: {CFG}')
    return yaml.safe_load(CFG.read_text(encoding='utf-8-sig'))


def verificar_config(cfg):
    def unico(x):
        return x[0] if isinstance(x, list) and len(x) == 1 else x
    checks = {
        'dados.fracao_base = 0.1': float(cfg['dados']['fracao_base']) == .1,
        'reducao.metodo = PCA': cfg['reducao']['metodo'].upper() == 'PCA',
        'gmm.n_components = [4]': int(unico(cfg['gmm']['n_components'])) == 4,
        'gmm.n_init = 5': int(cfg['gmm']['n_init']) == 5,
        'gmm.covariance_type = full': cfg['gmm']['covariance_type'].lower() == 'full',
        'amostragem.modo = RUS': cfg['amostragem']['modo'].upper() == 'RUS',
        'amostragem.razao_negativos_positivos = 16': int(cfg['amostragem']['razao_negativos_positivos']) == 16,
        'experimento.semente = 42': int(cfg['experimento']['semente']) == 42,
        'divisao.qual_fold_teste = 10': int(cfg['divisao']['qual_fold_teste']) == 10,
        'execucao.proteger_resultados = true': cfg['execucao']['proteger_resultados'] is True,
    }
    for rotulo, ok in checks.items():
        exigir(ok, f'Configuracao inesperada: {rotulo}')
    exigir((ROOT/'src'/'orquestracao_oof.py').is_file(), 'Orquestrador nao encontrado')
    exigir((ROOT / cfg['dados']['arquivo']).is_file(), 'Parquet original nao encontrado')
    exigir(OOF_ANTIGO.joinpath('vencedor','metricas','melhor_configuracao.json').is_file(),
           'OOF PCA 1D original nao encontrado')
    exigir(F10_RESULTADOS.joinpath('PCA_1D','metricas_f10.json').is_file(),
           'Metricas F10 existentes ausentes; NUNCA rerode o F10 para esse teste')
    for chave, nome in F10_NOMES.items():
        exigir((F10_FIGURAS/'PCA_1D'/nome).is_file(), f'Figura F10 existente ausente: {chave}/{nome}')
    exigir((F10_FIGURAS/'index.html').is_file(), 'Galeria F10 original ausente')
    exigir((OOF_ANTIGO/'vencedor'/'visualizacao'/'galeria'/'index.html').is_file(),
           'Galeria OOF 1D original ausente')
    print('[OK] PCA 1D K4 n_init5 FULL RUS16, fracao=10%, seed=42')
    print('[OK] Galerias OOF original e F10 ja salvo encontradas')
    print('[PROTECAO] Nenhuma leitura do Parquet, sem inferencia F10, sem recalibracao')


def conferir_oof(folder: Path, expected_n: int):
    m = ler_json(folder/'manifesto.json')
    escolha = ler_json(folder/'vencedor'/'metricas'/'melhor_configuracao.json')
    cfg = yaml.safe_load((folder/'configuracao.yaml').read_text(encoding='utf-8-sig'))
    exigir(m.get('status') == 'concluido_06_03', 'OOF novo nao foi concluido')
    exigir(m.get('tipo') == 'ensaio' and int(m.get('registros_desenvolvimento', -1)) == expected_n,
           f'OOF novo nao tem {expected_n} registros')
    exigir(cfg['pca']['n_components'] == [1], 'OOF novo nao e PCA 1D')
    exigir(cfg['gmm']['n_components'] == [4] and cfg['gmm']['n_init'] == 5,
           'OOF novo nao usa K4, n_init=5')
    exigir(float(cfg['dados']['fracao_base']) == .1, 'OOF novo nao e 10%')
    exigir(escolha.get('status') == 'selecionado_em_ensaio', 'Selecao OOF inesperada')
    exigir(escolha.get('teste_final_usado') is False, 'OOF novo diz ter utilizado F10!')
    origem = ler_json(OOF_ANTIGO/'manifesto.json')
    for campo in ('sha256_parquet', 'sha256_folds'):
        exigir(origem[campo] == m[campo], f'OOF novo nao usa a mesma origem: {campo}')
    return m, escolha


def identificar_imagens_oof(folder: Path):
    base = folder/'vencedor'/'visualizacao'
    pngs = sorted(base.rglob('*.png'))
    # Nomes canonicos recebem prioridade. Fallbacks sao heuristicas, e rotulados como tal.
    exact = {
        'painel': ['00_painel_score_composto.png', '00_painel_inicial.png'],
        'pr': ['01_curva_precision_recall.png'],
        'scores': ['03_distribuicao_scores.png','02_distribuicao_scores.png'],
        'cm': ['04_matrizes_confusao_050_vs_tau_oof.png',
               '04_matriz_confusao.png', 'matriz_confusao_comparativa.png'],
    }
    palavras = {
        'painel': [('painel', 'score'), ('painel', 'inicial')],
        'pr': [('precision', 'recall'), ('curva', 'pr')],
        'scores': [('distribuicao', 'score'), ('histograma', 'score')],
        'cm': [('matriz', 'confus'), ('matrizes', 'confus')],
    }
    escolhidos, modo = {}, {}
    for chave in NOMES:
        candidatos = [p for p in pngs if p.name.lower() in exact[chave]]
        if candidatos:
            candidatos.sort(key=lambda p: (exact[chave].index(p.name.lower()),len(p.parts)))
            escolhidos[chave] = candidatos[0]
            modo[chave] = 'arquivo canônico'
            continue
        ranked = []
        for p in pngs:
            s = p.name.lower()
            hits = sum(1 for pares in palavras[chave] if all(w in s for w in pares))
            if hits:
                # Preferir imagem no conjunto OOF e nome curto; nao a de geometria.
                prioridade = 0 if 'oof' in (part.lower() for part in p.parts) else 1
                ranked.append((-hits, prioridade, len(s), p))
        if ranked:
            ranked.sort(key=lambda x: x[:3])
            escolhidos[chave] = ranked[0][3]
            modo[chave] = 'aproximação por nome; conferir o conteúdo'
        else:
            escolhidos[chave] = None
            modo[chave] = 'sem equivalente localizado automaticamente'
    return pngs, escolhidos, modo


def href_de(p: Path, base: Path):
    return Path(os.path.relpath(p, base)).as_posix()


def link(p: Path, base: Path, legenda: str):
    return f'<a href="{escape(href_de(p,base),quote=True)}" target="_blank" rel="noopener">{escape(legenda)}</a>'


def html_comparacao(destino: Path, pasta_oof: Path, modo_novo: bool):
    galeria_oof = pasta_oof/'vencedor'/'visualizacao'/'galeria'/'index.html'
    exigir(galeria_oof.is_file(), f'Galeria OOF ausente: {galeria_oof}')
    PNGs, pares, correspondencias = identificar_imagens_oof(pasta_oof)
    f10_metrics = ler_json(F10_RESULTADOS/'PCA_1D'/'metricas_f10.json')
    escolha = ler_json(pasta_oof/'vencedor'/'metricas'/'melhor_configuracao.json')
    tau_oof = float(escolha['limiar_mcc_oof'])
    tau_f10 = float(f10_metrics['limiar_mcc_oof_congelado'])
    original_tau = float(ler_json(OOF_ANTIGO/'vencedor'/'metricas'/'melhor_configuracao.json')['limiar_mcc_oof'])
    # Nao comparar pixels nem afirmar identica forma grafica: amostras e pipeline diferem.
    blocos = []
    for chave, rotulo in NOMES.items():
        p_oof = pares[chave]
        p_f10 = F10_FIGURAS/'PCA_1D'/F10_NOMES[chave]
        def quadro(p, label):
            if p is None:
                return '<div class="sem">Não localizado automaticamente no catálogo OOF; confira a galeria completa.</div>'
            rel = escape(href_de(p,destino), quote=True)
            return f'<a class="fig" href="{rel}" target="_blank" rel="noopener"><img loading="lazy" src="{rel}" alt="{escape(label,quote=True)}"><span>Abrir em tamanho original</span></a>'
        blocos.append(f'''<section class="pair"><h2>{escape(rotulo)}</h2><p class="hint">Correspondência: {escape(correspondencias[chave])}. Mesma finalidade estatística não garante mesma aparência, elementos ou valores.</p>
          <div class="twocol"><article><h3>Campeão OOF — F1–F9</h3>{quadro(p_oof,'OOF '+rotulo)}</article>
          <article><h3>Teste final — F10 10% já salvo</h3>{quadro(p_f10,'F10 '+rotulo)}</article></div></section>''')
    usados = {p.resolve() for p in pares.values() if p is not None}
    extras = [p for p in PNGs if p.resolve() not in usados]
    # Evita duplicatas na lista de todas as figuras, que podem estar em subpastas.
    lista_html = ''.join(f'<li>{link(p,destino,href_de(p,pasta_oof))}</li>' for p in extras)
    versao = 'OOF NOVO TREINADO DO ZERO' if modo_novo else 'OOF JÁ EXISTENTE — COMPARAÇÃO SEM TREINAR'
    pagina = f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GMM PCA 1D — Campeão OOF x F10</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;color:#243449;font:15px/1.5 system-ui,Segoe UI,sans-serif}}
header{{padding:30px max(24px,3vw);background:#162b47;color:white}}h1{{font-size:27px;margin:0 0 8px}}header p{{max-width:1100px}}
main{{max-width:1700px;margin:25px auto;padding:0 20px}}.notice{{background:white;border-left:4px solid #346fab;padding:16px 20px;border-radius:6px;margin-bottom:18px}}
.pair,.all{{background:white;padding:17px 21px;margin:23px 0;border:1px solid #d5dee9;border-radius:10px}}
h2{{margin:0 0 6px;font-size:20px}}.hint{{color:#64738b;margin:0 0 13px;font-size:13px}}.twocol{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
article{{border:1px solid #d7e2eb;border-radius:8px;padding:12px;min-width:0}}h3{{font-size:16px;margin:0 0 12px}}
a.fig{{display:block;color:#17539a;text-decoration:none}}a.fig img{{width:100%;height:auto;object-fit:contain;border:1px solid #eef1f6;background:white}}a.fig span{{display:block;text-align:center;padding:8px}}
a{{color:#17539a}}iframe{{width:100%;height:650px;border:1px solid #bfcbd8;border-radius:8px}}.sem{{padding:40px;background:#fafafa;color:#6b7280}}
.meta{{font-variant-numeric:tabular-nums}}details li{{margin:4px 0;word-break:break-word}}@media(max-width:860px){{.twocol{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Teste PCA 1D — campeão OOF × teste final F10</h1><p>{versao} · GMM K=4 · n_init=5 · covariância FULL · RUS 16:1 · 10% · semente 42</p></header>
<main><div class="notice"><strong>O que significa comparar os gráficos?</strong> O OOF é calculado nos folds F1–F9; o F10 é o holdout já avaliado. Não são os mesmos registros nem os mesmos gráficos em quantidade. Os quatro pares abaixo são comparações por <em>finalidade</em>, não equivalência exata de layout. Nenhum resultado F10 foi recalculado.</div>
<div class="notice meta"><strong>Limiares:</strong> τ* no OOF mostrado = {tau_oof:.17g}; τ* congelado no F10 = {tau_f10:.17g}; τ* do OOF original de referência = {original_tau:.17g}.<br><strong>Atenção:</strong> mesmo que o novo OOF produza um τ* diferente, o F10 permanece com o limiar original congelado. Isso não autoriza uma segunda avaliação.</div>
<div class="notice"><strong>Catálogos completos:</strong> {link(galeria_oof,destino,'Abrir galeria OOF PCA 1D')} &nbsp; | &nbsp; {link(F10_FIGURAS/'index.html',destino,'Abrir galeria F10 original (1D, 2D e 3D)')}</div>
{''.join(blocos)}
<section class="all"><h2>Demais gráficos exclusivos ou adicionais do OOF ({len(extras)} arquivos PNG)</h2><p>O catálogo OOF inclui diagnósticos de treinamento e geometria que não foram gerados na galeria F10 de quatro gráficos.</p><details><summary>Mostrar inventário dos arquivos OOF adicionais</summary><ul>{lista_html}</ul></details></section>
<section class="all"><h2>Galeria OOF completa embutida</h2><iframe src="{escape(href_de(galeria_oof,destino),quote=True)}" title="Galeria OOF 1D completa" loading="lazy"></iframe></section>
</main></body></html>'''
    destino.mkdir(parents=True,exist_ok=False)
    (destino/'COMPARAR_OOF_VS_F10_PCA1D.html').write_text(pagina,encoding='utf-8')
    resumo = {
        'objetivo': 'COMPARACAO_VISUAL_OOF_VS_F10_SEM_REAVALIAR_F10',
        'novo_oof_treinado': modo_novo,
        'origem_oof': str(pasta_oof.relative_to(ROOT)),
        'origem_f10': str(F10_FIGURAS.relative_to(ROOT)),
        'modelo_final_f10_reajustado': False,
        'f10_reexecutado': False,
        'tau_oof_exibido': tau_oof,
        'tau_f10_congelado': tau_f10,
        'pca_1d_figuras_oof_encontradas': len(PNGs),
        'correspondencias': {ch: {'oof': str(p.relative_to(ROOT)) if p else None,
                                  'f10': str((F10_FIGURAS/'PCA_1D'/F10_NOMES[ch]).relative_to(ROOT)),
                                  'criterio': correspondencias[ch]} for ch,p in pares.items()},
    }
    (destino/'manifesto_comparacao.json').write_text(json.dumps(resumo,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return destino/'COMPARAR_OOF_VS_F10_PCA1D.html', resumo


def novo_oof():
    import copy
    original = CFG.read_bytes()
    cfg = yaml.safe_load(original.decode('utf-8-sig'))
    cfg = copy.deepcopy(cfg)
    cfg['pca']['n_components'] = [1]
    cfg['execucao']['diretorio_retomada'] = None
    cfg['execucao']['retomar'] = False
    # Sem alteracao de hiperparametros: novo ajuste OOF, nao refinamento com F10.
    verificar_config(cfg)
    # Bloqueio exclusivo: nao sobrescreve lock ou resultados anteriores.
    trava = OUT/'.TESTE_1D_OOF_VS_F10.lock'
    try:
        fd = os.open(str(trava),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f'Outro teste pode estar rodando (lock presente): {trava}; verifique processos') from exc
    with os.fdopen(fd,'w',encoding='utf-8') as f:
        f.write(f'PID={os.getpid()}\nINICIO={datetime.now().isoformat()}\n')
    # Proteger caso existam outras execucoes concorrentes: lock desta ferramenta
    # nao substitui exclusao mutua entre outros runners da mesma raiz.
    controle = OUT/('controle_novo_teste_pca1d_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    try:
        controle.mkdir(exist_ok=False)
        (controle/'yaml_original.sha256').write_text(sha256(original).hexdigest()+'\n',encoding='utf-8')
        (controle/'YAML_PRINCIPAL_ORIGINAL.yaml').write_bytes(original)
        antes = {p.resolve() for p in OUT.glob('ensaio_pca_gmm_*') if p.is_dir()}
        temp = yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False).encode('utf-8')
        log_out, log_err = controle/'treino_oof.stdout.log',controle/'treino_oof.stderr.log'
        erro = None
        try:
            CFG.write_bytes(temp)
            exigir(yaml.safe_load(CFG.read_text(encoding='utf-8-sig'))['pca']['n_components'] == [1],
                   'Falha ao preparar YAML PCA 1D')
            print('[OOF] Treinamento NOVO, PCA 1D, 10%, 9 folds F1-F9; sem avaliacao F10',flush=True)
            print('[LOG]',log_out,flush=True)
            env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1',MPLBACKEND='Agg')
            with log_out.open('w',encoding='utf-8') as std, log_err.open('w',encoding='utf-8') as err:
                resultado = subprocess.run([sys.executable,'-u','-m','src.orquestracao_oof','--executar'],
                                           cwd=ROOT,env=env,stdout=std,stderr=err,check=False)
            print('\n=== ULTIMAS LINHAS DO TREINAMENTO ===', flush=True)
            print('\n'.join(log_out.read_text(encoding='utf-8',errors='replace').splitlines()[-35:]),flush=True)
            if resultado.returncode:
                print('\n=== ULTIMOS ERROS / AVISOS ===',flush=True)
                print('\n'.join(log_err.read_text(encoding='utf-8',errors='replace').splitlines()[-25:]),flush=True)
                erro = f'Orquestrador retornou {resultado.returncode}; OOF eventualmente concluido foi preservado.'
        finally:
            CFG.write_bytes(original)
            exigir(CFG.read_bytes()==original,'CRITICO: YAML principal nao foi restaurado corretamente!')
            print('[YAML] Original restaurado byte a byte.',flush=True)
        depois = {p.resolve() for p in OUT.glob('ensaio_pca_gmm_*') if p.is_dir()}
        novas = sorted(depois-antes)
        if erro:
            print('[ALERTA] Nao rode novamente automaticamente. O log pode indicar OOF concluido e apenas graficos pendentes.',flush=True)
            raise RuntimeError(erro)
        exigir(len(novas)==1, f'Esperada uma pasta nova de ensaio PCA 1D, encontradas {len(novas)}')
        nova = novas[0]
        manifest, escolha = conferir_oof(nova,166716)
        exigir((nova/'vencedor'/'visualizacao'/'galeria'/'index.html').is_file(),
               'OOF concluido mas galeria ausente; nao repetir treino. Retomar apenas visualizacoes se necessario.')
        print(f'[OOF APROVADO] Pasta: {nova}',flush=True)
        print(f'[OOF] 166716 registros, tau*={escolha["limiar_mcc_oof"]:.17g}',flush=True)
        return nova
    finally:
        trava.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Comparar graficos do PCA 1D OOF e F10 sem repetir F10.')
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument('--verificar',action='store_true',help='Somente conferir configuracao e arquivos existentes.')
    grupo.add_argument('--novo-oof',action='store_true',help='Novo treinamento OOF PCA 1D com 10%; gerar painel de comparacao.')
    grupo.add_argument('--comparar-existente',action='store_true',help='Gerar painel usando OOF original, sem treinar.')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    verificar_config(ler_yaml())
    if args.verificar:
        print('[APROVADO] Pronto. Use --novo-oof para novo treino OOF; F10 permanece intocado.')
        return 0
    pasta = novo_oof() if args.novo_oof else OOF_ANTIGO
    destino = OUT/('comparacao_1D_OOF_vs_F10_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    pagina, resumo = html_comparacao(destino,pasta,args.novo_oof)
    print(f'[CONCLUIDO] Comparador HTML: {pagina}',flush=True)
    print(f'[INVENTARIO] OOF={resumo["pca_1d_figuras_oof_encontradas"]} imagens; F10=4 imagens',flush=True)
    print('[F10] Somente as figuras e metricas ja salvas foram lidas. Nenhuma reavaliacao.',flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f'\n[FALHA] {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        sys.exit(1)
