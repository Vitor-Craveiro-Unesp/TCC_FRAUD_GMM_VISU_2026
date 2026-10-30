# -*- coding: utf-8 -*-
"""Consolida resultados PCA 1D F10 ja salvos; nao executa F10 nem altera OOF.

    python scripts/execucao/CONSOLIDAR_GALERIA_F10_PCA1D.py --verificar
    python scripts/execucao/CONSOLIDAR_GALERIA_F10_PCA1D.py --gerar

--verificar apenas identifica itens existentes/faltantes e confere hashes.
--gerar exige todos os complementos e escreve uma NOVA galeria HTML e manifesto.
Nunca substitui a galeria principal index.html ou algum grafico/CSV existente.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENSAIO_REL = Path('outputs/ensaio_pca_gmm_20261029_173123')
F10_REL = ENSAIO_REL / 'vencedor/teste_final'
NOME_HTML = 'CONSOLIDADA_PCA1D_F10.html'
NOME_MANIFESTO = 'manifesto_galeria_consolidada_pca1d_f10.json'

PREDITIVOS = (
    ('00_painel_score_composto.png', '00 — Painel inicial'),
    ('01_curva_precision_recall.png', '01 — Precision–Recall'),
    ('02_mcc_por_limiar.png', '02 — MCC por limiar'),
    ('03_matriz_confusao_05.png', '03 — Matriz de confusão τ=0,50'),
    ('04_matriz_confusao_mcc.png', '04 — Matriz de confusão τ* OOF'),
    ('05_distribuicao_probabilidades.png', '05 — Distribuição das probabilidades'),
)
GEOMETRICOS = (
    ('12_pca_projecao_PC1.png', '12 — Projeção PCA PC1'),
    ('PCA_OBSERVACOES_PC1.png', 'PCA — Observações PC1'),
    ('GMM01_densidade_marginal_1d_PC1.png', 'GMM01 — Densidade marginal'),
    ('GMM04_dominancia_1d.png', 'GMM04 — Dominância das gaussianas'),
)
FRONTEIRAS = (
    ('GMM07_sensibilidade_1d_comparativo.png', 'GMM07 — Sensibilidade da fronteira'),
    ('GMM08_mapa_risco_1d_comparativo.png', 'GMM08 — Mapa de risco e gaussianas 95%'),
)
PRE01_REL = Path('diagnosticos/PRE01_F10/PRE01_F10_diagnostico_padronizacao_completo.html')
PRE01_MANIFESTO_REL = Path('diagnosticos/PRE01_F10/PRE01_F10_manifesto.json')
MANIFESTOS = (
    'manifesto_espelho_graficos_f10.json',
    'manifesto_complemento_geometria_pca1d_f10.json',
    'manifesto_complemento_gmm07_gmm08_pca1d_f10.json',
)
METRICAS = 'metricas_f10.json'
PREDICOES = 'predicoes_f10.csv'
OOF_REFERENCIAS = (
    ('oof/06_comparacao_hiperparametros.png', '06 — Comparação de hiperparâmetros (OOF)'),
    ('oof/07_metricas_por_fold.png', '07 — Métricas por fold (OOF)'),
    ('geometria_desenvolvimento/09_painel_parametros_gaussianos.png', '09 — Parâmetros GMM (F1–F9)'),
    ('geometria_desenvolvimento/11_pca_cotovelo.png', '11 — PCA cotovelo (F1–F9)'),
    ('geometria_desenvolvimento/GMM18_convergencia_em_inicializacoes.png', 'GMM18 — Convergência EM (F1–F9)'),
    ('geometria_desenvolvimento/PCA_BIPLOT_1D.png', 'PCA — Biplot 1D (F1–F9)'),
    ('geometria_desenvolvimento/PCA_COEFICIENTES_PC1.png', 'PCA — Coeficientes PC1 (F1–F9)'),
)


def exigir(cond, mensagem):
    if not cond:
        raise RuntimeError(mensagem)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def ler_json(path: Path) -> dict:
    exigir(path.is_file(), f'Manifesto ausente: {path}')
    d = json.loads(path.read_text(encoding='utf-8-sig'))
    exigir(isinstance(d, dict), f'Manifesto invalido: {path}')
    return d


def verificar_arquivo(f10: Path, caminho: str, esperada: str | None = None):
    f = f10 / caminho
    exigir(f.is_file() and f.stat().st_size > 0, f'Arquivo ausente/vazio: {f}')
    h = sha256(f)
    if esperada is not None:
        exigir(h == esperada, f'SHA-256 divergente: {f}')
    return h


def levantar(raiz: Path, estrito: bool = False) -> dict:
    raiz = Path(raiz).resolve()
    base = raiz / ENSAIO_REL / 'vencedor'
    f10 = raiz / F10_REL
    figuras = f10 / 'graficos_gerados'
    galeria_oof = base / 'visualizacao/galeria'
    exigir(base.is_dir(), f'Ensaio PCA 1D ausente: {base}')

    faltantes = []
    for rel in [METRICAS, PREDICOES, *MANIFESTOS, str(PRE01_REL), str(PRE01_MANIFESTO_REL)]:
        if not (f10 / rel).is_file():
            faltantes.append(rel)
    for nome, _ in (*PREDITIVOS, *GEOMETRICOS, *FRONTEIRAS):
        if not (figuras / nome).is_file():
            faltantes.append(f'graficos_gerados/{nome}')
    if not (galeria_oof / 'index.html').is_file():
        faltantes.append('OOF: vencedor/visualizacao/galeria/index.html')

    print('=== PCA 1D — INVENTARIO F10, SEM REAVALIAR ===')
    for titulo, grupo in [('Preditivos 00–05', PREDITIVOS), ('Geometria 1D', GEOMETRICOS), ('Fronteiras GMM07/08', FRONTEIRAS)]:
        presentes = sum((figuras / f).is_file() for f, _ in grupo)
        print(f'[{"OK" if presentes == len(grupo) else "PENDENTE"}] {titulo}: {presentes}/{len(grupo)} figuras')
    print(f'[{"OK" if (f10 / PRE01_REL).is_file() else "PENDENTE"}] PRE01 interativo F10')
    if faltantes:
        print('[FALTANTES]')
        for item in faltantes:
            print('  -', item)
        print('[SEM ESCRITA] A consolidacao exige todos os complementos.')
        if estrito:
            raise RuntimeError('Complementos ausentes: gere somente o que falta antes da consolidacao.')
        return dict(pronto=False, faltantes=faltantes, f10=f10)

    espelho = ler_json(f10 / MANIFESTOS[0])
    geo = ler_json(f10 / MANIFESTOS[1])
    fronteiras = ler_json(f10 / MANIFESTOS[2])
    pre01 = ler_json(f10 / PRE01_MANIFESTO_REL)
    metricas = ler_json(f10 / METRICAS)
    # Confere os elementos imutaveis e a origem comum, sem recalcular metricas.
    n = int(espelho['n'])
    tau = float(espelho['limiar_mcc_congelado_oof'])
    exigir(n == 18523 == int(metricas['n']) == int(geo['n_f10']) == int(fronteiras['n_f10']) == int(pre01['n_f10']),
           'As etapas nao se referem a mesma amostra F10.')
    exigir(tau == float(metricas['limiar_mcc_oof_congelado']) == float(geo['tau_star_oof']) ==
           float(fronteiras['tau_star_oof_congelado']) == float(pre01['tau_star_oof']),
           'Tau OOF alterado entre os arquivos.')
    exigir(metricas['limiar_otimizado_no_f10'] is False, 'Limiar F10 reotimizado indevidamente.')
    exigir(str(espelho['modelo_origem_oof']).replace('\\', '/') == ENSAIO_REL.as_posix(),
           'O F10 nao foi gerado pelo ensaio PCA 1D esperado.')

    hashes = {}
    for nome in (PREDICOES, METRICAS):
        esperado = espelho['predicoes_sha256'] if nome == PREDICOES else espelho['metricas_sha256']
        hashes[nome] = verificar_arquivo(f10, nome, esperado)
    for nome in MANIFESTOS:
        hashes[nome] = verificar_arquivo(f10, nome)
    for nome, _ in PREDITIVOS:
        esperado = espelho.get('graficos', {}).get(nome)
        exigir(esperado is not None, f'Manifesto espelho sem hash: {nome}')
        hashes[f'graficos_gerados/{nome}'] = verificar_arquivo(f10, f'graficos_gerados/{nome}', esperado)
    for nome, esperado in geo.get('graficos', {}).items():
        hashes[f'graficos_gerados/{nome}'] = verificar_arquivo(f10, f'graficos_gerados/{nome}', esperado)
    for nome, _ in GEOMETRICOS:
        exigir(nome in geo.get('graficos', {}), f'Manifesto geometria sem hash: {nome}')
    for nome, esperado in fronteiras.get('arquivos', {}).items():
        hashes[f'graficos_gerados/{nome}'] = verificar_arquivo(f10, f'graficos_gerados/{nome}', esperado)
    for nome, _ in FRONTEIRAS:
        exigir(nome in fronteiras.get('arquivos', {}), f'Manifesto GMM07/08 sem hash: {nome}')
    hashes[str(PRE01_REL)] = verificar_arquivo(f10, PRE01_REL.as_posix())
    hashes[str(PRE01_MANIFESTO_REL)] = verificar_arquivo(f10, PRE01_MANIFESTO_REL.as_posix())
    for nome, hash_original in pre01.get('arquivos_sha256', {}).items():
        hashes[f'diagnosticos/PRE01_F10/{nome}'] = verificar_arquivo(f10, f'diagnosticos/PRE01_F10/{nome}', hash_original)
    # Inventariar TODOS os CSV/JSON atuais do teste final, sem arquivos de dados originais.
    for arquivo in f10.rglob('*'):
        if arquivo.is_file() and arquivo.suffix.lower() in {'.csv', '.json'}:
            relativo = arquivo.relative_to(f10).as_posix()
            hashes.setdefault(relativo, sha256(arquivo))

    exigir(not (f10 / 'galeria' / NOME_HTML).exists() and not (f10 / NOME_MANIFESTO).exists(),
           'Uma galeria consolidada ja existe. Nada sera sobrescrito.')
    print(f'[APROVADO] {n} registros, tau OOF={tau:.17g}; 12 graficos F10 verificados por SHA-256.')
    print('[APROVADO] PRE01, origens e manifestos; nenhum modelo ou Parquet acessado.')
    return dict(pronto=True, f10=f10, oof=galeria_oof, n=n, tau=tau, hashes=hashes)


def html_consolidado(estado: dict) -> str:
    f10 = estado['f10']
    oof = estado['oof']
    esc = lambda s: html.escape(str(s), quote=True)

    def cards(grupo):
        return ''.join(
            '<article><a href="../graficos_gerados/'+esc(nome)+'" target="_blank" rel="noopener">'
            '<img loading="lazy" alt="'+esc(titulo)+'" src="../graficos_gerados/'+esc(nome)+'">'
            '<div class="titulo">'+esc(titulo)+'</div></a></article>'
            for nome, titulo in grupo
        )

    refs = []
    for relativo, titulo in OOF_REFERENCIAS:
        if (oof / relativo).is_file():
            href = '../../visualizacao/galeria/' + relativo
            refs.append(f'<article><a href="{esc(href)}" target="_blank" rel="noopener">'
                        f'<img loading="lazy" src="{esc(href)}" alt="{esc(titulo)}">'
                        f'<div class="titulo">{esc(titulo)}</div></a></article>')
    ref_html = ''.join(refs)
    pre01 = '../' + PRE01_REL.as_posix()
    # Links apenas para arquivos presentes, sem copiar nem modificar dados.
    downloads = []
    for caminho in sorted(f10.rglob('*')):
        if not caminho.is_file() or caminho.suffix.lower() not in {'.csv', '.json'}:
            continue
        rel = caminho.relative_to(f10).as_posix()
        downloads.append(f'<li><a href="../{esc(rel)}">{esc(rel)}</a></li>')
    downloads_html = ''.join(downloads)
    return f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PCA 1D — Galeria F10 consolidada</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;background:#f5f7fa;color:#172236;max-width:1420px;margin:auto;padding:24px}}
h1{{margin-bottom:6px}}h2{{font-size:21px;margin:35px 0 9px}}p{{max-width:1100px}}
.selo{{padding:13px;background:#e8edf4;border-left:4px solid #355877;border-radius:5px}}
.grade{{display:grid;grid-template-columns:repeat(auto-fit,minmax(325px,1fr));gap:16px}}
article{{background:white;border:1px solid #d7dee6;border-radius:10px;padding:13px;box-shadow:0 2px 8px #0001}}
article a{{text-decoration:none;color:#233a58}}article img{{width:100%;height:270px;object-fit:contain}}
.titulo{{margin-top:6px;font-weight:600}}.links{{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}}
.links a{{background:white;border:1px solid #cad5e2;color:#245386;padding:9px 13px;border-radius:8px;text-decoration:none}}
@media(max-width:740px){{body{{padding:12px}}.grade{{grid-template-columns:1fr}}}}
</style></head><body><h1>PCA 1D — Galeria consolidada do teste F10</h1>
<p class="selo">F10 amostral exploratório: {estado['n']:,} observações. PCA/GMM treinados em F1–F9.
Limiar τ* = {estado['tau']:.17g} definido exclusivamente em OOF. Nenhum reajuste ou nova seleção de modelos.</p>
<div class="links"><a href="index.html">Comparação OOF × F10 (seis gráficos)</a>
<a href="complemento_geometria_pca1d_f10.html">Galeria das projeções e geometria</a>
<a href="complemento_gmm07_gmm08_pca1d_f10.html">Galeria GMM07/GMM08</a>
<a href="{esc(pre01)}">Abrir PRE01 F10 interativo</a>
<a href="../../visualizacao/galeria/index.html">Abrir galeria OOF original</a></div>
<section><h2>00–05 — Validação preditiva no F10</h2><p>Mesmo padrão gráfico do campeão OOF, com scores e classes da amostra F10 já avaliada.</p><div class="grade">{cards(PREDITIVOS)}</div></section>
<section><h2>12, PCA e GMM01/GMM04 — Projeção e geometria F10</h2>
<p>PC1 e observações da amostra F10 aplicadas ao PCA congelado. Densidades e dominância correspondem ao GMM fixado em F1–F9 no domínio de PC1 de F10.</p><div class="grade">{cards(GEOMETRICOS)}</div></section>
<section><h2>GMM07/GMM08 — Fronteiras e mapa de risco</h2>
<p>Comparações τ=0,50 versus τ* OOF fixo, com MCC F10 já registrado e gaussianas estimadas em F1–F9.</p><div class="grade">{cards(FRONTEIRAS)}</div></section>
<section><h2>PRE01 — Diagnóstico interativo do F10</h2><p>Parâmetros de treinamento congelados e análise descritiva da distribuição F10, sem ajustar novamente encoder, padronizador ou PCA.</p>
<div class="links"><a href="{esc(pre01)}">Abrir painel PRE01 F10</a></div></section>
<section><h2>Diagnósticos exclusivos do desenvolvimento</h2>
<p>Esses gráficos não são resultados de uma nova avaliação F10. Consulte-os como documentação do modelo F1–F9 e dos folds OOF.</p>
<div class="grade">{ref_html}</div><p><a href="../../visualizacao/galeria/index.html">Galeria OOF completa</a></p></section>
<section><h2>Resultados, CSVs e manifestos do teste</h2><p>Arquivos existentes no diretório teste_final. Nenhuma cópia adicional foi criada.</p><ul>{downloads_html}</ul></section>
<p class="selo">Esta página apenas organiza PNGs, HTMLs e CSVs existentes. Os arquivos de origem e a galeria anterior permanecem inalterados.</p>
</body></html>'''


def consolidar(estado: dict):
    exigir(estado['pronto'], 'Complementos incompletos; consolidacao bloqueada.')
    f10 = estado['f10']
    destino = f10 / 'galeria' / NOME_HTML
    meta_destino = f10 / NOME_MANIFESTO
    exigir(not destino.exists() and not meta_destino.exists(), 'Destino ja existe; preservado.')
    texto = html_consolidado(estado)
    manifesto = {
        'versao': 'CONSOLIDADA_PCA1D_F10_V1',
        'fase': 'teste_final',
        'ensaio_oof': ENSAIO_REL.as_posix(),
        'tipo': 'GALERIA_DE_ARQUIVOS_PREEXISTENTES',
        'n_f10': estado['n'],
        'tau_star_congelado_oof': estado['tau'],
        'sem_retreino': True, 'sem_nova_inferencia': True,
        'sem_nova_avaliacao_f10': True, 'sem_leitura_parquet': True,
        'arquivos_origem_sha256': estado['hashes'],
        'galeria': 'galeria/' + NOME_HTML,
        'data_utc': datetime.now(timezone.utc).isoformat(),
    }
    # Criar exclusivamente: nem a pagina original index.html nem os PNGs sao tocados.
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open('x', encoding='utf-8') as f:
        f.write(texto)
    try:
        with meta_destino.open('x', encoding='utf-8') as f:
            json.dump(manifesto, f, ensure_ascii=False, indent=2)
            f.write('\n')
    except Exception:
        # Mantem a pagina caso o segundo arquivo falhe, para auditoria.
        print('[AVISO] HTML criado mas manifesto falhou. Nao repetir sem conferir os arquivos.', flush=True)
        raise
    print('[CONCLUIDO] 12 PNGs F10 + PRE01 integrados. OOF e arquivos anteriores preservados.')
    print('[GALERIA]', destino)
    print('[MANIFESTO]', meta_destino)


def main():
    parser = argparse.ArgumentParser()
    mod = parser.add_mutually_exclusive_group(required=True)
    mod.add_argument('--verificar', action='store_true')
    mod.add_argument('--gerar', action='store_true')
    args = parser.parse_args()
    estado = levantar(ROOT, estrito=args.gerar)
    if args.verificar:
        print('[VERIFICACAO] Nenhuma escrita; nenhuma nova avaliacao F10.')
    elif estado['pronto']:
        consolidar(estado)


if __name__ == '__main__':
    main()
