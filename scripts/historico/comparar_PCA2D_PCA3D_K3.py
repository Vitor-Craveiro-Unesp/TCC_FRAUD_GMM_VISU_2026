#!/usr/bin/env python3
"""Dois ensaios comparaveis de TCC: PCA 2D e PCA 3D, GMM K=3.

Execute na raiz do TCC_EXPERIMENTAL_V3:
  python comparar_PCA2D_PCA3D_K3.py

A configuracao original e restaurada byte a byte mesmo se houver erro.
OOF usa F1-F9; nao avalia o F10. Resultados anteriores sao preservados.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import argparse
import csv
import html
import json
import os
import re
import subprocess
import sys

import numpy as np
import yaml


ARQUIVO_YAML = Path('config_TCC_EXPERIMENTAL_V3_atualizado.yaml')
PONTOS = ('ap', 'mcc', 'logloss', 'score', 'limiar_mcc', 'mcc_limiar_05')


def exigir(condicao, mensagem):
    if not condicao:
        raise RuntimeError(mensagem)


def salvar_atomico(caminho: Path, dados: bytes):
    temporario = caminho.with_name(caminho.name + '.K3_TEMPORARIO')
    exigir(not temporario.exists(), f'Arquivo temporario ja existe: {temporario}')
    with temporario.open('xb') as arq:
        arq.write(dados)
        arq.flush()
        os.fsync(arq.fileno())
    os.replace(temporario, caminho)


def alterar_campo(texto: str, secao: str, novo_valor: int) -> str:
    """Troca apenas n_components dentro da secao PCA ou GMM, sem reformatar YAML."""
    linhas = texto.splitlines(keepends=True)
    ativo = False
    n_alterados = 0
    for i, linha in enumerate(linhas):
        limpa = linha.rstrip('\r\n')
        if re.fullmatch(rf'{re.escape(secao)}\s*:\s*(?:#.*)?', limpa):
            ativo = True
            continue
        if ativo and re.match(r'^[a-zA-Z_][a-zA-Z_0-9-]*\s*:', limpa):
            ativo = False
        if ativo:
            m = re.fullmatch(
                r'([ \t]+n_components[ \t]*:[ \t]*)'
                r'(?:\[[^\]]+\]|\d+)'
                r'([ \t]*(?:#.*)?)', limpa
            )
            if m:
                linhas[i] = (m.group(1) + f'[{novo_valor}]' + m.group(2)
                            + linha[len(limpa):])
                n_alterados += 1
    exigir(n_alterados == 1,
           f'Nao encontrei exatamente um {secao}.n_components em uma unica linha.')
    return ''.join(linhas)


def variante_bytes(original: bytes, dimensao: int, base: dict) -> bytes:
    com_bom = original.startswith(b'\xef\xbb\xbf')
    texto = original.decode('utf-8-sig')
    texto = alterar_campo(texto, 'pca', dimensao)
    texto = alterar_campo(texto, 'gmm', 3)
    cfg = yaml.safe_load(texto)
    esperado = deepcopy(base)
    esperado['pca']['n_components'] = [dimensao]
    esperado['gmm']['n_components'] = [3]
    exigir(cfg == esperado,
           'A troca textual modificaria outros hiperparametros: operacao bloqueada.')
    return (b'\xef\xbb\xbf' if com_bom else b'') + texto.encode('utf-8')


def config_ensaio(pasta: Path) -> dict:
    return yaml.safe_load((pasta / 'configuracao.yaml').read_text(encoding='utf-8-sig'))


def pastas_finalizadas(outputs: Path) -> set[Path]:
    return {p.resolve() for p in outputs.glob('ensaio_pca_gmm_*') if p.is_dir()}


def impedir_retomada_involuntaria(outputs: Path, variante: dict):
    # O orquestrador retoma automaticamente uma pasta em_andamento compativel.
    # Como foram pedidos ensaios NOVOS, bloqueamos esse reaproveitamento.
    for pasta in outputs.glob('em_andamento_ensaio_pca_gmm_*'):
        conf = pasta / 'configuracao.yaml'
        if not conf.is_file():
            continue
        try:
            if config_ensaio(pasta) == variante:
                raise RuntimeError(
                    f'Ja existe ensaio inacabado com configuracao identica: {pasta}. '
                    'Nao vou retomar nem sobrescrever automaticamente. '
                    'Confira a pasta antes de executar novos ensaios.'
                )
        except (OSError, yaml.YAMLError):
            pass


def comando_python(argumentos: list[str], log: Path, titulo: str):
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    comando = [sys.executable, '-u', '-m', 'src.orquestracao_oof', *argumentos]
    print(f'\n===== {titulo} =====', flush=True)
    print('Comando:', ' '.join(comando), flush=True)
    with log.open('w', encoding='utf-8') as arquivo:
        arquivo.write(' '.join(comando) + '\n\n')
        arquivo.flush()
        proc = subprocess.Popen(
            comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding='utf-8', errors='replace', bufsize=1, env=env
        )
        try:
            assert proc.stdout is not None
            for linha in proc.stdout:
                print(linha, end='', flush=True)
                arquivo.write(linha)
                arquivo.flush()
            retorno = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    exigir(retorno == 0,
           f'Falha em {titulo} (codigo {retorno}). Log: {log}. '
           'Nao exclua a pasta em_andamento/ ou o OOF que eventualmente terminou.')


def validar_pasta(pasta: Path, dim: int, base: dict):
    c = config_ensaio(pasta)
    esperado = deepcopy(base)
    esperado['pca']['n_components'] = [dim]
    esperado['gmm']['n_components'] = [3]
    exigir(c == esperado, f'YAML congelado divergiu em {pasta}')
    man = json.loads((pasta / 'manifesto.json').read_text(encoding='utf-8-sig'))
    exigir(man.get('status') == 'concluido_06_03', f'OOF incompleto em {pasta}')
    exigir(man.get('visualizacoes') == 'concluidas', f'Galeria incompleta em {pasta}')
    exigir(man.get('tipo') == 'ensaio', f'Nao e um ensaio amostral: {pasta}')
    escolhido = json.loads((pasta / 'vencedor' / 'metricas' /
                            'melhor_configuracao.json').read_text(encoding='utf-8-sig'))
    exigir(chosen_dim(escolhido) == dim, f'O vencedor nao e PCA {dim}D')
    exigir(int(escolhido['configuracao']['gmm']['n_components']) == 3,
           'O vencedor nao tem tres gaussianas.')
    exigir(escolhido.get('teste_final_usado') is False,
           'Selecao indica uso do F10!')
    metricas = json.loads((pasta / 'vencedor' / 'metricas' /
                           'metricas.json').read_text(encoding='utf-8-sig'))
    pagina = pasta / 'vencedor' / 'visualizacao' / 'galeria' / 'index.html'
    exigir(pagina.is_file(), f'Galeria nao encontrada: {pagina}')
    for nome in PONTOS:
        exigir(nome in metricas, f'Metrica nao encontrada: {nome} em {pasta}')
    exigir(metricas.get('tipo_avaliacao') == 'OOF_AMOSTRAL_NAO_OFICIAL'
           and metricas.get('quantidade_folds_cv') == 9,
           f'A metrica nao foi avaliada em OOF amostral F1-F9: {pasta}')
    return man, escolhido, metricas, pagina


def chosen_dim(escolha: dict) -> int:
    return int(escolha['configuracao']['reducao']['n_components'])


def checar_mesmas_observacoes(p1: Path, p2: Path, m1: dict, m2: dict):
    for campo in ('sha256_parquet', 'sha256_folds', 'registros_desenvolvimento'):
        exigir(m1.get(campo) == m2.get(campo),
               f'As execucoes diferiram em {campo}!')
    caminhos = [(p / 'vencedor' / 'oof' / 'predicoes.npz') for p in (p1, p2)]
    with np.load(caminhos[0], allow_pickle=False) as a, \
         np.load(caminhos[1], allow_pickle=False) as b:
        for campo in ('linha_original', 'fold', 'y_true'):
            exigir(np.array_equal(a[campo], b[campo]),
                   f'OOF PCA 2D x 3D nao tem exatamente os mesmos {campo}.')
        exigir(not np.any(a['fold'] == 10) and not np.any(b['fold'] == 10),
               'F10 identificado em dados OOF!')


def criar_relatorio(diretorio: Path, p1: Path, p2: Path,
                   e1: dict, e2: dict, m1: dict, m2: dict,
                   g1: Path, g2: Path):
    linhas = []
    for d, pasta, esc, met in ((2, p1, e1, m1), (3, p2, e2, m2)):
        linha = {'dimensao': f'PCA {d}D', 'K': 3,
                 'id_vencedor': esc['id_candidato'],
                 'pasta': str(pasta.resolve())}
        for nome in PONTOS:
            linha[nome] = f'{float(met[nome]):.7f}'
        linhas.append(linha)
    csv_path = diretorio / 'comparacao_metricas_OOF_PCA2D_PCA3D_K3.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as arq:
        escritor = csv.DictWriter(arq, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)

    def caminho(p: Path) -> str:
        return Path(os.path.relpath(p.resolve(), diretorio.resolve())).as_posix()

    colunas = [('AP', 'ap'), ('MCC OOF', 'mcc'),
               ('LogLoss', 'logloss'), ('Score composto', 'score'),
               ('Limiar otimo MCC', 'limiar_mcc'), ('MCC em 0,50', 'mcc_limiar_05')]
    trs = ''.join(
        '<tr><th>' + html.escape(rotulo) + '</th>'
        + ''.join(f'<td>{float(m[key]):.7f}</td>' for m in (m1, m2)) + '</tr>'
        for rotulo, key in colunas
    )
    comparador = f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Comparacao PCA 2D x 3D | GMM K=3</title>
<style>
body{{font-family:system-ui,Arial,sans-serif;background:#f5f7fa;color:#203047;margin:22px}}
h1{{margin-bottom:4px}}p{{color:#52657d}}table{{border-collapse:collapse;width:100%;max-width:780px;background:white}}
th,td{{border:1px solid #d7dee6;padding:10px 14px;text-align:right}}th:first-child{{text-align:left}}
header{{margin-bottom:24px}}.galerias{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:22px}}
article{{min-width:0;background:white;border:1px solid #ddd;border-radius:9px;padding:8px}}
article h2{{margin:8px}}iframe{{width:100%;height:72vh;border:1px solid #d7dee6}}
@media(max-width:940px){{.galerias{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>Comparacao PCA 2D x PCA 3D — GMM K=3</h1>
<p>Dois novos ensaios amostrais: mesma configuracao (exceto dimensionalidade),
mesmas observacoes e folds F1–F9. F10 nao utilizado.</p>
<table><thead><tr><th>Metrica OOF</th><th>PCA 2D</th><th>PCA 3D</th></tr></thead><tbody>{trs}</tbody></table>
<p>As galerias aparecem abaixo. Abra em aba propria para visualizar em tela inteira.</p></header>
<div class="galerias">
<article><h2>PCA 2D | K=3</h2><p><a href="{html.escape(caminho(g1),quote=True)}" target="_blank">Abrir galeria PCA 2D</a></p>
<iframe title="Galeria PCA 2D" src="{html.escape(caminho(g1),quote=True)}"></iframe></article>
<article><h2>PCA 3D | K=3</h2><p><a href="{html.escape(caminho(g2),quote=True)}" target="_blank">Abrir galeria PCA 3D</a></p>
<iframe title="Galeria PCA 3D" src="{html.escape(caminho(g2),quote=True)}"></iframe></article>
</div></body></html>'''
    html_path = diretorio / 'comparacao_galerias_PCA2D_PCA3D_K3.html'
    html_path.write_text(comparador, encoding='utf-8')
    manifesto = {'pca_2d': str(p1.resolve()), 'pca_3d': str(p2.resolve()),
                 'k_gmm': 3, 'f10_utilizado': False,
                 'galeria_2d': str(g1.resolve()), 'galeria_3d': str(g2.resolve()),
                 'metricas_csv': str(csv_path.resolve())}
    (diretorio / 'manifesto_comparacao.json').write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('\n========== COMPARACAO OOF - K=3 ==========')
    print(f'{"METRICA":<24} {"PCA 2D":>14} {"PCA 3D":>14}')
    for rotulo, nome in colunas:
        print(f'{rotulo:<24} {float(m1[nome]):>14.7f} {float(m2[nome]):>14.7f}')
    print('PCA 2D:', g1.resolve())
    print('PCA 3D:', g2.resolve())
    print('Comparacao HTML:', html_path.resolve())
    print('Comparacao CSV:', csv_path.resolve())
    if os.name == 'nt':
        try:
            os.startfile(str(html_path.resolve()))
        except OSError as exc:
            print('Nao foi possivel abrir o navegador automaticamente:', exc)
    return html_path


def main():
    argp = argparse.ArgumentParser(description='Comparar duas novas galerias PCA 2D e PCA 3D com GMM K=3.')
    argp.add_argument('--verificar', action='store_true',
                      help='Somente verifica ambiente e integracao; NAO executa OOF nem modifica YAML.')
    argp.add_argument('--por-fold', type=int, default=1000,
                      help='Maximo de registros por fold no modo --ensaio (padrao: 1000).')
    a = argp.parse_args()
    exigir(a.por_fold >= 20, '--por-fold precisa ser >= 20 para esta auditoria.')
    raiz = Path.cwd()
    exigir((raiz / 'src' / 'orquestracao_oof.py').is_file(),
           'Abra o terminal na raiz do TCC_EXPERIMENTAL_V3.')
    exigir(ARQUIVO_YAML.is_file(), 'YAML principal nao encontrado.')
    exigir((raiz / 'outputs').is_dir(), 'Pasta outputs/ nao encontrada.')
    # Exigir integracao de visualizacoes PCA3D antes do custo do OOF.
    modulos = (
        'visualizacao_gmm01_3d.py',
        'visualizacao_pca3d_complementos.py',
        'visualizacao_gmm04_3d.py',
        'visualizacao_gmm07_3d.py',
        'visualizacao_gmm08_3d.py',
    )
    ausentes = [nome for nome in modulos if not (raiz / 'src' / nome).is_file()]
    exigir(not ausentes,
           'Integracoes PCA3D nao encontradas em src/: ' + ', '.join(ausentes))
    # Nao alterar o pipeline ou recuperar imagens de testes sintéticos: todas as galerias
    # devem ser geradas pelos seus proprios ensaios OOF.


    original = ARQUIVO_YAML.read_bytes()
    base = yaml.safe_load(original.decode('utf-8-sig'))
    exigir(base['reducao']['metodo'].upper() == 'PCA',
           'O YAML deve estar configurado para metodo PCA.')
    exigir(Path(base['saidas']['diretorio_base']) == Path('outputs'),
           'Este script exige saidas.diretorio_base=outputs.')
    fracao = float(base['dados']['fracao_base'])
    exigir(0 < fracao < 1,
           'Este script preserva a fracao amostral do YAML; requer 0 < dados.fracao_base < 1.')
    exigir(base['execucao'].get('diretorio_retomada') is None,
           'Existe execucao.diretorio_retomada explicito; limpe-o antes de criar NOVOS ensaios.')
    exigir(base['execucao']['proteger_resultados'] is True,
           'Protecao de resultados precisa estar ligada.')
    exigir(int(base['divisao']['qual_fold_teste']) == 10,
           'F10 precisa continuar reservado.')
    exigir(base['avaliacao']['metrica_pr'] == 'average_precision'
           and base['avaliacao']['selecao_limiar'] == 'mcc_oof'
           and all(base['avaliacao'][k] is True for k in
                   ('calcular_mcc', 'calcular_logloss', 'calcular_composite')),
           'Metricas OOF esperadas (AP, MCC, LogLoss, Score) nao estao ativas.')

    variantes = {dim: variante_bytes(original, dim, base) for dim in (2, 3)}
    cfgs = {dim: yaml.safe_load(dados.decode('utf-8-sig'))
            for dim, dados in variantes.items()}
    c2 = deepcopy(cfgs[2]); c3 = deepcopy(cfgs[3])
    c3['pca']['n_components'] = [2]
    exigir(c2 == c3, 'As configuracoes 2D e 3D divergem alem da dimensionalidade.')

    # Auditoria preventiva: nao gastar OOF se a integracao PCA2D ficou
    # incompleta durante as instalacoes sucessivas dos complementos PCA3D.
    exigidos_2d = {
        raiz / 'src' / 'visualizacao_pca_biplot.py': (
            'PCA_COEFICIENTES_PC1_PC2',
        ),
        raiz / 'src' / 'visualizacao_integrada.py': (
            'PCA_COEFICIENTES_PC1_PC2',
            'GMM01_densidade_conjunta_2d_PC1_PC2',
            'GMM01_densidade_conjunta_superficie_PCA2D',
        ),
    }
    lacunas = []
    for modulo, nomes in exigidos_2d.items():
        texto = modulo.read_text(encoding='utf-8-sig') if modulo.is_file() else ''
        for nome in nomes:
            if nome not in texto:
                # Aceitar GMM01 2D implementado em modulo auxiliar,
                # desde que o gerador oficial realmente o chame.
                if modulo.name == 'visualizacao_gmm01.py':
                    auxiliar = raiz / 'src' / 'visualizacao_gmm01_2d.py'
                    if (auxiliar.is_file()
                            and 'gerar_complementares_2d' in texto
                            and nome in auxiliar.read_text(encoding='utf-8-sig')):
                        continue
                lacunas.append(f'{modulo.relative_to(raiz)}: {nome}')
    if lacunas:
        print('\nATENCAO - PCA 2D: nao encontrei estas referencias no codigo atual:', flush=True)
        for item in lacunas:
            print('  -', item, flush=True)
        raise RuntimeError(
            'Integracao PCA 2D possivelmente incompleta. Envie os tres arquivos '            'src/visualizacao_pca_biplot.py, src/visualizacao_gmm01.py e '            'src/visualizacao_integrada.py para adequarmos ANTES dos ensaios.'
        )
    # PRECHECK_GMM01_PCA2D_FUNCIONAL_V2
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    import pandas as pd
    from sklearn.mixture import GaussianMixture
    from src import visualizacao_gmm01 as gmm01

    rng = np.random.default_rng(20261027)

    centros = np.array([
        [-2.4, -1.2],
        [ 2.3, -0.6],
        [ 0.0,  2.4]
    ])

    X = np.vstack([
        rng.normal(loc=c, scale=.8, size=(75, 2))
        for c in centros
    ])

    coordenadas = pd.DataFrame(
        X, columns=['PC1', 'PC2']
    )

    modelo = GaussianMixture(
        n_components=3,
        covariance_type='full',
        n_init=2,
        max_iter=200,
        random_state=20261027
    ).fit(X)

    exigir(modelo.converged_, 'GMM de teste nao convergiu.')

    gmm = SimpleNamespace(
        modelo_=modelo,
        saidas_=['G1', 'G2', 'G3'],
        colunas_=['PC1', 'PC2']
    )

    esperados = [
        'GMM01_densidade_marginal_2d_PC1.png',
        'GMM01_densidade_marginal_2d_PC2.png',
        'GMM01_densidade_conjunta_2d_PC1_PC2.png',
        'GMM01_densidade_conjunta_superficie_PCA2D.png'
    ]

    with TemporaryDirectory(
        prefix='tcc_preflight_gmm01_2d_'
    ) as temporario:

        destino = Path(temporario)

        gmm01.gerar(
            cfgs[2],
            {'coord': coordenadas, 'gmm': gmm},
            destino,
            'png',
            85,
            False
        )

        faltantes = [
            nome for nome in esperados
            if not (destino / nome).is_file()
            or (destino / nome).stat().st_size == 0
        ]

        exigir(
            not faltantes,
            'GMM01 2D nao gerou: ' + ', '.join(faltantes)
        )

    print(
        'GMM01 PCA 2D: 4 PNGs validados funcionalmente.',
        flush=True
    )

    print('PREFLIGHT APROVADO: configs PCA 2D / PCA 3D, K=3 e integracoes esperadas.', flush=True)
    if a.verificar:
        print('Somente verificacao; YAML e outputs nao modificados.', flush=True)
        return

    saida = raiz / 'outputs' / (
        'comparacao_PCA2D_PCA3D_K3_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    saida.mkdir(parents=True, exist_ok=False)
    (saida / 'YAML_ORIGINAL_ANTES_DOS_ENSAIOS.bak').write_bytes(original)
    for dim in (2, 3):
        (saida / f'config_PCA{dim}D_K3.yaml').write_bytes(variantes[dim])

    print('Projeto:', raiz)
    print('Relatorios e backup:', saida)
    print('Fracao-base maxima por fold:', fracao)
    print('Gaussianas por GMM: K=3 nos dois ensaios.')
    print('Modo: --ensaio; limite por fold:', a.por_fold, 'observacoes (ate 9 folds).')
    print('Inicializacoes EM (n_init):', base['gmm']['n_init'])
    print('Amostragem:', base['amostragem']['modo'])
    print('Semente:', base['experimento']['semente'])
    print('Metricas: AP, MCC OOF, LogLoss, Score composto e MCC em 0,50.')
    print('F10: reservado. Os ensaios anteriores nao serao alterados.')

    resultados = {}
    try:
        for dim in (2, 3):
            print(f'\n{"="*60}\n NOVO ENSAIO PCA {dim}D / K=3\n{"="*60}', flush=True)
            impedir_retomada_involuntaria(raiz / 'outputs', cfgs[dim])
            exigir(ARQUIVO_YAML.read_bytes() in (original, variantes[2], variantes[3]),
                   'YAML foi alterado por outro processo durante estes ensaios.')
            salvar_atomico(ARQUIVO_YAML, variantes[dim])
            exigir(yaml.safe_load(ARQUIVO_YAML.read_text(encoding='utf-8-sig')) == cfgs[dim],
                   'YAML temporario inesperado.')
            comando_python(['--planejar'], saida / f'planejamento_PCA{dim}D.log',
                           f'PLANEJAMENTO PCA {dim}D')
            antes = pastas_finalizadas(raiz / 'outputs')
            comando_python(['--ensaio', '--por-fold', str(a.por_fold)], saida / f'execucao_PCA{dim}D.log',
                           f'EXECUCAO PCA {dim}D K3')
            depois = pastas_finalizadas(raiz / 'outputs')
            novos = sorted(depois - antes)
            exigir(len(novos) == 1,
                   f'Esperava uma unica pasta nova do PCA {dim}D; encontrei {len(novos)}. '
                   f'Verifique o log e outputs/.')
            pasta = novos[0]
            validar_pasta(pasta, dim, base)
            resultados[dim] = pasta
            print(f'\nPCA {dim}D CONCLUIDO:', pasta, flush=True)
    finally:
        # Fundamental: devolver EXATAMENTE os bytes originais ao YAML do projeto.
        if ARQUIVO_YAML.read_bytes() != original:
            salvar_atomico(ARQUIVO_YAML, original)
            print('\nYAML principal restaurado exatamente ao estado original.', flush=True)
        else:
            print('\nYAML principal ja estava no estado original.', flush=True)

    p1, p2 = resultados[2], resultados[3]
    m1, e1, v1, g1 = validar_pasta(p1, 2, base)
    m2, e2, v2, g2 = validar_pasta(p2, 3, base)
    checar_mesmas_observacoes(p1, p2, m1, m2)
    criar_relatorio(saida, p1, p2, e1, e2, v1, v2, g1, g2)
    print('\nSUCESSO: dois novos ensaios AMOSTRAIS comparaveis, K=3, F1-F9 equivalentes.')
    print('As galerias anteriores foram preservadas. F10 nao utilizado.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrompido. O YAML foi restaurado. '
              'As pastas ja geradas foram preservadas.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print('\nERRO:', exc, file=sys.stderr)
        print('Consulte os logs em outputs/comparacao_PCA2D_PCA3D_K3_*/.',
              file=sys.stderr)
        sys.exit(1)
