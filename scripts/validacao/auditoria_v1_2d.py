# -*- coding: utf-8 -*-
"""V1: auditoria estatica, SOMENTE LEITURA, para dois ensaios reais 3% 2D.

Nao importa nenhuma rotina de treinamento. Nao abre Parquet ou folds.npy,
nao gera arquivos, nao acessa F10 e nao altera dados, YAML ou artefatos.
"""
from __future__ import annotations
import ast
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
OBRIGATORIOS = [
    'src/orquestracao_oof.py',
    'src/validacao_oof.py',
    'src/reducao_tsne.py',
    'src/treinamento_final_tsne.py',
    'src/treinamento_final_pca.py',
    'src/ponte_oof_final_tsne.py',
    'src/visualizacao_oof.py',
    'src/visualizacao_geometrica.py',
    'src/visualizacao_tsne_pdf.py',
    'src/visualizacao_gmm01.py',
    'src/visualizacao_gmm04.py',
    'src/visualizacao_gmm07.py',
    'src/visualizacao_integrada.py',
    'src/garantia_layout_tsne.py',
    'scripts/execucao/experimentos_reais_3pct_1d.py',
]
AVISOS_V2 = [
    ('EXECUTOR_3PCT_FIXO_1D', 'Adaptar 1D -> 2D em selecao, persistencia, predicoes F10, geometria e galeria.'),
    ('GMM01_SUPERFICIE_ROTULADA_PCA', 'Usar TSNE1/TSNE2 e novo stem da superficie sem alterar PCA 2D.'),
    ('GALERIA_CONJUNTAS_SOMENTE_PCA', 'Adicionar validacao/copia das duas imagens conjuntas tambem em TSNE 2D.'),
    ('TSNE13_2D_PENDENTE_HOMOLOGACAO', 'Homologar scatter 2D OOF/F10 e suas legendas/rodape antes do OOF real.'),
]

def rel(caminho):
    return str(caminho.relative_to(ROOT)).replace('\\', '/')

def source(caminho):
    return (ROOT / caminho).read_text(encoding='utf-8-sig')

def ok(codigo, descricao):
    print(f'[OK] {codigo}: {descricao}')

def aviso(codigo, descricao):
    print(f'[AJUSTAR_V2] {codigo}: {descricao}')

def erro(codigo, descricao):
    print(f'[FALHA_V1] {codigo}: {descricao}')

def check(texto, marcadores):
    return all(m in texto for m in marcadores)

def run():
    falhas = []
    print('=== V1 | AUDITORIA ARQUITETURA TSNE 2D + PCA 2D | SOMENTE LEITURA ===')
    print('RAIZ=' + str(ROOT))
    print('PYTHON=' + sys.version.split()[0])
    print('PYTHON_EXECUTAVEL=' + sys.executable)
    if sys.prefix == sys.base_prefix or ROOT / '.venv' not in Path(sys.executable).parents:
        falhas.append('venv')
        erro('PYTHON_VENV', 'Executar com .venv/Scripts/python.exe do projeto.')
    else:
        ok('PYTHON_VENV', 'Executavel pertence ao .venv deste projeto.')

    versoes = {}
    for pacote in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'openTSNE',
                   'matplotlib', 'PyYAML', 'joblib', 'pyarrow'):
        try:
            versoes[pacote] = version(pacote)
        except PackageNotFoundError:
            versoes[pacote] = 'AUSENTE'
            falhas.append('pacote:' + pacote)
    print('PACOTES=' + '; '.join(f'{k}={v}' for k, v in versoes.items()))

    fontes = {}
    for nome in OBRIGATORIOS:
        arq = ROOT / nome
        if not arq.is_file():
            falhas.append(nome)
            erro('FONTE_AUSENTE', nome)
            continue
        try:
            texto = source(nome)
            ast.parse(texto, filename=str(arq))
        except (SyntaxError, UnicodeError, OSError) as exc:
            falhas.append(nome)
            erro('PYTHON_INVALIDO', f'{nome}: {exc}')
            continue
        fontes[nome] = texto
    print(f'FONTES_PYTHON_VALIDAS={len(fontes)}/{len(OBRIGATORIOS)}')

    yaml_arq = ROOT / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    if not yaml_arq.is_file():
        falhas.append('yaml_base')
        erro('YAML_BASE', 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml ausente')
        cfg = {}
    else:
        try:
            cfg = yaml.safe_load(yaml_arq.read_text(encoding='utf-8-sig'))
            if not isinstance(cfg, dict):
                raise ValueError('YAML nao e mapa')
            ok('YAML_BASE', 'Legivel. Auditoria nao alterou o YAML.')
        except Exception as exc:
            falhas.append('yaml_invalido')
            erro('YAML_BASE', str(exc))
            cfg = {}
    if cfg:
        esperado = [
            (cfg['experimento']['semente'] == 42, 'semente=42'),
            (cfg['divisao']['total_folds'] == 10, '10 folds'),
            (cfg['divisao']['qual_fold_teste'] == 10, 'F10 reservado'),
            (cfg['divisao']['estratificada'] is True, 'estratificado'),
            (cfg['gmm']['covariance_type'] == 'full', 'GMM FULL'),
            (cfg['gmm']['n_components'] == [4], 'K=4'),
            (cfg['gmm']['n_init'] == 5, 'n_init=5'),
            (cfg['amostragem']['modo'] == 'RUS', 'RUS'),
            (cfg['amostragem']['razao_negativos_positivos'] == 16, 'RUS 16:1'),
            (cfg['avaliacao']['selecao_limiar'] == 'mcc_oof', 'MCC OOF'),
        ]
        for valido, mensagem in esperado:
            if not valido:
                falhas.append(mensagem)
                erro('CONTRATO_METODO', mensagem)
        if all(v for v, _ in esperado):
            ok('CONTRATO_METODO', '10 folds, F10, seed 42, RUS16, GMM FULL K4 n_init5, MCC OOF.')
        t = cfg.get('tsne', {})
        n = t.get('n_components')
        if n not in (2, [2]):
            aviso('YAML_BASE_NAO_E_TSNE2D', f'tsne.n_components atual={n!r}; V2 gerara YAML 2D independente.')
        print('YAML_BASE_METODO=' + str(cfg.get('reducao', {}).get('metodo')))
        print('YAML_BASE_FRACAO=' + str(cfg.get('dados', {}).get('fracao_base')))
        print('TSNE_GRID_PERPLEXITY=' + str(t.get('perplexity')))
        print('TSNE_GRID_INITIALIZATION=' + str(t.get('initialization')))

    tsne = fontes.get('src/reducao_tsne.py', '')
    if check(tsne, ('def grade_tsne', 'def transform_treino', 'def transform(self, X)',
                    'p["n_components"] > 3', 'embedding_.transform')):
        ok('REDUTOR_TSNE_2D', 'Grade suporta 1/2/3D; fit treino e transform OOS distintos.')
    else:
        falhas.append('reducao_tsne_contrato')
        erro('REDUTOR_TSNE_2D', 'Nao confirmado suporte 2D e extensao OOS.')

    modelo = fontes.get('src/treinamento_final_tsne.py', '')
    if check(modelo, ('def ajustar_final_f1_f9', 'def prever_sem_rotulos',
                      'self.tsne.transform(x_pre)', 'tsne.transform_treino()')):
        ok('MODELO_FINAL_TSNE', 'Rotinas congeladas F1..F9 e previsao F10 OOS identificadas.')
    else:
        falhas.append('modelo_final_tsne')
        erro('MODELO_FINAL_TSNE', 'Integracao final ou OOS nao encontrada.')

    guard = fontes.get('src/garantia_layout_tsne.py', '')
    orq = fontes.get('src/orquestracao_oof.py', '')
    geo = fontes.get('src/visualizacao_geometrica.py', '')
    executor = fontes.get('scripts/execucao/experimentos_reais_3pct_1d.py', '')
    tsne_pdf = fontes.get('src/visualizacao_tsne_pdf.py', '')
    guard_ok = check(guard, ('PATCH_LAYOUT_TSNE1D_OOF_20261103',
                              'fig.tight_layout(rect=(.015, .145, .985, .91))',
                              'fig.subplots_adjust(left=.055, right=.945, top=.81, bottom=.30, wspace=.085)',
                              'fig.subplots_adjust(left=.10, right=.96, top=.83, bottom=.24)'))
    render_ok = check(tsne_pdf, ('PATCH_LAYOUT_TSNE1D_OOF_20261103',
                                 'fig.tight_layout(rect=(.015, .145, .985, .91))',
                                 'fig.subplots_adjust(left=.055, right=.945, top=.81, bottom=.30, wspace=.085)'))
    if not check(geo, ('PATCH_LAYOUT_TSNE1D_OOF_20261103',
                        'fig.subplots_adjust(left=.10, right=.96, top=.83, bottom=.24)')):
        render_ok = False
    entradas = (orq.count('verificar_layout_tsne(') >= 2
                and geo.count('verificar_layout_tsne(') >= 1
                and 'verificar_layout_tsne(' in executor)
    if guard_ok and render_ok and entradas:
        ok('GUARDA_LAYOUT', 'TSNE02/TSNE14 e 1D TSNE13 protegidos no YAML, OOF, geometria e executor 3%.')
    else:
        falhas.append('guarda_layout')
        erro('GUARDA_LAYOUT', 'Protecao ou renderizadores divergentes. Nao iniciar OOF.')

    if cfg:
        print('PARQUET_EXISTE=' + str((ROOT / cfg['dados']['arquivo']).is_file()).upper())
        print('PASTAS_SPLITS_MANIFESTOS=' + str(len(list((ROOT / 'data/splits').glob('*/manifesto_folds.json')))))
        if not (ROOT / cfg['dados']['arquivo']).is_file():
            falhas.append('parquet_ausente')
            erro('BASE_RAW', 'Parquet nao encontrado no caminho YAML.')
        if not list((ROOT / 'data/splits').glob('*/manifesto_folds.json')):
            falhas.append('splits_ausentes')
            erro('FOLDS_MASTER', 'Nenhum manifesto de folds; V2 deve validar hashes antes do OOF.')

    # Adaptacoes previstas e NAO tratadas como falha da V1.
    gmm01 = fontes.get('src/visualizacao_gmm01.py', '')
    integrada = fontes.get('src/visualizacao_integrada.py', '')
    print('\n=== ADAPTACOES OBRIGATORIAS PARA V2 (SEM EXECUTAR) ===')
    if "cfg['tsne']['n_components'] = [1]" in executor or "int(x['reducao']['n_components']) == 1" in executor:
        aviso(*AVISOS_V2[0])
    else:
        ok('EXECUTOR_2D', 'Nao encontrado bloqueio explicito 1D (ainda exige testes).')
    if check(gmm01, ("ax3d.set_xlabel('PC1'", "ax3d.set_ylabel('PC2'",
                      'GMM01_densidade_conjunta_superficie_PCA2D')):
        aviso(*AVISOS_V2[1])
    else:
        ok('GMM01_SUPERFICIE_METODO', 'Nao encontrados rotulos PCA fixos no GMM01.')
    if check(integrada, ("and escolha['configuracao']['metodo'].upper() == 'PCA'",
                         'GMM01_densidade_conjunta_2d_PC1_PC2')):
        aviso(*AVISOS_V2[2])
    else:
        ok('GALERIA_GMM01_2D', 'Nao encontrada restricao PCA exclusiva (exige testes).')
    aviso(*AVISOS_V2[3])
    print('\n=== RESULTADO V1 ===')
    print('TREINAMENTO_EXECUTADO=NAO')
    print('F10_LIDO=NAO')
    print('YAML_MODIFICADO=NAO')
    print('ARQUIVOS_GERADOS_PELA_AUDITORIA=NAO')
    if falhas:
        print('V1_STATUS=BLOQUEADA_AMBIENTE_OU_FONTES')
        print('FALHAS=' + ','.join(sorted(set(falhas))))
        return 2
    print('V1_STATUS=APROVADA_COM_AJUSTES_OBRIGATORIOS_NA_V2')
    return 0

if __name__ == '__main__':
    try:
        sys.exit(run())
    except Exception as exc:
        print(f'[FALHA_V1] ERRO_NAO_PREVISTO: {type(exc).__name__}: {exc}')
        print('V1_STATUS=BLOQUEADA_AMBIENTE_OU_FONTES')
        sys.exit(3)
