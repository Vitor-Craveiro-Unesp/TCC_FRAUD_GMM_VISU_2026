"""Instala o GMM18 de forma conservadora (sem executar treinamento).

Extrair o ZIP na raiz do TCC_EXPERIMENTAL_V3 e executar:
  python instalar_grafico_em.py

Faz backup dos arquivos alterados fora do repositorio.
Se qualquer ancora nao corresponder ao codigo atual, nao altera nada.
"""
from pathlib import Path
from datetime import datetime
import shutil
import sys
import ast
import yaml


def unico(codigo: str, antigo: str, novo: str, nome: str) -> str:
    numero = codigo.count(antigo)
    if numero != 1:
        raise RuntimeError(
            f'{nome}: procurado um unico trecho de integracao, encontrados {numero}. '
            'Nenhum arquivo foi modificado. Envie esse arquivo para ajustar o patch.'
        )
    return codigo.replace(antigo, novo, 1)


def main():
    raiz = Path.cwd().resolve()
    pacote = Path(__file__).resolve().parent
    arquivos = {
        'src/gmm.py': raiz/'src/gmm.py',
        'src/visualizacao_geometrica.py': raiz/'src/visualizacao_geometrica.py',
        'src/visualizacao_integrada.py': raiz/'src/visualizacao_integrada.py',
        'config_TCC_EXPERIMENTAL_V3_atualizado.yaml': raiz/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml',
    }
    novos = ['src/gmm_convergencia_em.py', 'src/visualizacao_gmm18.py']
    for nome, caminho in arquivos.items():
        if not caminho.is_file():
            raise FileNotFoundError(f'Arquivo nao encontrado na raiz do TCC: {nome}')
    for nome in novos:
        if not (pacote/nome).is_file():
            raise FileNotFoundError(f'O pacote de instalacao esta incompleto: {nome}')
        if (raiz/nome).exists() and (raiz/nome).resolve() != (pacote/nome).resolve():
            raise FileExistsError(f'Arquivo novo ja existe no projeto: {nome}. Nao sobrescrever.')
    textos = {nome: p.read_text(encoding='utf-8-sig') for nome, p in arquivos.items()}
    if 'convergencia_em' in textos['src/visualizacao_geometrica.py']:
        raise RuntimeError('GMM18 ja aparece na geometria. Nao aplicar duas vezes.')

    # GMM: mesma execucao interna do sklearn; instrumentacao opt-in.
    nome = 'src/gmm.py'
    t = textos[nome]
    t = unico(t,
        'from sklearn.mixture import GaussianMixture\n',
        'from sklearn.mixture import GaussianMixture\n'
        'from src.gmm_convergencia_em import GaussianMixtureComHistorico\n', nome)
    t = unico(t,
        '    def fit(self, X):\n',
        '    def fit(self, X, *, registrar_historico_em=False):\n', nome)
    t = unico(t,
        '        modelo = GaussianMixture(\n',
        '        classe_gmm = GaussianMixtureComHistorico if registrar_historico_em else GaussianMixture\n'
        '        modelo = classe_gmm(\n', nome)
    textos[nome] = t

    # Geometria: registrar APENAS no ajuste final do vencedor.
    nome = 'src/visualizacao_geometrica.py'
    t = textos[nome]
    t = unico(t,
        "    'sensibilidade_fronteira', 'densidades_marginais', 'regioes_dominancia',\n)",
        "    'sensibilidade_fronteira', 'densidades_marginais', 'regioes_dominancia',\n"
        "    'convergencia_em',\n)", nome)
    t = unico(t,
        'NUM = {',
        "NUM = {'convergencia_em': 18, ", nome)
    t = unico(t,
        "    modelo = GMMNaoSupervisionado(config,cand['gmm']).fit(coord.iloc[amostra])\n",
        "    registrar_em = bool(config.get('visualizacao', {}).get('graficos', {})\n"
        "                       .get('convergencia_em', {}).get('habilitar', False))\n"
        "    modelo = GMMNaoSupervisionado(config,cand['gmm']).fit(\n"
        "        coord.iloc[amostra], registrar_historico_em=registrar_em)\n", nome)
    t = unico(t,
        '    from src import visualizacao_pca_biplot as biplot\n',
        '    from src import visualizacao_pca_biplot as biplot\n'
        '    from src import visualizacao_gmm18 as em_grafico\n', nome)
    t = unico(t,
        "        'regioes_dominancia':lambda:dominancia.gerar(cfg,data,destino,ext,dpi,csv),\n",
        "        'regioes_dominancia':lambda:dominancia.gerar(cfg,data,destino,ext,dpi,csv),\n"
        "        'convergencia_em':lambda:em_grafico.gerar(cfg,data,destino,ext,dpi,csv),\n", nome)
    textos[nome] = t

    # Galeria: exibir arquivo do GMM18 no grupo geometrico.
    nome = 'src/visualizacao_integrada.py'
    t = textos[nome]
    t = unico(t,
        "    'sensibilidade_fronteira': 'GMM07_sensibilidade_',\n",
        "    'sensibilidade_fronteira': 'GMM07_sensibilidade_',\n"
        "    'convergencia_em': 'GMM18_convergencia_em_inicializacoes',\n", nome)
    # Ajusta titulo OOF existente, se ainda estiver com a numeracao antiga.
    if '00–06 — Painel e validacao OOF' in t:
        t = t.replace('00–06 — Painel e validacao OOF',
                      '00–07 — Painel e validacao OOF', 1)
    textos[nome] = t

    # Habilitar junto aos graficos geometricos do desenvolvimento.
    nome = 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    t = textos[nome]
    trecho = (
        '    # GMM 01 — DENSIDADES MARGINAIS POR DIMENSAO (PDF)'
    )
    bloco = (
        '    # GMM 18 — CONVERGENCIA DO EM (TRES INICIALIZACOES)\n'
        '    # Usa o MESMO ajuste GMM do vencedor, apenas no desenvolvimento.\n'
        '    # Nao muda n_init, RNG, criterio OOF nem acessa F10.\n'
        '    convergencia_em:\n'
        '      habilitar: true\n'
        '      gerar_em: ["teste_final"]\n'
        '\n'
        '    # --------------------------------------------------------\n'
    )
    t = unico(t, trecho, bloco + trecho, nome)
    textos[nome] = t

    # Teste legado: o numero de tipos geometricos aumentou de 8 para 9.
    caminho_teste = raiz/'tests/test_visualizacao_geometrica.py'
    if caminho_teste.is_file():
        codigo_teste = caminho_teste.read_text(encoding='utf-8-sig')
        if "assert len(r['graficos'])==8" in codigo_teste:
            nome_teste = 'tests/test_visualizacao_geometrica.py'
            arquivos[nome_teste] = caminho_teste
            textos[nome_teste] = unico(
                codigo_teste,
                "assert len(r['graficos'])==8",
                "assert len(r['graficos'])==9", nome_teste)

    # Conferir tudo em memoria ANTES de escrever.
    for nome, texto in textos.items():
        if nome.endswith('.py'):
            ast.parse(texto, filename=nome)
    for nome in novos:
        ast.parse((pacote/nome).read_text(encoding='utf-8'), filename=nome)
    config = yaml.safe_load(textos['config_TCC_EXPERIMENTAL_V3_atualizado.yaml'])
    assert config['visualizacao']['graficos']['convergencia_em']['habilitar'] is True
    assert config['visualizacao']['graficos']['convergencia_em']['gerar_em'] == ['teste_final']

    # Copia dos fontes originais fora do repositorio.
    destino_backup = raiz.parent / 'TCC_EXPERIMENTAL_V3_BACKUPS' / (
        'antes_gmm18_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    destino_backup.mkdir(parents=True, exist_ok=False)
    for nome, caminho in arquivos.items():
        alvo = destino_backup/nome
        alvo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(caminho, alvo)

    # Agora efetuar mudancas; nao tocar em outputs.
    for nome, texto in textos.items():
        caminho = arquivos[nome]
        caminho.write_text(texto, encoding='utf-8', newline='\n')
        print('ATUALIZADO:', nome)
    for nome in novos:
        origem = pacote/nome
        destino = raiz/nome
        if origem.resolve() != destino.resolve():
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origem, destino)
        print('NOVO:', nome)
    print('\nBACKUP ORIGINAL:', destino_backup)
    print('GMM18 INSTALADO NO CODIGO. Nenhum modelo foi treinado.')
    print('Agora execute os testes antes do commit e do ensaio do zero.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('INSTALACAO NAO CONCLUIDA:', exc, file=sys.stderr)
        raise
