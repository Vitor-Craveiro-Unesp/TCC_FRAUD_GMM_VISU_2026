import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, log_loss, matthews_corrcoef

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/execucao/ESPELHAR_VISUAIS_F10_NO_ENSAIO.py'
spec = importlib.util.spec_from_file_location('espelho', SCRIPTS)
espelho = importlib.util.module_from_spec(spec)
spec.loader.exec_module(espelho)


def test_catalogo_seis_figuras():
    assert len(espelho.GRAFICOS) == 6
    assert '02_mcc_por_limiar.png' in espelho.GRAFICOS


def test_nome_html_respeita_layout():
    html = espelho.gerar_html(Path('nada'), Path('nada'), 'a'*64, 'b'*64)
    assert '../../visualizacao/oof/01_curva_precision_recall.png' in html
    assert '../graficos_gerados/01_curva_precision_recall.png' in html
    assert 'F10' in html


def test_nao_substitui_arquivo(tmp_path):
    o = tmp_path / 'origem.txt'; d = tmp_path / 'destino.txt'
    o.write_text('o',encoding='utf-8');d.write_text('d',encoding='utf-8')
    try:
        espelho.copiar_sem_sobrescrever(o,d)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Sobrescrita deveria ter sido bloqueada')
    assert d.read_text() == 'd'


def test_rotulos_trocados_sem_renomear_corte_oof():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.text(.3,.3,'Resultados da validação OOF; τ* MCC')
    espelho.alterar_textos_figura(fig)
    texts=[t.get_text() for t in fig.texts]
    assert any('Resultados do teste F10' in t for t in texts)
    plt.close(fig)


def test_seis_graficos_reais_com_dados_sinteticos(tmp_path):
    # Teste isolado; não carrega dados do F10 real.
    projeto = Path(__file__).resolve().parents[1]
    source = projeto if (projeto / 'src/visualizacao_oof.py').is_file() else Path('/mnt/data/source_tcc')
    if not (source / 'src/visualizacao_oof.py').is_file():
        import pytest
        pytest.skip('O codigo de visualizacao_oof.py precisa estar instalado no projeto')
    sys.path.insert(0,str(source))
    try:
        rng=np.random.default_rng(432)
        n=450
        y=np.r_[np.zeros(430,dtype=int),np.ones(20,dtype=int)]
        p=np.clip(rng.beta(.5,30,n),1e-8,.6)
        p[-20:]=np.clip(rng.beta(1.5,12,20),1e-8,.6)
        tau=.01
        met={
            'n':n, 'ap':float(average_precision_score(y,p)),
            'mcc_limiar_oof':float(matthews_corrcoef(y,p>=tau)),
            'logloss':float(log_loss(y,p,labels=[0,1])),
        }
        met['score_composto_limiar_oof']=(met['ap']+(met['mcc_limiar_oof']+1)/2+np.exp(-met['logloss']))/3
        cfg = projeto / 'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
        if not cfg.is_file():
            cfg = Path('/mnt/data/config_TCC_EXPERIMENTAL_V3_atualizado(7).yaml')
        config=yaml.safe_load(cfg.read_text(encoding='utf-8-sig'))
        dados={'config':config,'escolha':{'limiar_mcc_oof':tau}, 'tabela':pd.DataFrame({'y_true':y,'prob_fraude':p}), 'met':met}
        espelho.figuras_usando_mesmo_codigo(dados,tmp_path)
        for fname in espelho.GRAFICOS:
            assert (tmp_path/fname).stat().st_size>1000
    finally:
        sys.path.remove(str(source))


def test_bloqueia_vincular_f10_a_outro_ensaio(tmp_path, monkeypatch):
    root=tmp_path
    antigo=root/'outputs/ensaio_original'; antigo.mkdir(parents=True)
    escolha=antigo/'vencedor/metricas/melhor_configuracao.json'
    escolha.parent.mkdir(parents=True)
    escolha.write_text(json.dumps({'id_candidato':'id_1', 'limiar_mcc_oof':0.01}),encoding='utf-8')
    pro=root/'outputs/pre/manifesto.json'; pro.parent.mkdir(parents=True)
    pro.write_text(json.dumps({'modelos':[{'pca_dimensao':1,'candidato_oof':'id_1',
           'limiar_mcc_oof_congelado':0.01,'experimento_oof':'outputs/ensaio_original'}]}),encoding='utf-8')
    fake=root/'outputs/ensaio_novo';fake.mkdir()
    monkeypatch.setattr(espelho,'RAIZ',root)
    try:
        espelho.conferir_vinculo({'referencia_protocolo':'outputs/pre/manifesto.json'},fake)
    except RuntimeError as exc:
        assert 'OUTRO ENSAIO' in str(exc)
    else:
        raise AssertionError('F10 de um outro experimento foi indevidamente aceito')


def test_raiz_disponivel_para_importacao_src_sem_pythonpath(tmp_path):
    """Simula execucao direta em projeto isolado, sem gravar no TCC real."""
    import subprocess
    import os
    import shutil

    projeto = Path(__file__).resolve().parents[1]
    src = projeto / 'src'
    if not (src / 'visualizacao_oof.py').is_file():
        import pytest
        pytest.skip('Disponivel no TCC real com src instalado')

    raiz_teste = tmp_path / 'projeto'
    script_dir = raiz_teste / 'scripts/execucao'
    src_dir = raiz_teste / 'src'
    script_dir.mkdir(parents=True)
    src_dir.mkdir()
    shutil.copy2(projeto/'scripts/execucao/ESPELHAR_VISUAIS_F10_NO_ENSAIO.py',
                 script_dir/'ESPELHAR_VISUAIS_F10_NO_ENSAIO.py')
    for nome in ('__init__.py', 'estrutura_saida.py', 'visualizacao_oof.py'):
        shutil.copy2(src/nome, src_dir/nome)
    runner = script_dir / 'teste_import.py'
    runner.write_text(
        'import ESPELHAR_VISUAIS_F10_NO_ENSAIO\n'
        'from src import visualizacao_oof\n'
        'print("IMPORT_OK")\n', encoding='utf-8')
    env = {k:v for k,v in os.environ.items() if k != 'PYTHONPATH'}
    resultado = subprocess.run(
        [sys.executable, str(runner)], cwd=str(raiz_teste),
        capture_output=True, text=True, env=env, timeout=60)
    assert resultado.returncode == 0, resultado.stderr
    assert 'IMPORT_OK' in resultado.stdout
