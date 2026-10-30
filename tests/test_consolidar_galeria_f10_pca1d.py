from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import pytest

MOD_PATH = Path(__file__).resolve().parents[1] / 'scripts/execucao/CONSOLIDAR_GALERIA_F10_PCA1D.py'
spec = importlib.util.spec_from_file_location('consolidador', MOD_PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def grava(f: Path, b: bytes = b'conteudo de teste'):
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b)
    return sha256(b).hexdigest()


def json_file(path: Path, obj: dict):
    grava(path, json.dumps(obj, ensure_ascii=False).encode())


def fixture_projeto(tmp_path, *, com_gmm=True):
    f10 = tmp_path / m.F10_REL
    oof = tmp_path / m.ENSAIO_REL / 'vencedor/visualizacao/galeria'
    grava(oof/'index.html', b'<html>OOF</html>')
    tau = .010524627848033968
    esp = {'n':18523,'limiar_mcc_congelado_oof':tau,'modelo_origem_oof':m.ENSAIO_REL.as_posix(), 'graficos':{}}
    geo = {'n_f10':18523,'tau_star_oof':tau,'graficos':{}}
    gmm = {'n_f10':18523,'tau_star_oof_congelado':tau,'arquivos':{}}
    pre = {'n_f10':18523,'tau_star_oof':tau,'arquivos_sha256':{}}
    met = {'n':18523,'limiar_mcc_oof_congelado':tau, 'limiar_otimizado_no_f10':False}
    esp['predicoes_sha256'] = grava(f10/m.PREDICOES, b'linha_original,fold\n1,10')
    esp['metricas_sha256'] = grava(f10/m.METRICAS, json.dumps(met).encode())
    for nome,_ in m.PREDITIVOS:
        esp['graficos'][nome] = grava(f10/'graficos_gerados'/nome)
    for nome,_ in m.GEOMETRICOS:
        geo['graficos'][nome] = grava(f10/'graficos_gerados'/nome)
    if com_gmm:
        for nome,_ in m.FRONTEIRAS:
            gmm['arquivos'][nome] = grava(f10/'graficos_gerados'/nome)
        json_file(f10/m.MANIFESTOS[2],gmm)
    grava(f10/m.PRE01_REL,b'<html>PRE01</html>')
    csv = 'PRE01_F10_todas_variaveis.csv'
    pre['arquivos_sha256'][csv] = grava(f10/'diagnosticos/PRE01_F10'/csv)
    json_file(f10/m.PRE01_MANIFESTO_REL, pre)
    json_file(f10/m.MANIFESTOS[0], esp)
    json_file(f10/m.MANIFESTOS[1], geo)
    return f10


def test_faltando_gmm_verificacao_nao_escreve(tmp_path):
    f10 = fixture_projeto(tmp_path, com_gmm=False)
    r = m.levantar(tmp_path)
    assert not r['pronto']
    assert any('GMM07' in x for x in r['faltantes'])
    assert not (f10/'galeria'/m.NOME_HTML).exists()


def test_estrito_bloqueia_sem_gmm(tmp_path):
    fixture_projeto(tmp_path, com_gmm=False)
    with pytest.raises(RuntimeError, match='Complementos ausentes'):
        m.levantar(tmp_path, estrito=True)


def test_gerar_preserva_originais_e_cria_galeria(tmp_path):
    f10 = fixture_projeto(tmp_path)
    original = (f10/'graficos_gerados'/m.PREDITIVOS[0][0]).read_bytes()
    estado = m.levantar(tmp_path, estrito=True)
    assert estado['pronto']
    m.consolidar(estado)
    html = (f10/'galeria'/m.NOME_HTML).read_text(encoding='utf-8')
    meta = json.loads((f10/m.NOME_MANIFESTO).read_text(encoding='utf-8'))
    assert 'PRE01 F10' in html
    assert 'GMM08' in html
    assert 'MCC' in html
    assert 'predicoes_f10.csv' in html
    assert meta['sem_nova_avaliacao_f10'] is True
    assert len(meta['arquivos_origem_sha256']) >= 12
    assert (f10/'graficos_gerados'/m.PREDITIVOS[0][0]).read_bytes() == original
    assert not (f10/'galeria/index.html').exists()


def test_sobrescrita_bloqueada(tmp_path):
    fixture_projeto(tmp_path)
    estado=m.levantar(tmp_path,estrito=True)
    m.consolidar(estado)
    with pytest.raises(RuntimeError, match='ja existe'):
        m.levantar(tmp_path,estrito=True)


def test_hash_diferente_bloqueia(tmp_path):
    f10=fixture_projeto(tmp_path)
    (f10/'graficos_gerados'/m.PREDITIVOS[0][0]).write_bytes(b'alterado')
    with pytest.raises(RuntimeError,match='SHA-256 divergente'):
        m.levantar(tmp_path,estrito=True)


def test_limiar_divergente_bloqueia(tmp_path):
    f10=fixture_projeto(tmp_path)
    f=f10/m.MANIFESTOS[2]
    data=json.loads(f.read_text())
    data['tau_star_oof_congelado']=.5
    json_file(f,data)
    with pytest.raises(RuntimeError,match='Tau OOF alterado'):
        m.levantar(tmp_path,estrito=True)


def test_fonte_oof_referenciada_sem_copiar(tmp_path):
    f10 = fixture_projeto(tmp_path)
    oof = tmp_path/m.ENSAIO_REL/'vencedor/visualizacao/galeria'
    grava(oof/'geometria_desenvolvimento'/'11_pca_cotovelo.png')
    estado = m.levantar(tmp_path,estrito=True)
    html = m.html_consolidado(estado)
    assert '11_pca_cotovelo.png' in html
    assert '../../visualizacao/galeria/geometria_desenvolvimento/11_pca_cotovelo.png' in html
    assert not (f10/'galeria'/m.NOME_HTML).exists()
