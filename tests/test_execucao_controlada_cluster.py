"""Tests without loading data, fitting models or using a real SLURM cluster."""
from datetime import datetime
import os
from pathlib import Path

import pytest

from src import execucao_controlada as ctl
from src import orquestracao_oof as orq
from src.recursos_cluster import n_jobs_efetivo


def test_cpu_cap_respeita_slurm_sem_mudar_yaml(monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '3')
    assert n_jobs_efetivo(-1) == 3
    assert n_jobs_efetivo(2) == 2
    with pytest.raises(RuntimeError, match='excede'):
        n_jobs_efetivo(4)
    monkeypatch.delenv('SLURM_CPUS_PER_TASK')
    assert n_jobs_efetivo(2) == 2


def test_snapshot_preserva_original_apos_edicao(tmp_path):
    origem = tmp_path / ctl.NOME_YAML
    origem.write_bytes(b'reducao:\n  metodo: PCA\n')
    snapshot = ctl.criar_snapshot(tmp_path, origem)
    origem.write_bytes(b'reducao:\n  metodo: TSNE\n')
    assert snapshot.read_bytes() == b'reducao:\n  metodo: PCA\n'
    assert (snapshot.parent / 'snapshot.json').exists()


def test_bloqueio_impede_duas_execucoes_no_mesmo_projeto(tmp_path):
    with ctl.bloquear_execucoes(tmp_path):
        with pytest.raises(RuntimeError, match='Outro processo'):
            with ctl.bloquear_execucoes(tmp_path):
                pass
    with ctl.bloquear_execucoes(tmp_path):
        pass


def test_lancador_nova_execucao_congela_yaml_e_envia_orquestrador(tmp_path, monkeypatch):
    raiz = tmp_path
    (raiz / 'src').mkdir()
    monkeypatch.setattr(ctl, '__file__', str(raiz / 'src' / 'execucao_controlada.py'))
    fonte = raiz / ctl.NOME_YAML
    fonte.write_text('execucao:\n  n_jobs: -1\n  retomar: false\n  diretorio_retomada: null\n', encoding='utf-8')
    monkeypatch.setattr(orq, 'validar_operacao', lambda cfg: True)
    chamadas = []

    def simular_orquestrador(argv, *, raiz):
        chamadas.append((list(argv), raiz))
        assert '--nova-execucao' in argv
        path = Path(argv[argv.index('--config') + 1])
        assert path.read_text(encoding='utf-8') == fonte.read_text(encoding='utf-8')
        assert path != fonte
        assert 'outputs/.submissoes' in path.as_posix()

    monkeypatch.setattr(orq, 'main', simular_orquestrador)
    assert ctl.main(['--executar']) == 0
    assert len(chamadas) == 1
    assert fonte.read_text(encoding='utf-8').startswith('execucao:')


def test_retoma_somente_com_permissao_e_snapshot(tmp_path, monkeypatch):
    raiz = tmp_path
    (raiz / 'src').mkdir()
    monkeypatch.setattr(ctl, '__file__', str(raiz / 'src' / 'execucao_controlada.py'))
    yaml = raiz / 'original.yaml'
    yaml.write_text('execucao:\n  n_jobs: -1\n  retomar: true\n  usar_cache: true\n  diretorio_retomada: null\n', encoding='utf-8')
    exp = raiz / 'outputs' / 'em_andamento_ensaio_pca_gmm_20261104_120000'
    exp.mkdir(parents=True)
    monkeypatch.setattr(orq, 'validar_operacao', lambda cfg: True)
    chamado = []
    monkeypatch.setattr(orq, 'main', lambda argv, *, raiz: chamado.append(list(argv)))
    assert ctl.main(['--retomar', '--config', str(yaml), '--experimento', str(exp)]) == 0
    assert '--retomar-execucao' in chamado[0]
    with pytest.raises(SystemExit):
        ctl.main(['--retomar', '--experimento', str(exp)])


def _config_saida():
    return {
        'saidas': {'diretorio_base': 'outputs', 'timezone': 'UTC'},
        'reducao': {'metodo': 'TSNE'},
        'dados': {'fracao_base': 0.01},
        'execucao': {'retomar': False, 'diretorio_retomada': None, 'n_jobs': -1}
    }


def test_nova_pasta_mesmo_segundo_nao_colide(tmp_path, monkeypatch):
    monkeypatch.setattr(orq, '_agora', lambda config: datetime(2026, 11, 4, 12, 0, 0))
    cfg = _config_saida()
    a, _, _ = orq._abrir_execucao(tmp_path, cfg, 'chave1', ensaio=False, nova_execucao=True)
    b, _, _ = orq._abrir_execucao(tmp_path, cfg, 'chave2', ensaio=False, nova_execucao=True)
    assert a != b
    assert a.is_dir() and b.is_dir()


def test_finalizacao_nao_substitui_diretorio_preexistente(tmp_path, monkeypatch):
    monkeypatch.setattr(orq, '_agora', lambda config: datetime(2026, 11, 4, 12, 0, 0))
    cfg = _config_saida()
    (tmp_path / 'outputs').mkdir()
    origem_a = tmp_path / 'outputs' / 'em_andamento_a'
    origem_b = tmp_path / 'outputs' / 'em_andamento_b'
    origem_a.mkdir(); origem_b.mkdir()
    manifest_a = {'tipo': 'ensaio'}
    manifest_b = {'tipo': 'ensaio'}
    a = orq._finalizar(origem_a, manifest_a, cfg)
    (a / 'arquivo_testemunha.txt').write_text('nao sobrescrever', encoding='utf-8')
    b = orq._finalizar(origem_b, manifest_b, cfg)
    assert a != b
    assert (a / 'arquivo_testemunha.txt').read_text(encoding='utf-8') == 'nao sobrescrever'
    assert (b / 'manifesto.json').exists()


def test_grade_oof_interpreta_seis_modalidades_sem_fit():
    import yaml
    from src.validacao_oof import grade_candidatos_oof
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / ctl.NOME_YAML).read_text(encoding='utf-8-sig'))
    for method in ('PCA', 'TSNE'):
        for ndim in (1, 2, 3):
            cfg['reducao']['metodo'] = method
            cfg['pca']['n_components'] = [ndim]
            cfg['tsne']['n_components'] = ndim
            candidatos = list(grade_candidatos_oof(cfg))
            assert len(candidatos) == (1 if method == 'PCA' else 2)
            assert all(c['reducao']['n_components'] == ndim for c in candidatos)
    cfg['reducao']['metodo'] = 'PCA'
    cfg['gmm']['n_components'] = [2, 4, 6]
    assert [c['gmm']['n_components'] for c in grade_candidatos_oof(cfg)] == [2, 4, 6]
