"""Teste de seguranca: --executar nao deve prometer F10 t-SNE incompleto."""
from pathlib import Path
import pytest
import yaml
from src import orquestracao_oof as orq


def test_tsne_f10_3d_planejamento_sem_criar_outputs(tmp_path, capsys):
    raiz=tmp_path
    arq=raiz/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    modelo=Path(__file__).resolve().parents[1]/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    cfg=yaml.safe_load(modelo.read_text(encoding='utf-8-sig'))
    cfg['reducao']['metodo']='TSNE'
    cfg['tsne']['n_components']=[3]
    cfg['execucao']['finalizacao_pca1d_automatica']=True
    arq.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    assert orq.main(['--planejar'],raiz=raiz) is None
    assert 'ETAPA FINAL TSNE 3D' in capsys.readouterr().out
    assert not (raiz/'outputs').exists()


def test_tsne_planejar_sem_f10_nao_cria_artefatos(tmp_path,capsys):
    raiz=tmp_path
    arq=raiz/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    modelo=Path(__file__).resolve().parents[1]/'config_TCC_EXPERIMENTAL_V3_atualizado.yaml'
    cfg=yaml.safe_load(modelo.read_text(encoding='utf-8-sig'))
    cfg['reducao']['metodo']='TSNE'
    cfg['tsne']['n_components']=[2]
    cfg['execucao']['finalizacao_pca1d_automatica']=False
    arq.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    assert orq.main(['--planejar'],raiz=raiz) is None
    mensagem=capsys.readouterr().out
    assert 'nao executa modelo final/F10' in mensagem
    assert 'Nenhum modelo foi treinado' in mensagem
    assert not (raiz/'outputs').exists()
