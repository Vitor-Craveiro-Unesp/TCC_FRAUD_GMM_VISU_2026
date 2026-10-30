# -*- coding: utf-8 -*-
"""Ensaios sinteticos da instalacao e do isolamento do F10."""
import ast
from pathlib import Path
import tempfile
import types
import sys
import importlib.util
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def importar_isolado(path,mod_name):
    spec=importlib.util.spec_from_file_location(mod_name,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_modulos_compilam_e_orquestrador_toma_decisao_apos_oof():
    for nome in ('orquestracao_oof.py','finalizacao_automatica_pca1d.py'):
        ast.parse((ROOT/'PACOTE_INSTALACAO'/nome).read_text(encoding='utf-8'))
    tree=ast.parse((ROOT/'PACOTE_INSTALACAO/orquestracao_oof.py').read_text(encoding='utf-8'))
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    # Garantir ordem: treino/OOF -> visualizacao -> teste final
    txt=ast.unparse(f)
    assert txt.index('executar_grade(')<txt.index('executar_visualizacoes(final)')<txt.index('concluir_ensaio(final)')
    assert 'if automatizar_f10 and a.executar:' in txt
    assert '--concluir-teste-final' in txt


def test_yaml_preserva_parametros_e_ativa_somente_pca1d():
    i=importar_isolado(ROOT/'INSTALAR_AUTOMACAO_PCA1D.py','installer_unit')
    # Os dois formatos 1D/2D são convertidos, preservando campos adicionais
    base='''pca:\n  n_components:\n  - 2\n  whiten: false\nexecucao:\n  n_jobs: -1\n  usar_cache: true\n'''
    # func exige campos de metodologia, usar config real do upload de referência
    import yaml
    raw=(ROOT/'tests/configuracao_referencia.yaml').read_bytes()
    updated=i.corrigir_yaml(raw)
    cfg=yaml.safe_load(updated)
    assert cfg['pca']['n_components']==[1]
    assert cfg['execucao']['finalizacao_pca1d_automatica'] is True
    assert yaml.safe_load(raw)['dados']['fracao_base']==cfg['dados']['fracao_base']


def test_amostra_f10_estratificada_deterministica_sem_acesso_treino():
    p=importar_isolado(ROOT/'PACOTE_INSTALACAO/finalizacao_automatica_pca1d.py','fin_unit')
    with tempfile.TemporaryDirectory() as t:
        root=Path(t)
        folded=np.r_[np.tile(np.arange(1,10),10),np.repeat(10,200)]
        y=np.zeros(len(folded),dtype=np.uint8)
        # 150 legitimate and 50 fraud inside F10
        y[-50:]=1
        fake=types.ModuleType('src.orquestracao_oof')
        fake._ler_alvo=lambda parquet,alvo:y
        src=types.ModuleType('src');src.orquestracao_oof=fake
        orig_src=sys.modules.get('src');orig_orq=sys.modules.get('src.orquestracao_oof')
        sys.modules['src']=src;sys.modules['src.orquestracao_oof']=fake
        try:
            state={'orig_folds':folded,'config':{'dados':{'coluna_alvo':'TARGET_OMEGA'}},'parquet':'dummy'}
            a=p.selecionar_amostra_f10(state,root)
            b=p.selecionar_amostra_f10(state,root)
            assert np.array_equal(a,b)
            assert len(a)==20 and int(y[a].sum())==5
            assert np.all(folded[a]==10)
            save=root/'vencedor/teste_final/indices_f10_amostra.npy'
            save.parent.mkdir(parents=True)
            np.save(save,a)
            fake._ler_alvo=lambda *args:(_ for _ in ()).throw(AssertionError('TARGET reaberto'))
            assert np.array_equal(p.selecionar_amostra_f10(state,root),a)
        finally:
            if orig_src is None:sys.modules.pop('src',None)
            else:sys.modules['src']=orig_src
            if orig_orq is None:sys.modules.pop('src.orquestracao_oof',None)
            else:sys.modules['src.orquestracao_oof']=orig_orq


def test_instalador_recusa_orquestrador_divergente():
    i=importar_isolado(ROOT/'INSTALAR_AUTOMACAO_PCA1D.py','installer_guard')
    assert i.sha(b'arquivo_modificado') not in {i.ORIGINAL_HASH,i.sha(i.TEMPLATE_ORQ.read_bytes())}


def test_padrao_de_saida_por_ensaio_sem_caminho_historico_fixo():
    txt=(ROOT/'PACOTE_INSTALACAO/finalizacao_automatica_pca1d.py').read_text(encoding='utf-8')
    assert 'avaliacao_final_f10_pca_10pct_20261029' not in txt
    assert "folder/'vencedor/teste_final'" in txt
    assert "'galeria/index.html'" in txt
    assert 'limiar_mcc_oof' in txt
