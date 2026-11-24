"""Passo 03.03 — NORMAL/RUS para ajuste do GMM no TCC_EXPERIMENTAL_V3.

Este modulo recebe APENAS os rotulos do treino de um fold e devolve
POSICOES (indices .iloc) do subconjunto utilizado no ajuste do GMM.

IMPORTANTE:
- O proprio pipeline deve fornecer apenas F1-F8 (CV) ou F1-F9 (ajuste final).
- O RUS NAO modifica X/y originais e NAO deve ser aplicado a validacao/teste.
- Encoders e scaler sao ajustados no treino ORIGINAL antes da selecao do GMM.
- As responsabilidades e os riscos historicos RH_k devem ser calculados
  usando TODOS os registros originais do treino, nao apenas os selecionados.
- Este modulo nao treina o GMM, nao usa rotulos para atribuir risco e nao
  faz nenhuma correcao de prior.
"""

import numpy as np


def _opcao(configurado, escolhido, nome):
    """Resolve um parametro escalar/vetor sem escolher grid ambiguo."""
    opcoes = configurado if isinstance(configurado, list) else [configurado]
    if not opcoes:
        raise ValueError(f"{nome}: lista de opcoes vazia.")
    if escolhido is None:
        if len(opcoes) != 1:
            raise ValueError(f"{nome}: escolha um valor da grade {opcoes}.")
        return opcoes[0]
    if escolhido not in opcoes:
        raise ValueError(f"{nome}: {escolhido!r} nao consta no YAML: {opcoes!r}.")
    return escolhido


def selecionar_indices_gmm(y_treino, config, *, modo=None, razao=None, fold_validacao=None):
    """Devolve indices POSICIONAIS ordenados para X_treino.iloc[indices].

    NORMAL: devolve todas as linhas do treino na ordem original.
    RUS: preserva TODAS as positivas e sorteia ate
         razao_negativos_positivos * numero_de_positivas negativas sem reposicao.

    Parameters
    ----------
    y_treino : array-like, shape (n,)
        Apenas os rotulos do treinamento ORIGINAL do fold, antes do RUS.
    config : dict
        YAML oficial do projeto, com amostragem, dados e experimento.
    modo, razao : opcionais
        Se a configuracao contiver listas, explicitar a opcao escolhida.
        Em NORMAL, a razao e ignorada e nao constitui hiperparametro.
    fold_validacao : int | None
        Numero do fold de validacao, usado exclusivamente na semente;
        None/0 corresponde ao ajuste final do modelo de desenvolvimento.

    Returns
    -------
    np.ndarray[int64]
        Posicoes do treino original; NAO sao NID_ALPHA nem indices de DataFrame.
    """
    amostragem = config["amostragem"]
    if amostragem.get("aplicar_somente_treino") is not True:
        raise ValueError("RUS/NORMAL deve ser aplicado somente ao treino.")
    escolhido = _opcao(amostragem["modo"], modo, "amostragem.modo")
    if escolhido not in ("NORMAL", "RUS"):
        raise ValueError("amostragem.modo deve ser NORMAL ou RUS.")

    y = np.asarray(y_treino)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y_treino deve ser um vetor unidimensional nao vazio.")
    positiva = config["dados"]["classe_positiva"]
    if positiva != 1 or not np.issubdtype(y.dtype, np.number):
        raise ValueError("Este TCC exige alvo numerico 0/1, positivo=1.")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("y_treino deve conter somente 0 e 1, sem ausentes.")

    positivas = np.flatnonzero(y == 1)
    negativas = np.flatnonzero(y == 0)
    if not len(positivas) or not len(negativas):
        raise ValueError("O treino deve conter exemplos de ambas as classes.")

    if escolhido == "NORMAL":
        if razao is not None:
            raise ValueError("Em NORMAL, nao informe razao; ela nao integra a grade.")
        return np.arange(y.size, dtype=np.int64)

    proporcao = _opcao(
        amostragem["razao_negativos_positivos"],
        razao,
        "amostragem.razao_negativos_positivos",
    )
    if isinstance(proporcao, (bool, np.bool_)) or not isinstance(
        proporcao, (int, np.integer)
    ) or proporcao < 1:
        raise ValueError("RUS: razao_negativos_positivos deve ser inteiro >= 1.")
    alvo_negativas = min(len(negativas), int(proporcao) * len(positivas))
    if alvo_negativas == len(negativas):
        return np.arange(y.size, dtype=np.int64)  # Nunca repetir negativas.

    semente = config["experimento"]["semente"]
    if isinstance(semente, bool) or not isinstance(semente, (int, np.integer)) or semente < 0:
        raise ValueError("experimento.semente deve ser inteiro nao negativo.")
    if fold_validacao is None:
        fold_validacao = 0
    if isinstance(fold_validacao, bool) or not isinstance(
        fold_validacao, (int, np.integer)
    ) or fold_validacao < 0:
        raise ValueError("fold_validacao deve ser inteiro nao negativo.")

    gerador = np.random.default_rng(
        np.random.SeedSequence([int(semente), int(fold_validacao)])
    )
    negativas_selecionadas = gerador.choice(
        negativas, size=alvo_negativas, replace=False
    )
    return np.sort(np.concatenate((positivas, negativas_selecionadas))).astype(
        np.int64, copy=False
    )
