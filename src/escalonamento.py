import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessamento import criar_preprocessador


def preparar_dados_modelos(X_train, X_test):

    # Fase 5: Preparação para o KNN
    # Reutilizamos o pré-processador da Fase 4.
    preprocessador_knn = criar_preprocessador(
        usar_nova_feature=True
    )

    # As seis primeiras colunas são numéricas.
    # As demais são categorias do OneHotEncoder.
    escalonador = ColumnTransformer(
        transformers=[
            (
                "numericas",
                StandardScaler(),
                list(range(6))
            )
        ],
        remainder="passthrough",
        sparse_threshold=0
    )

    pipeline_knn = Pipeline(steps=[
        ("preprocessamento", preprocessador_knn),
        ("escalonamento", escalonador)
    ])

    # fit_transform: aprende somente no treino.
    X_train_knn = pipeline_knn.fit_transform(X_train)

    # transform: aplica no teste os parâmetros
    # aprendidos durante o treinamento.
    X_test_knn = pipeline_knn.transform(X_test)

    # Preparação para a Árvore de Decisão
    # A Árvore usa limites de divisão, não
    # distâncias. Portanto não necessita escala.
    pipeline_arvore = criar_preprocessador(
        usar_nova_feature=True
    )

    X_train_arvore = pipeline_arvore.fit_transform(
        X_train
    )

    X_test_arvore = pipeline_arvore.transform(
        X_test
    )

    # Validação
    print("\n==== FASE 5 — ESCALONAMENTO ====")

    print("\nKNN:")
    print("Treino:", X_train_knn.shape)
    print("Teste:", X_test_knn.shape)

    print("\nÁrvore de Decisão:")
    print("Treino:", X_train_arvore.shape)
    print("Teste:", X_test_arvore.shape)

    print("\nMédias das 6 variáveis escalonadas:")
    print(
        np.asarray(X_train_knn[:, :6])
        .mean(axis=0).round(4)
    )

    print("\nDesvios-padrão:")
    print(
        np.asarray(X_train_knn[:, :6])
        .std(axis=0).round(4)
    )

    return (
        X_train_knn,
        X_test_knn,
        X_train_arvore,
        X_test_arvore
    )
