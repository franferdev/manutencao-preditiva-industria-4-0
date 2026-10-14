
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score

from src.preprocessamento import criar_preprocessador


def treinar_knn(X_train, X_test, y_train, y_test):

    print("\n==== FASE 6 — TREINAMENTO KNN ====")

    valores_k = [3, 5, 7]
    resultados = []

    validacao = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    for k in valores_k:

        escalonador = ColumnTransformer(
            transformers=[
                ("numericas", StandardScaler(), list(range(6)))
            ],
            remainder="passthrough",
            sparse_threshold=0
        )

        # O pipeline completo evita vazamento de dados
        # durante a validação cruzada.
        modelo = Pipeline(steps=[
            ("preprocessamento", criar_preprocessador()),
            ("escalonamento", escalonador),
            ("knn", KNeighborsClassifier(n_neighbors=k))
        ])

        # Seleção usando somente dados de treinamento
        scores_cv = cross_val_score(
            modelo,
            X_train,
            y_train,
            cv=validacao,
            scoring="balanced_accuracy"
        )

        # Treinamento com todos os dados de treino
        modelo.fit(X_train, y_train)

        previsoes_treino = modelo.predict(X_train)
        previsoes_teste = modelo.predict(X_test)

        acuracia_treino = accuracy_score(
            y_train, previsoes_treino
        )

        acuracia_teste = accuracy_score(
            y_test, previsoes_teste
        )

        recall_falhas = recall_score(
            y_test,
            previsoes_teste,
            zero_division=0
        )

        resultados.append({
            "k": k,
            "acuracia_treino": acuracia_treino,
            "acuracia_teste": acuracia_teste,
            "recall_falhas": recall_falhas,
            "cv_balanceada": scores_cv.mean(),
            "modelo": modelo
        })

        print(f"\nK = {k}")
        print(f"Acurácia treino: {acuracia_treino:.4f}")
        print(f"Acurácia teste: {acuracia_teste:.4f}")
        print(f"Recall falhas: {recall_falhas:.4f}")
        print(f"CV balanceada: {scores_cv.mean():.4f}")

    melhor = max(
        resultados,
        key=lambda resultado: resultado["cv_balanceada"]
    )

    print("\nMelhor KNN pela validação cruzada:")
    print(f"K = {melhor['k']}")

    return melhor, resultados
