
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score

from src.preprocessamento import criar_preprocessador


def treinar_arvore(X_train, X_test, y_train, y_test):

    print("\n==== FASE 6 — ÁRVORE DE DECISÃO ====")

    profundidades = [3, 5, None]
    resultados = []

    validacao = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    for profundidade in profundidades:

        # O balanceamento é aplicado apenas
        # no treinamento da Árvore.
        modelo = Pipeline(steps=[
            ("preprocessamento", criar_preprocessador()),
            ("arvore", DecisionTreeClassifier(
                max_depth=profundidade,
                class_weight="balanced",
                random_state=42
            ))
        ])

        scores_cv = cross_val_score(
            modelo,
            X_train,
            y_train,
            cv=validacao,
            scoring="balanced_accuracy"
        )

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
            "profundidade": profundidade,
            "acuracia_treino": acuracia_treino,
            "acuracia_teste": acuracia_teste,
            "recall_falhas": recall_falhas,
            "cv_balanceada": scores_cv.mean(),
            "modelo": modelo
        })

        print(f"\nProfundidade: {profundidade}")
        print(f"Acurácia treino: {acuracia_treino:.4f}")
        print(f"Acurácia teste: {acuracia_teste:.4f}")
        print(f"Recall falhas: {recall_falhas:.4f}")
        print(f"CV balanceada: {scores_cv.mean():.4f}")

    melhor = max(
        resultados,
        key=lambda resultado: resultado["cv_balanceada"]
    )

    print("\nMelhor Árvore pela validação cruzada:")
    print(f"Profundidade: {melhor['profundidade']}")

    return melhor, resultados
