
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def avaliar_modelos(
    melhor_knn,
    melhor_arvore,
    X_test,
    y_test
):

    print("\n==== FASE 7 — AVALIAÇÃO FINAL ====")

    modelos = {
        "KNN": melhor_knn["modelo"],
        "Árvore de Decisão": melhor_arvore["modelo"]
    }

    resultados_finais = {}

    for nome, modelo in modelos.items():

        # Prever a classe dos registros de teste
        previsoes = modelo.predict(X_test)

        # Calcular as métricas de desempenho
        acuracia = accuracy_score(y_test, previsoes)

        acuracia_balanceada = balanced_accuracy_score(
            y_test, previsoes
        )

        precisao = precision_score(
            y_test, previsoes, zero_division=0
        )

        recall = recall_score(
            y_test, previsoes, zero_division=0
        )

        f1 = f1_score(
            y_test, previsoes, zero_division=0
        )

        matriz = confusion_matrix(
            y_test, previsoes, labels=[0, 1]
        )

        resultados_finais[nome] = {
            "acuracia": acuracia,
            "acuracia_balanceada": acuracia_balanceada,
            "precisao": precisao,
            "recall": recall,
            "f1": f1,
            "matriz": matriz
        }

        print(f"\n==== {nome} ====")

        print(f"Acurácia: {acuracia:.4f}")
        print(
            f"Acurácia balanceada: "
            f"{acuracia_balanceada:.4f}"
        )
        print(f"Precisão: {precisao:.4f}")
        print(f"Recall falhas: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        print("\nMatriz de confusão:")
        print(matriz)

        print("\nRelatório de classificação:")
        print(
            classification_report(
                y_test,
                previsoes,
                zero_division=0
            )
        )

    return resultados_finais


def comparar_modelos(resultados):

    print("\n==== COMPARAÇÃO FINAL ====")

    for nome, metricas in resultados.items():
        print(f"\n{nome}")
        print(f"Acurácia: {metricas['acuracia']:.2%}")
        print(f"Recall falhas: {metricas['recall']:.2%}")
        print(f"F1-score: {metricas['f1']:.2%}")

    # Identifica a maior acurácia geral
    melhor_acuracia = max(
        resultados,
        key=lambda nome: resultados[nome]["acuracia"]
    )

    # Para manutenção preditiva, priorizamos
    # a detecção das máquinas com falha.
    # Este é um critério de negócio assumido,
    # que deverá ser validado pela empresa.
    melhor_deteccao = max(
        resultados,
        key=lambda nome: resultados[nome]["recall"]
    )

    print("\n==== VEREDITO FINAL ====")

    print(
        f"Maior acurácia geral: {melhor_acuracia}"
    )

    print(
        f"Maior recall de falhas: {melhor_deteccao}"
    )

    print(
        "\nModelo recomendado para o cenário "
        "em que detectar falhas é prioritário:"
    )
    print(melhor_deteccao)

    print(
        "\nJustificativa: o recall mede a proporção "
        "de falhas reais identificadas. "
        "A recomendação considera a necessidade "
        "de reduzir falhas não detectadas, mas "
        "deve também levar em conta os falsos "
        "alarmes e seus custos operacionais."
    )
