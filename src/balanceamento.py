from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def analisar_balanceamento(y_train):

    print("\n==== BALANCEAMENTO ====")

    print("\nDistribuição original:")
    print(y_train.value_counts().sort_index())

    classes = np.unique(y_train)

    pesos = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    pesos_classes = dict(zip(classes, pesos))

    print("\nPesos calculados:")
    for classe, peso in pesos_classes.items():
        print(f"Classe {classe}: {peso:.4f}")

    print("\nEstratégia:")
    print("Árvore de Decisão: class_weight='balanced'")
    print("KNN: inicialmente sem balanceamento")

    return pesos_classes