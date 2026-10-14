import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# Remove a feature para recalculá-la após a imputação
class PrepararVariaveis(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(
            columns=["diferenca_temperatura_k"],
            errors="ignore"
        ).copy()


# Preenche os valores ausentes antes de calcular a diferença
class ImputarNumericas(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.colunas_media = [
            "temperatura_ar_k",
            "temperatura_processo_k",
            "torque_nm"
        ]

        self.colunas_mediana = [
            "velocidade_rotacao_rpm"
        ]

        self.imputador_media = SimpleImputer(strategy="mean")
        self.imputador_mediana = SimpleImputer(strategy="median")

        self.imputador_media.fit(X[self.colunas_media])
        self.imputador_mediana.fit(X[self.colunas_mediana])

        return self

    def transform(self, X):
        X = X.copy()

        X[self.colunas_media] = self.imputador_media.transform(
            X[self.colunas_media]
        )

        X[self.colunas_mediana] = self.imputador_mediana.transform(
            X[self.colunas_mediana]
        )

        return X


# Calcula opcionalmente a nova variável
class DiferencaTemperatura(BaseEstimator, TransformerMixin):

    def __init__(self, ativar=True):
        self.ativar = ativar

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.ativar:
            X["diferenca_temperatura_k"] = (
                X["temperatura_processo_k"]
                - X["temperatura_ar_k"]
            )

        return X


# Configura o pipeline completo
def criar_preprocessador(X_train=None, usar_nova_feature=True):

    colunas_numericas = [
        "temperatura_ar_k",
        "temperatura_processo_k",
        "velocidade_rotacao_rpm",
        "torque_nm",
        "desgaste_ferramenta_min"
    ]

    if usar_nova_feature:
        colunas_numericas.append("diferenca_temperatura_k")

    transformador_colunas = ColumnTransformer(
        transformers=[
            ("numericas", "passthrough", colunas_numericas),
            ( "categoricas", OneHotEncoder(handle_unknown="ignore"), ["tipo"])
],
        remainder="drop"
    )

    return Pipeline(steps=[
        ("preparar", PrepararVariaveis()),
        ("imputar", ImputarNumericas()),
        ("nova_feature", DiferencaTemperatura(ativar=usar_nova_feature)),
        ("colunas", transformador_colunas)
    ])


# Mostra as configurações
def validar_preprocessador(preprocessador, X_train):

    print("\n==== PRÉ-PROCESSAMENTO ====")
    print("Quantidade de registros:", len(X_train))

    print("\nVariáveis de entrada:")
    print(X_train.columns.tolist())

    print("\nValores ausentes antes do tratamento:")
    print(X_train.isna().sum())

    print("\nTransformações configuradas:")

    for nome, transformador, colunas in (
        preprocessador.named_steps["colunas"].transformers
    ):
        print(f"{nome}: {colunas}")


# Testa os dois cenários
def testar_preprocessamento(preprocessador, X_train, X_test):

    print("\n==== TESTE DO PRÉ-PROCESSAMENTO ====")

    for usar_feature in [False, True]:

        nome = "COM" if usar_feature else "SEM"

        print(f"\n--- EXPERIMENTO {nome} NOVA FEATURE ---")

        pipeline = criar_preprocessador(
            X_train,
            usar_nova_feature=usar_feature
        )

        # Aprende apenas com treinamento
        X_train_processado = pipeline.fit_transform(X_train)

        # Aplica as mesmas regras ao teste
        X_test_processado = pipeline.transform(X_test)

        print("Formato treino:", X_train_processado.shape)
        print("Formato teste:", X_test_processado.shape)

        print(
            "Ausentes treino:",
            np.isnan(X_train_processado.toarray()).sum()
            if hasattr(X_train_processado, "toarray")
            else np.isnan(X_train_processado).sum()
        )

        print(
            "Ausentes teste:",
            np.isnan(X_test_processado.toarray()).sum()
            if hasattr(X_test_processado, "toarray")
            else np.isnan(X_test_processado).sum()
        )
