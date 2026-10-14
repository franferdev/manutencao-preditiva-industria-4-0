import pandas as pd


def carregar_dados(caminho):
    df = pd.read_csv(caminho)
    return df