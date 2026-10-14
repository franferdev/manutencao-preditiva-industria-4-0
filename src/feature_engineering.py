def criar_features(df):
    """
    Cria novas variáveis para análise preditiva.
    """

    df = df.copy()

    # diferença entre temperatura do processo e do ar
    df["diferenca_temperatura_k"] = (df["temperatura_processo_k"] - df["temperatura_ar_k"])

    print("\n==== FEATURE ENGINEERING ====")

    print(df[[
        "temperatura_ar_k",
        "temperatura_processo_k",
        "diferenca_temperatura_k"
    ]].head())

    print("\n==== VALIDAÇÃO DA NOVA FEATURE ====")

    print("Total de registros:", len(df))

    print(
        "Valores ausentes:",
        df["diferenca_temperatura_k"].isna().sum()
    )

    print("\nEstatísticas da diferença de temperatura:")
    print(df["diferenca_temperatura_k"].describe())

    return df