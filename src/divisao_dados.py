from sklearn.model_selection import train_test_split

def dividir_dados(df):
    y = df["falha_maquina"] # variavel alvo
    X = df.drop(columns=["falha_maquina",
        "falha_twf",
        "falha_hdf",
        "falha_pwf",
        "falha_osf",
        "falha_rnf",
        "udi",
        "id_produto"
    ]).copy() # x = variaveis preditoras - drop. --> retirar o alvo e colunas que revelam a falha, e colunas que não são uteis para o modelo (udi e id_produto)

    # Separar treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # stratify=y --> garante que a proporção de classes seja mantida em ambos os conjuntos    
    

    print("\n==== DIVISÃO DE DADOS ====")
    print (f"treinamento: {X_train.shape}")
    print (f"teste: {X_test.shape}")

    print("\n==== DISTRIBUICAO DO ALVO ====")
    print(f"treinamento: {y_train.value_counts(normalize=True)}")
    print(f"teste: {y_test.value_counts(normalize=True)}")

    print("\n==== VARIAVEIS UTILIZADAS ====")
    print(f"variaveis preditoras: {X_train.columns.tolist()}")
    
    print("\n==== TIPOS DAS VARIÁVEIS ====")
    print(X.dtypes)
    
    print("\n==== QUANTIDADE POR CLASSE ====")
    print("Treinamento:")
    print(y_train.value_counts().sort_index())

    print("\nTeste:")
    print(y_test.value_counts().sort_index())
    
    return X_train, X_test, y_train, y_test