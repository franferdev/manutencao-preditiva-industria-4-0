def analisar_dados(df):
    print("\n====LINHAS DUPLICADAS====")
    print(f"Quantidade de linhas duplicadas: {df.duplicated().sum()}")

    print("\n==== ANÁLISE ESTATÍSTICA ====")
    colunas = ["temperatura_ar_k",
        "temperatura_processo_k",
        "velocidade_rotacao_rpm",
        "torque_nm"]
    print(df[colunas].describe())

    print("\n=== ASSIMETRIA ===")
    for col in colunas:
        print(f"{col}: {df[col].skew()}")
        
    #estrategia de tratamento de valores ausentes definida#
    print("\n==== ESTRATEGIA DE TRATAMENTO DE VALORES AUSENTES ====")
    estrategias = {
        "temperatura_ar_k": "mean",
        "temperatura_processo_k": "mean",
        "velocidade_rotacao_rpm": "median",
        "torque_nm": "mean"
    }
    for col, estrategia in estrategias.items():
        print(f"{col}: {estrategia}")
        
    #identificar possiveis outliers com base no desvio padrão#
    print("\n==== POSSÍVEIS OUTLIERS ====")
    for col in colunas:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        
        outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)]
        print(f"{col}: {len(outliers)} possíveis outliers")