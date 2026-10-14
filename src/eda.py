import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analisar_estrutura(df):
    """
    Analisa a estrutura do DataFrame fornecido e retorna informações sobre colunas, tipos de dados e valores ausentes.
    """
    # mostrar dimensões
    print(f"Dimensões do DataFrame: {df.shape}")
    # mostrar colunas
    print(f"Colunas do DataFrame: {df.columns.tolist()}")
    # mostrar tipos
    print(f"Tipos de dados do DataFrame: {df.dtypes.to_dict()})")
    # mostrar resumo estatístico
    print(f"Resumo estatístico do DataFrame:\n{df.describe()}")
    
#verificar valores ausentes
def verificar_valores_ausentes(df):
    print(f"Valores ausentes no DataFrame:\n{df.isnull().sum()}")
    
def analisar_variavel_alvo(df):
    """
    Analisa a variável alvo do DataFrame fornecido e retorna informações sobre sua distribuição.
    
    # calcular quantidade de 0 e 1
    qtd_0 = (df['falha_maquina'] == 0).sum()
    qtd_1 = (df['falha_maquina'] == 1).sum()
    # exibir o resultado
    print(f"Quantidade de 0: {qtd_0}")
    print(f"Quantidade de 1: {qtd_1}")
    
    dá o mesmo resultado que df['falha_maquina'].value_counts() que mostra a distribuição da variável alvo
    """  
    # mostrar distribuição da variável alvo
    print(f"Distribuição da variável alvo:\n{df['falha_maquina'].value_counts()}")
    

def plotar_distribuicao_variavel_alvo(df):
    """
    Plota a distribuição da variável alvo do DataFrame fornecido.
    """
    #grafico de barras da variável alvo
    contagem = df['falha_maquina'].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    contagem.plot(kind='bar')
    plt.title('Distribuição da Variável Alvo')
    plt.xlabel('Falha na Máquina')
    plt.ylabel('Contagem')
    plt.xticks(rotation=0)
    plt.show()
    
def plotar_histograma_rotacao(df):
    """
    Plota o histograma da variável 'velocidade_rotacao_rpm' do DataFrame fornecido.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(df['velocidade_rotacao_rpm'].dropna(), bins=30, edgecolor="black", color='blue', alpha=0.7)
    plt.title('Distribuição da Velocidade de Rotação')
    plt.xlabel('Velocidade de Rotação (RPM)')
    plt.ylabel('Frequência')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    
def plotar_grafico_correlacao(df):
    correlacao = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlacao, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title('Mapa de Correlação entre Variáveis Numéricas')
    plt.show()
