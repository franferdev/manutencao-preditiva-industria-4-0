"""
Projeto de Manutenção Preditiva - Indústria 4.0
Módulo 1 - Análise Preditiva com Python
"""

from src.eda import analisar_estrutura, verificar_valores_ausentes, analisar_variavel_alvo, plotar_distribuicao_variavel_alvo, plotar_histograma_rotacao, plotar_grafico_correlacao
from src.data_prep import analisar_dados
from src.feature_engineering import criar_features
from src.divisao_dados import dividir_dados
from src.preprocessamento import criar_preprocessador, testar_preprocessamento, validar_preprocessador
from src.carregamento import carregar_dados
from src.balanceamento import analisar_balanceamento
from src.escalonamento import preparar_dados_modelos
from src.treinamento_knn import treinar_knn
from src.treinamento_arvore import treinar_arvore

caminho_dataset = "data/manutencao_preditiva.csv"

df = carregar_dados(caminho_dataset)

analisar_estrutura(df)
verificar_valores_ausentes(df)
analisar_variavel_alvo(df)
plotar_distribuicao_variavel_alvo(df)
plotar_histograma_rotacao(df)
plotar_grafico_correlacao(df)
analisar_dados(df)
# Fase 3 - Feature Engineering
df = criar_features(df)

# Fase 4 - Divisão dos dados
X_train, X_test, y_train, y_test = dividir_dados(df)

# Fase 4 - Configuração do pré-processamento
preprocessador = criar_preprocessador(X_train)

# Validação da configuração
validar_preprocessador(preprocessador, X_train)

# Testar imputação e codificação das variáveis
testar_preprocessamento(
    preprocessador,
    X_train,
    X_test
)

# Fase 4 - Análise do balanceamento
analisar_balanceamento(y_train)

# Fase 5 - Escalonamento das variáveis
(
    X_train_knn,
    X_test_knn,
    X_train_arvore,
    X_test_arvore
) = preparar_dados_modelos(X_train, X_test)

# Fase 6 — Treinamento e ajuste de parâmetros

melhor_knn, resultados_knn = treinar_knn(
    X_train, X_test, y_train, y_test
)

melhor_arvore, resultados_arvore = treinar_arvore(
    X_train, X_test, y_train, y_test
)
