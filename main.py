"""
Projeto de Manutenção Preditiva - Indústria 4.0
Módulo 1 - Análise Preditiva com Python
"""

from src.carregamento import carregar_dados
from src.eda import analisar_estrutura, verificar_valores_ausentes, analisar_variavel_alvo, plotar_distribuicao_variavel_alvo, plotar_histograma_rotacao, plotar_grafico_correlacao


caminho_dataset = "data/manutencao_preditiva.csv"

df = carregar_dados(caminho_dataset)

analisar_estrutura(df)
verificar_valores_ausentes(df)
analisar_variavel_alvo(df)
plotar_distribuicao_variavel_alvo(df)
plotar_histograma_rotacao(df)
plotar_grafico_correlacao(df)