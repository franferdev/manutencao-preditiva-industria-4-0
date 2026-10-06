"""
Projeto de Manutenção Preditiva - Indústria 4.0
Módulo 1 - Análise Preditiva com Python
"""

from src.carregamento import carregar_dados


caminho_dataset = "data/manutencao_preditiva.csv"

df = carregar_dados(caminho_dataset)

print(df.head())