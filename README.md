# manutencao-preditiva-industria-4-0
Pipeline de Machine Learning para previsão de falhas mecânicas em equipamentos industriais.

## Fase 1 — Análise Exploratória dos Dados

### Estrutura do dataset
- 10.000 linhas
- 14 colunas

### Valores ausentes
Foram identificados valores ausentes nas seguintes variáveis:

- `temperatura_ar_k`: 500 valores ausentes
- `temperatura_processo_k`: 500 valores ausentes
- `velocidade_rotacao_rpm`: 500 valores ausentes
- `torque_nm`: 500 valores ausentes

As demais colunas não apresentaram valores ausentes.

### Variável alvo
A variável alvo do projeto é `falha_maquina`.

Ela possui duas classes:
- `0`: funcionamento normal da máquina
- `1`: ocorrência de falha mecânica

A distribuição encontrada foi:
- Classe 0: 9.661 registros — 96,61%
- Classe 1: 339 registros — 3,39%

### Principais padrões observados
Os dados da variável alvo estão fortemente desbalanceados, pois a maior parte dos registros pertence à classe 0.

O histograma da variável `velocidade_rotacao_rpm` mostrou maior concentração de valores aproximadamente entre 1.400 e 1.650 RPM, com uma distribuição assimétrica à direita e presença de alguns valores mais altos.

No mapa de correlação, foram observadas algumas relações importantes entre as variáveis numéricas:

- `temperatura_ar_k` e `temperatura_processo_k` apresentaram forte correlação positiva, aproximadamente 0,88;
- `velocidade_rotacao_rpm` e `torque_nm` apresentaram forte correlação negativa, aproximadamente -0,88;
- `torque_nm` apresentou correlação positiva fraca com `falha_maquina`, aproximadamente 0,19;
- `desgaste_ferramenta_min` apresentou correlação positiva fraca com `falha_maquina`, aproximadamente 0,11.

As colunas `falha_twf`, `falha_hdf`, `falha_pwf`, `falha_osf` e `falha_rnf` apresentaram relação com a variável alvo, porém não serão utilizadas como variáveis preditoras, pois representam motivos históricos das falhas e foram indicadas pelo Departamento de Engenharia apenas para consulta.

### Impacto na modelagem
A análise exploratória indica alguns cuidados importantes para as próximas etapas do projeto.

O forte desbalanceamento da variável alvo mostra que será necessário utilizar uma técnica de balanceamento durante o treinamento do modelo, para evitar que ele favoreça excessivamente a classe 0.

Os valores ausentes encontrados nas variáveis numéricas precisarão ser tratados na etapa de limpeza dos dados. A escolha entre média ou mediana deverá considerar a distribuição de cada variável.

A assimetria observada na variável `velocidade_rotacao_rpm` também deverá ser considerada na análise de outliers e na escolha da técnica de imputação.

As correlações identificadas ajudam a entender as relações entre as variáveis, mas não serão utilizadas isoladamente para decidir quais atributos serão importantes para o modelo.


## Execução do projeto

Execute os comandos na raiz do repositório. O arquivo de dados utilizado é `data/manutencao_preditiva.csv`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

A execução normal abre os gráficos da análise exploratória; feche as janelas para continuar. Para executar sem janelas no PowerShell:

```powershell
$env:MPLBACKEND = "Agg"
python main.py
```

Nesse modo, avisos sobre a impossibilidade de exibir gráficos são esperados. O programa apresenta os resultados no terminal. Esta versão não salva automaticamente gráficos ou modelos.

## Fases implementadas

1. **Análise exploratória:** estrutura, valores ausentes, distribuição do alvo, histogramas e correlações.
2. **Preparação dos dados:** diagnóstico de duplicatas, assimetria, outliers e estratégias de imputação.
3. **Feature engineering:** diferença entre temperatura de processo e temperatura do ar. No pipeline, essa diferença é recalculada após a imputação.
4. **Divisão e pré-processamento:** divisão estratificada de 80% para treino e 20% para teste, com `random_state=42`; imputação por média para temperaturas e torque e por mediana para rotação; codificação de `tipo` com OneHotEncoder.
5. **Escalonamento:** padronização das variáveis numéricas para KNN e preparação das entradas para os modelos.
6. **Treinamento:** KNN com k em 3, 5 e 7; árvore com profundidades 3, 5 e sem limite. Seleção por acurácia balanceada em validação cruzada estratificada de três folds. A árvore usa `class_weight="balanced"`. Pré-processamento e escalonamento do KNN ficam dentro do pipeline durante a validação cruzada.
7. **Avaliação final:** acurácia, acurácia balanceada, precisão, recall, F1, matrizes de confusão e comparação dos modelos no conjunto de teste.

`udi`, `id_produto`, o alvo e as colunas dos tipos de falha não entram como preditores. Os pipelines são ajustados com os dados de treinamento.

## Resultados finais

Resultados da execução com 8.000 registros de treinamento e 2.000 de teste, incluindo 68 falhas no teste:

| Modelo selecionado | Acurácia | Acurácia balanceada | Precisão de falhas | Recall de falhas | F1 de falhas |
| --- | --- | --- | --- | --- | --- |
| KNN, k=3 | 97,35% | 66,00% | 75,86% | 32,35% | 45,36% |
| Árvore, profundidade=5 | 93,75% | 85,41% | 32,30% | 76,47% | 45,41% |

O KNN identificou 22 das 68 falhas e gerou 7 falsos alarmes. A árvore identificou 52 das 68 falhas e gerou 109 falsos alarmes.

Para o cenário assumido em que detectar falhas é prioritário, a árvore é a recomendação: deixa passar 16 falhas, contra 46 do KNN. Essa escolha exige validar com a empresa os custos das falhas não detectadas e dos falsos alarmes. A acurácia isolada é insuficiente devido ao desbalanceamento do alvo.

## Limitações e conclusão

As sete fases estão implementadas e o fluxo completo é executável. Os resultados representam um experimento com divisão fixa dos dados; não demonstram desempenho em operação industrial. O script também imprime métricas de teste para cada candidato na fase 6, embora a seleção automática use somente a validação cruzada do treino. Esses resultados de teste não devem orientar novos ajustes de parâmetros.

A comparação com e sem a nova feature verifica o pré-processamento; não constitui um estudo comparativo de desempenho dos modelos. Implantação, monitoramento, persistência dos modelos e validação econômica ficam como trabalhos futuros.