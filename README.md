# Manutenção Preditiva — Indústria 4.0

## Objetivo

O **Manutenção Preditiva — Indústria 4.0** é um projeto de análise preditiva desenvolvido em Python como atividade do **Módulo 1 — Análise Preditiva com Python**.

O objetivo é analisar dados de equipamentos industriais e construir modelos de classificação para identificar a ocorrência de falhas mecânicas. O projeto reúne análise exploratória, preparação dos dados, engenharia de atributos, pré-processamento, treinamento e avaliação de modelos de Machine Learning.

São comparados dois algoritmos: **K-Nearest Neighbors (KNN)** e **Árvore de Decisão**, considerando tanto o desempenho geral quanto a capacidade de detectar máquinas com falha.

---

## O que o projeto analisa

A análise utiliza o arquivo `data/manutencao_preditiva.csv`, com **10.000 registros e 14 colunas**.

Entre as análises realizadas estão:

- inspeção da estrutura e dos tipos das variáveis;
- identificação de valores ausentes e registros duplicados;
- distribuição da variável alvo;
- estatísticas descritivas, assimetria e possíveis outliers;
- distribuição da velocidade de rotação;
- correlação entre variáveis numéricas;
- criação da diferença entre as temperaturas de processo e do ar;
- divisão estratificada em treinamento e teste;
- imputação, codificação e escalonamento das variáveis;
- análise do desbalanceamento das classes;
- treinamento e seleção de parâmetros por validação cruzada;
- avaliação e comparação dos modelos.

### Variável alvo

A variável `falha_maquina` possui duas classes:

| Classe | Significado | Registros | Proporção |
| --- | --- | ---: | ---: |
| 0 | Funcionamento normal | 9.661 | 96,61% |
| 1 | Ocorrência de falha | 339 | 3,39% |

O desbalanceamento exige avaliar métricas além da acurácia, principalmente o recall da classe de falhas.

### Variáveis preditoras

- `tipo`;
- `temperatura_ar_k`;
- `temperatura_processo_k`;
- `velocidade_rotacao_rpm`;
- `torque_nm`;
- `desgaste_ferramenta_min`;
- `diferenca_temperatura_k`, criada durante o processamento.

Os identificadores `udi` e `id_produto` não são usados no treinamento. As colunas `falha_twf`, `falha_hdf`, `falha_pwf`, `falha_osf` e `falha_rnf` também são excluídas, pois descrevem os tipos de falha e poderiam revelar informação associada ao alvo.

---

## Fases do projeto

### Fase 1 — Análise exploratória

Inspeção das dimensões, colunas, tipos, estatísticas, valores ausentes e distribuição do alvo. São exibidos gráficos de distribuição das classes, histograma de rotação e mapa de correlação.

Foram encontrados **500 valores ausentes em cada uma** das variáveis de temperatura do ar, temperatura de processo, velocidade de rotação e torque. As demais colunas originais não apresentaram valores ausentes. Não foram identificadas linhas duplicadas.

### Fase 2 — Preparação dos dados

Análise de duplicatas, estatísticas, assimetria e possíveis outliers para orientar o tratamento dos dados. A rotação apresenta assimetria à direita. O diagnóstico de outliers não implica sua remoção automática.

### Fase 3 — Engenharia de atributos

Criação da variável:

```text
diferenca_temperatura_k = temperatura_processo_k - temperatura_ar_k
```

No pipeline de modelagem, a diferença é recalculada após a imputação das temperaturas.

### Fase 4 — Divisão e pré-processamento

Divisão estratificada com `test_size=0.2` e `random_state=42`:

| Conjunto | Total | Sem falha | Com falha |
| --- | ---: | ---: | ---: |
| Treinamento | 8.000 | 7.729 | 271 |
| Teste | 2.000 | 1.932 | 68 |

O pré-processamento aplica:

- imputação pela **média** nas temperaturas e no torque;
- imputação pela **mediana** na velocidade de rotação;
- codificação de `tipo` com **OneHotEncoder**, ignorando categorias desconhecidas;
- cálculo da diferença de temperatura após a imputação.

Também são verificados os formatos e a ausência de valores nulos após o processamento, com e sem a nova variável.

### Fase 5 — Escalonamento

As variáveis numéricas do KNN são padronizadas com **StandardScaler**, pois o algoritmo usa distâncias. A Árvore de Decisão utiliza as variáveis pré-processadas sem padronização.

### Fase 6 — Treinamento e ajuste de parâmetros

São avaliados:

- **KNN:** `n_neighbors` igual a 3, 5 e 7;
- **Árvore de Decisão:** `max_depth` igual a 3, 5 e `None`.

A seleção utiliza a maior **acurácia balanceada média** em validação cruzada estratificada de três folds, com embaralhamento e `random_state=42`.

A árvore usa `class_weight="balanced"`. O KNN é treinado sem balanceamento explícito das classes. O pré-processamento e o escalonamento do KNN ficam dentro do pipeline para serem ajustados em cada fold de treinamento.

### Fase 7 — Avaliação final

Os modelos selecionados são avaliados no conjunto de teste com:

- acurácia;
- acurácia balanceada;
- precisão da classe de falhas;
- recall da classe de falhas;
- F1-score da classe de falhas;
- matriz de confusão;
- relatório de classificação.

---

## Resultados obtidos

Na execução validada, os modelos selecionados foram **KNN com k=3** e **Árvore de Decisão com profundidade máxima 5**.

| Métrica | KNN | Árvore de Decisão |
| --- | ---: | ---: |
| Acurácia balanceada média na validação cruzada | 67,79% | 87,67% |
| Acurácia no teste | 97,35% | 93,75% |
| Acurácia balanceada no teste | 66,00% | 85,41% |
| Precisão de falhas | 75,86% | 32,30% |
| Recall de falhas | 32,35% | 76,47% |
| F1 de falhas | 45,36% | 45,41% |

### Matrizes de confusão

As linhas representam as classes reais e as colunas as classes previstas, na ordem **0 — sem falha** e **1 — com falha**.

```text
KNN                    Árvore de Decisão
[[1925,   7],          [[1823, 109],
 [  46,  22]]           [  16,  52]]
```

| Resultado no teste | KNN | Árvore de Decisão |
| --- | ---: | ---: |
| Falhas identificadas corretamente | 22 | 52 |
| Falhas não detectadas | 46 | 16 |
| Falsos alarmes | 7 | 109 |

### Conclusão da comparação

O KNN apresenta maior acurácia e menos falsos alarmes. A árvore identifica uma proporção maior das falhas e deixa passar menos equipamentos com falha.

Para o cenário assumido de manutenção preditiva em que **detectar falhas é prioritário**, a **Árvore de Decisão** é a recomendação deste experimento. Sua adoção depende da avaliação dos custos de falsos alarmes e de falhas não detectadas pela empresa.

---

## Conceitos aplicados

- funções, parâmetros e retornos;
- organização modular do código;
- classes e transformadores personalizados;
- manipulação de DataFrames com Pandas;
- operações numéricas com NumPy;
- análise estatística e visualização de dados;
- tratamento de valores ausentes;
- engenharia de atributos;
- codificação de variáveis categóricas;
- padronização de variáveis numéricas;
- pipelines de Machine Learning;
- divisão e validação cruzada estratificadas;
- ponderação das classes;
- seleção de parâmetros e métricas de classificação;
- cuidados para evitar vazamento de dados no pré-processamento;
- versionamento com Git e integração por pull requests.

---

## Como executar localmente

### 1. Clone o repositório

```bash
git clone https://github.com/franferdev/manutencao-preditiva-industria-4-0.git
cd manutencao-preditiva-industria-4-0
```

### 2. Crie e ative um ambiente virtual

No Windows, usando PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instale as dependências

```bash
python -m pip install -r requirements.txt
```

### 4. Verifique o dataset

O arquivo deve estar disponível em `data/manutencao_preditiva.csv`.

### 5. Execute o projeto

```bash
python main.py
```

Execute a partir da raiz do repositório. O arquivo `main.py` coordena o fluxo completo. Feche as janelas dos gráficos quando necessário para continuar a execução.

Para executar sem abrir janelas no PowerShell:

```powershell
$env:MPLBACKEND = "Agg"
python main.py
```

Nesse modo, avisos de que os gráficos não podem ser exibidos são esperados.

---

## Como executar no Google Colab

Faça upload da pasta do projeto, preservando `src/`, `data/`, `main.py` e `requirements.txt`. No notebook, entre na pasta que contém `main.py` e execute:

```python
!pip install -r requirements.txt
!MPLBACKEND=Agg python main.py
```

Essa execução apresenta os resultados no terminal do notebook, sem abrir janelas de gráficos. Pode ser necessário ajustar o diretório de trabalho conforme o local do upload. A validação registrada deste projeto foi realizada localmente.

---

## Estrutura de pastas

```text
manutencao-preditiva-industria-4-0/
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── manutencao_preditiva.csv
└── src/
    ├── __init__.py
    ├── carregamento.py
    ├── eda.py
    ├── data_prep.py
    ├── feature_engineering.py
    ├── divisao_dados.py
    ├── preprocessamento.py
    ├── balanceamento.py
    ├── escalonamento.py
    ├── treinamento_knn.py
    ├── treinamento_arvore.py
    └── avaliacao_modelos.py
```

---

## Decisões técnicas

### Organização modular

`main.py` coordena as etapas e os arquivos em `src/` concentram as funções e os transformadores de cada fase. O fluxo principal é:

```text
Carregar dados
      ↓
Analisar estrutura, distribuição e correlações
      ↓
Diagnosticar qualidade dos dados
      ↓
Criar atributos
      ↓
Dividir treinamento e teste
      ↓
Configurar e verificar pré-processamento
      ↓
Analisar balanceamento e escalonamento
      ↓
Treinar e selecionar KNN e Árvore de Decisão
      ↓
Avaliar no teste e comparar os resultados
```

### Pré-processamento dentro dos modelos

Os modelos selecionados contêm seus pipelines de pré-processamento e recebem os dados de teste no formato original das variáveis preditoras. As transformações aprendem parâmetros somente a partir do treinamento.

### Critério de seleção

A acurácia balanceada na validação cruzada orienta a seleção dos parâmetros. O recall de falhas orienta a recomendação final para o cenário de negócio assumido.

### Saídas do projeto

As análises, métricas e matrizes de confusão são apresentadas no terminal. Os gráficos são exibidos durante a execução interativa. A versão entregue não exporta automaticamente imagens, relatórios ou modelos treinados.

---

## Ferramentas utilizadas

- **Python** — desenvolvimento do pipeline;
- **Pandas** — leitura, manipulação e análise dos dados;
- **NumPy** — operações numéricas e verificações;
- **Matplotlib** — criação de gráficos;
- **Seaborn** — visualizações estatísticas;
- **Scikit-learn** — pré-processamento, treinamento, validação e métricas;
- **Visual Studio Code** — desenvolvimento local;
- **Git e GitHub** — versionamento e disponibilização do projeto;
- **Google Colab** — alternativa de ambiente de execução.

As dependências estão descritas em `requirements.txt`.

---

## Validação e limitações

O comando `python main.py` executou as sete fases com código de saída **0** em ambiente local, usando `MPLBACKEND=Agg`. Os valores apresentados neste README correspondem a essa execução.

Os resultados são de um experimento com divisão fixa dos dados e não comprovam desempenho em operação industrial. A fase 6 imprime métricas de teste para cada candidato, mas a seleção automática usa a validação cruzada do treino; as métricas de teste não devem orientar ajustes posteriores.

A verificação com e sem a diferença de temperatura compara o pré-processamento, sem medir o ganho de desempenho dessa feature. A instalação em um ambiente virtual novo e a execução no Colab não foram verificadas nessa validação.

---

## Próximas melhorias

- avaliar os custos de falsos alarmes e falhas não detectadas;
- medir o impacto da nova feature sobre o desempenho;
- comparar outras estratégias para classes desbalanceadas;
- validar os modelos em novos dados e cenários industriais;
- persistir os pipelines treinados;
- automatizar a exportação de relatórios e gráficos;
- implementar monitoramento do desempenho em produção.

---

## Vídeo de apresentação

Link da apresentação: **a adicionar**.

---

## Autoria

Projeto desenvolvido como atividade prática do **Módulo 1 — Análise Preditiva com Python**, com foco em manutenção preditiva e Indústria 4.0.

Repositório: [franferdev/manutencao-preditiva-industria-4-0](https://github.com/franferdev/manutencao-preditiva-industria-4-0).
