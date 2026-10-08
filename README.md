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

