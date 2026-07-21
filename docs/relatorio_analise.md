# Relatório de Análise: UE's Throughput Prediction em Redes O-RAN

**Disciplina:** Aplicações de Inteligência Artificial e Machine Learning em RIC  
**Curso:** Pós-Graduação em OpenRAN  
**Data:** Julho de 2026  
**Professor:** Julio Tesolin  

---

## 1. Introdução

### 1.1 Contexto: O-RAN e o Near-RT RIC

A arquitetura Open RAN (O-RAN), padronizada pela O-RAN Alliance, propõe a desagregação dos componentes de rede de acesso rádio (RAN) em elementos interoperáveis conectados por interfaces abertas. Um componente central desta arquitetura é o **RAN Intelligent Controller (RIC)**, subdividido em:

- **Non-RT RIC**: opera em escala de tempo superior a 1 segundo, hospeda rApps para otimização de políticas de longo prazo.
- **Near-RT RIC**: opera em escala de 10 ms a 1 segundo, hospeda **xApps** para controle em tempo quase real.

O Near-RT RIC é o locus natural para a implementação de algoritmos de Machine Learning voltados à otimização de KPIs de rádio, incluindo a predição de throughput de UEs (User Equipments).

### 1.2 O Caso de Uso: UE's Throughput Prediction

De acordo com o **O-RAN WG1 Use Cases Detailed Specification v20.00**, a predição de throughput de UEs é um caso de uso fundamental para:

- **QoS Steering**: direcionar UEs para células ou recursos com maior capacidade prevista.
- **Scheduling inteligente**: alimentar o scheduler da gNB (via E2 interface) com estimativas de throughput para priorização de recursos de rádio.
- **Previsão de congestionamento**: antecipar degradações de desempenho antes que afetem usuários.

O fluxo de trabalho segue o modelo descrito no **O-RAN WG2 AI/ML Workflow Description and Requirements 1.03**:

```
[Coleta de KPIs via E2] → [Feature Engineering] → [Modelo ML no xApp] → [Ação de Controle via E2]
```

### 1.3 Objetivo

Aplicar e comparar **duas técnicas distintas de Machine Learning** para resolver o problema de regressão de predição de throughput de UEs, avaliando seu potencial de implantação em xApps do Near-RT RIC.

---

## 2. Dados

### 2.1 Datasets Utilizados

O projeto utiliza dois datasets sintéticos representativos de métricas de rádio coletadas em células LTE/NR:

| Dataset | Arquivo | Amostras | Features | Target |
|---|---|---|---|---|
| **Principal** | `kNN_Practice_100rows.csv` | 100 | 7 KPIs de rádio | Throughput (Mbps) |
| **Temporal** | `traffic_prediction.csv` | 16 | 3 KPIs + hora | Throughput (Mbps) |

### 2.2 Descrição das Features (Dataset Principal)

| Feature | Unidade | Descrição |
|---|---|---|
| `PRB_Usage` | % | Utilização de Physical Resource Blocks |
| `Active_Users` | UEs | Número de UEs ativos na célula |
| `SINR` | dB | Signal-to-Interference-plus-Noise Ratio |
| `RSRQ` | dB | Reference Signal Received Quality |
| `Packet_Loss` | % | Taxa de perda de pacotes |
| `Latency` | ms | Latência de transmissão |
| `CPU_Load` | % | Carga de CPU da gNB/DU |
| `Throughput` | **Mbps** | **Target** — Throughput do UE |

### 2.3 Análise Exploratória dos Dados (EDA)

#### 2.3.1 Visão Geral do Dataset

```
Shape          : 100 amostras × 10 colunas
Valores nulos  : 0
```

**Distribuição por Cell_Class:**

| Cell_Class | Amostras |
|---|---|
| Normal | 34 |
| Congested | 33 |
| Degraded | 33 |

O dataset é balanceado entre as três classes, representando diferentes estados operacionais da célula.

#### 2.3.2 Estatísticas Descritivas

| Feature | Média | Desvio Padrão | Mínimo | Máximo |
|---|---|---|---|---|
| PRB_Usage | 64.08% | 20.80 | 28.00 | 100.00 |
| Active_Users | 59.22 | 36.95 | 11.00 | 146.00 |
| SINR | 13.81 dB | 7.92 | 0.00 | 29.00 |
| RSRQ | -13.38 dB | 3.44 | -19.60 | -8.10 |
| Packet_Loss | 2.67% | 2.37 | 0.04 | 7.84 |
| Latency | 30.60 ms | 18.64 | 8.30 | 79.90 |
| CPU_Load | 58.78% | 22.45 | 20.00 | 95.00 |
| **Throughput** | **126.02 Mbps** | **61.38** | **20.70** | **249.10** |

A alta variabilidade do target (σ = 61.38 Mbps, amplitude de 228.4 Mbps) reflete os diferentes regimes operacionais capturados (Normal, Congested, Degraded).

#### 2.3.3 Throughput por Classe de Célula

| Cell_Class | Throughput Médio | Desvio Padrão | Mínimo | Máximo |
|---|---|---|---|---|
| Normal | **188.93 Mbps** | 32.04 | 131.20 | 249.10 |
| Congested | 127.66 Mbps | 36.15 | 74.80 | 179.40 |
| Degraded | 59.55 Mbps | 22.98 | 20.70 | 88.10 |

A diferença de ~130 Mbps entre células Normal e Degraded evidencia a severidade da degradação e o valor de um modelo preditivo para acionar ações corretivas a tempo.

#### 2.3.4 Correlação das Features com o Target (Throughput)

| Feature | Correlação de Pearson | Interpretação |
|---|---|---|
| `Packet_Loss` | **-0.8407** | Forte correlação negativa — perda de pacotes degrada throughput |
| `RSRQ` | **+0.8216** | Forte correlação positiva — melhor qualidade de sinal → maior throughput |
| `SINR` | **+0.7807** | Forte correlação positiva — maior SINR → maior eficiência espectral |
| `Latency` | **-0.7532** | Forte correlação negativa — alta latência coexiste com baixo throughput |
| `CPU_Load` | -0.3009 | Correlação moderada negativa |
| `PRB_Usage` | -0.2369 | Correlação fraca negativa |
| `Active_Users` | -0.1594 | Correlação fraca negativa |

> **Interpretação técnica**: As quatro features de maior correlação (`Packet_Loss`, `RSRQ`, `SINR`, `Latency`) são KPIs diretamente relacionados às condições do canal rádio e da qualidade do enlace. Isso indica que modelos baseados nesses indicadores têm boa fundamentação física para a predição de throughput.

---

## 3. Metodologia

### 3.1 Pipeline de Processamento

O projeto adota o seguinte pipeline:

```
[Dados CSV] → [EDA] → [StandardScaler] → [Modelo ML] → [Avaliação] → [Comparação]
```

**Decisões de design:**
- **StandardScaler** embutido em `sklearn.Pipeline` para evitar *data leakage* entre folds de cross-validation.
- **GridSearchCV** para seleção automática de hiperparâmetros.
- **K-Fold Cross-Validation (k=5)** para estimativa robusta de desempenho em datasets pequenos.
- **Hold-out de teste (20%)** fixo para avaliação final independente.

### 3.2 Particionamento dos Dados

| Conjunto | Amostras | Percentual |
|---|---|---|
| Treino | 80 | 80% |
| Teste (hold-out) | 20 | 20% |

`random_state=42` utilizado para reprodutibilidade.

### 3.3 Métricas de Avaliação

| Métrica | Fórmula | Interpretação |
|---|---|---|
| **MAE** | `mean(|y_real - y_pred|)` | Erro médio absoluto em Mbps |
| **RMSE** | `sqrt(mean((y_real - y_pred)²))` | Penaliza erros grandes — em Mbps |
| **R²** | `1 - SS_res/SS_tot` | Proporção da variância explicada (0 a 1) |

---

## 4. Técnica 1: Random Forest Regressor

### 4.1 Fundamentação Teórica

O **Random Forest** é um método de aprendizado ensemble baseado em múltiplas árvores de decisão treinadas com *bootstrap aggregating* (bagging). Para regressão, a predição final é a média das predições das árvores individuais.

**Princípio central (Burkov, 2019):** Cada árvore é treinada em uma amostra aleatória dos dados com reposição, e em cada nó a divisão é feita usando apenas um subconjunto aleatório das features. Isso reduz a variância do modelo composto em relação a uma única árvore.

**Vantagens para este caso de uso:**
- Robusto ao overfitting mesmo sem regularização explícita.
- Fornece medidas de importância de features, úteis para interpretabilidade em contextos de telecomunicações.
- Não requer normalização prévia (embora incluída no pipeline por padronização).
- Lida bem com relações não-lineares entre KPIs de rádio.

**Hiperparâmetros otimizados via GridSearchCV:**

| Hiperparâmetro | Valores testados | Melhor valor |
|---|---|---|
| `n_estimators` | 50, 100, 200 | **200** |
| `max_depth` | None, 5, 10 | **None** |
| `min_samples_split` | 2, 5 | **2** |
| `min_samples_leaf` | 1, 2 | **1** |

O `max_depth=None` indica que as árvores crescem até a separação máxima, favorável dado o dataset pequeno onde overfitting por profundidade é mitigado pelo ensemble.

### 4.2 Importância das Features

| Feature | Importância | Percentual |
|---|---|---|
| `Packet_Loss` | 0.4620 | **46.20%** |
| `RSRQ` | 0.2125 | **21.25%** |
| `SINR` | 0.1268 | **12.68%** |
| `Latency` | 0.0764 | 7.64% |
| `CPU_Load` | 0.0477 | 4.77% |
| `PRB_Usage` | 0.0417 | 4.17% |
| `Active_Users` | 0.0328 | 3.28% |

**Observação técnica**: As três features de maior importância (`Packet_Loss`, `RSRQ`, `SINR`) acumulam **80.13%** da importância total. Isso é consistente com a análise de correlação da EDA e com o conhecimento de domínio em redes rádio — a qualidade do enlace e a taxa de erro têm impacto direto no throughput percebido pelo UE.

### 4.3 Resultados

**Conjunto de teste hold-out (20 amostras):**

| Métrica | Valor |
|---|---|
| MAE | **21.64 Mbps** |
| RMSE | **26.11 Mbps** |
| R² | **0.7847** |

**Cross-Validation k=5 (dataset completo):**

| Métrica | Média | Desvio Padrão |
|---|---|---|
| MAE | 28.40 Mbps | ± 4.23 |
| RMSE | 34.07 Mbps | ± 4.68 |
| R² | 0.6266 | ± 0.1928 |

> **Análise**: O R² de 0.78 no teste e 0.63 no CV indica variância considerável, esperada dado o tamanho do dataset (100 amostras). O desvio padrão do R² (±0.19) reflete a sensibilidade dos resultados à partição dos dados — uma limitação inerente ao regime de baixo volume de dados.

---

## 5. Técnica 2: Support Vector Regression (SVR)

### 5.1 Fundamentação Teórica

O **Support Vector Regression (SVR)** é uma extensão do Support Vector Machine (SVM) para problemas de regressão. O objetivo é encontrar uma função `f(x)` que aproxime os valores reais com uma margem de tolerância `ε` (epsilon), minimizando apenas os erros que excedam esse limiar.

**Formulação do problema de otimização:**

```
minimizar: ½||w||² + C·Σ(ξᵢ + ξᵢ*)
sujeito a: yᵢ - f(xᵢ) ≤ ε + ξᵢ
           f(xᵢ) - yᵢ ≤ ε + ξᵢ*
```

O **kernel RBF** (Radial Basis Function) mapeia os dados para um espaço de características de alta dimensionalidade, permitindo capturar relações não-lineares entre os KPIs de rádio e o throughput.

**Vantagens para este caso de uso:**
- Excelente desempenho em datasets pequenos com alta dimensionalidade relativa.
- O kernel RBF captura as relações não-lineares complexas entre SINR, PRB_Usage e Throughput.
- Robustez a outliers (os pontos de suporte são a minoria dos dados).

**Hiperparâmetros otimizados via GridSearchCV:**

| Hiperparâmetro | Valores testados | Melhor valor | Significado |
|---|---|---|---|
| `kernel` | rbf, linear | **rbf** | Mapeia para espaço não-linear |
| `C` | 0.1, 1, 10, 100 | **100** | Alta penalidade por erros — ajuste fino |
| `epsilon` | 0.01, 0.1, 1.0 | **0.01** | Margem de tolerância estreita (Mbps normalizados) |
| `gamma` | scale, auto | **scale** | Escala automática pelo número de features |

O `C=100` indica que o modelo prefere ajustar bem os dados de treino, aceitável dado que o SVR é intrinsecamente regularizado pela margem ε.

### 5.2 Resultados

**Conjunto de teste hold-out (20 amostras):**

| Métrica | Valor |
|---|---|
| MAE | **23.42 Mbps** |
| RMSE | **29.22 Mbps** |
| R² | **0.7302** |

**Cross-Validation k=5 (dataset completo):**

| Métrica | Média | Desvio Padrão |
|---|---|---|
| MAE | 28.83 Mbps | ± 3.55 |
| RMSE | 34.79 Mbps | ± 3.51 |
| R² | 0.6036 | ± 0.2210 |

> **Análise**: O SVR apresenta menor variabilidade no RMSE (±3.51 vs ±4.68 do RF), o que pode indicar maior estabilidade em diferentes partições. Contudo, o R² médio inferior (0.60 vs 0.63) sugere menor capacidade preditiva geral neste dataset.

---

## 6. Análise no Dataset Temporal

### 6.1 Metodologia de Validação Temporal

O dataset `traffic_prediction.csv` contém 16 observações horárias (08h–23h) com três features de rede (`ActiveUsers`, `AvgSINR`, `PRBUtilization`). Devido ao tamanho muito pequeno, foi utilizada **Leave-One-Out Cross-Validation (LOO-CV)**, em que cada observação é usada como teste uma vez.

Modelos específicos foram re-treinados com as features disponíveis para demonstrar a adaptabilidade dos algoritmos a diferentes granularidades de dados disponíveis em ambientes O-RAN reais.

### 6.2 Resultados LOO-CV no Dataset Temporal

| Modelo | MAE (Mbps) | RMSE (Mbps) | R² |
|---|---|---|---|
| Random Forest | **7.16** | **8.06** | **0.9520** |
| SVR | 12.46 | 18.36 | 0.7513 |

### 6.3 Predições vs. Valores Reais (Primeiras 8 horas)

| Hora | Real (Mbps) | RF Pred. | SVR Pred. |
|---|---|---|---|
| 08h | 62.0 | 79.0 | 110.1 |
| 09h | 70.0 | 78.8 | 100.9 |
| 10h | 84.0 | 89.9 | 95.7 |
| 11h | 93.0 | 98.2 | 99.1 |
| 12h | 108.0 | 111.4 | 108.9 |
| 13h | 120.0 | 128.0 | 122.0 |
| 14h | 132.0 | 133.3 | 134.2 |
| 15h | 145.0 | 154.5 | 147.7 |

No dataset temporal, o Random Forest supera significativamente o SVR (R²=0.95 vs 0.75), possivelmente por conseguir capturar melhor a estrutura monotônica do tráfego horário com poucas árvores de decisão.

---

## 7. Comparação dos Modelos

### 7.1 Métricas Consolidadas

#### Conjunto de Teste Hold-Out (20%)

| Modelo | MAE (Mbps) | RMSE (Mbps) | R² |
|---|---|---|---|
| **Random Forest** | **21.64** | **26.11** | **0.7847** |
| SVR | 23.42 | 29.22 | 0.7302 |

#### Cross-Validation k=5 (Dataset Completo)

| Modelo | MAE CV | RMSE CV | R² CV |
|---|---|---|---|
| **Random Forest** | 28.40 ± 4.23 | 34.07 ± 4.68 | **0.6266 ± 0.1928** |
| SVR | 28.83 ± 3.55 | 34.79 ± 3.51 | 0.6036 ± 0.2210 |

### 7.2 Amostra de Predições vs. Reais (Conjunto de Teste)

| Real (Mbps) | RF Pred. | \|Erro RF\| | SVR Pred. | \|Erro SVR\| |
|---|---|---|---|---|
| 58.3 | 51.2 | 7.1 | 61.0 | 2.7 |
| 178.8 | 145.0 | 33.8 | 117.2 | 61.6 |
| 52.1 | 58.8 | 6.7 | 67.1 | 15.0 |
| 129.0 | 132.1 | 3.1 | 149.6 | 20.6 |
| 99.7 | 106.1 | 6.4 | 113.6 | 13.9 |
| 156.2 | 118.2 | 38.0 | 105.9 | 50.3 |
| 166.2 | 192.3 | 26.1 | 191.7 | 25.5 |
| 84.9 | 62.8 | 22.1 | 81.1 | 3.8 |
| 235.3 | 199.2 | 36.1 | 206.9 | 28.4 |
| 171.3 | 212.8 | 41.5 | 208.8 | 37.5 |

### 7.3 Análise Comparativa

| Critério | Random Forest | SVR | Vantagem |
|---|---|---|---|
| R² (teste) | 0.7847 | 0.7302 | RF |
| MAE (teste) | 21.64 Mbps | 23.42 Mbps | RF |
| RMSE (teste) | 26.11 Mbps | 29.22 Mbps | RF |
| R² (CV) | 0.6266 | 0.6036 | RF |
| Estabilidade RMSE | ±4.68 | ±3.51 | SVR |
| Interpretabilidade | Feature importance | Vetores de suporte | RF |
| Custo computacional | Moderado | Alto (C=100) | RF |

**O Random Forest supera o SVR em 5 de 6 critérios avaliados.**

### 7.4 Discussão

**Por que o Random Forest teve melhor desempenho?**

1. **Natureza dos dados**: O dataset é caracterizado por três regimes distintos (Normal, Congested, Degraded) com fronteiras relativamente claras. Árvores de decisão capturam bem fronteiras por limiares de features.
2. **Feature importance**: O RF identificou `Packet_Loss` como responsável por 46% da variância explicada — uma relação que pode ser representada por partições de árvore de forma direta.
3. **Tamanho do dataset**: Em datasets pequenos, ensembles de árvores tendem a ser mais estáveis do que SVR com kernels complexos que requerem mais dados para calibrar o espaço de características.

**Por que o SVR ainda é relevante?**

O SVR apresentou menor variabilidade entre folds (σ_RMSE = 3.51 vs 4.68 do RF), o que pode ser preferível em cenários de produção onde a previsibilidade do modelo é tão importante quanto a acurácia média.

---

## 8. Contexto O-RAN: Implantação como xApp

### 8.1 Arquitetura de Implantação

No contexto O-RAN, o modelo treinado seria implantado como um **xApp no Near-RT RIC**, com o seguinte fluxo operacional:

```
gNB/DU → [E2 Interface] → Near-RT RIC
                               ↓
                    [xApp: Throughput Prediction]
                               ↓ (KPIs → modelo → throughput estimado)
                    [xApp: QoS/Scheduling Optimizer]
                               ↓
                    [E2 Interface] → gNB/DU (ação de controle)
```

### 8.2 Integração com o Workflow de AI/ML (O-RAN WG2)

Conforme o **O-RAN WG2 AI/ML Workflow Description and Requirements 1.03**, o ciclo de vida do modelo compreende:

1. **Data Collection**: KPIs coletados via interface E2 (relatórios de medição PM — *Performance Measurement*).
2. **Training**: Executado no SMO (*Service Management and Orchestration*) ou Non-RT RIC.
3. **Deployment**: Modelo serializado (pickle/ONNX) transferido para o xApp no Near-RT RIC.
4. **Inference**: Predição em tempo real com latência < 1 segundo.
5. **Monitoring & Retraining**: Detecção de *concept drift* e retreinamento periódico.

### 8.3 KPIs Necessários via E2

Os KPIs utilizados como features são coletados via:
- **E2SM-KPM** (*Key Performance Measurements*): PRB_Usage, Active_Users, Throughput
- **E2SM-RC** (*RAN Control*): SINR, RSRQ via relatórios de medição do UE
- **Métricas de QoS**: Packet_Loss, Latency

---

## 9. Limitações e Trabalhos Futuros

### 9.1 Limitações Identificadas

| Limitação | Impacto | Mitigação |
|---|---|---|
| Dataset pequeno (100 amostras) | Alta variância nas métricas de CV (σ_R² ≈ 0.20) | Coletar dados reais de telemetria de gNBs em produção |
| Dados sintéticos | Podem não capturar correlações complexas de redes reais | Utilizar datasets de benchmarks como o OpenAirInterface ou dados de operadoras |
| Ausência de dimensão temporal | Impossibilita captura de padrões de tráfego (e.g., horário de pico) | Modelos sequenciais (LSTM, Transformer) com janelas temporais |
| Features estáticas | Não considera mobilidade do UE, handover, beamforming | Incorporar métricas de mobilidade (e.g., serving cell changes) |
| Ausência de multi-célula | Throughput é predito por célula isolada | Modelagem com contexto inter-célula (interferência, load balancing) |

### 9.2 Trabalhos Futuros

- **Modelos sequenciais**: LSTM ou GRU para captura de padrões temporais de tráfego.
- **Aprendizado por Reforço**: Agente no Near-RT RIC para otimização adaptativa de recursos (Sutton & Barto, 2018).
- **Transfer Learning**: Treinar em dados sintéticos e fine-tunar com dados reais de produção.
- **Federated Learning**: Treinar modelos distribuídos em múltiplas gNBs sem centralizar dados sensíveis.

---

## 10. Conclusão

Este trabalho aplicou e comparou duas técnicas de Machine Learning — **Random Forest Regressor** e **Support Vector Regression (SVR)** — ao caso de uso de predição de throughput de UEs em redes O-RAN.

**Principais achados:**

1. O **Random Forest** superou o SVR em todas as métricas de acurácia, obtendo R²=0.78 no conjunto de teste e R²=0.63 na validação cruzada (k=5).

2. A análise de **feature importance** revelou que `Packet_Loss` (46%), `RSRQ` (21%) e `SINR` (13%) são responsáveis por mais de 80% da variância explicada do throughput — resultado coerente com o conhecimento de domínio em sistemas de comunicações rádio.

3. No dataset temporal (série horária), o Random Forest obteve R²=0.95, demonstrando excelente capacidade de captura de padrões temporais simples com apenas 3 features.

4. Ambos os modelos demonstram viabilidade para implantação como **xApps no Near-RT RIC** da arquitetura O-RAN, com latência de inferência compatível com o requisito de < 1 segundo.

5. A principal limitação é o **tamanho reduzido do dataset**, que limita a generalização. Em produção, datasets com milhares de amostras coletadas via interface E2 permitiriam modelos com desempenho significativamente superior.

---

## Referências

1. O-RAN Alliance. (2026). *O-RAN Working Group 1 Use Cases Detailed Specification v20.00*.
2. O-RAN Alliance. (2026). *O-RAN Working Group 1 Use Cases Analysis Report v20.00*.
3. O-RAN Alliance. (2021). *O-RAN Working Group 2 AI/ML Workflow Description and Requirements 1.03*.
4. Burkov, A. (2019). *The Hundred-Page Machine Learning Book*. Andriy Burkov.
5. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. A Bradford Book, MIT Press.
6. 3GPP. (2025). *AI/ML for NG-RAN*. Disponível em: https://www.3gpp.org/news-events/3gpp-news/ai-ml-2025
7. 3GPP TS 32.500. *Telecommunication management; Self-Organizing Networks (SON)*.
8. Scikit-learn Developers. (2024). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.

---

*Todos os resultados apresentados neste relatório foram gerados pelo script `src/throughput_prediction.py`, reprodutíveis através do ambiente virtual configurado no projeto.*
