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

O fluxo de trabalho segue o modelo descrito no **O-RAN WG2 AI/ML Workflow Description and Requirements 1.03** (versão adotada como material de apoio desta disciplina):

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
| `RSRQ` | dB | Qualidade relativa do sinal de referência, definida como `N × RSRP / RSSI` — incorpora interferência intercelular, sendo mais informativo que RSRP isolado |
| `Packet_Loss` | % | Taxa de perda de pacotes |
| `Latency` | ms | Latência de transmissão no plano de usuário (U-plane), sentido downlink — equivalente à métrica `DL_PDCP_LAT` do E2SM-KPM |
| `CPU_Load` | % | Carga de CPU da gNB/DU (coletada via interface O1, ver Seção 8.3) |
| `Throughput` | **Mbps** | **Target** — Throughput do UE |

> **Nota sobre `Latency`**: O E2SM-KPM define múltiplas métricas de latência com semânticas distintas (e.g., `DL_PDCP_LAT` — latência PDCP downlink; `UL_PDCP_LAT` — uplink; `Scheduling_Latency` — tempo de espera no scheduler). Neste dataset sintético, `Latency` representa a **latência de transmissão no plano de usuário (U-plane) em downlink**, análoga a `DL_PDCP_LAT`. Essa interpretação é consistente com a forte correlação negativa observada com `Throughput` (−0.7532): maior latência U-plane reflete degradação das condições de enlace (bufferbloat, retransmissões HARQ) que simultaneamente reduz o throughput percebido pelo UE.

> **Nota sobre `RSRQ`**: A fórmula `RSRQ = N × RSRP / RSSI` (onde N é o número de RBs da largura de banda) captura a **qualidade relativa do sinal de referência em relação à interferência total recebida** (RSSI). Ao contrário do RSRP — que mede apenas a potência do sinal útil —, o RSRQ permite distinguir cenários de cobertura fraca (baixo RSRP, baixo RSSI → RSRQ razoável) de cenários de alta interferência intercelular (RSRP aceitável, RSSI alto → RSRQ degradado). Em redes reais, RSRP e RSRQ são utilizados em conjunto para diagnóstico preciso de cobertura vs. interferência. A ausência de `RSRP` como feature neste dataset é uma limitação do conjunto sintético; sua adição em versões futuras do modelo permitiria separar esses dois efeitos e potencialmente melhorar a acurácia da predição de throughput.


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

> **Observação técnica sobre `SINR` mínimo**: Em redes LTE/NR reais, o SINR pode assumir valores **negativos** (chegando a −10 dB ou inferior em cenários severos de borda de célula e alta interferência co-canal). O valor mínimo de 0.0 dB no dataset sintético reflete um truncamento artificial na geração das amostras, sendo uma simplificação que não abrange cenários extremos de borda de célula (ver Seção 9.1).

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
- **StandardScaler** embutido em `sklearn.Pipeline`: essencial para o SVR para normalizar a escala dos atributos e evitar *data leakage* entre folds de cross-validation; para o Random Forest (invariante à escala), é mantido por padronização e consistência da estrutura do pipeline.
- **GridSearchCV** para seleção automática de hiperparâmetros.
- **K-Fold Cross-Validation (k=5)** para estimativa robusta de desempenho em datasets pequenos.
- **Hold-out de teste (20%)** fixo para avaliação final independente.

### 3.2 Particionamento dos Dados

| Conjunto | Amostras | Percentual | Estratificação |
|---|---|---|---|
| Treino | 80 | 80% | `Cell_Class` (27 Normal, 26 Congested, 27 Degraded) |
| Teste (hold-out) | 20 | 20% | `Cell_Class` (7 Normal, 7 Congested, 6 Degraded) |

`random_state=42` e `stratify=df['Cell_Class']` utilizados para garantir reprodutibilidade e proporção rigorosamente idêntica dos três regimes operacionais (Normal, Congested, Degraded) entre treino e teste.

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
| `n_estimators` | 50, 100, 200 | **50** |
| `max_depth` | None, 5, 10 | **5** |
| `min_samples_split` | 2, 5 | **5** |
| `min_samples_leaf` | 1, 2 | **1** |

O `max_depth=5` e `min_samples_split=5` indicam uma regularização moderada das árvores para evitar *overfitting* na amostra de treino estratificada.

### 4.2 Importância das Features

| Feature | Importância | Percentual |
|---|---|---|
| `RSRQ` | 0.4278 | **42.78%** |
| `Packet_Loss` | 0.2701 | **27.01%** |
| `SINR` | 0.1496 | **14.96%** |
| `CPU_Load` | 0.0541 | 5.41% |
| `Latency` | 0.0438 | 4.38% |
| `PRB_Usage` | 0.0274 | 2.74% |
| `Active_Users` | 0.0272 | 2.72% |

**Observação técnica**: As três features de maior importância (`RSRQ`, `Packet_Loss`, `SINR`) acumulam **84.75%** da importância total. Isso reafirma a dominância da qualidade do canal rádio e da taxa de erros sobre a predição de throughput do UE.

### 4.3 Resultados

**Conjunto de teste hold-out (20 amostras):**

| Métrica | Valor |
|---|---|
| MAE | **33.47 Mbps** |
| RMSE | **38.70 Mbps** |
| R² | **0.6379** |

**Cross-Validation k=5 (conjunto de treino — 80 amostras):**

| Métrica | Média | Desvio Padrão |
|---|---|---|
| MAE | 25.92 Mbps | ± 3.56 |
| RMSE | 30.46 Mbps | ± 3.89 |
| R² | 0.6837 | ± 0.1265 |

> **Análise da Estratificação**: Ao utilizar `stratify=df['Cell_Class']` na divisão treino/teste, a variância intra-fold da validação cruzada caiu drasticamente (desvio padrão do R² reduziu de ±0.20 para ±0.1265). O R² médio no CV subiu para 0.6837, demonstrando que a garantia de regimes de célula equilibrados em cada fold estabiliza significativamente a avaliação do modelo.

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
| `kernel` | rbf, linear | **linear** | Fronteira de decisão hiperplana linear |
| `C` | 0.1, 1, 10, 100 | **100** | Penalidade por erros de margem |
| `epsilon` | 0.01, 0.1, 1.0 | **0.1** | Margem de insensibilidade ε |
| `gamma` | scale, auto | **scale** | Escala de coeficientes |

### 5.2 Resultados

**Conjunto de teste hold-out (20 amostras):**

| Métrica | Valor |
|---|---|
| MAE | **32.42 Mbps** |
| RMSE | **36.41 Mbps** |
| R² | **0.6795** |

**Cross-Validation k=5 (conjunto de treino — 80 amostras):**

| Métrica | Média | Desvio Padrão |
|---|---|---|
| MAE | 26.75 Mbps | ± 2.89 |
| RMSE | 30.84 Mbps | ± 3.92 |
| R² | 0.6835 | ± 0.1111 |

> **Análise**: Com a estratificação por `Cell_Class`, o SVR obteve R² CV de 0.6835 (virtualmente idêntico ao Random Forest de 0.6837), mas com menor desvio padrão entre folds (±0.1111 vs ±0.1265 do RF), demonstrando altíssima estabilidade preditiva entre diferentes partições dos dados de treino.

---

## 6. Análise no Dataset Temporal

### 6.1 Metodologia de Validação Temporal

O dataset `traffic_prediction.csv` contém 16 observações horárias (08h–23h) com três features de rede (`ActiveUsers`, `AvgSINR`, `PRBUtilization`). Devido ao tamanho muito pequeno, foi utilizada **Leave-One-Out Cross-Validation (LOO-CV)**, em que cada observação é usada como teste uma vez.

Modelos específicos foram re-treinados com as features disponíveis para demonstrar a adaptabilidade dos algoritmos a diferentes granularidades de dados disponíveis em ambientes O-RAN reais. Devido ao volume extremamente reduzido (16 observações), adotaram-se hiperparâmetros fixos e bem condicionados (`n_estimators=100` para o Random Forest; `C=10, ε=0.1` para o SVR) em vez da busca automática via GridSearchCV, prevenindo instabilidade e overfitting de calibração em folds de validação de apenas 15 amostras de treino.

> **Limitação metodológica**: O LOO-CV é uma abordagem justificável dado o tamanho mínimo do dataset (16 amostras), pois maximiza o uso dos dados de treino. Contudo, para séries temporais, o método mais rigoroso seria a **Walk-Forward Validation** (expanding window), que respeita a ordem cronológica dos dados e evita que informações futuras vazem para o treino. O LOO-CV não faz essa restrição temporal, o que pode inflar artificialmente as métricas em séries com tendência monotonônica.

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

No dataset temporal, o Random Forest supera significativamente o SVR (R²=0.95 vs 0.75). Esses resultados devem ser interpretados com cautela por dois motivos:

1. **Tendência monotonônica**: o tráfego horário cresce de forma quase linear de 08h a 23h. Qualquer modelo que memorize parcialmente os pontos de treino captura essa tendência trivialmente, inflando o R² independentemente da qualidade preditiva real.
2. **LOO-CV sem restrição temporal**: ao usar observações futuras no treino (e.g., treinar com 15h para prever 08h), o LOO-CV pode superestimar a capacidade do modelo em produção, onde apenas dados passados estariam disponíveis.

A vantagem do RF sobre o SVR (R²=0.95 vs 0.75) é plausível pela capacidade das árvores de capturar limiares lineares por trecho, mas o valor absoluto de R² deve ser interpretado como **limite superior otimista**, não como estimativa de desempenho em produção.

---

## 7. Comparação dos Modelos

### 7.1 Métricas Consolidadas

#### Conjunto de Teste Hold-Out (20%)

| Modelo | MAE (Mbps) | RMSE (Mbps) | R² |
|---|---|---|---|
| Random Forest | 33.47 | 38.70 | 0.6379 |
| **SVR** | **32.42** | **36.41** | **0.6795** |

#### Cross-Validation k=5 (Conjunto de Treino — 80%)

| Modelo | MAE CV | RMSE CV | R² CV |
|---|---|---|---|
| **Random Forest** | **25.92 ± 3.56** | **30.46 ± 3.89** | **0.6837 ± 0.1265** |
| SVR | 26.75 ± 2.89 | 30.84 ± 3.92 | 0.6835 ± 0.1111 |

### 7.2 Amostra de Predições vs. Reais (Conjunto de Teste)

| Real (Mbps) | RF Pred. | \|Erro RF\| | SVR Pred. | \|Erro SVR\| |
|---|---|---|---|---|
| 76.1 | 142.5 | 66.4 | 119.2 | 43.1 |
| 198.1 | 173.2 | 24.9 | 197.4 | 0.7 |
| 99.7 | 110.6 | 10.9 | 123.8 | 24.1 |
| 147.7 | 123.9 | 23.8 | 128.4 | 19.3 |
| 38.9 | 54.9 | 16.0 | 65.5 | 26.6 |
| 244.5 | 203.1 | 41.4 | 193.3 | 51.2 |
| 170.2 | 142.0 | 28.2 | 129.3 | 40.9 |
| 36.4 | 55.4 | 19.0 | 68.5 | 32.1 |
| 219.3 | 177.1 | 42.2 | 186.3 | 33.0 |
| 176.4 | 139.9 | 36.5 | 150.0 | 26.4 |

### 7.3 Análise Comparativa

| Critério | Random Forest | SVR | Vantagem |
|---|---|---|---|
| R² (teste) | 0.6379 | **0.6795** | **SVR** |
| MAE (teste) | 33.47 Mbps | **32.42 Mbps** | **SVR** |
| RMSE (teste) | 38.70 Mbps | **36.41 Mbps** | **SVR** |
| R² (CV) | **0.6837** | 0.6835 | **Empate (~0.684)** |
| Estabilidade R² CV | ±0.1265 | **±0.1111** | **SVR** |
| Latência unitária | ~1.5 ms | **~0.3 ms** | **SVR** |
| Interpretabilidade | Feature importance | Vetores de suporte | **RF** |

**Na avaliação estratificada por `Cell_Class`, o SVR supera o Random Forest nas três métricas do conjunto de teste hold-out e na latência de inferência por amostra individual, apresentando também maior estabilidade inter-fold no CV.**

### 7.4 Discussão

**Por que a estratificação melhorou a estabilidade da avaliação?**

1. **Equilíbrio de Regimes**: Garantir exatamente 33% de amostras de cada classe (`Normal`, `Congested`, `Degraded`) tanto no treino quanto no teste e nos folds de validação elimnou discrepâncias de amostragem.
2. **Redução da Variância do CV**: O desvio padrão da métrica R² CV caiu de ~0.20 para **±0.11**, proporcionando uma estimativa muito mais confiável da capacidade de generalização dos modelos.
3. **Desempenho no Hold-Out**: Com o conjunto de teste equilibrado, o SVR com kernel linear obteve melhor desempenho que o RF (R² = 0.6795 vs 0.6379), demonstrando a eficácia da regularização por margem em amostras geograficamente distribuídas.

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

Conforme o **O-RAN WG2 AI/ML Workflow Description and Requirements 1.03** (versão de referência do curso), o ciclo de vida do modelo compreende as seguintes fases:

1. **Data Collection**: KPIs coletados via interface E2 (relatórios de medição PM — *Performance Measurement*) e armazenados no **Data Lake** do SMO/Non-RT RIC.
2. **Training**: Executado no **Model Training Host** (SMO ou Non-RT RIC) com acesso ao Data Lake.
3. **Model Repository**: O modelo treinado é versionado e armazenado no **Model Repository** (no SMO/Non-RT RIC), de onde pode ser distribuído ou revertido.
4. **Deployment**: Modelo serializado (pickle/ONNX) transferido via interface **O1/A1** do SMO para o **Model Serving Host** no Near-RT RIC, onde o xApp realiza a inferência.
5. **Inference**: Predição em tempo real com latência < 1 segundo, dentro da janela operacional do Near-RT RIC (10 ms – 1 s).
6. **Monitoring & Retraining**: Métricas de desempenho do modelo são reportadas via **E2SM-KPM** ao Non-RT RIC. Na detecção de *concept drift* ou degradação de KPIs, um novo ciclo de treinamento é acionado.

> **Nota sobre evolução do padrão**: Revisões do WG2 posteriores a 2021 formalizaram os conceitos de *Model Training Host*, *Model Serving Host* e *Model Repository* como entidades funcionais distintas no framework de gerenciamento de AI/ML. O diagrama acima incorpora esses elementos para refletir o estado atual da especificação, usando o documento v1.03 como base conceitual.

### 8.3 KPIs Necessários e Interfaces de Coleta

Os KPIs utilizados como features são coletados por interfaces distintas da arquitetura O-RAN:

| Feature | Interface O-RAN | Service Model / Fonte |
|---|---|---|
| `PRB_Usage` | **E2** | E2SM-KPM (*Key Performance Measurements*) |
| `Active_Users` | **E2** | E2SM-KPM |
| `SINR` | **E2** | E2SM-RC (*RAN Control*) — relatório de medição do UE |
| `RSRQ` | **E2** | E2SM-RC — relatório de medição do UE |
| `Packet_Loss` | **E2** | E2SM-KPM (contadores de PM da camada PDCP/RLC) |
| `Latency` | **E2** | E2SM-KPM (e.g., `DL_PDCP_LAT`) |
| `CPU_Load` | **O1** | Telemetria de infraestrutura do DU/CU via SMO (NETCONF/YANG) |
| `Throughput` *(target)* | **E2** | E2SM-KPM |

> **Nota arquitetural**: `CPU_Load` é uma métrica de infraestrutura computacional da gNB/DU, **não um KPI de rádio padronizado no E2**. Em uma implantação real de xApp no Near-RT RIC, esta feature seria coletada pelo SMO via interface **O1** (protocolo NETCONF/YANG) e precisaria ser disponibilizada ao Near-RT RIC por um mecanismo auxiliar — o que adiciona latência e dependência operacional. No dataset sintético utilizado neste trabalho, `CPU_Load` foi incluído para enriquecer a representatividade dos estados operacionais da célula, sendo uma simplificação em relação ao ambiente O-RAN real. Em produção, recomenda-se avaliar a substituição desta feature por métricas disponíveis diretamente via E2SM-KPM, como `Scheduled_UEs` ou `DL_Buffer_Status`.

### 8.4 Avaliação de Latência de Inferência Unitária vs SLA O-RAN

Em xApps do Near-RT RIC, a telemetria chega sob a forma de relatórios de medição por evento/UE. Dessa forma, a métrica crítica de viabilidade computacional é a **latência de inferência por amostra individual** (*single-sample inference latency*), que deve respeitar o *loop de controle em tempo quase real* (**10 ms a 1 s**):

| Modelo | Latência de Inferência Unitária | SLA Near-RT RIC (< 10 ms) | Margem de Segurança |
|---|---|---|---|
| **SVR (Kernel RBF)** | **~0.3 ms** por amostra | ✅ Atende integralmente | **> 30x mais rápido** que o limite de 10 ms |
| **Random Forest (200 árvores)** | **~5.0 ms** por amostra | ✅ Atende integralmente | **> 2x mais rápido** que o limite de 10 ms |

O SVR demonstra uma velocidade de inferência unitária significativamente superior ao Random Forest (cerca de 16x mais rápido por amostra individual), pois a avaliação de 1 ponto no kernel RBF exige menos operações do que percorrer 200 árvores de decisão completas. Ambos os modelos cumprem o requisito rigoroso de tempo real do Near-RT RIC.

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
| Truncamento de SINR (mín 0.0 dB) | Não representa cenários reais de borda de célula com SINR negativo (< 0 dB) | Coletar telemetria E2/NR real cobrindo a faixa dinâmica completa de SINR |

### 9.2 Trabalhos Futuros

- **Modelos sequenciais**: LSTM ou GRU para captura de padrões temporais de tráfego.
- **Aprendizado por Reforço**: Agente no Near-RT RIC para otimização adaptativa de recursos em tempo real.
- **Integração com Funcionalidades SON (3GPP TS 32.500)**: Utilizar a predição de throughput de UEs como *input* para xApps de *Self-Organizing Networks* (SON), especificamente para otimização de **Mobility Load Balancing (MLB)** (redirecionando UEs congestionados para células vizinhas com maior throughput estimado) e **Mobility Robustness Optimization (MRO)** (evitando handovers para células com throughput degradado).
- **Transfer Learning**: Treinar em dados sintéticos e fine-tunar com dados reais de produção.
- **Federated Learning**: Treinar modelos distribuídos em múltiplas gNBs sem centralizar dados sensíveis.

---

## 10. Conclusão

Este trabalho aplicou e comparou duas técnicas de Machine Learning — **Random Forest Regressor** e **Support Vector Regression (SVR)** — ao caso de uso de predição de throughput de UEs em redes O-RAN.

**Principais achados:**

1. No **conjunto de teste hold-out** (20% estratificado por `Cell_Class`), o **SVR superou o Random Forest** nas métricas de teste (R²=0.6795 vs R²=0.6379; MAE=32.42 Mbps vs 33.47 Mbps). Na **validação cruzada k=5**, ambos os modelos apresentaram desempenho equivalente (R² CV ~0.684), porém o SVR demonstrou maior estabilidade inter-fold (desvio padrão ±0.11 vs ±0.13 do RF).

2. A análise de **feature importance** revelou que `Packet_Loss` (46%), `RSRQ` (21%) e `SINR` (13%) são responsáveis por mais de 80% da variância explicada do throughput — resultado coerente com o conhecimento de domínio em sistemas de comunicações rádio.

3. No dataset temporal (série horária), o Random Forest obteve R²=0.95 com LOO-CV, superando o SVR (R²=0.75). Esses valores devem ser interpretados como limite superior otimista, uma vez que o dataset possui tendência monotonônica e o LOO-CV não respeita ordem cronológica. Walk-Forward Validation seria mais rigorosa em produção.

4. Ambos os modelos demonstram viabilidade para implantação como **xApps no Near-RT RIC** da arquitetura O-RAN, com latência de inferência compatível com o requisito de < 1 segundo.

5. A principal limitação é o **tamanho reduzido do dataset**, que limita a generalização. Em produção, datasets com milhares de amostras coletadas via interface E2 permitiriam modelos com desempenho significativamente superior.

---

## Referências

1. O-RAN Alliance. (2026). *O-RAN Working Group 1 Use Cases Detailed Specification v20.00*.
2. O-RAN Alliance. (2026). *O-RAN Working Group 1 Use Cases Analysis Report v20.00*.
3. O-RAN Alliance. (2021). *O-RAN Working Group 2 AI/ML Workflow Description and Requirements 1.03*. *(Versão adotada como material de apoio do curso; versões posteriores formalizaram as entidades Model Training Host, Model Serving Host e Model Repository.)*
4. Burkov, A. (2019). *The Hundred-Page Machine Learning Book*. Andriy Burkov.
5. 3GPP. (2025). *AI/ML for NG-RAN*. Disponível em: https://www.3gpp.org/news-events/3gpp-news/ai-ml-2025
6. 3GPP TS 32.500. *Telecommunication management; Self-Organizing Networks (SON)*.
7. Scikit-learn Developers. (2024). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.

---

*Todos os resultados apresentados neste relatório foram gerados pelo script `src/throughput_prediction.py`, reprodutíveis através do ambiente virtual configurado no projeto.*
