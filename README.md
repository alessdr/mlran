# UE's Throughput Prediction — O-RAN / Near-RT RIC

Projeto de avaliação final da disciplina **Aplicações de Inteligência Artificial e Machine Learning em RIC**, do curso de Pós-Graduação em OpenRAN.

## Caso de Uso

**UE's Throughput Prediction** — Predição do throughput de User Equipments (UEs) em células de redes O-RAN, utilizando KPIs de rádio coletados via interface E2 do Near-RT RIC.

## Técnicas de ML Aplicadas

| Técnica | Tipo | R² (Teste) |
|---|---|---|
| Random Forest Regressor | Ensemble (Bagging) | 0.7847 |
| Support Vector Regression (SVR) | Kernel Method (RBF) | 0.7302 |

## Requisitos do Sistema

- **Python**: 3.10 ou superior
- **Sistema Operacional**: macOS, Linux ou Windows
- **Memória RAM**: mínimo 2 GB (recomendado 4 GB)
- **Espaço em disco**: ~500 MB (incluindo virtualenv)

## Estrutura do Projeto

```
MLRan/
├── .gitignore
├── README.md
├── requirements.txt
├── venv/                          # Virtualenv (ignorada pelo git)
├── src/
│   └── throughput_prediction.py   # Script principal
├── docs/
│   └── relatorio_analise.md       # Documentação e análise
└── Fontes/
    ├── kNN_Practice_100rows.csv   # Dataset principal (100 amostras)
    └── traffic_prediction.csv    # Dataset temporal (16 amostras)
```

## Instalação e Configuração

### 1. Clonar ou acessar o projeto

```bash
cd /caminho/para/MLRan
```

### 2. Criar o ambiente virtual

```bash
python3 -m venv venv
```

### 3. Ativar o ambiente virtual

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Instalar as dependências

```bash
pip install -r requirements.txt
```

## Execução

Com o ambiente virtual ativo, execute:

```bash
python src/throughput_prediction.py
```

O script executa automaticamente todas as etapas:

1. **EDA** — Estatísticas descritivas, correlações e análise por classe de célula
2. **Pré-processamento** — Normalização e particionamento treino/teste
3. **Random Forest** — Otimização via GridSearchCV e avaliação com k-fold CV
4. **SVR** — Otimização via GridSearchCV e avaliação com k-fold CV
5. **Análise temporal** — Validação no dataset horário com LOO-CV
6. **Comparação** — Tabela consolidada de métricas e conclusões

### Tempo estimado de execução

| Etapa | Tempo Estimado |
|---|---|
| EDA e pré-processamento | < 1 segundo |
| GridSearchCV Random Forest | ~30–60 segundos |
| GridSearchCV SVR | ~60–120 segundos |
| Análise temporal (LOO-CV) | ~5–10 segundos |
| **Total** | **~2–3 minutos** |

## Dependências

| Biblioteca | Versão | Uso |
|---|---|---|
| `numpy` | 2.2.6 | Operações numéricas |
| `pandas` | 2.2.3 | Manipulação de dados |
| `scikit-learn` | 1.6.1 | Modelos ML, pipelines, métricas |
| `matplotlib` | 3.10.3 | Visualização (disponível para extensões) |
| `seaborn` | 0.13.2 | Visualização estatística (disponível para extensões) |
| `tabulate` | 0.9.0 | Formatação de tabelas no terminal |

## Datasets

| Arquivo | Amostras | Features | Target |
|---|---|---|---|
| `Fontes/kNN_Practice_100rows.csv` | 100 | PRB_Usage, Active_Users, SINR, RSRQ, Packet_Loss, Latency, CPU_Load | Throughput (Mbps) |
| `Fontes/traffic_prediction.csv` | 16 | ActiveUsers, AvgSINR, PRBUtilization | Throughput (Mbps) |

## Resultados Resumidos

### Métricas no Conjunto de Teste (20%)

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 21.64 Mbps | 26.11 Mbps | 0.7847 |
| SVR | 23.42 Mbps | 29.22 Mbps | 0.7302 |

### Cross-Validation k=5

| Modelo | R² CV | 
|---|---|
| Random Forest | 0.6266 ± 0.1928 |
| SVR | 0.6036 ± 0.2210 |

> Para a análise completa, consulte [`docs/relatorio_analise.md`](docs/relatorio_analise.md).

## Referências

- O-RAN Alliance. (2026). *O-RAN WG1 Use Cases Detailed Specification v20.00*
- O-RAN Alliance. (2021). *O-RAN WG2 AI/ML Workflow Description and Requirements 1.03*
- Burkov, A. (2019). *The Hundred-Page Machine Learning Book*
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
- 3GPP. (2025). *AI/ML for NG-RAN* — https://www.3gpp.org/news-events/3gpp-news/ai-ml-2025
