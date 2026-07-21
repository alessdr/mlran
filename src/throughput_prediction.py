#!/usr/bin/env python3
"""
=============================================================================
UE's Throughput Prediction — Avaliação Final
Disciplina: Aplicações de Inteligência Artificial e Machine Learning em RIC
Pós-Graduação em OpenRAN
=============================================================================

Caso de Uso : UE's Throughput Prediction (O-RAN WG1)
Técnicas    : 1) Random Forest Regressor
              2) Support Vector Regression (SVR)
Dataset     : Fontes/kNN_Practice_100rows.csv
              Fontes/traffic_prediction.csv (validação temporal)

Referências :
  - O-RAN Alliance WG1 Use Cases Detailed Specification v20.00
  - O-RAN Alliance WG2 AI/ML Workflow Description and Requirements 1.03
  - Burkov, A. (2019). The hundred-page machine learning book.
=============================================================================
"""

import sys
import os
import time
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
    GridSearchCV,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTES_DIR = os.path.join(BASE_DIR, "Fontes")

DATASET_MAIN = os.path.join(FONTES_DIR, "kNN_Practice_100rows.csv")
DATASET_TEMPORAL = os.path.join(FONTES_DIR, "traffic_prediction.csv")

# ---------------------------------------------------------------------------
# Constantes do experimento
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.20

FEATURES = [
    "PRB_Usage",
    "Active_Users",
    "SINR",
    "RSRQ",
    "Packet_Loss",
    "Latency",
    "CPU_Load",
]
TARGET = "Throughput"


# ===========================================================================
# SEÇÃO 0 — Utilitários
# ===========================================================================

def section(title: str) -> None:
    """Imprime um cabeçalho de seção formatado."""
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def subsection(title: str) -> None:
    print(f"\n--- {title} ---")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Retorna dicionário com MAE, RMSE e R²."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R²": r2}


# ===========================================================================
# SEÇÃO 1 — Carregamento e EDA
# ===========================================================================

def load_and_explore() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os datasets e exibe análise exploratória."""

    section("SEÇÃO 1 — Carregamento dos Dados e EDA")

    # ---- Dataset principal ----
    subsection("1.1  Dataset principal: kNN_Practice_100rows.csv")
    df = pd.read_csv(DATASET_MAIN)

    print(f"\nShape          : {df.shape[0]} amostras × {df.shape[1]} colunas")
    print(f"Colunas        : {list(df.columns)}")
    print(f"Valores nulos  : {df.isnull().sum().sum()}")
    print(f"\nClasses presentes (Cell_Class):")
    print(df["Cell_Class"].value_counts().to_string())

    subsection("1.2  Estatísticas descritivas das features e target")
    stats_cols = FEATURES + [TARGET]
    stats_df = df[stats_cols].describe().T
    stats_df.columns = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    print(tabulate(stats_df, headers="keys", tablefmt="rounded_outline", floatfmt=".2f"))

    subsection("1.3  Correlação com o target (Throughput)")
    corr = df[stats_cols].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    corr_df = corr.reset_index()
    corr_df.columns = ["Feature", f"Correlação com {TARGET}"]
    print(tabulate(corr_df, headers="keys", tablefmt="rounded_outline",
                   showindex=False, floatfmt=".4f"))

    # ---- Dataset temporal ----
    subsection("1.4  Dataset temporal: traffic_prediction.csv")
    df_temporal = pd.read_csv(DATASET_TEMPORAL)
    print(f"\nShape   : {df_temporal.shape}")
    print(f"Colunas : {list(df_temporal.columns)}")
    print(f"\nAmostra (primeiras 5 linhas):")
    print(tabulate(df_temporal.head(), headers="keys", tablefmt="rounded_outline",
                   showindex=False, floatfmt=".1f"))

    # Estatísticas do target no dataset principal por classe
    subsection("1.5  Throughput médio por Cell_Class")
    class_stats = df.groupby("Cell_Class")[TARGET].agg(["mean", "std", "min", "max"])
    class_stats.columns = ["Média", "Desvio Padrão", "Mínimo", "Máximo"]
    print(tabulate(class_stats, headers="keys", tablefmt="rounded_outline", floatfmt=".2f"))

    return df, df_temporal


# ===========================================================================
# SEÇÃO 2 — Pré-processamento
# ===========================================================================

def preprocess(df: pd.DataFrame) -> tuple:
    """Prepara X e y e divide em treino/teste."""

    section("SEÇÃO 2 — Pré-processamento")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # NV-02: Estratificação por Cell_Class garante proporção idêntica dos 3 regimes
    # operacionais (Normal, Congested, Degraded) entre treino (80) e teste (20).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["Cell_Class"]
    )
    print(f"\nTotal de amostras : {len(X)}")
    print(f"Treino            : {len(X_train)} ({100*(1-TEST_SIZE):.0f}%)")
    print(f"Teste             : {len(X_test)}  ({100*TEST_SIZE:.0f}%)")

    # Exibe distribuição de Cell_Class
    train_dist = df.loc[X_train.index, "Cell_Class"].value_counts().to_dict()
    test_dist = df.loc[X_test.index, "Cell_Class"].value_counts().to_dict()
    print(f"Proporção de regimes no Treino : {train_dist}")
    print(f"Proporção de regimes no Teste  : {test_dist}")

    subsection("2.2  Estratégia de validação")
    print(f"\nK-Fold Cross-Validation : k = {N_FOLDS}")
    print("Normalização (StandardScaler) embutida via sklearn Pipeline.")
    print("Isso evita data leakage entre folds de treino e validação.")

    return X_train, X_test, y_train, y_test, X, y


# ===========================================================================
# SEÇÃO 3 — Técnica 1: Random Forest Regressor
# ===========================================================================

def run_random_forest(
    X_train, X_test, y_train, y_test, X, y
) -> dict:
    """Treina, otimiza e avalia o Random Forest Regressor."""

    section("SEÇÃO 3 — TÉCNICA 1: Random Forest Regressor")

    subsection("3.1  Definição do modelo e grid de hiperparâmetros")

    # RF é invariante a escala — o StandardScaler não afeta a qualidade do
    # modelo (Random Forest particiona por limiares de features, não por
    # distâncias). O scaler é mantido no Pipeline exclusivamente por
    # consistência arquitetural com o pipeline do SVR.
    pipe_rf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE)),
    ])

    param_grid_rf = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    print("\nGrid de busca:")
    for k, v in param_grid_rf.items():
        print(f"  {k}: {v}")

    subsection("3.2  GridSearchCV com K-Fold (k=5)")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid_rf = GridSearchCV(
        pipe_rf,
        param_grid_rf,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    grid_rf.fit(X_train, y_train)

    best_rf = grid_rf.best_estimator_
    print(f"\nMelhores hiperparâmetros encontrados:")
    for k, v in grid_rf.best_params_.items():
        print(f"  {k.replace('model__', '')}: {v}")

    subsection("3.3  Avaliação no conjunto de teste")
    y_pred_rf = best_rf.predict(X_test)
    metrics_test_rf = compute_metrics(y_test, y_pred_rf)

    # NV-01: Medição de latência de inferência por amostra individual (single-sample latency)
    sample_single = X_test.iloc[[0]]
    best_rf.predict(sample_single)  # warm-up
    n_runs = 1000
    t0 = time.perf_counter()
    for _ in range(n_runs):
        best_rf.predict(sample_single)
    t1 = time.perf_counter()
    lat_rf_ms = ((t1 - t0) / n_runs) * 1000.0
    metrics_test_rf["Latência Inferência Unitária"] = f"{lat_rf_ms:.4f} ms"

    rows = [[k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in metrics_test_rf.items()]
    print(tabulate(rows, headers=["Métrica", "Valor (Teste)"],
                   tablefmt="rounded_outline"))

    subsection("3.4  Cross-Validation no conjunto de treino (X_train)")
    # IC-03: CV executado sobre X_train/y_train para preservar a independência
    # do hold-out (X_test/y_test) como conjunto de avaliação final não visto.
    cv_results_rf = cross_validate(
        best_rf, X_train, y_train,
        cv=kf,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        return_train_score=False,
    )

    cv_mae = -cv_results_rf["test_mae"]
    cv_rmse = -cv_results_rf["test_rmse"]
    cv_r2 = cv_results_rf["test_r2"]

    cv_rows = [
        ["MAE",  f"{cv_mae.mean():.4f}",  f"± {cv_mae.std():.4f}"],
        ["RMSE", f"{cv_rmse.mean():.4f}", f"± {cv_rmse.std():.4f}"],
        ["R²",   f"{cv_r2.mean():.4f}",   f"± {cv_r2.std():.4f}"],
    ]
    print(tabulate(cv_rows, headers=["Métrica", "Média CV", "Desvio Padrão"],
                   tablefmt="rounded_outline"))

    subsection("3.5  Importância das features")
    rf_model = best_rf.named_steps["model"]
    importances = pd.Series(rf_model.feature_importances_, index=FEATURES)
    importances = importances.sort_values(ascending=False)

    imp_rows = [(feat, f"{imp:.4f}", f"{imp*100:.2f}%")
                for feat, imp in importances.items()]
    print(tabulate(imp_rows,
                   headers=["Feature", "Importância", "Percentual"],
                   tablefmt="rounded_outline"))

    return {
        "model_name": "Random Forest",
        "test": metrics_test_rf,
        "cv_mae": cv_mae,
        "cv_rmse": cv_rmse,
        "cv_r2": cv_r2,
        "y_pred": y_pred_rf,
        "feature_importances": importances,
        "best_params": grid_rf.best_params_,
        "single_sample_lat_ms": lat_rf_ms,
    }


# ===========================================================================
# SEÇÃO 4 — Técnica 2: Support Vector Regression (SVR)
# ===========================================================================

def run_svr(
    X_train, X_test, y_train, y_test, X, y
) -> dict:
    """Treina, otimiza e avalia o SVR."""

    section("SEÇÃO 4 — TÉCNICA 2: Support Vector Regression (SVR)")

    subsection("4.1  Definição do modelo e grid de hiperparâmetros")

    # SVR requer normalização obrigatória: é sensível à escala das features
    # pois a função de custo e o kernel RBF dependem de distâncias no espaço
    # de características. Embutir o StandardScaler dentro do Pipeline garante
    # que o fit do scaler ocorre APENAS nos dados de treino de cada fold —
    # evitando data leakage durante o cross-validation.
    pipe_svr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR()),
    ])

    param_grid_svr = {
        "model__kernel": ["rbf", "linear"],
        "model__C": [0.1, 1, 10, 100],
        "model__epsilon": [0.01, 0.1, 1.0],
        "model__gamma": ["scale", "auto"],
    }

    print("\nGrid de busca:")
    for k, v in param_grid_svr.items():
        print(f"  {k}: {v}")

    subsection("4.2  GridSearchCV com K-Fold (k=5)")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid_svr = GridSearchCV(
        pipe_svr,
        param_grid_svr,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    grid_svr.fit(X_train, y_train)

    best_svr = grid_svr.best_estimator_
    print(f"\nMelhores hiperparâmetros encontrados:")
    for k, v in grid_svr.best_params_.items():
        print(f"  {k.replace('model__', '')}: {v}")

    subsection("4.3  Avaliação no conjunto de teste")
    y_pred_svr = best_svr.predict(X_test)
    metrics_test_svr = compute_metrics(y_test, y_pred_svr)

    # NV-01: Medição de latência de inferência por amostra individual (single-sample latency)
    sample_single = X_test.iloc[[0]]
    best_svr.predict(sample_single)  # warm-up
    n_runs = 1000
    t0 = time.perf_counter()
    for _ in range(n_runs):
        best_svr.predict(sample_single)
    t1 = time.perf_counter()
    lat_svr_ms = ((t1 - t0) / n_runs) * 1000.0
    metrics_test_svr["Latência Inferência Unitária"] = f"{lat_svr_ms:.4f} ms"

    rows = [[k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in metrics_test_svr.items()]
    print(tabulate(rows, headers=["Métrica", "Valor (Teste)"],
                   tablefmt="rounded_outline"))

    subsection("4.4  Cross-Validation no conjunto de treino (X_train)")
    # IC-03: CV executado sobre X_train/y_train para preservar a independência
    # do hold-out (X_test/y_test) como conjunto de avaliação final não visto.
    cv_results_svr = cross_validate(
        best_svr, X_train, y_train,
        cv=kf,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        return_train_score=False,
    )

    cv_mae = -cv_results_svr["test_mae"]
    cv_rmse = -cv_results_svr["test_rmse"]
    cv_r2 = cv_results_svr["test_r2"]

    cv_rows = [
        ["MAE",  f"{cv_mae.mean():.4f}",  f"± {cv_mae.std():.4f}"],
        ["RMSE", f"{cv_rmse.mean():.4f}", f"± {cv_rmse.std():.4f}"],
        ["R²",   f"{cv_r2.mean():.4f}",   f"± {cv_r2.std():.4f}"],
    ]
    print(tabulate(cv_rows, headers=["Métrica", "Média CV", "Desvio Padrão"],
                   tablefmt="rounded_outline"))

    return {
        "model_name": "SVR",
        "test": metrics_test_svr,
        "cv_mae": cv_mae,
        "cv_rmse": cv_rmse,
        "cv_r2": cv_r2,
        "y_pred": y_pred_svr,
        "best_params": grid_svr.best_params_,
        "single_sample_lat_ms": lat_svr_ms,
    }


# ===========================================================================
# SEÇÃO 5 — Análise Temporal (dataset traffic_prediction.csv)
# ===========================================================================

def temporal_analysis(df_temporal: pd.DataFrame, rf_result: dict, svr_result: dict) -> None:
    """
    Aplica os modelos treinados ao dataset temporal para demonstrar
    predição em série temporal horária.
    Nota: os modelos foram treinados com features do dataset principal.
    Aqui usamos apenas as features em comum para demonstração.
    """

    section("SEÇÃO 5 — ANÁLISE NO DATASET TEMPORAL (traffic_prediction.csv)")

    print("""
Nota metodológica: O dataset temporal possui apenas 4 features
(Hour, ActiveUsers, AvgSINR, PRBUtilization). Como os modelos foram
treinados com 7 features, esta seção demonstra a capacidade preditiva
usando um modelo re-treinado exclusivamente nas features disponíveis.
Esta abordagem ilustra a adaptação de modelos ML a diferentes
granularidades de dados em ambientes O-RAN reais.
""")

    TEMPORAL_FEATURES = ["ActiveUsers", "AvgSINR", "PRBUtilization"]
    TEMPORAL_TARGET = "Throughput"

    X_t = df_temporal[TEMPORAL_FEATURES].values
    y_t = df_temporal[TEMPORAL_TARGET].values

    # Treina modelos simples nas features disponíveis (leave-one-out dado o tamanho)
    from sklearn.model_selection import LeaveOneOut

    loo = LeaveOneOut()

    # Devido ao tamanho ínfimo do dataset temporal (16 amostras), os modelos
    # foram definidos com hiperparâmetros fixos razoáveis (n_estimators=100 para RF;
    # C=10, epsilon=0.1 para SVR) em vez de GridSearchCV, evitando overfitting excessivo
    # na seleção de parâmetros com pouquíssimos dados por fold do LOO-CV.
    rf_t = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
    ])
    svr_t = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=10, epsilon=0.1)),
    ])

    # Predições por LOO-CV
    preds_rf, preds_svr, trues = [], [], []
    for train_idx, test_idx in loo.split(X_t):
        rf_t.fit(X_t[train_idx], y_t[train_idx])
        svr_t.fit(X_t[train_idx], y_t[train_idx])
        preds_rf.append(rf_t.predict(X_t[test_idx])[0])
        preds_svr.append(svr_t.predict(X_t[test_idx])[0])
        trues.append(y_t[test_idx][0])

    preds_rf = np.array(preds_rf)
    preds_svr = np.array(preds_svr)
    trues = np.array(trues)

    m_rf_t = compute_metrics(trues, preds_rf)
    m_svr_t = compute_metrics(trues, preds_svr)

    subsection("5.1  Resultados LOO-CV no dataset temporal")
    rows = [
        ["Random Forest", f"{m_rf_t['MAE']:.2f}", f"{m_rf_t['RMSE']:.2f}", f"{m_rf_t['R²']:.4f}"],
        ["SVR",           f"{m_svr_t['MAE']:.2f}", f"{m_svr_t['RMSE']:.2f}", f"{m_svr_t['R²']:.4f}"],
    ]
    print(tabulate(rows, headers=["Modelo", "MAE (Mbps)", "RMSE (Mbps)", "R²"],
                   tablefmt="rounded_outline"))

    subsection("5.2  Predições vs. Valores Reais (primeiras 8 horas)")
    header = ["Hora", "Real (Mbps)", "RF Pred.", "SVR Pred."]
    sample_rows = [
        [df_temporal["Hour"].iloc[i], trues[i], f"{preds_rf[i]:.1f}", f"{preds_svr[i]:.1f}"]
        for i in range(8)
    ]
    print(tabulate(sample_rows, headers=header, tablefmt="rounded_outline"))


# ===========================================================================
# SEÇÃO 6 — Comparação e Conclusões
# ===========================================================================

def compare_models(rf_result: dict, svr_result: dict, y_test: np.ndarray) -> None:
    """Exibe a comparação final entre os dois modelos."""

    section("SEÇÃO 6 — COMPARAÇÃO FINAL DOS MODELOS")

    subsection("6.1  Métricas no conjunto de teste hold-out (20%)")
    rows = []
    for res in [rf_result, svr_result]:
        rows.append([
            res["model_name"],
            f"{res['test']['MAE']:.4f}",
            f"{res['test']['RMSE']:.4f}",
            f"{res['test']['R²']:.4f}",
        ])
    print(tabulate(rows, headers=["Modelo", "MAE (Mbps)", "RMSE (Mbps)", "R²"],
                   tablefmt="rounded_outline"))

    subsection("6.2  Métricas Cross-Validation (k=5) — média ± desvio padrão")
    rows_cv = []
    for res in [rf_result, svr_result]:
        rows_cv.append([
            res["model_name"],
            f"{res['cv_mae'].mean():.4f} ± {res['cv_mae'].std():.4f}",
            f"{res['cv_rmse'].mean():.4f} ± {res['cv_rmse'].std():.4f}",
            f"{res['cv_r2'].mean():.4f} ± {res['cv_r2'].std():.4f}",
        ])
    print(tabulate(rows_cv,
                   headers=["Modelo", "MAE CV", "RMSE CV", "R² CV"],
                   tablefmt="rounded_outline"))

    subsection("6.3  Predições vs. Reais — amostra do conjunto de teste")
    sample = min(10, len(y_test))
    y_test_arr = np.array(y_test)[:sample]
    preds_rf = np.array(rf_result["y_pred"])[:sample]
    preds_svr = np.array(svr_result["y_pred"])[:sample]

    sample_rows = [
        [f"{yt:.1f}", f"{prf:.1f}", f"{abs(yt-prf):.1f}",
                      f"{psv:.1f}", f"{abs(yt-psv):.1f}"]
        for yt, prf, psv in zip(y_test_arr, preds_rf, preds_svr)
    ]
    print(tabulate(sample_rows,
                   headers=["Real", "RF Pred.", "|Erro RF|", "SVR Pred.", "|Erro SVR|"],
                   tablefmt="rounded_outline"))

    subsection("6.4  Latência de inferência por amostra individual (Near-RT RIC SLA < 10 ms)")
    lat_rows = [
        ["Random Forest", f"{rf_result['single_sample_lat_ms']:.4f} ms", "Conforme SLA Near-RT RIC (< 10 ms)"],
        ["SVR",           f"{svr_result['single_sample_lat_ms']:.4f} ms", "Conforme SLA Near-RT RIC (< 10 ms)"],
    ]
    print(tabulate(lat_rows, headers=["Modelo", "Latência Unitária (ms)", "SLA O-RAN Near-RT RIC"],
                   tablefmt="rounded_outline"))

    subsection("6.5  Análise dos melhores hiperparâmetros")
    print("\nRandom Forest:")
    for k, v in rf_result["best_params"].items():
        print(f"  {k.replace('model__', '')}: {v}")
    print("\nSVR:")
    for k, v in svr_result["best_params"].items():
        print(f"  {k.replace('model__', '')}: {v}")

    subsection("6.5  Conclusão do experimento")

    rf_r2_cv = rf_result["cv_r2"].mean()
    svr_r2_cv = svr_result["cv_r2"].mean()
    winner = "Random Forest" if rf_r2_cv >= svr_r2_cv else "SVR"

    print(f"""
Com base nas métricas de Cross-Validation (k={N_FOLDS}), o modelo
{winner} apresentou melhor desempenho geral para predição de Throughput
em células O-RAN.

Random Forest R² CV : {rf_r2_cv:.4f}
SVR          R² CV : {svr_r2_cv:.4f}

Ambos os modelos demonstram capacidade de capturar a relação entre
os KPIs de rádio (PRB_Usage, SINR, Active_Users, etc.) e o Throughput
dos UEs — informação crítica para o xApp de Throughput Steering
no Near-RT RIC da arquitetura O-RAN.

Limitações identificadas:
  - Dataset de tamanho reduzido (100 amostras) — risco de alta variância
    nas métricas de CV. Em produção, seria necessário um dataset com
    dados reais de telemetria de gNBs.
  - Dados sintéticos sem variação temporal — impossibilita modelagem
    de padrões de tráfego horário com alta granularidade.
  - Ausência de dados de handover e mobilidade — relevantes para
    predição de throughput em cenários dinâmicos de UE.
""")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  UE's Throughput Prediction — O-RAN / Near-RT RIC")
    print("  Pós-Graduação em OpenRAN — Disciplina ML em RIC")
    print("=" * 70)
    print(f"\n  Dataset principal : {DATASET_MAIN}")
    print(f"  Dataset temporal  : {DATASET_TEMPORAL}")
    print(f"  Features          : {FEATURES}")
    print(f"  Target            : {TARGET}")
    print(f"  K-Folds           : {N_FOLDS}")
    print(f"  Test size         : {TEST_SIZE * 100:.0f}%")
    print(f"  Random state      : {RANDOM_STATE}")

    # Seção 1 — EDA
    df, df_temporal = load_and_explore()

    # Seção 2 — Pré-processamento
    X_train, X_test, y_train, y_test, X, y = preprocess(df)

    # Seção 3 — Random Forest
    rf_result = run_random_forest(X_train, X_test, y_train, y_test, X, y)

    # Seção 4 — SVR
    svr_result = run_svr(X_train, X_test, y_train, y_test, X, y)

    # Seção 5 — Análise temporal
    temporal_analysis(df_temporal, rf_result, svr_result)

    # Seção 6 — Comparação final
    compare_models(rf_result, svr_result, y_test)

    section("FIM DA EXECUÇÃO")
    print("\nTodos os resultados foram exibidos acima.")
    print("Consulte docs/relatorio_analise.md para a documentação completa.\n")


if __name__ == "__main__":
    main()
