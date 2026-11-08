"""
Análisis Comparativo de Métodos de Aprendizaje Automático
Clasificación Tabular de Exoplanetas (NASA Exoplanet Archive)
Proyecto Integrador - Fase 3
Archivo: evaluacion_modelos_exoplanetas.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def main() -> None:
    # 1. Carga y Filtrado de Datos
    print("Cargando datos del archivo CSV...")
    df = pd.read_csv("cumulative_2026.10.19_20.00.40.csv", comment="#")

    target_col = "koi_disposition"
    df = df[df[target_col].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()

    # Codificación binaria (0 = FALSE POSITIVE, 1 = CONFIRMED)
    df["target"] = df[target_col].map({"FALSE POSITIVE": 0, "CONFIRMED": 1})

    # 2. Selección de Características (13 variables físicas astronómicas)
    features = [
        "koi_period",
        "koi_duration",
        "koi_depth",
        "koi_prad",
        "koi_ror",
        "koi_steff",
        "koi_slogg",
        "koi_smet",
        "koi_srad",
        "koi_smass",
        "koi_model_snr",
        "koi_model_dof",
        "koi_model_chisq",
    ]

    features_present = [f for f in features if f in df.columns]
    X = df[features_present].values
    y = df["target"].values

    print(f"Características utilizadas: {len(features_present)}")

    # 3. Preprocesamiento (Imputación por mediana y Escalado Robusto)
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    # Partición estratificada (80% entrenamiento, 20% prueba)
    X_train, _X_test, y_train, _y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Tratamiento del desbalance con SMOTE
    print("Aplicando SMOTE para balanceo de clases...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 5. Definición de los 7 Modelos de la Suite
    models = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            random_state=42
        ),
    }

    # 6. Validación Cruzada Estratificada de 10 pliegues (10-fold CV)
    results = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    print("Entrenando modelos y evaluando con 10-fold CV...\n")
    for name, model in models.items():
        cv_res = cross_validate(
            model,
            X_train_res,
            y_train_res,
            cv=cv,
            scoring=["f1", "roc_auc"],
        )

        f1_mean = cv_res["test_f1"].mean()
        f1_std = cv_res["test_f1"].std()
        roc_mean = cv_res["test_roc_auc"].mean()
        roc_std = cv_res["test_roc_auc"].std()

        results.append(
            {
                "Condición o modelo": name,
                "n": len(X_train_res),
                "F1-Score": f"{f1_mean:.3f}",
                "IC (F1)": f"±{f1_std:.3f}",
                "ROC-AUC": f"{roc_mean:.3f}",
                "IC (AUC)": f"±{roc_std:.3f}",
            }
        )

    # Mostrar la tabla formateada en consola
    results_df = pd.DataFrame(results)
    print("--- Tabla 1: Comparativa de desempeño ---")
    print(results_df.to_string(index=False))
    print("\n")

    # 7. Visualización de Importancia de Características (Random Forest)
    print("Generando Gráfica de Importancia de características...")
    rf = models["Random Forest"]
    rf.fit(X_train_res, y_train_res)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=importances[indices],
        y=np.array(features_present)[indices],
        palette="viridis",
    )
    plt.title(
        "Figura 1. Importancia relativa de las características físicas (Random Forest)"
    )
    plt.xlabel("Importancia Relativa (Gini Impurity)")
    plt.ylabel("Característica Astrofísica")
    plt.tight_layout()

    # Guardar la gráfica para el reporte escrito
    plt.savefig("figura1_importancia.png", dpi=300)
    print("Gráfica guardada como 'figura1_importancia.png'.")
    plt.show()


if __name__ == "__main__":
    main()