# Evaluación Comparativa de Algoritmos de ML para Clasificación de Exoplanetas

Este repositorio contiene el código fuente, la configuración del entorno y los artefactos generados para la **Etapa 3 (Resultados)** del manuscrito de investigación titulado:

> **"Evaluación comparativa de algoritmos de aprendizaje automático para la clasificación de candidatos a exoplanetas"**  
> **Asignatura:** Investigación en Matemáticas Aplicadas y Computación (IMAC) — 7° Semestre, Grupo 1701  
> **Institución:** Facultad de Estudios Superiores Acatlán (UNAM)  
> **Autores:** González Sánchez Raúl Alejandro & Martínez Rodríguez Fernando  

---

## Descripción del Proyecto

El objetivo de este proyecto es evaluar y comparar el desempeño predictivo de 7 clasificadores de Aprendizaje Automático supervisado sobre la tabla acumulativa del **NASA Exoplanet Archive (Kepler Data Release 25 - DR25)**. 

Para evitar el sesgo conocido como *data leakage* (fuga de información), se han excluido rigurosamente todas las banderas de diagnóstico con intervención humana directa (`koi_fpflag_*`, `koi_score`, `kepler_name`), evaluando a los modelos basándose exclusivamente en **13 parámetros físicos, orbitales y astronómicos puros**.

### Modelos Evaluados (7 Algoritmos)
1. **Gaussian Naive Bayes**
2. **Árbol de Decisión**
3. **Random Forest**
4. **XGBoost**
5. **LightGBM**
6. **CatBoost**
7. **Histogram Gradient Boosting**

---

## Requisitos Previos e Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/FernandoMR10/Proyecto-Integrador-Codigos.git
cd Proyecto-Integrador-Codigos
```

### 2. Crear y activar el entorno virtual (`.venv`)

En **Windows (PowerShell)**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

En **Linux / macOS (Bash/Zsh)**:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## Estructura del Repositorio

```text
Proyecto-Integrador-Codigos/
├── .venv/                              # Entorno virtual de Python (ignorado por Git)
├── cumulative_2026.10.19_20.00.40.csv  # Base de datos del NASA Exoplanet Archive (DR25)
├── evaluacion_modelos_exoplanetas.py   # Script principal de preprocesamiento, SMOTE y entrenamiento
├── requirements.txt                    # Dependencias con versiones exactas
├── .gitignore                          # Configuración de exclusión para Git
├── figura1_importancia.png             # Gráfica generada de Importancia de Características
└── README.md                           # Documentación del proyecto
```

---

## Ejecución del Código

Asegúrate de tener el archivo de datos de Kepler (`cumulative_2026.10.19_20.00.40.csv`) en la raíz del proyecto y el entorno virtual activo. Luego ejecuta:

```bash
python evaluacion_modelos_exoplanetas.py
```

### ¿Qué hace el script?
1. **Preprocesamiento:** Depura el dataset eliminando variables con *data leakage* y filtra la clase objetivo en una clasificación binaria (`CONFIRMED` vs `FALSE POSITIVE`).
2. **Imputación y Escalado:** Imputa valores faltantes mediante la mediana de cada atributo y escala las variables con `RobustScaler`.
3. **Balanceo de Clases:** Aplica **SMOTE** sobre el subconjunto de entrenamiento (80%).
4. **Validación Cruzada:** Realiza una **Validación Cruzada Estratificada de 10 pliegues (10-fold CV)** reportando F1-Score y ROC-AUC con sus respectivos intervalos de dispersión ($\pm\sigma$).
5. **Generación de Gráficas:** Exporta automáticamente la gráfica `figura1_importancia.png` correspondiente a la reducción de impureza de Gini obtenida por el modelo de Random Forest.

---

## Resultados Obtenidos (Etapa 3)

| Condición o Modelo | $n$ | Métrica (F1 / ROC-AUC) | Resultado | IC o Incertidumbre |
| :--- | :---: | :---: | :---: | :---: |
| Gaussian Naive Bayes | 7,742 | F1-Score / ROC-AUC | 0.797 / 0.835 | $\pm$ 0.009 / $\pm$ 0.012 |
| Árbol de Decisión | 7,742 | F1-Score / ROC-AUC | 0.895 / 0.894 | $\pm$ 0.010 / $\pm$ 0.010 |
| Random Forest | 7,742 | F1-Score / ROC-AUC | 0.936 / 0.980 | $\pm$ 0.006 / $\pm$ 0.004 |
| XGBoost | 7,742 | F1-Score / ROC-AUC | 0.935 / 0.980 | $\pm$ 0.007 / $\pm$ 0.004 |
| LightGBM | 7,742 | F1-Score / ROC-AUC | 0.936 / 0.981 | $\pm$ 0.007 / $\pm$ 0.004 |
| CatBoost | 7,742 | F1-Score / ROC-AUC | 0.937 / 0.981 | $\pm$ 0.007 / $\pm$ 0.004 |
| **Hist Gradient Boosting** | **7,742** | **F1-Score / ROC-AUC** | **0.937 / 0.981** | **$\pm$ 0.008 / $\pm$ 0.004** |

---

## Licencia y Fuente de Datos
* **Fuente de Datos:** [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) (Kepler Cumulative KOI Table - DR25). Datos de acceso público y libre reutilización.
* **Licencia:** Este proyecto se distribuye bajo la licencia MIT.