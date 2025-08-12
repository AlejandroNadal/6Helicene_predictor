# 6Helicene Predictor · GA + Deep Learning
[![Made with Python](https://img.shields.io/badge/Python-3.10%2B-informational)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/) [![Notebooks](https://img.shields.io/badge/Jupyter-Notebooks-blue)](https://jupyter.org/) [![License](https://img.shields.io/badge/License-Choose--one-lightgrey)](#licencia)

Predicción de **fuerza rotatoria** ($R_{max}$, $R_{min}$) y **longitud de onda** ($\lambda$) para derivados **halogenados de [6]heliceno** (1–16 sustituyentes), combinando **modelos N-body** con **regresión cuantílica** para incertidumbre y un **algoritmo genético** (GA) que explora el espacio molecular de forma eficiente.

> Objetivo: identificar candidatos con alto $R_{max}$ (y controlar $R_{min}$) más allá de 6 sustituyentes, estimando intervalos de predicción para priorizar moléculas **prometedoras y confiables**.

---

## ✨ Características
- **Modelos de aprendizaje profundo (TensorFlow/Keras)** para $R_{max}$, $R_{min}$ y $\lambda$.
- **Incertidumbre** vía **regresión cuantílica** → intervalos (lower/upper) y media/σ.
- **Algoritmo genético** con **penalización** para n>6, capaz de listar **Top‑K candidatos por n**.
- **Reproducibilidad**: métricas, hiperparámetros y curvas de pérdida en `experiments/**/logs/`.
- **Modelos preentrenados** en `experiments/**/models/` (quantiles y CV).
- **Notebooks** listos para análisis, entrenamiento y validación cruzada.

---

## 📊 Resultados (resumen)
| Tarea   | MAE | RMSE | MAPE | R² |
|:-------:|----:|-----:|-----:|---:|
| Rmax    | 24.24 | 42.06 | 6.12% | 0.899 |
| Rmin    | 23.44 | 31.46 | 8.55% | 0.934 |
| λ       | 1.90 nm | 8.34 nm | 0.60% | 0.554 |

> Hiperparámetros óptimos (Keras Tuner):  
> **Rmax** → units=256, layers=1, dropout=0.0, lr=0.005  
> **Rmin** → units=256, layers=3, dropout=0.3, lr=0.005  
> **λ** → units=512, layers=1, dropout=0.0, lr=0.005

---

## 🗂️ Estructura del repositorio
```text
6Helicene_predictor/
├─ ga_with_uncertainty.py          # Búsqueda genética con intervalos (quantiles)
├─ experiments/
│  ├─ Rmax/
│  │  ├─ logs/                     # best_hp.json, final_metrics.json, loss_history.json
│  │  └─ models/                   # model_lower.h5, model_upper.h5, CV_Models/...
│  ├─ Rmin/
│  │  ├─ logs/
│  │  └─ models/                   # Rmin_model_lower.h5, Rmin_model_upper.h5, ...
│  └─ Lambda/
│     ├─ logs/
│     └─ models/                   # model_lower.h5, model_upper.h5, ...
├─ data/
│  └─ processed/
│     └─ Lambda_pd/
│        └─ Imax_stats.csv         # medias/std de λ (ejemplo)
├─ notebooks/
│  ├─ Rmax_nb/ ...                 # entrenamiento, CV y análisis
│  ├─ Rmin_nb/ ...
│  └─ Lambda_nb/ ...
└─ ROBERT/                         # utilidades GUI para informes (SHAP, etc.)
```

---

## 🚀 Instalación
Requiere **Python 3.10+**. Crea un entorno y instala dependencias mínimas:

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install tensorflow numpy pandas pillow scikit-learn matplotlib jupyter keras-tuner
```

> Si usas GPU, instala la distribución de TensorFlow adecuada a tu sistema.

---

## 🧪 Uso rápido

### 1) Replicar análisis/entrenamiento desde notebooks
Abre Jupyter y ejecuta, por ejemplo:
- notebooks/Rmax_nb/Rmax_invariant_Nbody_CV.ipynb — **Rmax**: arquitectura invariante y validación cruzada.
- notebooks/Rmax_nb/Quantile_interval_pred.ipynb — **Rmax**: **intervalos** por regresión cuantílica.
- notebooks/Rmax_nb/Data_analysis.ipynb — **Rmax**: análisis exploratorio.
> Hay notebooks análogos para **Rmin** y **λ** en sus respectivas carpetas.

### 2) Búsqueda genética con incertidumbre
El script `ga_with_uncertainty.py` evalúa genotipos con **media** y **σ** a partir de los modelos cuantílicos, y ejecuta un GA por cada número de sustituyentes *(n)*.

**Requisitos previos** (en el directorio del script):
- Modelos cuantílicos para Rmax:  
  `experiments/Rmax/models/model_lower.h5`, `experiments/Rmax/models/model_upper.h5`
- Estadísticos de escala (z‑score) para desnormalizar:  
  Un CSV `Rmax_stats.csv` con columnas: `Rmax_abs_mean`, `Rmax_abs_std`  
  *(si no lo tienes, puedes calcularlos con `notebooks/Rmax_nb/Data_analysis.ipynb`)*

**Ejecutar**:
```bash
python ga_with_uncertainty.py
# genera: ga_top1000_n7_16_penal.csv
```

**Codificación del genotipo (16 posiciones)**  
Vector de enteros en {0,1,2,3,4} para cada posición del [6]heliceno:  
`0=H, 1=F, 2=Cl, 3=Br, 4=I`  *(puedes ajustar el mapeo si cambias el conjunto de halógenos)*

**Salida**  
CSV con las mejores moléculas por *n*:  
`n_subs, rank, genotype, mu_real, sigma_real`  
donde `mu_real ± sigma_real` es la predicción desnormalizada de $R_{max}$.

---

## 📝 Notas de reproducibilidad
- Métricas finales: `experiments/**/logs/final_metrics.json`  
- Hiperparámetros: `experiments/**/logs/best_hp.json`  
- Modelos (SavedModel/H5): `experiments/**/models/`  
- Semillas fijadas en GA y predicción para resultados estables.

---

## 🤝 Contribuir
1. Abre un *issue* con la propuesta/mejora.
2. (Opcional) Crea una rama y adjunta un *pull request* con:
   - descripción clara,
   - pasos para reproducir,
   - ejemplos o capturas si aplica.

---

## 📚 Cita / referencia
Si usas este repositorio en trabajos académicos, cita así (ejemplo):
```
@misc{helicene_predictor,
  title  = {Predicciones de propiedades moleculares mediante Redes Neuronales},
  author = {Alejandro Nadal López-Cepero},
  year   = {2025},
  url    = {https://github.com/<usuario>/6Helicene_predictor}
}
```

---

## 🔐 Licencia

---

## 🙋 FAQ
**¿Qué es la “penalización” del GA para n>6?**  
Una regularización que favorece genotipos con propiedades plausibles/robustas al aumentar el número de sustituyentes, evitando sobreexploración de configuraciones espurias.

**¿Puedo añadir nuevos halógenos?**  
Sí. Amplía el espacio {H,F,Cl,Br,I} y ajusta el *one‑hot* y el mapeo de códigos.

**¿Dónde están los datos brutos?**  
Los CSV procesados están en `data/processed/`. Para datos grandes, usa *Releases* en GitHub.

