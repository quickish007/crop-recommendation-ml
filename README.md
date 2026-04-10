# Crop Recommendation ML Project

A machine learning project that recommends the most suitable crop to grow based on soil and climate conditions. Three classifiers are trained, tuned, and compared to find the best-performing model.

## Dataset

**File:** `Crop_Recommendation.csv`  
**Samples:** 2,200 (100 per crop class)  
**Classes:** 22 crops

| Feature | Description |
|---|---|
| Nitrogen | Nitrogen content in soil |
| Phosphorus | Phosphorus content in soil |
| Potassium | Potassium content in soil |
| Temperature | Ambient temperature (°C) |
| Humidity | Relative humidity (%) |
| pH_Value | Soil pH level |
| Rainfall | Rainfall (mm) |
| **Crop** | **Target label** |

**Supported crops:** Rice, Maize, ChickPea, KidneyBeans, PigeonPeas, MothBeans, MungBean, Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

## Project Workflow

1. **Exploratory Data Analysis** — distribution and class balance check
2. **Feature Selection** — correlation heatmap; `Phosphorus` dropped due to high correlation with other features
3. **Preprocessing** — `StandardScaler` applied to all remaining features
4. **Model Training & Tuning** — `GridSearchCV` with 5-fold cross-validation on three classifiers
5. **Evaluation** — Accuracy and weighted F1-score compared across models

## Models & Results

| Classifier | Best Hyperparameters | Accuracy | F1 Score (Weighted) |
|---|---|---|---|
| Decision Tree | criterion=gini, max_depth=None, min_samples_leaf=2, min_samples_split=5 | 96.14% | 0.9612 |
| XGBoost | learning_rate=0.1, max_depth=3, n_estimators=100 | **98.86%** | **0.9886** |
| Random Forest | max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=100 | 98.86% | 0.9863 |

**Winner: XGBoost** — highest weighted F1-score at 98.86%

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Usage

Open and run `crop_project.ipynb` in Jupyter Notebook or JupyterLab:

```bash
jupyter notebook crop_project.ipynb
```

## Repository Structure

```
Crop_Reco_ML_Project/
├── crop_project.ipynb        # Main notebook with full analysis
├── Crop_Recommendation.csv   # Dataset
└── README.md
```
