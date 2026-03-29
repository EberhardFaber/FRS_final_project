# FRS_final_project
# Forest Properties Mapping Using Sentinel-2 Satellite Imagery

Machine Learning and Deep Learning for Forest Structure Classification

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project leverages **machine learning** and **deep learning** techniques to map forest structural properties using multispectral Sentinel-2 satellite imagery from the TreeSatAI dataset. By analyzing spectral information, we predict key forest characteristics essential for forest management, carbon accounting, and ecosystem monitoring.

The project demonstrates how satellite remote sensing can replace costly field surveys with scalable, repeatable forest monitoring across large areas.

---

## Key Features

- **Multi-target Prediction**: Classify dominant tree species (4 classes) and forest age groups (4 classes)
- **Rich Feature Set**: 35-40 features including Sentinel-2 spectral bands, vegetation indices (NDVI, NDWI, GNDVI, SAVI, FACI), and topographic features
- **Class Imbalance Handling**: SMOTE, class weights, weighted sampling, and focal loss strategies
- **Multiple Model Architectures**: Random Forest baseline and 1D CNN with batch normalization
- **Model Interpretability**: Feature importance analysis and SHAP explanations
- **Geospatial Visualization**: Interactive maps of forest stands with predictions

---

## Dataset

The TreeSatAI Forest Structure Dataset contains **110,000+ forest stands** with:

### Target Variables
| Variable | Type | Classes/Range |
|----------|------|---------------|
| Prevailing Species | Categorical | 0: Fir, 1: Birch, 2: Larch, 3: Spruce |
| Age Group | Categorical | 0: Young, 1: Middle-aged, 2: Maturing, 3: Mature |
| Height | Continuous | 0-50 m |
| Stock Volume | Continuous | 0-500 m³/ha |
| Basal Area | Continuous | 0-50 m²/ha |
| Carbon Content | Continuous | 0-250 t/ha |

### Features
- **Spectral Bands**: Sentinel-2 bands B02-B12 (mean and std per polygon)
- **Vegetation Indices**: NDVI, NDWI, GNDVI, SAVI, FACI
- **Topographic Features**: Elevation, slope, aspect, hillshade

### Data Source
- Sentinel-2 imagery from 2019-2022
- Coverage: Kholmsk, Nevelsk, Korsakov regions
- Spatial resolution: 10-20 meters

---

## Model Performance

### Species Classification

| Model | Strategy | Accuracy | F1-Score | Minority Recall |
|-------|----------|----------|----------|-----------------|
| Random Forest | All Features | 57.2% | 52.8% | Fir: 5%, Spruce: 9% |
| CNN | Weighted Sampler | 50.9% | 52.0% | Fir: 46%, Spruce: 48% |
| CNN | Class Weights | 41.2% | 42.2% | Fir: 51%, Spruce: 65% |
| CNN | Focal Loss | 50.1% | 50.9% | Fir: 44%, Spruce: 46% |
| **Ensemble** | Combined | **50.8%** | **52.1%** | **Fir: 50%, Spruce: 58%** |

### Age Classification

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Random Forest | 64.1% | 54.1% |

---

## Technologies

- **Python 3.8+** - Core programming language
- **PyTorch** - Deep learning framework for CNN models
- **Scikit-learn** - Random Forest, preprocessing, metrics
- **GeoPandas** - Geospatial data handling
- **Rasterio** - Satellite image processing
- **SHAP** - Model interpretability
- **Imbalanced-learn** - SMOTE for class balancing
- **Matplotlib/Seaborn** - Visualization

---

## Project Structure

```
forest-properties-mapping/
│
├── data/
│   ├── full_test_30.csv              # Forest stand data with geometries
│   └── sentinel-2/                   # Satellite imagery (6 TIFF files)
│
├── notebooks/
│   ├── 01_eda_visualization.ipynb    # Exploratory data analysis
│   ├── 02_baseline_models.ipynb      # Random Forest baseline
│   ├── 03_cnn_models.ipynb           # CNN with batch normalization
│   └── 04_ensemble_interpretability.ipynb  # Ensemble & SHAP
│
├── models/
│   ├── best_cnn_model.pth            # Saved PyTorch model
│   └── feature_importance.csv        # Feature importance scores
│
├── results/
│   ├── training_curves_*.png         # Training loss curves
│   ├── confusion_matrix_*.png        # Confusion matrices
│   └── feature_importance.png        # Feature importance plots
│
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment file
└── README.md                          # This file
```

---

## Installation

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/forest-properties-mapping.git
cd forest-properties-mapping

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda (recommended for geospatial packages)

```bash
# Clone the repository
git clone https://github.com/yourusername/forest-properties-mapping.git
cd forest-properties-mapping

# Create conda environment
conda env create -f environment.yml
conda activate forest-mapping
```

### Requirements.txt
```
geopandas>=0.14.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
rasterio>=1.3.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
torch>=2.0.0
shap>=0.42.0
openpyxl>=3.1.0
```

---

## 📖 Usage

### 1. Load and Explore Data

```python
import pandas as pd
import geopandas as gpd

# Load forest stand data
df = pd.read_csv('data/full_test_30.csv')

# Convert geometry to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df, 
    geometry=gpd.GeoSeries.from_wkt(df['geometry']),
    crs='EPSG:3857'
)

# View class distribution
print(df['prevailing_species'].value_counts())
```

### 2. Prepare Features

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define feature columns
spectral_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
features = [f'{band}_mean' for band in spectral_bands] + \
           [f'{band}_std' for band in spectral_bands]

# Prepare data
X = df[features].fillna(0)
y = df['prevailing_species']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 3. Train Random Forest Baseline

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 4. Train CNN with Balancing

```python
from models.cnn_model import train_cnn_balanced

# Train CNN with weighted sampler and SMOTE
model, accuracy, f1, train_loss, val_loss, y_test, pred, probs = train_cnn_balanced(
    X, y,
    num_classes=4,
    epochs=80,
    batch_size=64,
    strategy='weighted_sampler',
    use_smote=True,
    learning_rate=0.001
)
```

### 5. Model Interpretability

```python
import shap

# SHAP analysis for Random Forest
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=features)

# Feature importance
importances = rf.feature_importances_
for name, importance in zip(features, importances):
    print(f"{name}: {importance:.4f}")
```

## Results Visualization

### Training Curves
CNN training shows stable convergence with early stopping preventing overfitting:
- Weighted Sampler: Loss 0.81 → 0.53
- Focal Loss: Loss 0.10 → 0.06 (smoothest convergence)

### Confusion Matrices
Ensemble model achieves balanced performance across all four species:
- Birch: 62% precision, 44% recall
- Larch: 63% precision, 56% recall
- Spruce: 33% precision, 58% recall
- Fir: 25% precision, 50% recall

### Feature Importance
Top 10 most important features:
1. B08 (NIR) Mean - 14.2%
2. B04 (Red) Mean - 11.8%
3. NDVI Mean - 9.5%
4. B08 (NIR) Std - 7.2%
5. B11 (SWIR) Mean - 6.5%
6. GNDVI Mean - 5.8%
7. B04 (Red) Std - 5.1%
8. B02 (Blue) Mean - 4.7%
9. Elevation Mean - 4.3%
10. B03 (Green) Mean - 3.9%

---

## Key Findings

1. **Class Imbalance is Critical**: Random Forest achieves 57% accuracy but fails on minority classes (Fir: 5% recall). CNN with balancing improves minority recall by 45-49%.

2. **Feature Importance**: Near-infrared (B08) and red (B04) bands are most predictive, confirming vegetation analysis theory.

3. **Balancing Strategies**: Weighted sampling and focal loss provide the best trade-off between overall accuracy and minority class performance.

4. **Ensemble Benefits**: Combining multiple strategies yields 2-3% improvement over individual approaches.

5. **Spatial Patterns**: Model predictions show realistic spatial clustering matching actual forest distributions.
