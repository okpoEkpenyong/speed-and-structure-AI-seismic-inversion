
***

# Hybrid Machine Learning Seismic Inversion Workflow

**Author:** Ekpenyong Okpo  
**Submission:** American Geophysical Union (AGU), 2025

## 📌 Overview
This repository contains the implementation of a **Hybrid Machine Learning Seismic Inversion** workflow designed to resolve seismic velocity distributions in highly heterogeneous  reservoirs.

Traditional deterministic inversion often struggles in thin-bed zones. This project utilizes a **Python-based ensemble model** to integrate seismic attributes with well-log data, achieving higher resolution and geological plausibility.

## 🚀 Key Results
- **Accuracy:** Achieved **~15% improvement** in correlation within thin-bed zones compared to commercial inversion packages.
- **Robustness:** Validated via **Blind Well Testing** and K-Fold Cross-Validation.
- **Geological Fidelity:** Successfully replicated complex features (salt domes, faults) with a MAPE of **0.033**.

## 🛠️ Installation

```bash
git clone https://github.com/okpoEkpenyong/speed-and-structure-AI-inversion.git
cd speed-and-structure-AI-inversion
pip install -r requirements.txt
```

## 📊 Workflow
1.  **Data Ingestion:** No formal cleaning of the dataset was required.
2.  **Feature Engineering:** Extracting instantaneous attributes like well log attributes.
3.  **Ensemble Training:** Combining Gradient Boosting (XGBoost/LightGBM) with Random Forests.
4.  **Blind Validation:** Predicting velocity on held-out wells to verify generalization.

## ⚠️ Dataset
Dataset is provided by Thinkonward which also owns full copyright. Since it was made public, it's hosted here with these links:
https://www.kaggle.com/datasets/okpoekpenyong/train-extra-2
https://www.kaggle.com/datasets/okpoekpenyong/train-extra-1
https://www.kaggle.com/datasets/okpoekpenyong/train-1000
https://www.kaggle.com/datasets/okpoekpenyong/train-500
https://www.kaggle.com/datasets/okpoekpenyong/train-300
https://www.kaggle.com/datasets/okpoekpenyong/train-200


***
#### **B. Reproducibility (The Seed)**
random seed is implemented to help in reproducibility:

```python
import numpy as np
import tensorflow as tf
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
# If using Scikit-Learn
model = RandomForestRegressor(random_state=SEED)
```
