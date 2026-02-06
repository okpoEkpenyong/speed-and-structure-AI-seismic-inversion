
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

## 🛠️ Installation & Run

```bash
git clone https://github.com/okpoEkpenyong/speed-and-structure-AI-inversion.git
cd speed-and-structure-AI-inversion
pip install -r requirements.txt
```
For ease of run, single entry points are created in the two notebooks:
1. ../notebooks/speed-and-structure-1st.ipynb and
2.  ../notebooks/speed-and-structure-30th.ipynb. 
The user can run each cell in each notebook to see the results. Since both Kaggle and Azure ML environments were used, slight modifications will be required if paths are changed.

## 📊 Workflow
1.  **Data Ingestion:** No formal cleaning of the dataset was required.
2.  **Feature Engineering:** Extracting instantaneous attributes like well log attributes.
3.  **Machine Learning Training:** Combining Hybrid Unet and Physics-aware losses.
4.  **Blind Validation:** Predicting velocity on held-out wells to verify generalization.

## ⚠️ Dataset
Dataset is provided by Thinkonward which also owns full copyright. Since it was made public, it's hosted here with these links:
1. https://www.kaggle.com/datasets/okpoekpenyong/train-extra-2
2. https://www.kaggle.com/datasets/okpoekpenyong/train-extra-1
3. https://www.kaggle.com/datasets/okpoekpenyong/train-1000
4. https://www.kaggle.com/datasets/okpoekpenyong/train-500
5. https://www.kaggle.com/datasets/okpoekpenyong/train-300
6. https://www.kaggle.com/datasets/okpoekpenyong/train-200


***
#### **B. Reproducibility (The Seed)**
random seed is implemented to help in reproducibility:
