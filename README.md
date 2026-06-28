# Network Intrusion Detection System

A hybrid intrusion detection system combining an unsupervised autoencoder with a supervised XGBoost multi-class classifier. The autoencoder's reconstruction error is injected as an engineered feature into XGBoost — this is the core architectural idea.

## Architecture

```
Raw traffic (NSL-KDD)
        ↓
  Preprocessing
  OHE + MinMaxScale → [0,1]
        ↓
   ┌────┴────┐
   │         │
Autoencoder  XGBoost
(normal only) (labeled)
   │         │
Recon error──┘
(feature 42)
   │
Hybrid decision
Normal / DoS / Probe / R2L / U2R
```

The autoencoder catches volume-based attacks (DoS) that look nothing like normal traffic. XGBoost catches behavioral attacks (R2L, U2R) using labeled patterns. Neither alone is sufficient.

## Dataset

NSL-KDD — cleaned version of KDD Cup 1999. Download from [Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd).

Place `KDDTrain+.txt` and `KDDTest+.txt` in `data/raw/`.

| Class  | Samples |
|--------|---------|
| Normal | 67,343  |
| DoS    | 45,927  |
| Probe  | 11,656  |
| R2L    | 995     |
| U2R    | 52      |

Primary metric: **macro F1** — accuracy is misleading due to class imbalance.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/intrusion-detection-system.git
cd intrusion-detection-system

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Run

Execute notebooks in order:

```
notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_autoencoder.ipynb
notebooks/04_xgboost.ipynb
notebooks/05_evaluation.ipynb
```

Then launch the Streamlit app:

```bash
streamlit run app/main.py
```

## Project Structure

```
├── data/
│   ├── raw/                  # KDDTrain+.txt, KDDTest+.txt (not pushed)
│   └── processed/            # numpy arrays, encoder artifacts (not pushed)
├── notebooks/                # run in order 01 → 05
├── src/                      # reusable pipeline modules
│   ├── preprocess.py
│   ├── autoencoder.py
│   ├── classifier.py
│   └── pipeline.py           # single predict() used by the app
├── models/                   # saved autoencoder + XGBoost (not pushed)
├── app/                      # Streamlit demo
│   ├── main.py
│   └── sections/
├── reports/                  # confusion matrix, SHAP plots
└── requirements.txt
```

## App Sections

| Section | Purpose |
|---------|---------|
| Data Explorer | Class distribution, feature stats, why macro F1 |
| Anomaly Detector | Reconstruction error histogram, live threshold slider |
| Live Classifier | Full pipeline on a single record, class probabilities |
| Explainability | SHAP feature importance, plain-English prediction summary |

## Stack

Python 3.11 · TensorFlow/Keras · XGBoost · scikit-learn · SHAP · Streamlit