# Credit Risk Probability Model

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Made with Love](https://img.shields.io/badge/made%20with-love-red.svg)]()

---

## 📌 Project Overview
This repository contains the **Credit Risk Probability Model**, designed to assess the likelihood of default for customers in a Buy-Now-Pay-Later (BNPL) service.  
It leverages **machine learning** techniques for feature engineering, model training, and probability scoring.

---

## ⚡ Features
- MLflow experiment tracking  
- Model comparison & hyperparameter tuning  
- Probability scoring for credit risk  
- Metrics: Accuracy, AUC, Precision, Recall, F1  
- Artifact & log management  

---

## 🏗️ Methodology
1. **Data Preprocessing** – Cleaning, handling missing values, feature engineering.  
2. **Exploratory Data Analysis (EDA)** – Visual insights into risk patterns.  
3. **Model Training** – Logistic Regression, Random Forest, Gradient Boosting.  
4. **Experiment Tracking** – Using MLflow for runs, metrics, and artifacts.  
5. **Evaluation** – AUC, ROC, and confusion matrices for performance.  
6. **Deployment** – Model saved as `.pkl` with reproducibility support.  

---

## 🚀 Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Addisu-Taye/Enhanced-credit-risk-model.git
cd credit-risk-model
```

2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```
3️⃣ Install Dependencies
```bash

pip install -r requirements.txt
```
4️⃣ Run Training
```bash

python train_model.py
```
📂 Repository Structure
```bash
Enhanced-credit-risk-model/
├── data/
│ ├── raw/
│ │ └── raw_data.csv
│ └── processed/
│ └── processed_data.csv
├── mlruns/ # MLflow experiment tracking
├── models/ # Saved scaler and feature columns
│ ├── scaler.joblib
│ └── feature_columns.json
├── src/
│ ├── data_processing.py # RFM, clustering, target creation
│ ├── train.py # Model training & MLflow logging
│ └── api/
│ └── main.py # FastAPI application
├── requirements.txt
├── README.md
└── Dockerfile
```

🧰 Tech Stack
Python 3.10+

Scikit-learn

MLflow

Pandas & NumPy

Matplotlib & Seaborn

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

👨‍💻 Author
Addisu Taye
📧 Contact: addtaye@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/addisu-taye/


