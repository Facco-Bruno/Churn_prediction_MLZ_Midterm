# 🧠 Customer Churn Prediction API (Telco Dataset)

This project builds and serves a **machine learning model** to predict customer churn for a telecommunications company using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

The model is trained using **Scikit-learn**, served via a **Flask API**, and containerized with **Docker** for easy deployment.

---

## 📘 Project Overview

**Goal:**  
Predict whether a customer is likely to churn (leave the company) based on demographic, service usage, and billing information.

**Key Steps:**
1. Data preprocessing and cleaning  
2. Model training and evaluation  
3. Exporting the model and vectorizer  
4. Serving predictions through a Flask REST API  
5. Containerizing the application with Docker

---

## 🗂️ Project Structure

```
streaming-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── train.py
├── predict.py
├── requirements.txt
├── Dockerfile
├── model.bin
├── dv.bin
└── README.md
```

---

## 🧰 Requirements

- Python 3.8+
- pip
- Docker (optional, for containerized execution)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ How to Run the Project Locally

1. **Activate your virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # (Linux / macOS)
   # or
   .\venv\Scripts\Activate    # (Windows)
   ```

2. **Train the model:**

   ```bash
   python train.py
   ```

   This script:
   - Loads and preprocesses the dataset
   - Trains Logistic Regression and Random Forest models
   - Evaluates them and selects the best one
   - Saves the model and DictVectorizer as `model.bin` and `dv.bin`

3. **Run the API locally:**

   ```bash
   python predict.py
   ```

   The service will start at:  
   👉 `http://0.0.0.0:9696/predict`

4. **Send a test prediction request:**

   ```bash
   curl -X POST      -H "Content-Type: application/json"      -d '{
       "gender": "Female",
       "SeniorCitizen": 0,
       "Partner": "Yes",
       "Dependents": "No",
       "tenure": 5,
       "PhoneService": "Yes",
       "MultipleLines": "No",
       "InternetService": "Fiber optic",
       "OnlineSecurity": "No",
       "OnlineBackup": "Yes",
       "DeviceProtection": "No",
       "TechSupport": "No",
       "StreamingTV": "Yes",
       "StreamingMovies": "No",
       "Contract": "Month-to-month",
       "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check",
       "MonthlyCharges": 70.70,
       "TotalCharges": 151.65
     }'      http://0.0.0.0:9696/predict
   ```

   **Expected output:**

   ```json
   {"churn": true, "churn_probability": 0.65}
   ```

---

## 🐳 Running with Docker

You can also build and run the project as a Docker container.

1. **Build the Docker image:**

   ```bash
   docker build -t churn-prediction-service .
   ```

2. **Run the container:**

   ```bash
   docker run -p 9696:9696 churn-prediction-service
   ```

3. **Send the same test request:**

   ```bash
   curl -X POST      -H "Content-Type: application/json"      -d '{
       "gender": "Female",
       "SeniorCitizen": 0,
       "Partner": "Yes",
       "Dependents": "No",
       "tenure": 5,
       "PhoneService": "Yes",
       "MultipleLines": "No",
       "InternetService": "Fiber optic",
       "OnlineSecurity": "No",
       "OnlineBackup": "Yes",
       "DeviceProtection": "No",
       "TechSupport": "No",
       "StreamingTV": "Yes",
       "StreamingMovies": "No",
       "Contract": "Month-to-month",
       "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check",
       "MonthlyCharges": 70.70,
       "TotalCharges": 151.65
     }'      http://0.0.0.0:9696/predict
   ```

   **Output:**
   ```json
   {"churn": true, "churn_probability": 0.65}
   ```

---

## 📊 Dataset

The dataset used in this project is the **Telco Customer Churn dataset**, originally provided by IBM and available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

To use it, place the CSV file in the `data/` directory as:

```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## 📈 Model Details

Two models were compared:
- **Logistic Regression** — simple and interpretable baseline
- **Random Forest** — ensemble model that performed better on ROC AUC

Metrics used:
- ROC AUC
- Cross-validation mean score

The best model and its vectorizer are saved as:
- `model.bin`
- `dv.bin`

---

## 🧩 Example Prediction Output

```json
{
  "churn": true,
  "churn_probability": 0.6546
}
```

---

## ☁️ Optional: Deploying to the Cloud

You can deploy this container on:
- [Render](https://render.com/)
- [Railway](https://railway.app/)
- [Fly.io](https://fly.io/)
- [AWS Lightsail](https://aws.amazon.com/lightsail/)

Each platform can run your Docker image directly with minimal setup.

