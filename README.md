# Airline Passenger Satisfaction Prediction Service

> **Capstone Project 1 - Machine Learning Zoomcamp 2025**
> *An End-to-End MLOps pipeline demonstrating reproducible training, experiment tracking, and containerized serving.*

## Project Overview

In the competitive airline industry, customer retention is critical. This project simulates an **Analytics Engineering** workflow to predict passenger satisfaction in real-time. By identifying "Neutral" or "Dissatisfied" passengers before they churn, Airline Operations can intervene with proactive customer service.

**Key Features:**

* **Multi-Model Pipeline:** Automatically trains and evaluates a **Logistic Regression** baseline against a **Tuned XGBoost** model.
* **Experiment Tracking:** Uses **MLflow** to log metrics (AUC), parameters, and artifacts for every run.
* **Production Serving:** Uses **BentoML** for high-performance model serving, replacing standard Flask wrappers.
* **Reproducibility:** Dependencies managed via **uv**, ensuring deterministic environments.
* **Containerization:** Fully Dockerized application using a multi-stage build optimization.

## Tech Stack

| Component | Tool | Reason for Choice |
| --- | --- | --- |
| **Models** | XGBoost & Logistic Regression | Comparison between linear baseline and gradient boosting tree performance. |
| **Tracking** | MLflow | Tracks experiments, parameters, and metrics to ensure model auditability. |
| **Serving** | BentoML | Standardizes model packaging and API generation (automatically generates Swagger UI). |
| **Container** | Docker | Ensures the service runs identically on local and production environments. |
| **Manager** | uv | Faster and more reliable than pip/poetry for dependency resolution. |

---

## How to Run Locally

### 1. Clone the Repository


### 2. Install Dependencies

This project uses `uv` for lightning-fast setup, but standard `pip` works too.

```bash
uv sync
```

### 3. Training the Pipeline

Run the training script. This script performs the following MLOps steps:

1. **ETL:** Loads and cleans the raw data.
2. **Baseline Training:** Trains a Logistic Regression model and logs AUC to MLflow.
3. **Champion Training:** Trains an XGBoost model with hyperparameter tuning.
4. **Artifact Promotion:** Automatically saves the best model (XGBoost) and the Preprocessor (`DictVectorizer`) to the **BentoML Model Store** for serving.

```bash
python train.py

```

*Output Confirmation:*

> `BentoML Model Saved: airline_satisfaction_model:latest`

### 4. Serve the Model (Local API)

Start the BentoML server locally to test predictions.

```bash
bentoml serve service.py:AirlineService

```

The server will start at `http://localhost:3000`.

---

## Running with Docker (Reproducibility)

This project is fully containerized to ensure it runs anywhere. We use a **slim Python 3.11** base image to optimize resource usage.

### 1. Build the Docker Image

We use BentoML to package the code, model, and system dependencies (including `libgomp1` for XGBoost optimization).

```bash
# Build the Bento bundle
bentoml build

# Containerize the bundle
bentoml containerize airline_satisfaction_service:latest

```

### 2. Run the Container

Run the service in an isolated Docker container, use the generated tag (e.g., `airline_satisfaction_service:xyz`).

```bash
docker run -it --rm -p 3000:3000 airline_satisfaction_service:xyz

```

---

## Testing the API

### Method 1: Curl Request

Send a POST request to the `/predict` endpoint.

```bash
curl -X POST \
   -H "Content-Type: application/json" \
   -d '{
     "gender": "Male",
     "customer_type": "Loyal Customer",
     "age": 30,
     "type_of_travel": "Business travel",
     "flight_class": "Business", 
     "flight_distance": 1000,
     "arrival_delay_in_minutes": 0.0
   }' \
   http://localhost:3000/predict

```

**Expected Response:**

```json
{
  "satisfaction_probability": 0.942,
  "is_satisfied": "SATISFIED"
}

```

### Method 2: Swagger UI

BentoML automatically generates interactive documentation.

1. Open `http://localhost:3000` in your browser.
2. Click on **Service APIs**.
3. Use the "Try it out" button to send test data interactively.

---

## Project Structure

```text
├── data/                  # Training data
├── notebooks/
│   └── eda.ipynb          # Exploratory Data Analysis & Feature Selection
├── train.py               # Training script (Logistic Regression vs XGBoost + MLflow)
├── service.py             # BentoML Service definition
├── bentofile.yaml         # Configuration for building the Docker image
├── requirements.txt       # Frozen dependencies
└── README.md              # Documentation

```

## Exploratory Data Analysis (EDA)

Extensive analysis was performed in `notebooks/eda.ipynb`, including:

* **Target Balance:** Checked for class imbalance to determine evaluation metrics.
* **Correlation Analysis:** Identified key drivers of satisfaction (e.g., "In-flight entertainment" vs "Departure Delay").
* **Feature Importance:** Analyzed XGBoost feature scores to confirm business intuition.