# MLOPS_Delivery_Time_Estimation-Project-Robust-Pipeline-

# Delivery ETA Error Prediction – End-to-End MLOps Project

## Overview

This project builds a **production-style MLOps pipeline** to predict the **ETA error for package deliveries**.

Instead of predicting the delivery time directly, the model predicts:

```
ETA Error = Actual Delivery Time − Expected Delivery Time
```

This tells whether a delivery will be **early or delayed**, and by **how many minutes**.

The system includes the full lifecycle of a machine learning system:

* Data ingestion from MongoDB
* Data validation using a schema
* Feature engineering and preprocessing
* Model training and evaluation
* Model storage in AWS S3
* Prediction API using FastAPI
* Containerization using Docker
* CI/CD pipeline with GitHub Actions
* Deployment on AWS EC2

The goal of this project is to demonstrate **how machine learning models move from experimentation to production systems**.

---

# System Architecture

The system consists of two main pipelines:

### Training Pipeline

```
MongoDB
   ↓
Data Ingestion
   ↓
Data Validation
   ↓
Data Transformation
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Pusher (S3)
```

### Prediction Pipeline

```
User Request
     ↓
FastAPI API
     ↓
Load Model from S3
     ↓
Prediction
     ↓
Return ETA Error
```

---

# Dataset

The dataset contains delivery information such as:

| Feature             | Description                   |
| ------------------- | ----------------------------- |
| delivery_partner    | Delivery company              |
| package_type        | Type of package               |
| vehicle_type        | Delivery vehicle              |
| delivery_mode       | Standard / Express / Same Day |
| region              | Delivery region               |
| weather_condition   | Weather during delivery       |
| distance_km         | Delivery distance             |
| package_weight_kg   | Package weight                |
| delivery_time_hours | Actual delivery time          |
| expected_time_hours | Expected delivery time        |

The target variable is created as:

```
eta_error_minutes = (actual_delivery_time − expected_delivery_time) × 60
```

Meaning:

* **Positive value → delivery delay**
* **Negative value → early delivery**

---

# Project Structure

```
MLOPS_Delivery_Time_Estimation

│
├── src
│
│   ├── components
│   │      data_ingestion.py
│   │      data_validation.py
│   │      data_transformation.py
│   │      model_trainer.py
│   │      model_evaluation.py
│   │      model_pusher.py
│
│   ├── configuration
│   │      aws_connection.py
│   │      mongo_db_connection.py
│
│   ├── cloud_storage
│   │      aws_storage.py
│
│   ├── data_access
│   │      proj1_data.py
│
│   ├── entity
│   │      config_entity.py
│   │      artifact_entity.py
│   │      estimator.py
│   │      s3_estimator.py
│
│   ├── pipeline
│   │      training_pipeline.py
│   │      prediction_pipeline.py
│
│   ├── utils
│   │      main_utils.py
│
│   └── constants
│
├── config
│      schema.yaml
│
├── templates
│      eta.html
│
├── static
│      css files
│
├── app.py
├── Dockerfile
├── requirements.txt
└── github workflow
```

---

# Pipeline Components

## Data Ingestion

* Reads delivery data from **MongoDB Atlas**
* Converts the data into a pandas dataframe
* Stores it in the **feature store**

Output:

```
artifact/data_ingestion/feature_store/data.csv
```

---

## Data Validation

Validates the dataset using `schema.yaml`.

Checks include:

* Missing columns
* Duplicate rows
* Duplicate columns
* Numeric column validity
* Categorical column uniqueness

A validation report is generated.

---

## Data Transformation

Responsible for preparing data for model training.

Steps:

1. Create target column `eta_error_minutes`
2. Drop unused columns
3. One-Hot encode categorical features
4. Scale numerical features
5. Split dataset into training and test sets

Artifacts generated:

```
X_train.npy
X_test.npy
y_train.npy
y_test.npy
preprocessing.pkl
```

---

## Model Training

The project uses a **Linear Regression model**.

Steps:

* Load transformed dataset
* Train regression model
* Evaluate using:

```
MAE
RMSE
R² Score
```

The preprocessing pipeline and model are wrapped together inside a custom class:

```
MyModel
```

This ensures the **same preprocessing is applied during prediction**.

---

## Model Evaluation

The trained model is evaluated using the **raw dataset**.

Metrics computed:

```
MAE
RMSE
R² Score
Improvement over baseline
SLA metrics
```

The model is accepted only if it improves performance beyond a defined threshold.

---

## Model Pusher

If the model passes evaluation:

* The trained model is uploaded to **AWS S3**
* Metrics are also stored in S3

Example S3 structure:

```
S3 Bucket
   └── model
        ├── model.pkl
        └── metrics.json
```

---

# Prediction System

The application provides a **FastAPI web interface** where users can enter delivery details.

Example input:

* Delivery partner
* Package type
* Vehicle type
* Delivery mode
* Region
* Weather condition
* Distance
* Package weight

The API then:

```
1. Loads the model from S3
2. Applies preprocessing
3. Generates prediction
4. Returns ETA error
```

Example output:

```
Delivery expected 120 minutes EARLY
```

or

```
Delivery expected 45 minutes LATE
```

---

# Deployment Architecture

The project uses **Docker and AWS services for deployment**.

### Deployment Flow

```
GitHub Push
     ↓
GitHub Actions
     ↓
Build Docker Image
     ↓
Push Image to AWS ECR
     ↓
EC2 Pulls Image
     ↓
FastAPI Service Runs
```

This enables automated **CI/CD deployment**.

---

# Running the Project Locally

### 1. Clone repository

```
git clone <repo-url>
cd project
```

### 2. Create virtual environment

```
python -m venv myenv
```

Activate environment:

```
myenv\Scripts\activate
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

### 4. Set environment variables

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
MONGODB_URL
```

---

### 5. Train the model

```
python demo.py
```

---

### 6. Run API server

```
python app.py
```

Open:

```
http://localhost:5000
```

---

# Technologies Used

* Python
* Scikit-learn
* Pandas
* MongoDB Atlas
* AWS S3
* FastAPI
* Docker
* GitHub Actions
* AWS EC2
* AWS ECR

---

# Future Improvements

Possible improvements include:

* Model versioning
* Experiment tracking (MLflow)
* Feature store integration
* Drift detection and monitoring
* Feeding more varied data
* Using advanced models like XGBoost or Random Forest

---

# Conclusion

This project demonstrates how to build a **complete machine learning system ready for production**, including:

* Data pipelines
* Model lifecycle management
* Cloud deployment
* Automated CI/CD workflows

It highlights how machine learning models can be transformed from simple experiments into **reliable production services**.