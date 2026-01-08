import os

# =========================
# MongoDB
# =========================
DATABASE_NAME = "Proj1"
DATA_INGESTION_COLLECTION_NAME = "Proj1-Data"
MONGODB_URL_KEY = "MONGODB_URL"

# =========================
# AWS
# =========================
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

# =========================
# Pipeline
# =========================
PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

# =========================
# Files
# =========================
FILE_NAME: str = "data.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"

# =========================
# Target
# =========================
TARGET_COLUMN = "eta_error_minutes"

# =========================
# Data Ingestion
# =========================
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

# =========================
# Data Validation
# =========================
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

# =========================
# Data Transformation
# =========================
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# =====================================================
# MODEL TRAINER (ETA REGRESSION – LINEAR REGRESSION)
# =====================================================

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "eta_linear_regression.pkl"


# -------------------------
# Model identification
# -------------------------
MODEL_TYPE: str = "linear_regression"


# -------------------------
# Linear Regression parameters
# (kept explicit for clarity & future extensibility)
# -------------------------
LINEAR_REG_FIT_INTERCEPT: bool = True
LINEAR_REG_COPY_X: bool = True
LINEAR_REG_N_JOBS = None   # Can be set to -1 if needed later


# -------------------------
# MODEL Evaluation related constants
# -------------------------
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "aws-my-model-mlopsproj"
MODEL_PUSHER_S3_KEY = "model-registry"


# -------------------------
# App related constants
# -------------------------
APP_HOST = "0.0.0.0"
APP_PORT = 5000
