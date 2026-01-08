# src/entity/config_entity.py

import os
from dataclasses import dataclass
from datetime import datetime
from src.constants import *

# =====================================================
# TIMESTAMP
# =====================================================
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


# =====================================================
# TRAINING PIPELINE CONFIG
# =====================================================
@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config = TrainingPipelineConfig()


# =====================================================
# DATA INGESTION CONFIG
# =====================================================
@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_INGESTION_DIR_NAME
    )

    feature_store_file_path: str = os.path.join(
        data_ingestion_dir,
        DATA_INGESTION_FEATURE_STORE_DIR,
        FILE_NAME
    )

    collection_name: str = DATA_INGESTION_COLLECTION_NAME


# =====================================================
# DATA VALIDATION CONFIG
# =====================================================
@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_VALIDATION_DIR_NAME
    )

    validation_report_file_path: str = os.path.join(
        data_validation_dir,
        DATA_VALIDATION_REPORT_FILE_NAME
    )

    schema_file_path: str = SCHEMA_FILE_PATH


# =====================================================
# DATA TRANSFORMATION CONFIG
# =====================================================
@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_TRANSFORMATION_DIR_NAME
    )

    transformed_data_dir: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR
    )

    transformed_object_dir: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR
    )

    # Transformed arrays
    X_train_path: str = os.path.join(transformed_data_dir, "X_train.npy")
    X_test_path: str = os.path.join(transformed_data_dir, "X_test.npy")
    y_train_path: str = os.path.join(transformed_data_dir, "y_train.npy")
    y_test_path: str = os.path.join(transformed_data_dir, "y_test.npy")

    # Preprocessing object
    preprocessing_object_path: str = os.path.join(
        transformed_object_dir,
        PREPROCESSING_OBJECT_FILE_NAME
    )


# =====================================================
# MODEL TRAINER CONFIG
# =====================================================
@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        MODEL_TRAINER_DIR_NAME
    )

    trained_model_file_path: str = os.path.join(
        model_trainer_dir,
        MODEL_TRAINER_TRAINED_MODEL_DIR,
        MODEL_FILE_NAME
    )

    model_type: str = MODEL_TYPE


# =====================================================
# MODEL EVALUATION CONFIG
# (NO S3 RESPONSIBILITY – OPTION A)
# =====================================================
@dataclass
class ModelEvaluationConfig:
    """
    Controls only model acceptance logic.
    No S3 paths by design.
    """
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE


# =====================================================
# MODEL PUSHER CONFIG
# (ALL S3 RESPONSIBILITY)
# =====================================================
@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = "model/model.pkl"
    s3_metrics_key_path: str = "model/metrics.json"
    improvement_threshold: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE


# =====================================================
# ETA PREDICTION CONFIG
# =====================================================
@dataclass
class ETAPredictorConfig:
    model_bucket_name: str = MODEL_BUCKET_NAME
    model_file_path: str = "model/model.pkl"