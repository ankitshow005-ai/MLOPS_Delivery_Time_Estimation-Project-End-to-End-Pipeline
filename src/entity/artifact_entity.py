# src/entity/artifact_entity.py

from dataclasses import dataclass
from typing import Dict


# =====================================================
# DATA INGESTION ARTIFACT
# =====================================================
@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str


# =====================================================
# DATA VALIDATION ARTIFACT
# =====================================================
@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    validation_report_file_path: str


# =====================================================
# DATA TRANSFORMATION ARTIFACT
# =====================================================
@dataclass
class DataTransformationArtifact:
    X_train_path: str
    X_test_path: str
    y_train_path: str
    y_test_path: str
    preprocessing_object_path: str


# =====================================================
# MODEL TRAINER ARTIFACT
# =====================================================
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str


# =====================================================
# MODEL EVALUATION ARTIFACT  ✅ SINGLE SOURCE OF TRUTH
# =====================================================
@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    baseline_mae: float
    model_mae: float
    improvement_pct: float
    sla_metrics: Dict
    trained_model_path: str
    metrics_file_path: str


# =====================================================
# MODEL PUSHER ARTIFACT
# =====================================================
@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str