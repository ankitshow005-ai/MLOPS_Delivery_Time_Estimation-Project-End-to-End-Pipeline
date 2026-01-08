# src/pipline/training_pipeline.py

import sys
from src.exception import MyException
from src.logger import logging

# =========================
# Components
# =========================
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

# =========================
# Configs
# =========================
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

# =========================
# Artifacts
# =========================
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)


class TrainPipeline:
    """
    End-to-end training pipeline for ETA Error Regression.

    FLOW:
    Data Ingestion
        → Data Validation
            → Data Transformation
                → Model Training
                    → Model Evaluation (RAW DATA)
                        → Model Pusher (acceptance + S3 upload)
    """

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    # =====================================================
    # STEP 1: DATA INGESTION
    # =====================================================
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting Data Ingestion")
            component = DataIngestion(self.data_ingestion_config)
            return component.initiate_data_ingestion()
        except Exception as e:
            raise MyException(e, sys)

    # =====================================================
    # STEP 2: DATA VALIDATION
    # =====================================================
    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info("Starting Data Validation")
            component = DataValidation(
                data_ingestion_artifact,
                self.data_validation_config
            )
            return component.initiate_data_validation()
        except Exception as e:
            raise MyException(e, sys)

    # =====================================================
    # STEP 3: DATA TRANSFORMATION
    # =====================================================
    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation")
            component = DataTransformation(
                data_ingestion_artifact,
                data_validation_artifact,
                self.data_transformation_config
            )
            return component.initiate_data_transformation()
        except Exception as e:
            raise MyException(e, sys)

    # =====================================================
    # STEP 4: MODEL TRAINING
    # =====================================================
    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logging.info("Starting Model Training")
            component = ModelTrainer(
                data_transformation_artifact,
                self.model_trainer_config
            )
            return component.initiate_model_trainer()
        except Exception as e:
            raise MyException(e, sys)

    # =====================================================
    # STEP 5: MODEL EVALUATION (RAW DATA)
    # =====================================================
    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting Model Evaluation")
            component = ModelEvaluation(
                self.model_evaluation_config,
                data_ingestion_artifact,   # ✅ RAW DATA ONLY
                model_trainer_artifact
            )
            return component.initiate_model_evaluation()
        except Exception as e:
            raise MyException(e, sys)

    # =====================================================
    # STEP 6: MODEL PUSHER
    # =====================================================
    def start_model_pusher(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact
    ) -> ModelPusherArtifact | None:
        try:
            logging.info("Starting Model Pusher")
            component = ModelPusher(
                model_evaluation_artifact,
                self.model_pusher_config
            )
            return component.initiate_model_pusher()
        except Exception as e:
            raise MyException(e, sys)

    # =====================================================
    # PIPELINE ORCHESTRATION
    # =====================================================
    def run_pipeline(self) -> None:
        try:
            logging.info("========== Training Pipeline Started ==========")

            ingestion_artifact = self.start_data_ingestion()

            validation_artifact = self.start_data_validation(
                ingestion_artifact
            )
            if not validation_artifact.validation_status:
                logging.error("Validation failed. Pipeline stopped.")
                return

            transformation_artifact = self.start_data_transformation(
                ingestion_artifact,
                validation_artifact
            )

            trainer_artifact = self.start_model_trainer(
                transformation_artifact
            )

            evaluation_artifact = self.start_model_evaluation(
                ingestion_artifact,
                trainer_artifact
            )

            # ✅ Acceptance handled INSIDE ModelPusher
            self.start_model_pusher(evaluation_artifact)

            logging.info("========== Training Pipeline Completed ==========")

        except Exception as e:
            raise MyException(e, sys)