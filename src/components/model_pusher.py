import sys
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import (
    ModelPusherArtifact,
    ModelEvaluationArtifact
)
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import Proj1Estimator


class ModelPusher:
    """
    Pushes approved model and metrics to S3.

    Acceptance logic is enforced here.
    """

    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig
    ):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config

            # ✅ model_path is fixed here
            self.model_estimator = Proj1Estimator(
                bucket_name=model_pusher_config.bucket_name,
                model_path=model_pusher_config.s3_model_key_path
            )

            self.metrics_estimator = Proj1Estimator(
                bucket_name=model_pusher_config.bucket_name,
                model_path=model_pusher_config.s3_metrics_key_path
            )

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact | None:
        try:
            logging.info("Starting Model Pusher stage")

            # -------------------------------
            # Hard gate
            # -------------------------------
            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model rejected. Skipping S3 push.")
                return None

            logging.info("Model accepted. Uploading model & metrics to S3.")

            # -------------------------------
            # Upload MODEL
            # -------------------------------
            self.model_estimator.save_model(
                from_file=self.model_evaluation_artifact.trained_model_path
            )

            # -------------------------------
            # Upload METRICS
            # -------------------------------
            self.metrics_estimator.save_model(
                from_file=self.model_evaluation_artifact.metrics_file_path
            )

            artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )

            logging.info(f"Model push successful: {artifact}")
            return artifact

        except Exception as e:
            raise MyException(e, sys)