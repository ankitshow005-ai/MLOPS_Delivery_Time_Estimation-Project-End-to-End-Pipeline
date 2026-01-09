import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object, read_yaml_file
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH


class ModelEvaluation:
    """
    Model Evaluation for ETA Regression.

    RAW DATA
      → recreate target
      → model.predict()
      → metrics
      → acceptance decision
      → metrics.json
    """

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def compute_sla(y_true, y_pred, windows=(30, 60, 120)):
        return {
            f"sla_pm_{w}_min": round(
                np.mean(np.abs(y_true - y_pred) <= w) * 100, 2
            )
            for w in windows
        }

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting Model Evaluation (RAW data -> model)")

            # --------------------------------------------------
            # 1. Load RAW data
            # --------------------------------------------------
            df = pd.read_csv(
                self.data_ingestion_artifact.feature_store_file_path
            )

            # --------------------------------------------------
            # 2. Recreate TARGET
            # --------------------------------------------------
            src_cols = self.schema["derived_source_columns"]

            df["actual_delivery_minutes"] = (
                df[src_cols[0]].apply(lambda x: int(x.split(".")[-1])) * 60
            )
            df["expected_delivery_minutes"] = (
                df[src_cols[1]].apply(lambda x: int(x.split(".")[-1])) * 60
            )

            df[TARGET_COLUMN] = (
                df["actual_delivery_minutes"]
                - df["expected_delivery_minutes"]
            )

            drop_cols = (
                self.schema["drop_columns"]
                + self.schema["derived_source_columns"]
                + ["actual_delivery_minutes", "expected_delivery_minutes"]
            )

            df.drop(columns=drop_cols, inplace=True, errors="ignore")

            X = df.drop(columns=[TARGET_COLUMN])
            y = df[TARGET_COLUMN].values

            # --------------------------------------------------
            # 3. Load trained model
            # --------------------------------------------------
            model = load_object(
                self.model_trainer_artifact.trained_model_file_path
            )

            # --------------------------------------------------
            # 4. Predict
            # --------------------------------------------------
            y_pred = model.predict(X)

            # --------------------------------------------------
            # 5. Metrics
            # --------------------------------------------------
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)

            baseline_mae = np.mean(np.abs(y))
            improvement_pct = (
                (baseline_mae - mae) / baseline_mae
            ) * 100

            sla_metrics = self.compute_sla(y, y_pred)

            metrics = {
                "baseline_mae": round(baseline_mae, 2),
                "model_mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r2_score": round(r2, 4),
                "improvement_pct": round(improvement_pct, 2),
                "sla_metrics": sla_metrics
            }

            logging.info(f"Evaluation metrics: {metrics}")

            # --------------------------------------------------
            # 6. Save metrics.json
            # --------------------------------------------------
            model_dir = os.path.dirname(
                self.model_trainer_artifact.trained_model_file_path
            )
            metrics_file_path = os.path.join(model_dir, "metrics.json")

            with open(metrics_file_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logging.info(f"Metrics saved at: {metrics_file_path}")

            # --------------------------------------------------
            # 7. ACCEPTANCE DECISION (THIS WAS MISSING)
            # --------------------------------------------------
            is_model_accepted = (
                improvement_pct >= self.model_eval_config.changed_threshold_score
            )

            logging.info(f"Model accepted: {is_model_accepted}")

            return ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                baseline_mae=metrics["baseline_mae"],
                model_mae=metrics["model_mae"],
                improvement_pct=metrics["improvement_pct"],
                sla_metrics=sla_metrics,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                metrics_file_path=metrics_file_path
            )

        except Exception as e:
            raise MyException(e, sys)