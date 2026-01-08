import sys
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    save_object
)
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from src.entity.estimator import MyModel


class ModelTrainer:
    """
    Model Trainer for ETA Error Regression.

    Responsibilities:
    - Load transformed X/y data
    - Train Linear Regression model
    - Evaluate using MAE, RMSE, R²
    - Save final model (preprocessor + regressor)
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Trains Linear Regression model on transformed data.
        """
        try:
            logging.info("========== Starting Model Trainer ==========")

            # --------------------------------------------------
            # 1. Load transformed data
            # --------------------------------------------------
            X_train = load_numpy_array_data(
                self.data_transformation_artifact.X_train_path
            )
            X_test = load_numpy_array_data(
                self.data_transformation_artifact.X_test_path
            )
            y_train = load_numpy_array_data(
                self.data_transformation_artifact.y_train_path
            )
            y_test = load_numpy_array_data(
                self.data_transformation_artifact.y_test_path
            )

            logging.info(
                f"Loaded data | X_train: {X_train.shape}, X_test: {X_test.shape}"
            )

            # --------------------------------------------------
            # 2. Train model
            # --------------------------------------------------
            model = LinearRegression()

            logging.info("Training Linear Regression model")
            model.fit(X_train, y_train)

            # --------------------------------------------------
            # 3. Evaluate model
            # --------------------------------------------------
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            logging.info(f"MAE (minutes): {mae:.2f}")
            logging.info(f"RMSE (minutes): {rmse:.2f}")
            logging.info(f"R2 Score: {r2:.4f}")

            # --------------------------------------------------
            # 4. Load preprocessing pipeline
            # --------------------------------------------------
            preprocessing_obj = load_object(
                self.data_transformation_artifact.preprocessing_object_path
            )

            # --------------------------------------------------
            # 5. Save final model (pipeline + regressor)
            # --------------------------------------------------
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True
            )

            final_model = MyModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=model
            )

            save_object(
                self.model_trainer_config.trained_model_file_path,
                final_model
            )

            logging.info("Final model saved successfully")

            # --------------------------------------------------
            # 6. Return artifact
            # --------------------------------------------------
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path
            )

            logging.info(
                f"Model Trainer completed: {model_trainer_artifact}"
            )

            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys)