import sys
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact
)
from src.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file
)
from src.constants import SCHEMA_FILE_PATH


class DataTransformation:
    """
    Data Transformation stage for ETA Error Prediction.

    Responsibilities:
    - Create target column (eta_error_minutes)
    - Drop unused columns (schema-driven)
    - Encode categorical features
    - Scale numerical features
    - Train-test split
    - Save transformed arrays & preprocessing pipeline
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def extract_encoded_hours(time_str: str) -> int:
        """
        Dataset encodes HOURS in fractional seconds.
        Example:
        1970-01-01 00:00:00.000000008 -> 8 hours
        """
        return int(time_str.split(".")[-1])

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Running data transformation checks")

            # --------------------------------------------------
            # Hard gate: validation must pass
            # --------------------------------------------------
            if not self.data_validation_artifact.validation_status:
                raise Exception(
                    f"Validation failed: {self.data_validation_artifact.message}"
                )

            # --------------------------------------------------
            # 1. Load raw data
            # --------------------------------------------------
            df = self.read_data(
                self.data_ingestion_artifact.feature_store_file_path
            )
            logging.info(f"Raw data loaded with shape: {df.shape}")

            # --------------------------------------------------
            # 2. Create target column (schema-driven)
            # --------------------------------------------------
            src_cols = self.schema["derived_source_columns"]

            df["actual_delivery_hours"] = df[src_cols[0]].apply(self.extract_encoded_hours)
            df["expected_delivery_hours"] = df[src_cols[1]].apply(self.extract_encoded_hours)

            df["actual_delivery_minutes"] = df["actual_delivery_hours"] * 60
            df["expected_delivery_minutes"] = df["expected_delivery_hours"] * 60

            target_col = self.schema["target_column"]
            df[target_col] = (
                df["actual_delivery_minutes"] - df["expected_delivery_minutes"]
            )

            logging.info("Target column eta_error_minutes created")

            # --------------------------------------------------
            # 3. Drop columns (schema-driven)
            # --------------------------------------------------
            drop_cols = (
                self.schema["drop_columns"]
                + self.schema["derived_source_columns"]
                + [
                    "actual_delivery_hours",
                    "expected_delivery_hours",
                    "actual_delivery_minutes",
                    "expected_delivery_minutes",
                ]
            )

            df.drop(columns=drop_cols, inplace=True, errors="ignore")
            logging.info(f"Columns after drop: {df.columns.tolist()}")

            # --------------------------------------------------
            # 4. Split features & target
            # --------------------------------------------------
            X = df.drop(columns=[target_col])
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            logging.info(
                f"Train-test split completed | "
                f"X_train: {X_train.shape}, X_test: {X_test.shape}"
            )

            # --------------------------------------------------
            # 5. ColumnTransformer (ENCODING + SCALING)
            # --------------------------------------------------
            categorical_cols = self.schema["categorical_columns"]
            numerical_cols = [
                col for col in self.schema["numerical_columns"]
                if col != target_col
            ]

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "categorical",
                        OneHotEncoder(
                            drop="first",
                            handle_unknown="ignore",
                            sparse_output=False
                        ),
                        categorical_cols
                    ),
                    (
                        "numerical",
                        MinMaxScaler(),
                        numerical_cols
                    )
                ],
                remainder="drop"
            )

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Encoding & scaling completed using ColumnTransformer")

            # --------------------------------------------------
            # 6. Save transformed arrays
            # --------------------------------------------------
            os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok=True)
            os.makedirs(self.data_transformation_config.transformed_object_dir, exist_ok=True)

            save_numpy_array_data(self.data_transformation_config.X_train_path, X_train_transformed)
            save_numpy_array_data(self.data_transformation_config.X_test_path, X_test_transformed)
            save_numpy_array_data(self.data_transformation_config.y_train_path, y_train.to_numpy())
            save_numpy_array_data(self.data_transformation_config.y_test_path, y_test.to_numpy())

            # --------------------------------------------------
            # 7. Save preprocessing pipeline
            # --------------------------------------------------
            save_object(
                self.data_transformation_config.preprocessing_object_path,
                preprocessor
            )

            logging.info("Data Transformation completed")

            return DataTransformationArtifact(
                X_train_path=self.data_transformation_config.X_train_path,
                X_test_path=self.data_transformation_config.X_test_path,
                y_train_path=self.data_transformation_config.y_train_path,
                y_test_path=self.data_transformation_config.y_test_path,
                preprocessing_object_path=self.data_transformation_config.preprocessing_object_path
            )

        except Exception as e:
            raise MyException(e, sys)