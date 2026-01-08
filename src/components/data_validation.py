import os
import sys
import json
import pandas as pd

from pandas import DataFrame
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def validate_columns_exist(self, df: DataFrame) -> list:
        expected_cols = set(self.schema["raw_columns"])
        actual_cols = set(df.columns)
        return list(expected_cols - actual_cols)

    def check_duplicate_rows(self, df: DataFrame) -> bool:
        return df.duplicated().any()

    def check_duplicate_columns(self, df: DataFrame) -> bool:
        return df.columns.duplicated().any()

    def validate_numerical_columns(self, df: DataFrame) -> list:
        errors = []
        for col in self.schema["numerical_columns"]:
            if col not in df.columns:
                continue
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.isnull().all():
                errors.append(
                    f"Numerical column '{col}' cannot be converted to numeric"
                )
        return errors

    def validate_categorical_columns(self, df: DataFrame) -> list:
        errors = []
        for col in self.schema["categorical_columns"]:
            if col not in df.columns:
                continue
            if df[col].nunique(dropna=True) < 2:
                errors.append(
                    f"Categorical column '{col}' has <2 unique values"
                )
        return errors

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Running data validation checks")

            df = self.read_data(
                self.data_ingestion_artifact.feature_store_file_path
            )

            errors = []

            # Column presence
            missing_cols = self.validate_columns_exist(df)
            if missing_cols:
                errors.append(f"Missing columns: {missing_cols}")

            # Duplicate checks
            if self.check_duplicate_rows(df):
                errors.append("Duplicate rows found")

            if self.check_duplicate_columns(df):
                errors.append("Duplicate columns found")

            # Type sanity checks
            errors.extend(self.validate_numerical_columns(df))
            errors.extend(self.validate_categorical_columns(df))

            validation_status = len(errors) == 0
            message = " | ".join(errors)

            # Save report
            os.makedirs(
                os.path.dirname(
                    self.data_validation_config.validation_report_file_path
                ),
                exist_ok=True
            )

            report = {
                "validation_status": validation_status,
                "errors": errors
            }

            with open(
                self.data_validation_config.validation_report_file_path, "w"
            ) as f:
                json.dump(report, f, indent=4)

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=message,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            logging.info(f"Data Validation completed: {artifact}")
            return artifact

        except Exception as e:
            raise MyException(e, sys)