import os
import sys
from pandas import DataFrame

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.proj1_data import Proj1Data


class DataIngestion:
    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        """
        Data Ingestion component:
        - Fetch raw data
        - Store it in feature store
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Fetch data from source and store it in feature store
        """
        try:
            logging.info("Starting data export from MongoDB")

            my_data = Proj1Data()
            dataframe = my_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )

            logging.info(f"Raw data shape: {dataframe.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)

            logging.info(
                f"Saving raw data to feature store: {feature_store_file_path}"
            )
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            return dataframe

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates data ingestion pipeline
        """
        logging.info("Started data ingestion")

        try:
            self.export_data_into_feature_store()

            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            )

            logging.info(f"Data ingestion completed: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys)