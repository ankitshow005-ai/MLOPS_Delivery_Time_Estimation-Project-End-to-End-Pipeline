import sys
from pandas import DataFrame
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ETAPredictorConfig
from src.entity.s3_estimator import Proj1Estimator


class ETAData:
    """
    Holds one ETA prediction request.
    """

    def __init__(
        self,
        delivery_partner: str,
        package_type: str,
        vehicle_type: str,
        delivery_mode: str,
        region: str,
        weather_condition: str,
        distance_km: float,
        package_weight_kg: float,
    ):
        try:
            self.delivery_partner = delivery_partner
            self.package_type = package_type
            self.vehicle_type = vehicle_type
            self.delivery_mode = delivery_mode
            self.region = region
            self.weather_condition = weather_condition
            self.distance_km = distance_km
            self.package_weight_kg = package_weight_kg
        except Exception as e:
            raise MyException(e, sys)

    def get_input_dataframe(self) -> DataFrame:
        try:
            return DataFrame([self.get_input_dict()])
        except Exception as e:
            raise MyException(e, sys)

    def get_input_dict(self) -> dict:
        logging.info("Creating ETA input dictionary")

        return {
            "delivery_partner": self.delivery_partner,
            "package_type": self.package_type,
            "vehicle_type": self.vehicle_type,
            "delivery_mode": self.delivery_mode,
            "region": self.region,
            "weather_condition": self.weather_condition,
            "distance_km": self.distance_km,
            "package_weight_kg": self.package_weight_kg,
        }


class ETAPredictor:
    """
    Loads model from S3 and predicts ETA error (minutes).
    """

    def __init__(self, config: ETAPredictorConfig = ETAPredictorConfig()):
        try:
            self.config = config
            self.estimator = Proj1Estimator(
                bucket_name=config.model_bucket_name,
                model_path=config.model_file_path
            )
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: DataFrame):
        try:
            logging.info("Running ETA prediction")
            return self.estimator.predict(dataframe)
        except Exception as e:
            raise MyException(e, sys)