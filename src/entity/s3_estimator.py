from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.entity.estimator import MyModel
import sys
from pandas import DataFrame


class Proj1Estimator:
    """
    Save, load, and predict using a model stored in S3.

    OPTION A DESIGN:
    - model_path is FIXED during initialization
    - save_model() does NOT accept model_path again
    """

    def __init__(self, bucket_name: str, model_path: str | None = None):
        """
        :param bucket_name: S3 bucket name
        :param model_path: S3 key where model/metrics will be stored
        """
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = SimpleStorageService()
        self.loaded_model: MyModel | None = None

    # --------------------------------------------------
    # Check if model exists in S3
    # --------------------------------------------------
    def is_model_present(self, model_path: str | None = None) -> bool:
        try:
            path = model_path or self.model_path
            if path is None:
                raise ValueError("model_path must be provided")

            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name,
                s3_key=path
            )
        except Exception:
            return False

    # --------------------------------------------------
    # Load model from S3
    # --------------------------------------------------
    def load_model(self, model_path: str | None = None) -> MyModel:
        try:
            path = model_path or self.model_path
            if path is None:
                raise ValueError("model_path must be provided")

            return self.s3.load_model(
                model_name=path,
                bucket_name=self.bucket_name
            )
        except Exception as e:
            raise MyException(e, sys)

    # --------------------------------------------------
    # Save model or metrics to S3  ✅ FIXED
    # --------------------------------------------------
    def save_model(self, from_file: str, remove: bool = False):
        """
        Uploads a local file to the S3 path defined at init time.
        """
        try:
            if self.model_path is None:
                raise ValueError("model_path must be set in constructor")

            self.s3.upload_file(
                from_filename=from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove
            )
        except Exception as e:
            raise MyException(e, sys)

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------
    def predict(self, dataframe: DataFrame):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe)
        except Exception as e:
            raise MyException(e, sys)