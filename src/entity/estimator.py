import numpy as np
import pandas as pd


class MyModel:
    """
    Wrapper class that bundles:
    - preprocessing object (ColumnTransformer)
    - trained ML model

    Used for inference & deployment.
    """

    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Perform prediction on raw input data.

        Steps:
        1. Apply preprocessing
        2. Apply trained model
        """

        X_transformed = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(X_transformed)

    def __repr__(self) -> str:
        return (
            f"MyModel("
            f"preprocessing_object={self.preprocessing_object}, "
            f"trained_model_object={self.trained_model_object})"
        )