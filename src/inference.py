import joblib
import pandas as pd

class ModelPredictor:
    """A class for loading a trained model and making predictions."""

    def __init__(self, model_path: str = 'models/random_forest_model.pkl'):
        """
        Initializes the InferenceModel with a trained machine learning model.

        Args:
            model_path (str, optional): Path to the serialized model file. Defaults to 'models/random_forest_model.pkl'.
        """
        self.model = joblib.load(model_path)

    def predict(self, X: pd.DataFrame):
        """
        Makes predictions using the loaded model and saves results to a CSV file.

        Args:
            X (pd.DataFrame): The input data for prediction.

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'predictions' column.
        """
        X['predictions'] = self.model.predict(X)  
        X.to_csv('predictions/predictions.csv', index=False)  
        return X