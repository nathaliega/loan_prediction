import joblib

class InferenceModel:
    def __init__(self, model_path='models/random_forest_model.pkl'):
        """Initialize with a trained model."""
        self.model = joblib.load(model_path)

    def predict(self, X):
        """Make predictions on the given unseen data."""
        return self.model.predict(X)
