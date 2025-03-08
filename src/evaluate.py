import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from constants import TARGET

class Evaluation:
    def __init__(self, model_path, test_data, output_path):
        """
        Initializes the Evaluation class with the trained model and test data.

        Parameters:
        - model_path: str, path to the trained model (e.g., a .pkl file)
        - X_test: DataFrame or ndarray, test features
        - y_test: DataFrame or ndarray, true labels for the test set
        - output_csv: str, path to save the evaluation metrics CSV (default is "evaluation_metrics.csv")
        """
        self.model = joblib.load(model_path)  
        self.X_test = test_data.drop(columns=[TARGET])
        self.y_test = test_data[TARGET]
        self.output_path = output_path  

    def _predict(self):
        """Make predictions on the test set."""
        return self.model.predict(self.X_test)

    def evaluate(self):
        """Evaluate the model on the test set and save metrics to CSV."""
        y_pred = self._predict()

        class_report = classification_report(self.y_test, y_pred, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()

        class_report_df.to_csv(self.output_path, mode='w', header=True)
