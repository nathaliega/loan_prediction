import joblib
import pandas as pd
from sklearn.metrics import classification_report
from src.constants import TARGET


def evaluate_model(model_path: str, test_data: pd.DataFrame, output_path: str):
    """
    Loads a trained model, makes predictions on test data, evaluates performance,
    and saves the evaluation metrics to a CSV file.

    Parameters:
    - model_path: str, path to the trained model (e.g., a .pkl file)
    - test_data: DataFrame, test dataset including features and target column
    - output_path: str, path to save the evaluation metrics CSV
    - target_column: str, the name of the target column in test_data
    """
    # Load the trained model
    model = joblib.load(model_path)
    
    # Split features and target
    X_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    
    # Save to CSV
    class_report_df.to_csv(output_path, mode="w", header=True)
    
    return class_report_df
