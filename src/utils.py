import joblib

def load_model(model_path):
    """
    Load a trained model from the given path.

    Parameters:
    - model_path: str, path to the trained model (e.g., a .pkl file)

    Returns:
    - model: the trained model
    """
    return joblib.load(model_path)