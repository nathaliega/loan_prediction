import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path: str):
    """
    Loads data from a CSV file, splits it into training and testing sets.

    Args:
        file_path (str): The path to the CSV file containing the dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing DataFrames.
    """
    df = pd.read_csv(file_path)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test
