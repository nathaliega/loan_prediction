import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from constants import NUMERICAL_COLS, CATEGORICAL_COLS

class Preprocessing:
    """Handles preprocessing of training and testing datasets."""
    
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, target_column: str):
        """
        Initializes the Preprocessing class with training and testing data.
        
        Args:
            train_data (pd.DataFrame): The training dataset.
            test_data (pd.DataFrame): The testing dataset.
            target_column (str): The name of the target column.
        """
        self.train_data = train_data
        self.test_data = test_data
        self.target_column = target_column

    def drop_columns(self):
        """
        Drops unnecessary columns from the datasets.
        """
        self.train_data = self.train_data.drop(columns=['Loan_ID'], axis=1)
        self.test_data = self.test_data.drop(columns=['Loan_ID'], axis=1)

    def handle_missing_values(self):
        """
        Handles missing values in numerical and categorical columns.
        Numerical columns are imputed using the mean, and categorical columns
        are imputed using the most frequent value.
        """
        imputer = SimpleImputer(strategy='mean')
        self.train_data[NUMERICAL_COLS] = imputer.fit_transform(self.train_data[NUMERICAL_COLS])
        self.test_data[NUMERICAL_COLS] = imputer.transform(self.test_data[NUMERICAL_COLS])

        imputer = SimpleImputer(strategy='most_frequent')
        self.train_data[CATEGORICAL_COLS] = imputer.fit_transform(self.train_data[CATEGORICAL_COLS])
        self.test_data[CATEGORICAL_COLS] = imputer.transform(self.test_data[CATEGORICAL_COLS])

    def encode_categorical_cols(self):
        """
        Encodes categorical columns using label encoding.
        """
        le = LabelEncoder()
        for col in CATEGORICAL_COLS:
            self.train_data[col] = le.fit_transform(self.train_data[col])
            self.test_data[col] = le.transform(self.test_data[col])

    def transform_data(self):
        """
        Applies log transformation to selected numerical columns to
        normalize their distributions.
        """
        log_transformer = FunctionTransformer(np.log1p, validate=True)
        log_columns = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]

        self.train_data[log_columns] = log_transformer.fit_transform(self.train_data[log_columns])
        self.test_data[log_columns] = log_transformer.transform(self.test_data[log_columns])

    def apply_smote(self):
        """
        Applies Synthetic Minority Over-sampling Technique (SMOTE) to balance
        the training dataset.
        """
        smote = SMOTE(random_state=42)
        
        X_train = self.train_data.drop(columns=[self.target_column])
        y_train = self.train_data[self.target_column]
        
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        self.train_data = pd.DataFrame(X_resampled, columns=X_train.columns)
        self.train_data[self.target_column] = y_resampled
    
    def scale_features(self):
        """
        Standardizes numerical features using StandardScaler.
        """
        scaler = StandardScaler()
        self.train_data[NUMERICAL_COLS] = scaler.fit_transform(self.train_data[NUMERICAL_COLS])
        self.test_data[NUMERICAL_COLS] = scaler.transform(self.test_data[NUMERICAL_COLS])


    def preprocess_data(self):
        """
        Executes the complete preprocessing pipeline including:
        - Dropping unnecessary columns
        - Handling missing values
        - Encoding categorical columns
        - Applying log transformation
        - Applying SMOTE
        - Scaling numerical features
        """
        self.drop_columns()
        self.handle_missing_values()
        self.encode_categorical_cols()
        self.transform_data()
        self.apply_smote()
        self.scale_features()

        return self.train_data, self.test_data