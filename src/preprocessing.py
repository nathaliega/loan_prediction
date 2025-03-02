import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from .constants import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS

class Preprocessing:
    def __init__(self, train_data, test_data, target_column):
        self.train_data = train_data
        self.test_data = test_data
        self.target_column = target_column

    def drop_columns(self):
        self.train_data = self.train_data.drop(columns=['Loan_ID'], axis=1)
        self.test_data = self.test_data.drop(columns=['Loan_ID'], axis=1)

    def handle_missing_values(self):
        imputer = SimpleImputer(strategy='mean')
        imputer.fit_transform(self.train_data[NUMERICAL_COLUMNS])
        self.test_data[NUMERICAL_COLUMNS] = imputer.transform(self.test_data[NUMERICAL_COLUMNS])

        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit(self.train_data[CATEGORICAL_COLUMNS])
        self.test_data[CATEGORICAL_COLUMNS] = imputer.transform(self.test_data[CATEGORICAL_COLUMNS])

    def encode_categorical_columns(self):
        le = LabelEncoder()
        for col in CATEGORICAL_COLUMNS:
            self.train_data[col] = le.fit_transform(self.train_data[col])
            self.test_data[col] = le.transform(self.test_data[col])

    def transform_data(self):
        log_transformer = FunctionTransformer(np.log1p, validate=True)
        log_columns = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]

        self.train_data[log_columns] = log_transformer.fit_transform(self.train_data[log_columns])

        self.test_data[log_columns] = log_transformer.transform(self.test_data[log_columns])

    def apply_smote(self):
        smote = SMOTE(random_state=42)
        
        X_train = self.train_data.drop(columns=[self.target_column])
        y_train = self.train_data[self.target_column]
        
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        self.train_data = pd.DataFrame(X_resampled, columns=X_train.columns)
        self.train_data[self.target_column] = y_resampled

    def preprocess(self):
        self.drop_columns()
        self.handle_missing_values()
        self.encode_categorical_columns()
        self.transform_data()
        self.apply_smote()