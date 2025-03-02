import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from .constants import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS

class Preprocessing:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def drop_columns(self):
        self.train_data = self.train_data.drop(columns=['Loan_ID'], axis=1)
        self.test_data = self.test_data.drop(columns=['Loan_ID'], axis=1)

    def hanlde_missing_values(self):
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
        