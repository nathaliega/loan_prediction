import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from constants import NUMERICAL_COLS, CATEGORICAL_COLS, TARGET

class Preprocessing:
    """Handles preprocessing of training and testing datasets."""
    
    def __init__(self, target_column: str = None, train_data: pd.DataFrame = None, test_data: pd.DataFrame = None, inference_data: pd.DataFrame = None):
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
        self.inference_data = inference_data

        # Initialize transformers
        self.imputer_num = SimpleImputer(strategy='mean')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        self.label_encoders = {}
        self.scaler = StandardScaler()

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
        self.imputer_num.fit(self.train_data[NUMERICAL_COLS])
        self.train_data[NUMERICAL_COLS] = self.imputer_num.transform(self.train_data[NUMERICAL_COLS])
        self.test_data[NUMERICAL_COLS] = self.imputer_num.transform(self.test_data[NUMERICAL_COLS])

        self.train_data[TARGET] = self.imputer_cat.fit_transform(self.train_data[[TARGET]]).ravel()
        self.test_data[TARGET] = self.imputer_cat.transform(self.test_data[[TARGET]]).ravel()

        self.imputer_cat.fit(self.train_data[CATEGORICAL_COLS])
        self.train_data[CATEGORICAL_COLS] = self.imputer_cat.transform(self.train_data[CATEGORICAL_COLS])
        self.test_data[CATEGORICAL_COLS] = self.imputer_cat.transform(self.test_data[CATEGORICAL_COLS])


    def encode_categorical_cols(self):
        """
        Encodes categorical columns using label encoding.
        """
        le = LabelEncoder()

        self.train_data[TARGET] = le.fit_transform(self.train_data[[TARGET]])
        self.test_data[TARGET] = le.transform(self.test_data[[TARGET]])

        # for col in CATEGORICAL_COLS:
        #     self.train_data[col] = le.fit_transform(self.train_data[col])
        #     self.test_data[col] = le.transform(self.test_data[col])
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            self.train_data[col] = le.fit_transform(self.train_data[col])
            self.test_data[col] = le.transform(self.test_data[col])
            self.label_encoders[col] = le


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
        self.scaler.fit(self.train_data[NUMERICAL_COLS])
        self.train_data[NUMERICAL_COLS] = self.scaler.transform(self.train_data[NUMERICAL_COLS])
        self.test_data[NUMERICAL_COLS] = self.scaler.transform(self.test_data[NUMERICAL_COLS])


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
        self.apply_smote()
        self.scale_features()

        # Save preprocessing parameters
        joblib.dump(self.imputer_num, "../params/imputer_num.pkl")
        joblib.dump(self.imputer_cat, "../params/imputer_cat.pkl")
        joblib.dump(self.label_encoders, "../params/label_encoders.pkl")
        joblib.dump(self.scaler, "../params/scaler.pkl")

        return self.train_data, self.test_data
    
    def preprocess_for_inference(self):
        """
        Preprocesses new data for inference using saved transformation parameters.
        """
        # Load preprocessing parameters
        imputer_num = joblib.load("../params/imputer_num.pkl")
        imputer_cat = joblib.load("../params/imputer_cat.pkl")
        label_encoders = joblib.load("../params/label_encoders.pkl")
        scaler = joblib.load("../params/scaler.pkl")
        
        self.inference_data = self.inference_data.drop(columns=['Loan_ID'])

        # Handle missing values
        self.inference_data[NUMERICAL_COLS] = imputer_num.transform(self.inference_data[NUMERICAL_COLS])
        self.inference_data[CATEGORICAL_COLS] = imputer_cat.transform(self.inference_data[CATEGORICAL_COLS])
        
        # Encode categorical columns
        for col in CATEGORICAL_COLS:
            self.inference_data[col] = label_encoders[col].transform(self.inference_data[col])
        
        # Scale numerical features
        self.inference_data[NUMERICAL_COLS] = scaler.transform(self.inference_data[NUMERICAL_COLS])
        
        return self.inference_data
