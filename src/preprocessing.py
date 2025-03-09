import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from constants import NUMERICAL_COLS, CATEGORICAL_COLS, TARGET
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pickle
from imblearn.pipeline import Pipeline as ImbPipeline  #

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'fitted_')
        return X.drop(columns=self.columns_to_drop)

    def get_feature_names_out(self, input_features=None):
        return [col for col in input_features if col not in self.columns_to_drop]


class PreprocessingPipeline:
    def __init__(self):
        self._preprocessor = self._make_preprocessor()
        self._target_pipeline = self._make_target_pipeline()


    def preprocess_data(self, train_data, test_data):
        """Applies transformations and SMOTE to training data, only transformations to test data."""
        # Separate features and target columns
        X_train = train_data.drop(columns=[TARGET])
        y_train = train_data[TARGET]
        X_test = test_data.drop(columns=[TARGET])
        y_test = test_data[TARGET]

        X_train = self._preprocessor.fit_transform(X_train)
        X_test = self._preprocessor.transform(X_test)

        y_train = self._target_pipeline.fit_transform(y_train.to_numpy().reshape(-1, 1))  # Convert to numpy and reshape
        y_test = self._target_pipeline.transform(y_test.to_numpy().reshape(-1, 1))  # Convert to numpy and reshape

        # Apply SMOTE for resampling (only on training data)
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)

        print(X_train.columns)
        # Update train_data with resampled y_train (SMOTE increases the number of samples)
        processed_train_data = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train, columns=[TARGET])], axis=1)

        # Update test_data with transformed y_test (test data does not undergo SMOTE)
        X_test[TARGET] = y_test

        self.save('../params/preprocessing_pipeline.pkl')

        return processed_train_data, X_test
    
    def transform_for_inference(self, X, y = None):
        loaded_pipeline = self.load('../params/preprocessing_pipeline.pkl')  # Load the fitted pipeline
        X = loaded_pipeline._preprocessor.transform(X)  # Use the loaded (fitted) preprocessor

        if y is not None:
            y = loaded_pipeline._target_pipeline.transform(y)  # Use the loaded (fitted) target pipeline
            return X, y
        return X


    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)


    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


    def _make_preprocessor(self):
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder()),
            ]
        )

        ct = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, NUMERICAL_COLS),
                ("cat", cat_pipeline, CATEGORICAL_COLS),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
            n_jobs=-1,
        )

        preprocessor = Pipeline(
            steps=[
                ("ct", ct),
                ("dropper", ColumnDropper(columns_to_drop=["Loan_ID"])),
            ]
        )
        preprocessor.set_output(transform="pandas")

        return preprocessor


    def _make_target_pipeline(self):
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder()),
            ]
        )
    