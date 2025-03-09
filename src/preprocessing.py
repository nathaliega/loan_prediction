import pandas as pd
from imblearn.over_sampling import SMOTE
from src.constants import NUMERICAL_COLS, CATEGORICAL_COLS, TARGET
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pickle
import logging


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Custom transformer to drop specified columns from a DataFrame.

    Attributes:
        columns_to_drop (list): List of column names to drop.
    """

    def __init__(self, columns_to_drop):
        """Initializes the ColumnDropper with columns to drop.

        Args:
            columns_to_drop (list): List of column names to remove.
        """
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        """Fits the transformer (no actual fitting is required).

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series, optional): Target variable. Defaults to None.

        Returns:
            ColumnDropper: Returns self.
        """
        self.fitted_ = True
        return self

    def transform(self, X):
        """Transforms the input DataFrame by dropping specified columns.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Transformed DataFrame with specified columns dropped.
        """
        check_is_fitted(self, 'fitted_')
        return X.drop(columns=self.columns_to_drop)

    def get_feature_names_out(self, input_features=None):
        """Gets the names of the output features after transformation.

        Args:
            input_features (list, optional): List of original feature names. Defaults to None.

        Returns:
            list: List of feature names after transformation.
        """
        return [col for col in input_features if col not in self.columns_to_drop]


class PreprocessingPipeline:
    """Pipeline for preprocessing data, including transformations and SMOTE resampling.

    Attributes:
        _preprocessor (Pipeline): Data transformation pipeline.
        _target_pipeline (Pipeline): Target variable transformation pipeline.
        _logger (logging.Logger): Logger for pipeline events.
    """

    def __init__(self):
        """Initializes the preprocessing pipeline by creating sub-pipelines."""
        self._preprocessor = self._make_preprocessor()
        self._target_pipeline = self._make_target_pipeline()
        self._logger = logging.getLogger(self.__class__.__name__)

    def preprocess_data(self, train_data, test_data):
        """Applies transformations and SMOTE to training data, and only transformations to test data.

        Args:
            train_data (pd.DataFrame): Training dataset including features and target.
            test_data (pd.DataFrame): Test dataset including features and target.

        Returns:
            tuple: Processed training data (pd.DataFrame), Processed test data (pd.DataFrame).
        """
        self._logger.info("Starting preprocessing pipeline")

        X_train = train_data.drop(columns=[TARGET])
        y_train = train_data[TARGET]
        X_test = test_data.drop(columns=[TARGET])
        y_test = test_data[TARGET]

        X_train = self._preprocessor.fit_transform(X_train)
        X_test = self._preprocessor.transform(X_test)

        y_train = self._target_pipeline.fit_transform(y_train.to_numpy().reshape(-1, 1))
        y_test = self._target_pipeline.transform(y_test.to_numpy().reshape(-1, 1))

        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
        processed_train_data = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train, columns=[TARGET])], axis=1)
        X_test[TARGET] = y_test

        self.save('params/preprocessing_pipeline.pkl')
        self._logger.info("Finished preprocessing pipeline")

        return processed_train_data, X_test
    
    def transform_for_inference(self, X, y=None):
        """Applies preprocessing transformations for inference.

        Args:
            X (pd.DataFrame): Feature dataset for inference.
            y (pd.Series, optional): Target values, if available. Defaults to None.

        Returns:
            pd.DataFrame or tuple: Processed features, and optionally processed target.
        """
        loaded_pipeline = self.load('params/preprocessing_pipeline.pkl')
        X = loaded_pipeline._preprocessor.transform(X)

        if y is not None:
            y = loaded_pipeline._target_pipeline.transform(y)
            return X, y
        return X

    def save(self, path):
        """Saves the preprocessing pipeline to a file.

        Args:
            path (str): File path to save the pipeline.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Loads a preprocessing pipeline from a file.

        Args:
            path (str): File path from which to load the pipeline.

        Returns:
            PreprocessingPipeline: Loaded preprocessing pipeline instance.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def _make_preprocessor(self):
        """Creates the feature preprocessing pipeline.

        Returns:
            Pipeline: Feature preprocessing pipeline.
        """
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
        """Creates the target variable preprocessing pipeline.

        Returns:
            Pipeline: Target variable transformation pipeline.
        """
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder()),
            ]
        )
