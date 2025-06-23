import joblib
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import optuna
from src.constants import TARGET
import pandas as pd


class ModelTrainer:
    """A class for training a machine learning model with hyperparameter optimization."""

    def __init__(self, train_data: pd.DataFrame) -> None:
        """
        Initializes the ModelTrainer with training data.

        Args:
            train_data (pd.DataFrame): The DataFrame containing the training data, including features and target.
        """
        self.X_train = train_data.drop(columns=[TARGET])
        self.y_train = train_data[TARGET]
        self.best_params = None  

    def _optimize_hyperparameters(self):
        """
        Perform hyperparameter optimization using Optuna.

        Returns:
            Dict[str, int]: The best hyperparameters found during optimization.
        """
        def objective(trial):
            # Hyperparameter search space
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 5, 50)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

            # Model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )

            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1')
            return cv_scores.mean()

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)
        return study.best_params

    def optimize(self) -> None:
        """
        Run the hyperparameter optimization process and store the best parameters.

        This method will set the `best_params` attribute of the class.
        """
        self.best_params = self._optimize_hyperparameters()

    def train(self):
        """
        Train the model on the entire training dataset using the optimized hyperparameters.

        Returns:
            RandomForestClassifier: The trained RandomForestClassifier model.

        Raises:
            ValueError: If the model has not been optimized yet.
        """
        if self.best_params is None:
            raise ValueError("Model has not been optimized yet. Please run optimize() first.")
        model = RandomForestClassifier(**self.best_params, random_state=42)
        model.fit(self.X_train, self.y_train)
        return model

    def save_model(self, model: RandomForestClassifier, model_path: str = 'models/model.pkl'):
        """
        Save the trained model to a file.

        Args:
            model (RandomForestClassifier): The trained model to be saved.
            model_path (str, optional): The path where the model will be saved. Defaults to 'models/model.pkl'.
        """
        joblib.dump(model, model_path)