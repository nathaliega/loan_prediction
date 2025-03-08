import joblib
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import optuna
from constants import TARGET

class TrainModel:
    def __init__(self, train_data):
        self.X_train = train_data.drop(columns=[TARGET])
        self.y_train = train_data[TARGET]
        self.best_params = None  # Initialize as None, will be set after optimization

    def _optimize_hyperparameters(self):
        """Optuna hyperparameter optimization."""
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

    def optimize(self):
        """Run the hyperparameter optimization explicitly."""
        self.best_params = self._optimize_hyperparameters()

    def train(self):
        """Train the model on the entire training dataset."""
        if self.best_params is None:
            raise ValueError("Model has not been optimized yet. Please run optimize() first.")
        model = RandomForestClassifier(**self.best_params, random_state=42)
        model.fit(self.X_train, self.y_train)
        return model

    def save_model(self, model, model_path='../models/model.pkl'):
        """Save the trained model to a file."""
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
