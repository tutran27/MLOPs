from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

import pandas as pd
import optuna

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("california_housing")

def suggest_gb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 7, step=1),
        # "min_samples_split": trial.suggest_int("min_samples_split", 2, 7, step=1),
        # "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 7, step=1),
        # "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }
    return params

def suggest_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 7, step=1),
        # "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        # "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }
    return params

MODEL_SPECS = {
    "gradient_boosting": (
        lambda p: GradientBoostingRegressor(**p, random_state=42),
        suggest_gb
    ),
    "random_forest": (
        lambda p: RandomForestRegressor(**p, random_state=42),
        suggest_rf
    )
}

def tune_model(model_name, X_train, y_train, X_test, y_test, n_trials=10):
    model_fn, suggest_fn = MODEL_SPECS[model_name]
    
    with mlflow.start_run(run_name=f"tune_{model_name}") as parent_run:
        mlflow.set_tag("type", "optimization")

        def objective(trial):
            params = suggest_fn(trial)

            with mlflow.start_run(run_name=f"{model_name}_trial_{trial.number}", nested=True):
                model = model_fn(params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                mlflow.log_params(params)
                mlflow.log_metrics({"mse": mse, "r2": r2})

                return mse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics({"best_mse": study.best_value})
        return {
            "model_name": model_name,
            "run_id": parent_run.info.run_id,
            "best_params": study.best_params,
            "best_val_rmse": study.best_value
        }

    
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load dataset
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
# Tune models
results = [tune_model(model_name, X_train, y_train, X_val, y_val) 
            for model_name in MODEL_SPECS.keys()]
winner = min(results, key=lambda x: x["best_val_rmse"])

build_fn, _ = MODEL_SPECS[winner["model_name"]]
best_model = build_fn(winner["best_params"])
best_model.fit(X_val, y_val)
y_test_pred = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

with mlflow.start_run(run_id=winner["run_id"]):
    mlflow.log_metrics({"test_mse": test_mse, "test_r2": test_r2})
    model_info = mlflow.sklearn.log_model(
        best_model,
        name="best_model",
        input_example=X_train.iloc[:5],
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

















