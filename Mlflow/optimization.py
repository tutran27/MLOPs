
import optuna
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Cancer-classification-optimization")

def objective(trial):
    n_estimates = trial.suggest_int("n_estimators", 50, 200, step=50)
    max_dept = trial.suggest_int("max_depth", 4, 10, step=2)

    with mlflow.start_run(
        run_name = f"Trial_{trial.number}",
        nested=True
    ):    
        model = RandomForestClassifier(
            n_estimators=n_estimates,
            max_depth=max_dept,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "recall": recall,
            "f1_score": f1
        })

        return f1, recall
with mlflow.start_run():
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(objective, n_trials=10)

    # Chọn trial có recall cao nhất trong nhóm tốt nhất (hoặc đổi values[0] để ưu tiên f1)
    best_trial = max(study.best_trials, key=lambda t: t.values[1])
    best_params = best_trial.params
    best_f1, best_recall = best_trial.values

    print(f"Best parameters: {best_params}")
    print(f"Best F1_score: {best_f1}")

    best_model = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1,
    )
    best_model.fit(X_train, y_train)

    mlflow.sklearn.log_model(best_model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
    mlflow.log_params(
        {f"best_{k}" : v for k, v in best_params.items()}
    )
    mlflow.log_metrics({
        "best_f1" : best_f1,
        "best_recall" : best_recall
    })
print("====================================")
print(best_params)