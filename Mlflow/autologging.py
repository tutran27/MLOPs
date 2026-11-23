import mlflow 
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.autolog()

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("use-autolog")


X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
n_estimates = [50, 100, 200]
for n in n_estimates:
    with mlflow.start_run(run_name=f"rf_run_{n}") as run:
        model = RandomForestClassifier(
            n_estimators=n,
            max_depth=10,
            random_state=42,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="macro")

        # mlflow.log_params({
        #     "n_estimators": n,
        #     "max_depth": 10,
        #     "random_state": 42,
        # })
        # mlflow.log_metrics({
        #     "accuracy": accuracy,
        #     "f1_score": f1,
        # })