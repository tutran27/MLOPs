import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from airflow.sdk import dag, task
from datetime import datetime

@dag(
    dag_id="experiment_tracking",
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["mlops", "experiment-tracking"],
)
def experiment_tracking_dag():
    
    @task
    def data_loading():
        X, y= load_iris(return_X_y=True)
        return {
            "X": X.tolist(),
            "y": y.tolist()
        }

    @task
    def split_dataset(data):
        X = data["X"]
        y = data["y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
        
    @task
    def train_model(train_test_dataset):
        X_train = train_test_dataset["X_train"]
        X_test = train_test_dataset["X_test"]
        y_train = train_test_dataset["y_train"]
        y_test = train_test_dataset["y_test"]

        mlflow.set_tracking_uri("sqlite:////opt/airflow/dags/mlflow.db")
        
        experiment_name = "iris-v3"

        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name,
                                     artifact_location="/opt/airflow/dags/mlruns")

        mlflow.set_experiment(experiment_name)
        
        for n_estimators in [25, 50, 100]:

            with mlflow.start_run():

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=42,
                )

                model.fit(X_train, y_train)

                predictions = model.predict(X_test)
                accuracy = accuracy_score(
                    y_test, predictions
                )

                mlflow.log_param(
                    "n_estimators", n_estimators
                )

                mlflow.log_metric(
                    "accuracy", accuracy
                )

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    skops_trusted_types=["sklearn.tree._tree.Tree"]
                )

    data = data_loading()
    train_test_data = split_dataset(data)
    train_model(train_test_data)

experiment_tracking_dag()