import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("breast-cancer-classification")

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

for c in [0.01, 0.1, 1.0]:
    with mlflow.start_run(run_name=f"logreg-C-{c}") as run:

        model = LogisticRegression(
            C=c,
            max_iter=2000,
            solver="liblinear",
            random_state=42,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        mlflow.log_params({
            "C": c,
            "solver": "liblinear",
            "max_iter": 2000,
        })

        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_score": f1,
        })

        mlflow.set_tag("model_type", "LogisticRegression")
        
        print(run.info.run_id, accuracy, f1)