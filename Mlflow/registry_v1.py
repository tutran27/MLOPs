import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score

from mlflow import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Registry-Demo")

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="LogisticRegression") as run:
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    predicts = model.predict(X_test)

    f1 = f1_score(y_test, predicts)
    recall = recall_score(y_test, predicts)
    
    model_info = mlflow.sklearn.log_model(model, name="model", 
                            registered_model_name="breast-cancer-model",
                            input_example=X_test[:5],
                            signature=mlflow.models.infer_signature(X_test, predicts))

    mlflow.log_metrics({"f1": f1, "recall": recall})



client = MlflowClient()
client.set_model_version_tag(name="breast-cancer-model", 
                               version=model_info.registered_model_version, 
                               key="f1_score", 
                               value=str(f1))
client.set_model_version_tag(name="breast-cancer-model", 
                               version=model_info.registered_model_version, 
                               key="recall_score", 
                               value=str(recall))
client.set_registered_model_alias(name="breast-cancer-model", 
                                  alias="prod", 
                                  version=model_info.registered_model_version)

# Thêm mô tả và tag kiểm duyệt
client.update_model_version(
    name="breast-cancer-model",
    version=str(model_info.registered_model_version),
    description=f"LogisticRegression, F1={f1:.4f}, Recall={recall:.4f}",
)