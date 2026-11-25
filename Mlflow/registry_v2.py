import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, recall_score

from mlflow import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Registry-Demo")

client = MlflowClient()

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimates = [50, 100, 200]

best_f1 = -1.0
best_recall = -1.0
best_model = None
best_run_id = None
best_es = None  
for es in n_estimates:
    with mlflow.start_run(run_name=f"RandomForest_{es}") as run:
        model = RandomForestClassifier(n_estimators=es, 
                                        max_depth=10, 
                                        random_state=42)
        model.fit(X_train, y_train)
        predicts = model.predict(X_test)

        f1 = f1_score(y_test, predicts)
        recall = recall_score(y_test, predicts)
        if f1 > best_f1:
            best_f1 = f1
            best_recall = recall
            best_model = model
            best_run_id = run.info.run_id
            best_es = es
        
        mlflow.log_params({
            "n_estimators": es,
            "max_depth": 10,
            "random_state": 42
        })
        mlflow.log_metrics({
            "f1": f1,
            "recall": recall
        })

print(f"Best F1: {best_f1}, Best Recall: {best_recall}, Best ES: {best_es}, Best Run ID: {best_run_id}")
print(f"Best Model: {best_model}")

with mlflow.start_run(run_id=best_run_id):
    model_info = mlflow.sklearn.log_model(best_model, name="model", 
                                input_example=X_test[:5],
                                signature=mlflow.models.infer_signature(X_test, predicts),
                                skops_trusted_types=["sklearn.tree._tree.Tree"])

    prod_model = mlflow.pyfunc.load_model("models:/breast-cancer-model@prod")
    pro_predict = prod_model.predict(X_test)
    f1_prod = f1_score(y_test, pro_predict)
    recall_prod = recall_score(y_test, pro_predict)
    print(f"Prod F1: {f1_prod}, Prod Recall: {recall_prod}")
    if best_f1 >= f1_prod and best_recall >= recall_prod:
        reg=mlflow.register_model(model_info.model_uri, "breast-cancer-model")
        print("Model registered")
        client.set_registered_model_alias("breast-cancer-model", "prod", reg.version)
        client.set_model_version_tag(name = "breast-cancer-model", 
                            version = reg.version, 
                            key= "f1_score", value = str(best_f1))
        client.set_model_version_tag(name = "breast-cancer-model", 
                            version = reg.version, 
                            key= "recall_score", value = str(best_recall))
        client.update_model_version(name="breast-cancer-model", 
        version = reg.version,
                            description="RandomForest classifier with f1: {} and recall: {}".format(best_f1, best_recall))

    else:
        print("Model not registered")


