from datetime import datetime
from airflow.sdk import dag, task, Variable, Connection

@dag(dag_id="xom-varriables-connections", start_date=datetime(2026, 1, 1), schedule=None, catchup=False)
def mlops_pipeline():

    @task
    def fetch_and_evaluate() -> dict:
        conn = Connection.get("mlflow_api")
        print(f"Connecting to MLflow: {conn.host}:{conn.port}")
        
        metrics = {"model_name": "xgboost_v1", "f1_score": 0.8}
        return metrics 

    @task
    def deploy_decision(metrics: dict):
        score = metrics["f1_score"]
        
        threshold = float(Variable.get("min_deploy_f1", default=0.85))
        
        if score >= threshold:
            print(f"Deploying model {metrics['model_name']} ({score} >= {threshold})")
        else:
            print(f"Model rejected ({score} < {threshold})")

    metrics_result = fetch_and_evaluate()
    deploy_decision(metrics_result)

mlops_pipeline()