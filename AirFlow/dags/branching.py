from airflow.sdk import dag, task
from datetime import datetime

@dag(
    dag_id="branching",
    schedule=None,
    start_date=datetime(2026,11,16),
    catchup=False
)

def branching():
    @task
    def compute_value():
        return {
            "f1_score": 0.91
        }

    @task.branch
    def accept_model(score):
        if score["f1_score"] > 0.9:
            return "deploy_model"
        else:
            return "reject_model"

    @task
    def deploy_model():
        print("deploying model")
    
    @task
    def reject_model():
        print("rejecting model")

    score = compute_value() 
    accept_model(score) >> [deploy_model(), reject_model()]
branching()
