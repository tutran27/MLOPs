from airflow.sdk import dag, task
from datetime import datetime

@dag(dag_id="mapping", start_date=datetime(2026,1,1), schedule=None, catchup=False)
def model_training_pipeline():
    
    @task
    def data_collection():
        return [
            {"id":1, "row": 1000},
            {"id":2, "row": 5000},
            {"id":3, "row": 2000}
        ]

    @task
    def process_data(data:list[dict]):
        return data["row"]
    
    @task
    def aggregate_data(data:list):
        print(f"aggregating data {sum(data)}")
        return sum(data)

    data_collection=data_collection()
    rows=process_data.expand(data=data_collection)
    total_rows=aggregate_data(rows)     

model_training_pipeline()