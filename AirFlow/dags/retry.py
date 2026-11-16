from airflow.sdk import dag, task
from datetime import datetime, timedelta
import random

@dag(
    dag_id="retry-dag",
    schedule=None,
    start_date=datetime(2026,11,16),
    catchup=False
)

def retry_demo():
    
    @task(retries=2, 
        retry_delay=timedelta(seconds=5), 
        retry_exponential_backoff=True,
        execution_timeout=timedelta(minutes=1))
    def call_external_api():
        if random.random() < 0.5:
            raise ValueError("API is down! Re")
        else:
            return "API is UP!"
    
    @task
    def process_response(response):
        print(f"response: {response}")

    res = call_external_api()
    process_response(res)

retry_demo()