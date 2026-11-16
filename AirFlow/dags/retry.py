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
    
    @task(
        retries=5, 
        retry_delay=timedelta(seconds=5), 
        retry_exponential_backoff=True,
        execution_timeout=timedelta(minutes=1)
    )
    def call_external_api(ti=None):
        attempt = ti.try_number if ti else 1
        print(f"Đang thử gọi API lần thứ: {attempt}")

        if attempt < 3:
            raise ValueError(f"API is down ở lần thử {attempt}! Airflow sẽ retry...")

        print("API is UP! Đã kết nối thành công ở lần thử này.")
        return "API is UP!"
    
    @task
    def process_response(response):
        print(f"response: {response}")

    res = call_external_api()
    process_response(res)

retry_demo()