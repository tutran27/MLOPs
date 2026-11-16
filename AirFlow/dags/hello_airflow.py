from datetime import datetime
from airflow.sdk import dag, task


@dag(
    dag_id="hello_airflow",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
)
def hello_airflow():

    @task
    def hello():
        print("Airflow is working!")

    hello()


hello_airflow()