from airflow.sdk import dag, task
from datetime import datetime

@dag(
    dag_id="task-flow",
    schedule=None,
    start_date=datetime(2026,11,16),
    catchup=False
)
def taskflow():

    @task
    def load_predict():
        print("Predict loading...")
        return {
            "predict": [0,1,1,1,0,0],
            "label": [1,1,1,1,1,0]
        }

    @task
    def compute_accuracy(data: dict) -> float:
        print("Computing accuracy...")
        correct = sum(p==l for p,l in zip(data["predict"],data["label"]))
        return correct / len(data["label"])
    
    @task
    def notify_accuracy(accuracy: float):
        print(f"Accuracy: {accuracy:.2f}")

    @task
    def save_accuracy(accuracy: float):
        with open("/tmp/accuracy.txt","w") as f:
            f.write(str(accuracy))

    data = load_predict()
    accuracy = compute_accuracy(data)
    t1 = notify_accuracy(accuracy)
    t2 = save_accuracy(accuracy)
    t1 >> t2
taskflow()