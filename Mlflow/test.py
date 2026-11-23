import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("demo")
for i in range(2):
    with mlflow.start_run() as run:
        print("Run ID:", run.info.run_id)
        print("Artifact URI:", mlflow.get_artifact_uri())
        mlflow.log_param("x", i)
        mlflow.log_metric("y", i * i)