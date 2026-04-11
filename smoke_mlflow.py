import mlflow
import time

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("smoke-test-v2")

with mlflow.start_run():
    mlflow.log_param("demo", "ok")
    mlflow.log_metric("metric_example", 1.0)
    mlflow.log_text(f"hello mlflow at {time.time()}", "note.txt")

print("Logged 1 run to MLflow. Open http://localhost:5000 to verify.")
