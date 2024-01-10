import os
import yaml
import subprocess
from datetime import datetime
from typing import Dict, Any

class MLOpsPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.project_name = self.config.get("project_name", "default-ml-project")
        self.model_name = self.config.get("model_name", "default-model")
        self.github_repo = self.config.get("github_repo", f"Witattly2/{self.project_name}")
        print(f"Initialized MLOpsPipeline for project: {self.project_name}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            print(f"Config file not found at {config_path}. Using default settings.")
            return {
                "project_name": "default-ml-project",
                "model_name": "default-model",
                "github_repo": "Witattly2/default-ml-project",
                "data_path": "data/raw/",
                "processed_data_path": "data/processed/",
                "model_path": "models/",
                "metrics_path": "metrics/",
                "training_script": "train.py",
                "evaluation_script": "evaluate.py"
            }
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _run_command(self, command: str, cwd: str = None) -> str:
        print(f"Executing: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise RuntimeError(f"Command failed: {command}")
        return result.stdout

    def setup_dvc(self):
        print("Setting up DVC for data versioning...")
        if not os.path.exists(".dvc"):
            self._run_command("dvc init")
        self._run_command(f"dvc add {self.config["data_path"]}")
        self._run_command("git add .dvcignore data/.dvc")
        self._run_command("git commit -m \"feat: Initialize DVC and add raw data\"")
        self._run_command("dvc push")
        print("DVC setup complete.")

    def run_training(self):
        print("Running model training...")
        # Simulate training script execution
        with open(os.path.join(self.config["model_path"], f"{self.model_name}.pkl"), "w") as f:
            f.write("Simulated trained model content")
        with open(os.path.join(self.config["metrics_path"], "train_metrics.json"), "w") as f:
            json.dump({"accuracy": 0.92, "loss": 0.15}, f)
        
        self._run_command(f"dvc add {self.config["model_path"]}")
        self._run_command(f"dvc add {self.config["metrics_path"]}")
        self._run_command("git add models/.dvc metrics/.dvc")
        self._run_command("git commit -m \"feat: Train model and version artifacts\"")
        self._run_command("dvc push")
        print("Training complete.")

    def run_evaluation(self):
        print("Running model evaluation...")
        # Simulate evaluation script execution
        with open(os.path.join(self.config["metrics_path"], "eval_metrics.json"), "w") as f:
            json.dump({"precision": 0.88, "recall": 0.90, "f1": 0.89}, f)
        
        self._run_command(f"dvc add {self.config["metrics_path"]}")
        self._run_command("git add metrics/.dvc")
        self._run_command("git commit -m \"feat: Evaluate model and version metrics\"")
        self._run_command("dvc push")
        print("Evaluation complete.")

    def deploy_model(self):
        print("Simulating model deployment...")
        # In a real scenario, this would involve deploying to a serving platform
        print(f"Model {self.model_name} deployed successfully!")

    def run_full_pipeline(self):
        print("\n--- Running Full MLOps Pipeline ---")
        self.setup_dvc()
        self.run_training()
        self.run_evaluation()
        self.deploy_model()
        print("--- MLOps Pipeline Completed Successfully ---")

if __name__ == "__main__":
    # Create dummy data and directories for simulation
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    with open("data/raw/sample_data.csv", "w") as f:
        f.write("col1,col2\n1,2\n3,4")

    # Create a dummy config.yaml
    dummy_config = {
        "project_name": "fraud-detection",
        "model_name": "xgboost-v1",
        "data_path": "data/raw/",
        "model_path": "models/",
        "metrics_path": "metrics/"
    }
    with open("config.yaml", "w") as f:
        yaml.dump(dummy_config, f)

    pipeline = MLOpsPipeline()
    # To run the full pipeline, DVC and Git must be configured and authenticated.
    # For this simulation, we'll just demonstrate the steps.
    # pipeline.run_full_pipeline() 
    print("\nNote: To run the full pipeline, DVC and Git must be installed and configured.")
    print("This script demonstrates the structure and calls for an MLOps pipeline.")

    # Clean up dummy files
    # os.remove("config.yaml")
    # os.remove("data/raw/sample_data.csv")
    # os.rmdir("data/raw")
    # os.rmdir("data/processed")
    # os.rmdir("models")
    # os.rmdir("metrics")
