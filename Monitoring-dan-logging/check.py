import mlflow
import os

def check_mlflow_runs():
    """Check available MLflow runs and their details"""
    
    # Set tracking URI if needed
    # mlflow.set_tracking_uri("file:///path/to/mlruns")  # Uncomment if needed
    
    try:
        # List all experiments
        experiments = mlflow.search_experiments()
        print("Available Experiments:")
        print("=" * 50)
        
        for exp in experiments:
            print(f"Experiment ID: {exp.experiment_id}")
            print(f"Name: {exp.name}")
            print(f"Lifecycle Stage: {exp.lifecycle_stage}")
            print("-" * 30)
            
            # Get runs for this experiment
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            
            if len(runs) > 0:
                print(f"Runs in experiment '{exp.name}':")
                for idx, run in runs.iterrows():
                    print(f"  Run ID: {run['run_id']}")
                    print(f"  Status: {run['status']}")
                    print(f"  Start Time: {run['start_time']}")
                    print(f"  Metrics: {dict(run['metrics']) if 'metrics' in run else 'None'}")
                    print(f"  Parameters: {dict(run['params']) if 'params' in run else 'None'}")
                    
                    # Check if model artifact exists
                    try:
                        model_uri = f"runs:/{run['run_id']}/model"
                        model_info = mlflow.models.get_model_info(model_uri)
                        print(f"  ✅ Model artifact found: {model_info.model_uri}")
                    except Exception as e:
                        print(f"  ❌ No model artifact: {str(e)}")
                    
                    print("  " + "-" * 40)
            else:
                print(f"  No runs found in experiment '{exp.name}'")
            
            print("=" * 50)
            
    except Exception as e:
        print(f"Error accessing MLflow: {e}")
        print("\nTrying to check local mlruns directory...")
        
        # Check local mlruns directory
        mlruns_path = "mlruns"
        if os.path.exists(mlruns_path):
            print(f"Found local mlruns directory: {mlruns_path}")
            for root, dirs, files in os.walk(mlruns_path):
                if "meta.yaml" in files:
                    run_id = os.path.basename(root)
                    print(f"Run ID: {run_id}")
                    
                    # Check for model artifacts
                    model_path = os.path.join(root, "artifacts", "model")
                    if os.path.exists(model_path):
                        print(f"  ✅ Model artifact found at: {model_path}")
                    else:
                        print(f"  ❌ No model artifact found")
        else:
            print("No local mlruns directory found")

if __name__ == "__main__":
    check_mlflow_runs()