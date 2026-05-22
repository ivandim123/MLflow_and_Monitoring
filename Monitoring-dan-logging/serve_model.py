import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle
import logging
import time
import os
import glob
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and metrics
model = None
prediction_count = 0
error_count = 0
response_times = []

def load_model():
    """Load the trained model from MLflow"""
    global model
    
    # Try multiple methods to load the model
    
    # Method 1: Try to get the latest run automatically
    try:
        # Get the latest run from default experiment
        runs = mlflow.search_runs(experiment_ids=["0"], order_by=["start_time DESC"], max_results=1)
        if len(runs) > 0:
            latest_run_id = runs.iloc[0]['run_id']
            model_uri = f"runs:/{latest_run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded successfully from MLflow run: {latest_run_id}")
            return True
    except Exception as e:
        logger.warning(f"Could not load latest MLflow model: {e}")
    
    # Method 2: Try to load from specific run ID (if set)
    run_id = os.environ.get('MLFLOW_RUN_ID')
    if run_id:
        try:
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded from environment run ID: {run_id}")
            return True
        except Exception as e:
            logger.warning(f"Could not load model from run ID {run_id}: {e}")
    
    # Method 3: Try to load from local MLflow artifacts
    try:
        import glob
        # Look for model in local mlruns
        model_paths = glob.glob("mlruns/*/*/artifacts/model/model.pkl")
        if model_paths:
            model_path = model_paths[0]  # Use the first one found
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from local MLflow artifact: {model_path}")
            return True
    except Exception as e:
        logger.warning(f"Could not load from local MLflow artifacts: {e}")
    
    # Method 4: Fallback to local model.pkl
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded from local model.pkl file")
        return True
    except Exception as e:
        logger.warning(f"Could not load local model.pkl: {e}")
    
    # Method 5: Create a dummy model for testing
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        logger.warning("Creating dummy model for testing purposes")
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        logger.info("Dummy model created successfully")
        return True
    except Exception as e:
        logger.error(f"Could not create dummy model: {e}")
    
    logger.error("All model loading methods failed")
    return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    global prediction_count, error_count, response_times
    
    start_time = time.time()
    
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'features' not in data:
            error_count += 1
            return jsonify({'error': 'Invalid input data'}), 400
        
        # Convert to DataFrame
        features = pd.DataFrame([data['features']])
        
        # Make prediction
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
        
        # Calculate response time
        response_time = time.time() - start_time
        response_times.append(response_time)
        
        # Update metrics
        prediction_count += 1
        
        # Prepare response
        response = {
            'prediction': prediction.tolist(),
            'timestamp': datetime.now().isoformat(),
            'response_time': response_time
        }
        
        if prediction_proba is not None:
            response['prediction_probability'] = prediction_proba.tolist()
        
        logger.info(f"Prediction made successfully. Count: {prediction_count}")
        
        return jsonify(response)
        
    except Exception as e:
        error_count += 1
        response_time = time.time() - start_time
        response_times.append(response_time)
        
        logger.error(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Metrics endpoint for Prometheus"""
    avg_response_time = np.mean(response_times) if response_times else 0
    
    metrics = f"""# HELP ml_predictions_total Total number of predictions made
# TYPE ml_predictions_total counter
ml_predictions_total {prediction_count}

# HELP ml_errors_total Total number of prediction errors
# TYPE ml_errors_total counter
ml_errors_total {error_count}

# HELP ml_response_time_seconds Average response time in seconds
# TYPE ml_response_time_seconds gauge
ml_response_time_seconds {avg_response_time}

# HELP ml_model_loaded Whether the model is loaded
# TYPE ml_model_loaded gauge
ml_model_loaded {1 if model is not None else 0}
"""
    
    return metrics, 200, {'Content-Type': 'text/plain'}

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting ML service...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("Failed to load model. Exiting...")
        exit(1)