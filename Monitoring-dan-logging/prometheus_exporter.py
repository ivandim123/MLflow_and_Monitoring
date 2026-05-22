import time
import requests
import psutil
import threading
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
ml_service_up = Gauge('ml_service_up', 'ML service availability (1 = up, 0 = down)')
ml_service_response_time = Histogram('ml_service_response_time_seconds', 'ML service response time')
ml_total_predictions = Counter('ml_total_predictions', 'Total predictions made')
ml_prediction_errors = Counter('ml_prediction_errors', 'Total prediction errors')

# System metrics
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage')

# Model performance metrics
model_accuracy = Gauge('ml_model_accuracy', 'Current model accuracy')
model_latency = Gauge('ml_model_latency_seconds', 'Model inference latency')
data_drift_score = Gauge('ml_data_drift_score', 'Data drift detection score')

class MLMetricsCollector:
    def __init__(self, ml_service_url='http://localhost:5000'):
        self.ml_service_url = ml_service_url
        self.running = True
        
    def check_ml_service_health(self):
        """Check if ML service is healthy"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.ml_service_url}/health", timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                ml_service_up.set(1)
                ml_service_response_time.observe(response_time)
                logger.info(f"ML service is healthy (response time: {response_time:.3f}s)")
                return True
            else:
                ml_service_up.set(0)
                logger.warning(f"ML service unhealthy: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            ml_service_up.set(0)
            logger.error(f"ML service check failed: {e}")
            return False
    
    def collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_usage.set(disk_percent)
            
            logger.info(f"System metrics - CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk_percent:.1f}%")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def simulate_model_metrics(self):
        """Simulate model performance metrics (replace with actual model monitoring)"""
        import random
        
        # Simulate accuracy (normally would come from model validation)
        accuracy = 0.85 + random.uniform(-0.1, 0.1)
        model_accuracy.set(max(0, min(1, accuracy)))
        
        # Simulate latency (normally would come from actual inference times)
        latency = random.uniform(0.01, 0.05)
        model_latency.set(latency)
        
        # Simulate data drift score (normally would come from drift detection)
        drift_score = random.uniform(0, 0.3)
        data_drift_score.set(drift_score)
        
        logger.info(f"Model metrics - Accuracy: {accuracy:.3f}, Latency: {latency:.3f}s, Drift: {drift_score:.3f}")
    
    def test_ml_prediction(self):
        """Test ML service with sample prediction"""
        try:
            # Sample data for testing (adjust based on your model)
            test_data = {
                "features": [1.0, 2.0, 3.0, 4.0, 5.0]  # Adjust based on your model features
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.ml_service_url}/predict",
                json=test_data,
                timeout=10
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                ml_total_predictions.inc()
                ml_service_response_time.observe(response_time)
                logger.info(f"Test prediction successful (response time: {response_time:.3f}s)")
            else:
                ml_prediction_errors.inc()
                logger.warning(f"Test prediction failed: HTTP {response.status_code}")
                
        except Exception as e:
            ml_prediction_errors.inc()
            logger.error(f"Test prediction error: {e}")
    
    def collect_metrics(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                # Collect all metrics
                self.check_ml_service_health()
                self.collect_system_metrics()
                self.simulate_model_metrics()
                
                # Test prediction every 30 seconds
                if int(time.time()) % 30 == 0:
                    self.test_ml_prediction()
                
                time.sleep(10)  # Collect metrics every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(10)
    
    def stop(self):
        """Stop metrics collection"""
        self.running = False

def main():
    """Main function to start the metrics exporter"""
    # Start metrics server
    start_http_server(8000)
    logger.info("Prometheus metrics exporter started on port 8000")
    
    # Start metrics collector
    collector = MLMetricsCollector()
    
    # Run metrics collection in a separate thread
    metrics_thread = threading.Thread(target=collector.collect_metrics)
    metrics_thread.daemon = True
    metrics_thread.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down metrics exporter...")
        collector.stop()

if __name__ == '__main__':
    main()