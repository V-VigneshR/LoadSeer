# app.py
"""Flask application and API endpoints"""

from flask import Flask, request, jsonify, Response
import requests
import signal
import numpy as np
from datetime import datetime
import threading
from collections import deque
from config import *
from aws_manager import AWSManager
from ml_predictor import PersistentTrafficPredictor
from load_balancer import LoadBalancer
from monitor import BackgroundMonitor

app = Flask(__name__, static_folder=None)

def create_app():
    """Application factory"""
    aws_manager = AWSManager()
    predictor = PersistentTrafficPredictor(aws_manager.s3)
    load_balancer = LoadBalancer(aws_manager, predictor)
    monitor = BackgroundMonitor(load_balancer)

    # Start background threads
    threading.Thread(target=monitor.background_scaler, daemon=True).start()
    threading.Thread(target=monitor.ml_background_monitor, daemon=True).start()

    # Initialize instances
    try:
        aws_manager.manage_instance('start', 'small')
        aws_manager.register_instances('small')

        for group in ['medium', 'large']:
            aws_manager.manage_instance('stop', group)
            aws_manager.deregister_instances(group)
    except Exception as e:
        print(f"Initialization failed: {str(e)}")

    return app, load_balancer, predictor

app, lb, predictor = create_app()

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST'])
@app.route('/<path:path>', methods=['GET', 'POST'])
def reverse_proxy(path):
    """Reverse proxy for backend requests"""
    if path != 'metrics':
        lb.request_count += 1
        print(f"[REQUEST] Count: {lb.request_count}, Path: /{path}, Rate: {lb.get_arrival_rate():.2f} req/sec")

    try:
        headers = dict(request.headers)
        headers['instance-type'] = lb.current_state
        url = f"{BACKEND_SERVER}/{path}"
        resp = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            params=request.args
        )

        return Response(resp.content, status=resp.status_code, headers=dict(resp.headers))
    except Exception as e:
        return f"Error routing request: {e}", 500

@app.route('/metrics')
def metrics():
    """Enhanced metrics endpoint"""
    def safe_convert(value):
        """Convert numpy types to native Python types"""
        if isinstance(value, (np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.int32, np.int64)):
            return int(value)
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    try:
        rate = lb.get_arrival_rate()
        features = predictor.extract_features(lb.traffic_history)
        predicted_rate = predictor.predict_traffic(features) if features else None

        performance_metrics = predictor.performance_tracker.get_current_accuracy_metrics()
        prediction_trend = predictor.performance_tracker.get_prediction_trend()

        response_data = {
            "arrival_rate": safe_convert(round(rate, 2)),
            "predicted_rate": safe_convert(round(predicted_rate, 2)) if predicted_rate else None,
            "active_group": lb.current_state,
            "ml_trained": predictor.is_trained,
            "training_samples": len(predictor.training_data),
            "last_retrain": safe_convert(predictor.last_retrain),
            "jmeter_patterns": {k: safe_convert(v) for k,v in predictor.jmeter_patterns.items()},
            "request_count": lb.request_count,
            "uptime_minutes": safe_convert(round((datetime.now() - lb.start_time).total_seconds() / 60, 2)),
            "debug_info": {
                "traffic_history_length": len(lb.traffic_history),
                "current_rate_calc": f"{lb.request_count} requests / {(datetime.now() - lb.start_time).total_seconds():.1f} seconds"
            }
        }

        if performance_metrics:
            response_data["model_performance"] = {k: safe_convert(v) for k,v in performance_metrics.items()}
            response_data["prediction_trend"] = prediction_trend
            response_data["model_quality"] = interpret_model_quality(performance_metrics)
            response_data["recommendations"] = get_model_recommendations(performance_metrics)

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route('/debug-prediction')
def debug_prediction():
    """Debug endpoint to see prediction pipeline"""
    try:
        current_rate = lb.get_arrival_rate()
        features = predictor.extract_features(lb.traffic_history)

        debug_info = {
            "current_rate": current_rate,
            "traffic_history_length": len(lb.traffic_history),
            "required_lags": predictor.sequence_lags,
            "model_trained": predictor.is_trained,
            "training_samples": len(predictor.training_data),
            "features": features,
            "features_length": len(features) if features else 0
        }

        if features:
            predicted_rate = predictor.predict_traffic(features)
            debug_info["predicted_rate"] = predicted_rate

            if len(features) >= predictor.sequence_lags + 3:
                debug_info["feature_breakdown"] = {
                    "lag_rates": features[:predictor.sequence_lags],
                    "hour": features[predictor.sequence_lags],
                    "day_of_week": features[predictor.sequence_lags + 1],
                    "minute": features[predictor.sequence_lags + 2],
                    "pattern_features": features[predictor.sequence_lags + 3:]
                }

        return jsonify(debug_info)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/calibrate-model')
def calibrate_model():
    """Force recalibration of the model"""
    try:
        if len(predictor.training_data) < 20:
            return jsonify({"status": "Need at least 20 samples", "samples": len(predictor.training_data)})
        
        predictor.train_model()
        
        predictor.jmeter_patterns = {
            'ramp_up_detected': False,
            'peak_detected': False,
            'ramp_down_detected': False,
            'pattern_start_time': None
        }
        
        return jsonify({
            "status": "Model recalibrated",
            "training_samples": len(predictor.training_data),
            "last_retrain": predictor.last_retrain.isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/prediction-chart')
def prediction_chart():
    """Get data for plotting predictions vs actual"""
    try:
        recent_history = list(lb.traffic_history)[-50:]

        chart_data = []
        for i, point in enumerate(recent_history):
            if i >= predictor.sequence_lags:
                features = predictor.extract_features(deque(recent_history[:i+1]))
                predicted = predictor.predict_traffic(features)

                chart_data.append({
                    "timestamp": point['timestamp'].isoformat(),
                    "actual_rate": point['rate'],
                    "predicted_rate": predicted,
                    "state": point['state']
                })

        return jsonify(chart_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset-patterns')
def reset_jmeter_patterns():
    """Reset JMeter pattern detection for new test"""
    lb.request_count = 0
    lb.start_time = datetime.now()
    lb.traffic_history.clear()

    predictor.jmeter_patterns = {
        'ramp_up_detected': False,
        'peak_detected': False,
        'ramp_down_detected': False,
        'pattern_start_time': None
    }
    return jsonify({"status": "JMeter patterns and counters reset"})

def interpret_model_quality(metrics):
    """Provide human-readable interpretation of model quality"""
    r2 = metrics.get('r2_score', 0)
    mape = metrics.get('mape', 100)
    accuracy_10 = metrics.get('accuracy_within_10_percent', 0)
    accuracy_25 = metrics.get('accuracy_within_25_percent', 0)
    scaling_accuracy = metrics.get('scaling_accuracy', 0)
    sample_size = metrics.get('sample_size', 0)

    if sample_size < 30:
        return "Insufficient Data - Need more predictions for reliable assessment"
    
    if mape < 3 and accuracy_10 >= 90 and scaling_accuracy >= 90:
        return "Excellent - World-class accuracy for load balancing"
    elif mape < 5 and accuracy_10 >= 85 and scaling_accuracy >= 85:
        return "Very Good - High accuracy, reliable for production"
    elif mape < 10 and accuracy_25 >= 80 and scaling_accuracy >= 80:
        return "Good - Solid performance, suitable for most scenarios"
    elif mape < 15 and accuracy_25 >= 70 and scaling_accuracy >= 70:
        return "Fair - Moderate accuracy, consider improvements"
    elif mape < 25 and accuracy_25 >= 50:
        return "Below Average - Needs improvement for reliable scaling"
    else:
        return "Poor - Significant issues, requires immediate attention"

def get_model_recommendations(metrics):
    """Provide specific recommendations based on model performance"""
    recommendations = []
    
    r2 = metrics.get('r2_score', 0)
    mape = metrics.get('mape', 100)
    accuracy_10 = metrics.get('accuracy_within_10_percent', 0)
    sample_size = metrics.get('sample_size', 0)
    scaling_accuracy = metrics.get('scaling_accuracy', 0)
    
    if sample_size < 50:
        recommendations.append("Need more training data - run longer tests for robust evaluation")
    
    if mape > 10:
        recommendations.append("High prediction error - consider feature engineering or different algorithm")
    elif mape > 5:
        recommendations.append("Moderate prediction error - model could benefit from more diverse training data")
    
    if scaling_accuracy < 80:
        recommendations.append("Poor scaling decisions - review threshold values or prediction logic")
    elif scaling_accuracy < 90:
        recommendations.append("Good scaling accuracy - minor threshold adjustments may help")
    
    if r2 < 0:
        recommendations.append("Negative R² suggests overfitting or model complexity issues")
    elif r2 < 0.3 and mape < 5:
        recommendations.append("Low R² with good MAPE is normal for steady traffic - consider longer evaluation period")
    elif r2 < 0.3 and mape > 10:
        recommendations.append("Low R² and high MAPE - model may need different approach or more features")
    
    if accuracy_10 < 70 and mape < 5:
        recommendations.append("Good average error but inconsistent - check for outliers or edge cases")
    
    if mape < 3 and scaling_accuracy > 95:
        recommendations.append("Excellent performance - system is production-ready")
    elif mape < 5 and scaling_accuracy > 90:
        recommendations.append("Very good performance - minor optimizations possible but not required")
    
    return recommendations if recommendations else ["Model performing well - no immediate action needed"]

def handle_shutdown(signal_received, frame):
    """Handle shutdown gracefully"""
    print("\n[SHUTDOWN] Saving to S3 and cleaning up...")
    predictor.save_to_s3()
    for group in TARGET_GROUPS:
        try:
            lb.aws.deregister_instances(group)
            lb.aws.manage_instance('stop', group)
        except:
            pass
    exit(0)

if __name__ == '__main__':
    try:
        print("Starting JMeter-optimized ML Load Balancer...")
        signal.signal(signal.SIGINT, handle_shutdown)

        print("Ready for JMeter testing!")
        print("IMPORTANT: Configure JMeter to target '/' , NOT '/metrics'")
        print("\nDEBUGGING ENDPOINTS ADDED:")
        print("- GET /debug-prediction - See why predictions might be None")
        print("- GET /prediction-chart - Get prediction vs actual data for plotting")
        print("- Enhanced logging in console every 50 seconds")

        app.run(host=FLASK_HOST, port=FLASK_PORT)

    except Exception as e:
        print(f"Startup failed: {str(e)}")
        for group in TARGET_GROUPS:
            try:
                lb.aws.deregister_instances(group)
                lb.aws.manage_instance('stop', group)
            except:
                pass