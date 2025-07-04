from flask import Flask, request, jsonify, Response
import requests
import threading
import signal
import os
from datetime import datetime
from config import *
from aws_manager import *
from ml_predictor import *
from monitor import *
from load_balancer import *
from performance_tracker import *

app = Flask(__name__, static_folder=None)

# Initialize components
predictor = PersistentTrafficPredictor()
monitor = TrafficMonitor(predictor)
load_balancer = LoadBalancer(monitor)

# Initialize AWS resources
initialize_aws_resources()

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST'])
@app.route('/<path:path>', methods=['GET', 'POST'])
def reverse_proxy(path):
    # Don't count metrics requests as load traffic
    if path != 'metrics':
        monitor.request_count += 1
        print(f"[REQUEST] Count: {monitor.request_count}, Path: /{path}, Rate: {monitor.get_arrival_rate():.2f} req/sec")

    try:
        headers = dict(request.headers)
        headers['instance-type'] = monitor.current_state
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
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(value, (np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.int32, np.int64)):
            return int(value)
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    try:
        rate = monitor.get_arrival_rate()
        features = predictor.extract_features(monitor.traffic_history)
        predicted_rate = predictor.predict_traffic(features) if features else None

        # Get model performance metrics
        performance_metrics = predictor.performance_tracker.get_current_accuracy_metrics()
        prediction_trend = predictor.performance_tracker.get_prediction_trend()

        # Build response with type conversion
        response_data = {
            "arrival_rate": safe_convert(round(rate, 2)),
            "predicted_rate": safe_convert(round(predicted_rate, 2)) if predicted_rate else None,
            "active_group": monitor.current_state,
            "ml_trained": predictor.is_trained,
            "training_samples": len(predictor.training_data),
            "last_retrain": safe_convert(predictor.last_retrain),
            "jmeter_patterns": {k: safe_convert(v) for k,v in predictor.jmeter_patterns.items()},
            "request_count": monitor.request_count,
            "uptime_minutes": safe_convert(round((datetime.now() - monitor.start_time).total_seconds() / 60, 2)),
            "debug_info": {
                "traffic_history_length": len(monitor.traffic_history),
                "current_rate_calc": f"{monitor.request_count} requests / {(datetime.now() - monitor.start_time).total_seconds():.1f} seconds"
            }
        }

        # Add performance metrics with type conversion
        if performance_metrics:
            response_data["model_performance"] = {k: safe_convert(v) for k,v in performance_metrics.items()}
            response_data["model_quality"] = interpret_model_quality(performance_metrics)
            response_data["recommendations"] = get_model_recommendations(performance_metrics)

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route('/debug-prediction')
def debug_prediction():
    """Debug endpoint to see prediction pipeline"""
    try:
        current_rate = monitor.get_arrival_rate()
        features = predictor.extract_features(monitor.traffic_history)

        debug_info = {
            "current_rate": current_rate,
            "traffic_history_length": len(monitor.traffic_history),
            "required_lags": predictor.sequence_lags,
            "model_trained": predictor.is_trained,
            "training_samples": len(predictor.training_data),
            "features": features,
            "features_length": len(features) if features else 0
        }

        if features:
            # Test prediction
            predicted_rate = predictor.predict_traffic(features)
            debug_info["predicted_rate"] = predicted_rate

            # Show feature breakdown
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
        
        # Retrain with current data
        predictor.train_model()
        
        # Reset patterns
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
        # Get recent traffic history
        recent_history = list(monitor.traffic_history)[-50:]  # Last 50 points

        chart_data = []
        for i, point in enumerate(recent_history):
            if i >= predictor.sequence_lags:  # Can make predictions
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
    monitor.request_count = 0
    monitor.start_time = datetime.now()
    monitor.traffic_history.clear()

    predictor.jmeter_patterns = {
        'ramp_up_detected': False,
        'peak_detected': False,
        'ramp_down_detected': False,
        'pattern_start_time': None
    }
    return jsonify({"status": "JMeter patterns and counters reset"})

@app.route('/force-s3-sync')
def force_s3_sync():
    """Force immediate S3 sync"""
    try:
        print(f"[FORCE-SYNC] Current training samples: {len(predictor.training_data)}")
        predictor.save_to_s3()
        return jsonify({
            "status": "S3 sync completed",
            "training_samples": len(predictor.training_data),
            "last_sync": predictor.last_s3_sync.isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def handle_shutdown(signal_received, frame):
    print("\n[SHUTDOWN] Saving to S3 and cleaning up...")
    predictor.save_to_s3()  # Final save
    for group in TARGET_GROUPS:
        try:
            deregister_instances(group)
            manage_instance('stop', group)
        except:
            pass
    exit(0)

if __name__ == '__main__':
    try:
        print("Starting JMeter-optimized ML Load Balancer...")

        # Start background threads
        threading.Thread(target=load_balancer.background_scaler, daemon=True).start()
        threading.Thread(target=monitor.ml_background_monitor, daemon=True).start()

        signal.signal(signal.SIGINT, handle_shutdown)

        print("Ready for JMeter testing!")
        print("IMPORTANT: Configure JMeter to target '/' or '/test', NOT '/metrics'")
        print("\nDEBUGGING ENDPOINTS ADDED:")
        print("- GET /debug-prediction - See why predictions might be None")
        print("- GET /prediction-chart - Get prediction vs actual data for plotting")
        print("- Enhanced logging in console every 50 seconds")

        app.run(host='0.0.0.0', port=8080)

    except Exception as e:
        print(f"Startup failed: {str(e)}")
        for group in TARGET_GROUPS:
            try:
                deregister_instances(group)
                manage_instance('stop', group)
            except:
                pass