# monitor.py
"""Background monitoring and scaling threads"""

import time
import threading
from datetime import datetime
from config import *

class BackgroundMonitor:
    """Handles background monitoring and scaling tasks"""
    
    def __init__(self, load_balancer):
        self.lb = load_balancer
        self.prediction_log_counter = 0

    def ml_background_monitor(self):
        """Enhanced monitoring with performance tracking"""
        last_prediction = None

        while True:
            try:
                # Safe rate calculation
                try:
                    current_rate = self.lb.get_arrival_rate()
                    if not isinstance(current_rate, (int, float)) or current_rate < 0:
                        print(f"[MONITOR] Invalid rate: {current_rate}, using 0")
                        current_rate = 0.0
                except Exception as e:
                    print(f"[MONITOR] Rate calculation error: {e}")
                    current_rate = 0.0

                # Safe traffic point creation
                try:
                    traffic_point = {
                        'rate': float(current_rate),
                        'timestamp': datetime.now(),
                        'state': self.lb.current_state
                    }
                    self.lb.traffic_history.append(traffic_point)
                except Exception as e:
                    print(f"[MONITOR] Traffic point creation error: {e}")
                    time.sleep(5)
                    continue

                # Safe feature extraction
                features = None
                try:
                    features = self.lb.predictor.extract_features(self.lb.traffic_history)
                except Exception as e:
                    print(f"[MONITOR] Feature extraction error: {e}")

                # Log prediction details every 10 iterations (50 seconds)
                if self.prediction_log_counter % 10 == 0:
                    print(f"\n[PREDICTION-LOG] Current rate: {current_rate:.2f}")
                    print(f"[PREDICTION-LOG] History length: {len(self.lb.traffic_history)}")
                    print(f"[PREDICTION-LOG] Features valid: {features is not None}")

                # Safe prediction
                current_prediction = None
                try:
                    current_prediction = self.lb.predictor.predict_traffic(features)
                except Exception as e:
                    print(f"[MONITOR] Prediction error: {e}")

                if self.prediction_log_counter % 10 == 0:
                    pred_str = f"{current_prediction:.2f}" if current_prediction is not None else "None"
                    print(f"[PREDICTION-LOG] Predicted rate: {pred_str}")
                    print(f"[PREDICTION-LOG] Model trained: {self.lb.predictor.is_trained}")
                    print(f"[PREDICTION-LOG] Training samples: {len(self.lb.predictor.training_data)}\n")

                # Track prediction accuracy
                if last_prediction is not None and current_rate > 0:
                    try:
                        self.lb.predictor.performance_tracker.add_prediction(
                            last_prediction,
                            current_rate,
                            datetime.now()
                        )
                    except Exception as e:
                        print(f"[MONITOR] Performance tracking error: {e}")

                last_prediction = current_prediction
                self.prediction_log_counter += 1

                # Safe training data addition
                if features is not None:
                    try:
                        self.lb.predictor.add_training_data(features, current_rate)
                    except Exception as e:
                        print(f"[MONITOR] Training data addition error: {e}")

                # Safe scaling decision tracking
                try:
                    if len(self.lb.traffic_history) >= 2:
                        prev_rate = list(self.lb.traffic_history)[-2]['rate']
                        current_decision = self.lb.choose_load_balancer_ml()
                        was_correct = self.lb.evaluate_scaling_decision(current_decision, current_rate, prev_rate)
                        self.lb.predictor.performance_tracker.add_scaling_decision(
                            current_decision, current_rate, was_correct
                        )
                except Exception as e:
                    print(f"[MONITOR] Scaling decision tracking error: {e}")

                # Safe model training triggers
                try:
                    if len(self.lb.predictor.training_data) >= 50 and not self.lb.predictor.is_trained:
                        print("[ML] Initial training...")
                        self.lb.predictor.train_model()
                    elif len(self.lb.predictor.training_data) % 100 == 0 and len(self.lb.predictor.training_data) > 0:
                        print("[ML] Retraining with new data...")
                        self.lb.predictor.train_model()
                except Exception as e:
                    print(f"[MONITOR] Model training error: {e}")

                time.sleep(MONITOR_SLEEP_SECONDS)

            except Exception as e:
                print(f"[MONITOR] Main loop error: {str(e)}")
                time.sleep(ERROR_SLEEP_SECONDS)

    def background_scaler(self):
        """Handle background scaling decisions"""
        while True:
            try:
                target_state = self.lb.choose_load_balancer_ml()
                if target_state and target_state != self.lb.current_state and not self.lb.switching_event.is_set():
                    threading.Thread(target=self.lb.warmup_and_switch, args=(target_state,)).start()
                time.sleep(SCALER_SLEEP_SECONDS)
            except Exception as e:
                print(f"Background scaler error: {str(e)}")
                time.sleep(ERROR_SLEEP_SECONDS)
