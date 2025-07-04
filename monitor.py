import time
from datetime import datetime
from collections import deque
import threading
from config import *
from ml_predictor import *

class TrafficMonitor:
    def __init__(self, predictor):
        self.predictor = predictor
        self.traffic_history = deque(maxlen=500)
        self.request_count = 0
        self.start_time = datetime.now()
        self.switching_event = threading.Event()
        self.current_state = 'small'

    def get_arrival_rate(self):
        try:
            now = datetime.now()
            elapsed = (now - self.start_time).total_seconds()
            return self.request_count / elapsed if elapsed > 0 else 0
        except Exception as e:
            print(f"[ERROR] get_arrival_rate: {str(e)}")
            return 0

    def evaluate_scaling_decision(self, decision, current_rate, prev_rate):
        """Evaluate if scaling decision was correct"""
        try:
            # Simple logic: decision should match what thresholds suggest
            if current_rate > T2 + HYSTERESIS_BUFFER:
                return decision == 'large'
            elif current_rate > T1 + HYSTERESIS_BUFFER:
                return decision in ['medium', 'large']  # Either is acceptable
            elif current_rate < T1 - HYSTERESIS_BUFFER:
                return decision == 'small'
            else:
                return True  # In hysteresis zone, any decision is reasonable
        except:
            return True  # Default to correct if we can't evaluate

    def ml_background_monitor(self):
        """Enhanced monitoring with performance tracking"""
        last_prediction = None
        prediction_log_counter = 0
        
        

        while True:
            try:
                # Safe rate calculation
                try:
                    current_rate = self.get_arrival_rate()
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
                        'state': self.current_state
                    }
                    self.traffic_history.append(traffic_point)
                except Exception as e:
                    print(f"[MONITOR] Traffic point creation error: {e}")
                    time.sleep(5)
                    continue

                # Safe feature extraction
                features = None
                try:
                    features = self.predictor.extract_features(self.traffic_history)
                except Exception as e:
                    print(f"[MONITOR] Feature extraction error: {e}")

                # Add training data when we have valid features and rate
                if features is not None and current_rate >= 0:
                    try:
                        self.predictor.add_training_data(features, current_rate)
                        if len(self.predictor.training_data) % 25 == 0:
                            print(f"[MONITOR] Training data count: {len(self.predictor.training_data)}")
                    except Exception as e:
                        print(f"[MONITOR] Training data addition error: {e}")

                # Log prediction details every 10 iterations (50 seconds)
                if prediction_log_counter % 10 == 0:
                    print(f"\n[PREDICTION-LOG] Current rate: {current_rate:.2f}")
                    print(f"[PREDICTION-LOG] History length: {len(self.traffic_history)}")
                    print(f"[PREDICTION-LOG] Training samples: {len(self.predictor.training_data)}")
                    print(f"[PREDICTION-LOG] Features valid: {features is not None}")

                # Safe prediction
                current_prediction = None
                try:
                    current_prediction = self.predictor.predict_traffic(features)
                except Exception as e:
                    print(f"[MONITOR] Prediction error: {e}")

                if prediction_log_counter % 10 == 0:
                    pred_str = f"{current_prediction:.2f}" if current_prediction is not None else "None"
                    print(f"[PREDICTION-LOG] Predicted rate: {pred_str}")
                    print(f"[PREDICTION-LOG] Model trained: {self.predictor.is_trained}")
                    print(f"[PREDICTION-LOG] Last S3 sync: {self.predictor.last_s3_sync}")
                    print(f"[PREDICTION-LOG] S3 sync interval check: {datetime.now() - self.predictor.last_s3_sync}\n")

                # Track prediction accuracy
                if last_prediction is not None and current_rate > 0:
                    try:
                        self.predictor.performance_tracker.add_prediction(
                            last_prediction,
                            current_rate,
                            datetime.now()
                        )
                    except Exception as e:
                        print(f"[MONITOR] Performance tracking error: {e}")

                last_prediction = current_prediction
                prediction_log_counter += 1

                # Safe scaling decision tracking
                try:
                    if len(self.traffic_history) >= 2:
                        prev_rate = list(self.traffic_history)[-2]['rate']
                        current_decision = self.choose_load_balancer_ml()
                        was_correct = self.evaluate_scaling_decision(current_decision, current_rate, prev_rate)
                        self.predictor.performance_tracker.add_scaling_decision(
                            current_decision, current_rate, was_correct
                        )
                except Exception as e:
                    print(f"[MONITOR] Scaling decision tracking error: {e}")

                # Safe model training triggers
                try:
                    if len(self.predictor.training_data) >= 50 and not self.predictor.is_trained:
                        print("[ML] Initial training...")
                        self.predictor.train_model()
                    elif len(self.predictor.training_data) % 100 == 0 and len(self.predictor.training_data) > 0:
                        print("[ML] Retraining with new data...")
                        self.predictor.train_model()
                except Exception as e:
                    print(f"[MONITOR] Model training error: {e}")

                time.sleep(5)

            except Exception as e:
                print(f"[MONITOR] Main loop error: {str(e)}")
                import traceback
                traceback.print_exc()
                time.sleep(10)

    def choose_load_balancer_ml(self):
        """ML-based load balancer selection"""
        try:
            current_rate = self.get_arrival_rate()
            features = self.predictor.extract_features(self.traffic_history)
            predicted_rate = self.predictor.predict_traffic(features)
            perf_metrics = self.predictor.performance_tracker.get_current_accuracy_metrics() or {}
            
            
            # Simple blending based on model quality
            if predicted_rate is None or (perf_metrics and perf_metrics.get('r2_score', 0) < 0.3):
                effective_rate = current_rate
                print("[ML] Using current rate (poor model performance)")
            else:
                # Just blend - no multipliers
                blend_factor = 0.7  # Trust prediction 70%
                effective_rate = (predicted_rate * blend_factor) + (current_rate * (1 - blend_factor))
                
                # Sanity check
                if abs(predicted_rate - current_rate) > current_rate * 0.5:
                    effective_rate = current_rate
                    print(f"[ML] Prediction outlier - using current rate")

            if self.predictor.jmeter_patterns['ramp_up_detected']:
                buffer_multiplier = 1.1
                print("[ML] Ramp-up detected, aggresive thresholds applied")
            else:
                buffer_multiplier = 1.2

            if self.current_state == 'small' and effective_rate > T1 * buffer_multiplier:
                target = 'medium'
            elif self.current_state == 'medium' and effective_rate > T2 * buffer_multiplier:
                target = 'large'
            elif self.current_state == 'medium' and effective_rate < T1 / buffer_multiplier:
                target = 'small'
            elif self.current_state == 'large' and effective_rate < T2 / buffer_multiplier:
                target = 'medium'
            else:
                target = self.current_state

            pred_str = f"{predicted_rate:.1f}" if predicted_rate is not None else "None"
            print(f"[ML-DECISION] Current: {current_rate:.1f}, Predicted: {pred_str}")
            print(f"[ML-DECISION] Effective: {effective_rate:.1f}, Target: {target}")
            return target

        except Exception as e:
            print(f"[ML] Decision error: {str(e)}")
            return self.current_state