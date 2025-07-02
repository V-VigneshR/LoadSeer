# load_balancer.py
"""Core load balancer logic and state management"""

import threading
import time
from datetime import datetime
from collections import deque
from config import *

class LoadBalancer:
    """Manages load balancer state and scaling decisions"""
    
    def __init__(self, aws_manager, ml_predictor):
        self.aws = aws_manager
        self.predictor = ml_predictor
        self.current_state = 'small'
        self.switching_event = threading.Event()
        self.request_count = 0
        self.start_time = datetime.now()
        self.traffic_history = deque(maxlen=TRAFFIC_HISTORY_SIZE)

    def get_arrival_rate(self):
        """Calculate current request rate"""
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
            if current_rate > T2 + HYSTERESIS_BUFFER:
                return decision == 'large'
            elif current_rate > T1 + HYSTERESIS_BUFFER:
                return decision in ['medium', 'large']
            elif current_rate < T1 - HYSTERESIS_BUFFER:
                return decision == 'small'
            else:
                return True
        except:
            return True

    def choose_load_balancer_ml(self):
        """Make scaling decision using ML predictions"""
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

            # Simple thresholds with hysteresis
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

    def warmup_and_switch(self, target_group_to_use):
        """Handle instance warmup and switching"""
        if self.switching_event.is_set():
            return

        self.switching_event.set()
        print(f"[JMeter-ML] Switching to: {target_group_to_use}")

        try:
            self.aws.manage_instance('start', target_group_to_use)
            self.aws.register_instances(target_group_to_use)

            start_time = time.time()
            while time.time() - start_time < 60:
                if self.aws.check_target_health(target_group_to_use):
                    self.aws.deregister_instances(self.current_state)
                    self.aws.manage_instance('stop', self.current_state)
                    self.current_state = target_group_to_use
                    print(f"[SUCCESS] Switched to {target_group_to_use}")
                    return
                time.sleep(5)

            print(f"Health check timeout for {target_group_to_use}")

        except Exception as e:
            print(f"Transition failed: {str(e)}")
            if target_group_to_use != 'small':
                self.warmup_and_switch('small')
        finally:
            self.switching_event.clear()