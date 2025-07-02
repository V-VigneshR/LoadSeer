# performance_tracker.py
"""Model performance tracking and evaluation"""

import time
import numpy as np
from datetime import datetime
from collections import deque
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import PREDICTION_HISTORY_SIZE, ACCURACY_HISTORY_SIZE, SCALING_DECISIONS_SIZE


class ModelPerformanceTracker:
    """Tracks model performance and scaling decisions"""
    
    def __init__(self):
        self.predictions = deque(maxlen=PREDICTION_HISTORY_SIZE)
        self.accuracy_history = deque(maxlen=ACCURACY_HISTORY_SIZE)
        self.scaling_decisions = deque(maxlen=SCALING_DECISIONS_SIZE)
        self.correct_scales = 0
        self.total_scales = 0

    def add_prediction(self, predicted, actual, timestamp):
        """Track prediction vs actual"""
        if actual == 0:
            percent_error = 0 if predicted == 0 else 100
        else:
            percent_error = abs(predicted - actual) / abs(actual) * 100
        
        self.predictions.append({
            'predicted': predicted,
            'actual': actual,
            'timestamp': timestamp,
            'error': abs(predicted - actual),
            'percent_error': percent_error
        })

    def add_scaling_decision(self, decision, actual_rate, was_correct):
        """Track scaling decisions"""
        self.scaling_decisions.append({
            'decision': decision,
            'actual_rate': actual_rate,
            'was_correct': was_correct,
            'timestamp': datetime.now()
        })

        if was_correct:
            self.correct_scales += 1
        self.total_scales += 1

    def get_current_accuracy_metrics(self):
        """Calculate current model accuracy metrics with proper train/test separation"""
        if len(self.predictions) < 50:
            return None

        current_time = time.time()
        min_age_seconds = 300  # 5 minutes

        old_predictions = []
        for p in self.predictions:
            pred_time = p.get('timestamp')
            if pred_time:
                if isinstance(pred_time, datetime):
                    pred_timestamp = pred_time.timestamp()
                else:
                    pred_timestamp = pred_time

                if (current_time - pred_timestamp) >= min_age_seconds:
                    old_predictions.append(p)

        if len(old_predictions) < 30:
            return None

        eval_sample = old_predictions[-100:] if len(old_predictions) >= 100 else old_predictions

        predicted_values = [p['predicted'] for p in eval_sample]
        actual_values = [p['actual'] for p in eval_sample]

        # Calculate various accuracy metrics
        mae = mean_absolute_error(actual_values, predicted_values)
        mse = mean_squared_error(actual_values, predicted_values)
        rmse = np.sqrt(mse)

        try:
            r2 = r2_score(actual_values, predicted_values)
        except:
            r2 = 0

        mape = np.mean([p['percent_error'] for p in eval_sample])

        within_10_percent = sum(1 for p in eval_sample if p['percent_error'] <= 10) / len(eval_sample) * 100
        within_25_percent = sum(1 for p in eval_sample if p['percent_error'] <= 25) / len(eval_sample) * 100
        within_50_percent = sum(1 for p in eval_sample if p['percent_error'] <= 50) / len(eval_sample) * 100

        return {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'r2_score': round(r2, 3),
            'mape': round(mape, 2),
            'accuracy_within_10_percent': round(within_10_percent, 1),
            'accuracy_within_25_percent': round(within_25_percent, 1),
            'accuracy_within_50_percent': round(within_50_percent, 1),
            'sample_size': len(eval_sample),
            'scaling_accuracy': round(self.correct_scales / max(self.total_scales, 1) * 100, 1),
            'evaluation_method': f'Using predictions at least {min_age_seconds}s old to avoid data leakage'
        }

    def get_prediction_trend(self):
        """Get recent prediction trend"""
        if len(self.predictions) < 5:
            return "insufficient_data"

        recent_errors = [p['percent_error'] for p in list(self.predictions)[-10:]]
        avg_recent_error = np.mean(recent_errors)

        if avg_recent_error < 15:
            return "excellent"
        elif avg_recent_error < 30:
            return "good"
        elif avg_recent_error < 50:
            return "fair"
        else:
            return "poor"