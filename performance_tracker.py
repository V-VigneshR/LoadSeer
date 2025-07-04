from datetime import datetime
import time
import numpy as np
from collections import deque
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelPerformanceTracker:
    def __init__(self):
        self.predictions = deque(maxlen=1000)  # Store recent predictions vs actual
        self.accuracy_history = deque(maxlen=100)  # Store accuracy over time
        self.scaling_decisions = deque(maxlen=200)  # Track scaling decisions
        self.correct_scales = 0
        self.total_scales = 0

    def add_prediction(self, predicted, actual, timestamp):
        """Track prediction vs actual"""
        if actual == 0:
            # Handle zero actual values
            percent_error = 0 if predicted == 0 else 100
        else:
            percent_error = abs(predicted - actual) / abs(actual) * 100
        
        # Track rate of change
        if len(self.predictions) > 0:
            last_rate = self.predictions[-1]['actual']
            rate_change = (actual - last_rate) / last_rate if last_rate != 0 else 0
        else:
            rate_change = 0
            

        self.predictions.append({
            'predicted': predicted,
            'actual': actual,
            'timestamp': timestamp,
            'error': abs(predicted - actual),
            'percent_error': percent_error,
            'rate_change': rate_change
        })
        
    def detect_ramp_pattern(self):
        """Enhanced ramp pattern detection"""
        if len(self.predictions) < 5:
            return False
            
        recent = list(self.predictions)[-5:]
        rate_changes = [p['rate_change'] for p in recent]
        
        # Detect rapid increase
        if all(rc > 0.1 for rc in rate_changes[-3:]):  # 10%+ increase for last 3 samples
            return 'ramp_up'
        # Detect rapid decrease
        elif all(rc < -0.1 for rc in rate_changes[-3:]):
            return 'ramp_down'
        return None


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

        current_time = time.time()  # Float timestamp
        min_age_seconds = 300  # 5 minutes instead of 120 seconds
        
        old_predictions = []
        for p in self.predictions:
            pred_time = p.get('timestamp')
            if pred_time:
                # Convert datetime to timestamp if needed
                if isinstance(pred_time, datetime):
                    pred_timestamp = pred_time.timestamp()
                else:
                    pred_timestamp = pred_time  # Already a float
                
                # Now both are floats, safe to subtract
                if (current_time - pred_timestamp) >= min_age_seconds:
                    old_predictions.append(p)
        
        if len(old_predictions) < 30:
            return None
        
        # Take up to 100 old predictions for evaluation
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

def interpret_model_quality(metrics):
    """Provide human-readable interpretation of model quality - FIXED FOR LOAD BALANCING"""
    r2 = metrics.get('r2_score', 0)
    mape = metrics.get('mape', 100)
    accuracy_10 = metrics.get('accuracy_within_10_percent', 0)
    accuracy_25 = metrics.get('accuracy_within_25_percent', 0)
    scaling_accuracy = metrics.get('scaling_accuracy', 0)
    sample_size = metrics.get('sample_size', 0)

    # For load balancing, MAPE and scaling accuracy matter most
    # R² is less important for threshold-based decisions
    
    # Check for insufficient data first
    if sample_size < 30:
        return "Insufficient Data - Need more predictions for reliable assessment"
    
    # Primary assessment based on MAPE (most important for load balancing)
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
    
    # Sample size recommendations
    if sample_size < 50:
        recommendations.append("Need more training data - run longer tests for robust evaluation")
    
    # MAPE-based recommendations (most important)
    if mape > 10:
        recommendations.append("High prediction error - consider feature engineering or different algorithm")
    elif mape > 5:
        recommendations.append("Moderate prediction error - model could benefit from more diverse training data")
    
    # Scaling accuracy recommendations
    if scaling_accuracy < 80:
        recommendations.append("Poor scaling decisions - review threshold values or prediction logic")
    elif scaling_accuracy < 90:
        recommendations.append("Good scaling accuracy - minor threshold adjustments may help")
    
    # R² recommendations (less critical for load balancing)
    if r2 < 0:
        recommendations.append("Negative R² suggests overfitting or model complexity issues")
    elif r2 < 0.3 and mape < 5:
        recommendations.append("Low R² with good MAPE is normal for steady traffic - consider longer evaluation period")
    elif r2 < 0.3 and mape > 10:
        recommendations.append("Low R² and high MAPE - model may need different approach or more features")
    
    # Accuracy distribution analysis
    if accuracy_10 < 70 and mape < 5:
        recommendations.append("Good average error but inconsistent - check for outliers or edge cases")
    
    # Success case
    if mape < 3 and scaling_accuracy > 95:
        recommendations.append("Excellent performance - system is production-ready")
    elif mape < 5 and scaling_accuracy > 90:
        recommendations.append("Very good performance - minor optimizations possible but not required")
    
    return recommendations if recommendations else ["Model performing well - no immediate action needed"]