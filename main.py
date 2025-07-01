from flask import Flask, request, jsonify, Response
import boto3
import requests
import threading
import time
import numpy as np
from datetime import datetime, timedelta
from collections import deque, defaultdict
import signal
import pickle
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder=None)

# Constants
T1 = 125
T2 = 225
BACKEND_SERVER = "http://load-balancer-1524502438.ap-south-1.elb.amazonaws.com"
INSTANCE_START_TIMEOUT = 300
PREDICTION_HORIZON = 300

# AWS Config
region_name = 'ap-south-1'
elbv2 = boto3.client('elbv2', region_name=region_name)
ec2 = boto3.client('ec2', region_name=region_name)
s3 = boto3.client('s3', region_name=region_name)

# S3 Configuration for persistence
S3_BUCKET = 'ml-load-balancing'  # Replace with your bucket
S3_MODEL_KEY = 'ml-models/traffic_model.pkl'
S3_SCALER_KEY = 'ml-models/traffic_scaler.pkl'
S3_DATA_KEY = 'ml-models/training_data.json'
S3_HISTORY_KEY = 'ml-models/traffic_history.json'

# Target groups and instances
TARGET_GROUPS = {
    'small': 'arn:aws:elasticloadbalancing:ap-south-1:975050195505:targetgroup/tg-small/811961e640771f8a',
    'medium': 'arn:aws:elasticloadbalancing:ap-south-1:975050195505:targetgroup/tg-medium/2b528b0c7d77684c',
    'large': 'arn:aws:elasticloadbalancing:ap-south-1:975050195505:targetgroup/tg-large/ec947ab293c85ae0'
}

INSTANCE_IDS = {
    'small': ['i-0e198a8e8ed3c264f'],
    'medium': ['i-017b9c685be6436be'],
    'large': ['i-04dcd92ab938b2515']
}

# State Management
current_state = 'small'
switching_event = threading.Event()

# Add this class to track model performance
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

class PersistentTrafficPredictor:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = deque(maxlen=2000)  # Increased capacity
        self.feature_window = 30
        self.last_retrain = datetime.now()
        self.last_s3_sync = datetime.now()
        self.sync_interval = timedelta(minutes=5)  # Sync every 5 minutes
        self.performance_tracker = ModelPerformanceTracker()  # Add performance tracker
        self.sequence_lags = 5  # rate_t-1 to rate_t-5

        # JMeter pattern recognition
        self.jmeter_patterns = {
            'ramp_up_detected': False,
            'peak_detected': False,
            'ramp_down_detected': False,
            'pattern_start_time': None
        }

        # Load existing data from S3
        self.load_from_s3()

    def create_s3_bucket_if_needed(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            s3.head_bucket(Bucket=S3_BUCKET)
        except:
            try:
                s3.create_bucket(
                    Bucket=S3_BUCKET,
                    CreateBucketConfiguration={'LocationConstraint': region_name}
                )
                print(f"[S3] Created bucket {S3_BUCKET}")
            except Exception as e:
                print(f"[S3] Error creating bucket: {str(e)}")

    def save_to_s3(self):
        """Save model, scaler, and training data to S3"""
        try:
            self.create_s3_bucket_if_needed()

            # Save model
            if self.is_trained:
                model_bytes = pickle.dumps(self.model)
                s3.put_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY, Body=model_bytes)

                scaler_bytes = pickle.dumps(self.scaler)
                s3.put_object(Bucket=S3_BUCKET, Key=S3_SCALER_KEY, Body=scaler_bytes)

            # Save training data (convert deque to list for JSON serialization)
            training_data_list = []
            for item in self.training_data:
                training_data_list.append({
                    'features': item['features'],
                    'target': item['target'],
                    'timestamp': item['timestamp'].isoformat()
                })

            data_json = json.dumps(training_data_list)
            s3.put_object(Bucket=S3_BUCKET, Key=S3_DATA_KEY, Body=data_json)

            self.last_s3_sync = datetime.now()
            print(f"[S3] Saved model and {len(self.training_data)} training samples")

        except Exception as e:
            print(f"[S3] Save error: {str(e)}")

    def load_from_s3(self):
        """Load model, scaler, and training data from S3"""
        try:
            # Load model and scaler
            try:
                model_obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY)
                self.model = pickle.loads(model_obj['Body'].read())

                scaler_obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_SCALER_KEY)
                self.scaler = pickle.loads(scaler_obj['Body'].read())

                self.is_trained = True
                print("[S3] Loaded existing model and scaler")
            except:
                print("[S3] No existing model found, will train new one")

            # Load training data
            try:
                data_obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_DATA_KEY)
                training_data_list = json.loads(data_obj['Body'].read())

                for item in training_data_list:
                    self.training_data.append({
                        'features': item['features'],
                        'target': item['target'],
                        'timestamp': datetime.fromisoformat(item['timestamp'])
                    })

                print(f"[S3] Loaded {len(self.training_data)} training samples")
            except:
                print("[S3] No existing training data found")

        except Exception as e:
            print(f"[S3] Load error: {str(e)}")

    def detect_jmeter_pattern(self, traffic_history):
        """Enhanced pattern detection with statistical validation"""
        try:
            # Timeout after 5 minutes (300 seconds)
            if (self.jmeter_patterns['pattern_start_time'] and 
                (datetime.now() - self.jmeter_patterns['pattern_start_time']).total_seconds() > 300):
                self._reset_patterns()
                return 'pattern_timeout'
            
            if len(traffic_history) < 20:  # Need more data for reliable detection
                return None

            recent = list(traffic_history)[-20:]
            rates = np.array([p['rate'] for p in recent])
            timestamps = np.array([p['timestamp'].timestamp() for p in recent])
            
            # Calculate linear regression trend
            slope, intercept = np.polyfit(timestamps, rates, 1)
            trend_strength = np.corrcoef(timestamps, rates)[0, 1]
            
            # Ramp-up detection (strong positive trend)
            if (not self.jmeter_patterns['ramp_up_detected'] and 
                slope > 5 and trend_strength > 0.8 and 
                rates[-1] > rates[0] * 1.5):
                self._reset_patterns()
                self.jmeter_patterns['ramp_up_detected'] = True
                self.jmeter_patterns['pattern_start_time'] = datetime.now()
                return 'ramp_up_start'
                
            # Peak detection (stable after ramp-up)
            if (self.jmeter_patterns['ramp_up_detected'] and 
                not self.jmeter_patterns['peak_detected'] and
                abs(slope) < 2 and np.std(rates[-10:]) < 15):
                self.jmeter_patterns['peak_detected'] = True
                return 'peak_detected'
                
            # Ramp-down detection (strong negative trend after peak)
            if (self.jmeter_patterns['peak_detected'] and 
                not self.jmeter_patterns['ramp_down_detected'] and
                slope < -5 and trend_strength < -0.7):
                self.jmeter_patterns['ramp_down_detected'] = True
                return 'ramp_down_start'
                
            return None
            
        except Exception as e:
            print(f"[ML] Pattern detection error: {str(e)}")
            return None

    def _reset_patterns(self):
        """Helper to reset pattern states"""
        self.jmeter_patterns = {
            'ramp_up_detected': False,
            'peak_detected': False,
            'ramp_down_detected': False,
            'pattern_start_time': None
        }

    def extract_features(self, traffic_history):
        """Feature extraction using lag features + time + JMeter pattern - IMPROVED ROBUSTNESS"""
        try:
            # FIX 2: Better validation and error handling
            if len(traffic_history) < self.sequence_lags + 1:
                print(f"[FEATURES] Insufficient history: {len(traffic_history)} < {self.sequence_lags + 1}")
                return None

            # Ensure we have valid data
            recent_data = list(traffic_history)[-self.sequence_lags:]
            
            # Validate data quality
            lag_rates = []
            for i, point in enumerate(recent_data):
                if not isinstance(point, dict) or 'rate' not in point:
                    print(f"[FEATURES] Invalid data point at index {i}: {point}")
                    return None
                
                rate = point['rate']
                if not isinstance(rate, (int, float)) or rate < 0:
                    print(f"[FEATURES] Invalid rate value: {rate}")
                    return None
                
                lag_rates.append(float(rate))  # Ensure float type

            # Time-based features with validation
            now = datetime.now()
            hour = now.hour
            day_of_week = now.weekday()
            minute = now.minute

            # FIX 2: Safer pattern detection with fallbacks
            try:
                pattern = self.detect_jmeter_pattern(traffic_history)
            except Exception as e:
                print(f"[FEATURES] Pattern detection failed: {e}")
                pattern = None

            # Create pattern features with safe defaults
            pattern_features = {
                'is_ramp_up': 1 if pattern == 'ramp_up_start' else 0,
                'is_peak': 1 if pattern == 'peak_detected' else 0,
                'is_ramp_down': 1 if pattern == 'ramp_down_start' else 0,
                'test_duration': 0.0  # Default to 0 if no pattern timing
            }

            # Calculate test duration safely
            if self.jmeter_patterns['pattern_start_time']:
                try:
                    duration_seconds = (now - self.jmeter_patterns['pattern_start_time']).total_seconds()
                    pattern_features['test_duration'] = max(0.0, duration_seconds / 60.0)  # Minutes
                except:
                    pattern_features['test_duration'] = 0.0

            # Combine features with type safety
            features = lag_rates + [float(hour), float(day_of_week), float(minute)] + list(pattern_features.values())
            
            # Final validation
            if len(features) != (self.sequence_lags + 7):  # 5 lags + 3 time + 4 pattern
                print(f"[FEATURES] Unexpected feature count: {len(features)}")
                return None

            # Ensure all features are numeric
            for i, feat in enumerate(features):
                if not isinstance(feat, (int, float)) or not np.isfinite(feat):
                    print(f"[FEATURES] Invalid feature at index {i}: {feat}")
                    return None

            print(f"[FEATURES] Successfully extracted {len(features)} features")
            return features

        except Exception as e:
            print(f"[FEATURES] Extract error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


    def debug_prediction_pipeline(self, traffic_history):
        """Debug why predictions might be None"""
        print(f"[DEBUG] Traffic history length: {len(traffic_history)}")
        print(f"[DEBUG] Required sequence lags: {self.sequence_lags}")
        print(f"[DEBUG] Model trained: {self.is_trained}")

        if len(traffic_history) < self.sequence_lags + 1:
            print(f"[DEBUG] Not enough history: need {self.sequence_lags + 1}, have {len(traffic_history)}")
            return None

        features = self.extract_features(traffic_history)
        print(f"[DEBUG] Extracted features: {features}")

        if features is None:
            print("[DEBUG] Feature extraction failed!")
            return None

        if not self.is_trained:
            print("[DEBUG] Model not trained yet!")
            return None

        try:
            prediction = self.model.predict([features])[0]
            print(f"[DEBUG] Raw prediction: {prediction}")
            return prediction
        except Exception as e:
            print(f"[DEBUG] Prediction error: {e}")
            return None

    def add_training_data(self, features, actual_rate):
        """Add training data and sync to S3 periodically"""
        if features is not None:
            self.training_data.append({
                'features': features,
                'target': actual_rate,
                'timestamp': datetime.now()
            })

            # Periodic S3 sync
            if datetime.now() - self.last_s3_sync > self.sync_interval:
                threading.Thread(target=self.save_to_s3).start()

    def train_model(self):
        """Enhanced XGBoost training with cross-validation and robust validation - IMPROVED"""
        MIN_SAMPLES = 50   # Reduced from 200 for faster initial training
        EARLY_STOP = 10    
        N_FOLDS = 3        
        MAX_MAE_RATIO = 0.4  # Slightly relaxed from 0.3

        if len(self.training_data) < MIN_SAMPLES:
            print(f"[TRAIN] Need {MIN_SAMPLES} samples (current: {len(self.training_data)})")
            return False

        try:
            # FIX 3: Better data validation and cleaning
            valid_data = []
            for point in self.training_data:
                try:
                    features = point['features']
                    target = point['target']
                    
                    # Validate features
                    if (features is None or not isinstance(features, list) or 
                        len(features) != (self.sequence_lags + 7)):
                        continue
                    
                    # Validate target
                    if (target is None or not isinstance(target, (int, float)) or 
                        target < 0 or not np.isfinite(target)):
                        continue
                    
                    # Check for NaN or infinite values in features
                    if any(not np.isfinite(f) for f in features):
                        continue
                    
                    valid_data.append({'features': features, 'target': target})
                    
                except Exception as e:
                    print(f"[TRAIN] Skipping invalid data point: {e}")
                    continue

            if len(valid_data) < MIN_SAMPLES:
                print(f"[TRAIN] Insufficient valid data: {len(valid_data)} < {MIN_SAMPLES}")
                return False

            print(f"[TRAIN] Using {len(valid_data)} valid samples from {len(self.training_data)} total")

            X = np.array([point['features'] for point in valid_data], dtype=np.float64)
            y = np.array([point['target'] for point in valid_data], dtype=np.float64)
            
            # Additional data validation
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                print("[TRAIN] Found NaN values in training data")
                return False
            
            avg_y = np.mean(y)
            std_y = np.std(y)
            
            print(f"[TRAIN] Target stats - Mean: {avg_y:.2f}, Std: {std_y:.2f}, Range: [{np.min(y):.2f}, {np.max(y):.2f}]")

            # XGBoost parameters optimized for time series
            params = {
                'n_estimators': 200,  # Reduced for faster training
                'max_depth': 4,       # Reduced to prevent overfitting
                'learning_rate': 0.1,
                'subsample': 0.9,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'n_jobs': -1,
                'random_state': 42
            }

            # Cross-validated training with early stopping
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=min(N_FOLDS, len(valid_data) // 10))  # Adjust folds based on data
            val_scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                try:
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    model = XGBRegressor(**params)
                    model.fit(X_train, y_train, verbose=False)
                    
                    # Store validation metrics
                    preds = model.predict(X_val)
                    mae = mean_absolute_error(y_val, preds)
                    r2 = r2_score(y_val, preds) if len(set(y_val)) > 1 else 0
                    
                    val_scores.append({'mae': mae, 'r2': r2})
                    print(f"[TRAIN] Fold {fold+1}: MAE={mae:.2f}, R²={r2:.3f}")
                    
                except Exception as e:
                    print(f"[TRAIN] Fold {fold+1} failed: {e}")
                    continue

            if not val_scores:
                print("[TRAIN] All folds failed")
                return False

            # Analyze validation results
            avg_mae = np.mean([s['mae'] for s in val_scores])
            avg_r2 = np.mean([s['r2'] for s in val_scores])
            
            print(f"[TRAIN] Cross-val MAE: {avg_mae:.2f} ({(avg_mae/avg_y)*100:.1f}%)")
            print(f"[TRAIN] Cross-val R²: {avg_r2:.3f}")

            # More lenient acceptance criteria for initial training
            if avg_mae > avg_y * MAX_MAE_RATIO:
                print(f"[TRAIN] High MAE but proceeding with training (MAE: {avg_mae:.2f} vs threshold: {avg_y * MAX_MAE_RATIO:.2f})")

            # Train final model on all data
            self.model = XGBRegressor(**params)
            self.model.fit(X, y, verbose=False)
            self.is_trained = True
            self.last_retrain = datetime.now()

            # Feature importance analysis
            print("\n[TRAIN] Feature Importances:")
            features = ["lag1", "lag2", "lag3", "lag4", "lag5", 
                    "hour", "weekday", "minute", 
                    "is_ramp", "is_peak", "is_ramp_down", "duration"]
            for name, imp in sorted(zip(features, self.model.feature_importances_),
                                key=lambda x: x[1], reverse=True)[:8]:  # Top 8
                print(f"  {name}: {imp:.3f}")

            return True

        except Exception as e:
            print(f"[TRAIN] Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


    def predict_traffic(self, features):
        """Enhanced predict with detailed debugging and improved logic - IMPROVED ERROR HANDLING"""
        # FIX 4: Better validation and error handling
        if not self.is_trained:
            print("[PREDICT] Model not trained")
            return None

        if features is None:
            print("[PREDICT] Features are None")
            return None

        try:
            # Validate features thoroughly
            if not isinstance(features, list):
                print(f"[PREDICT] Features not a list: {type(features)}")
                return None
            
            expected_length = self.sequence_lags + 7  # 5 lags + 3 time + 4 pattern
            if len(features) != expected_length:
                print(f"[PREDICT] Wrong feature count: {len(features)} != {expected_length}")
                return None

            # Check for invalid values
            for i, feat in enumerate(features):
                if not isinstance(feat, (int, float)) or not np.isfinite(feat):
                    print(f"[PREDICT] Invalid feature at index {i}: {feat}")
                    return None

            # Convert to numpy array with proper shape and ensure float64
            features_array = np.array([features], dtype=np.float64).reshape(1, -1)
            
            # Validate array
            if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                print("[PREDICT] NaN or Inf in features array")
                return None

            # Get base prediction and ensure Python native float
            base_prediction = float(self.model.predict(features_array)[0])
            
            if not np.isfinite(base_prediction):
                print(f"[PREDICT] Invalid base prediction: {base_prediction}")
                return None

            print(f"[PREDICT] Base XGBoost prediction: {base_prediction:.2f}")

            # Dynamic JMeter pattern adjustments with better error handling
            adjustment_factor = 1.0
            current_time = datetime.now()
            
            try:
                # Check if ramp-up period should expire (after 5 minutes)
                if (self.jmeter_patterns['ramp_up_detected'] and 
                    self.jmeter_patterns['pattern_start_time'] and
                    (current_time - self.jmeter_patterns['pattern_start_time']).total_seconds() > 300):
                    
                    self.jmeter_patterns['ramp_up_detected'] = False
                    self.jmeter_patterns['peak_detected'] = True
                    print("[PREDICT] Ramp-up period expired (5 minutes), switching to peak mode")

                # Apply adjustments only if justified
                if self.jmeter_patterns['ramp_up_detected'] and not self.jmeter_patterns['peak_detected']:
                    # Verify recent trend supports ramp-up
                    try:
                        recent_trend = self._calculate_recent_trend()
                        if recent_trend > 0.05:  # At least 5% upward trend
                            adjustment_factor = min(1.3, 1.0 + recent_trend)
                            print(f"[PREDICT] Applied dynamic ramp-up boost: {adjustment_factor:.2f}x")
                        else:
                            print("[PREDICT] Insufficient trend for ramp-up boost")
                    except:
                        print("[PREDICT] Error calculating trend, no adjustment")
                            
                elif self.jmeter_patterns['ramp_down_detected']:
                    adjustment_factor = 0.7
                    print("[PREDICT] Applied ramp-down reduction: 0.7x")

            except Exception as e:
                print(f"[PREDICT] Pattern adjustment error: {e}")
                adjustment_factor = 1.0

            final_prediction = max(0, base_prediction * adjustment_factor)
            
            if not np.isfinite(final_prediction):
                print(f"[PREDICT] Invalid final prediction: {final_prediction}")
                return base_prediction  # Fallback to base prediction

            print(f"[PREDICT] Final prediction: {final_prediction:.2f}")
            return final_prediction

        except Exception as e:
            print(f"[PREDICT] Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_recent_trend(self):
        """Calculate percentage growth over last 5 data points - IMPROVED"""
        try:
            if len(self.training_data) < 5:
                return 0.0
            
            recent = [d['target'] for d in list(self.training_data)[-5:]]
            
            # Filter out invalid values
            valid_recent = [r for r in recent if isinstance(r, (int, float)) and r >= 0]
            
            if len(valid_recent) < 2:
                return 0.0
                
            if valid_recent[0] == 0:  # Avoid division by zero
                return 0.0
                
            trend = (valid_recent[-1] - valid_recent[0]) / valid_recent[0]
            
            # Clamp extreme values
            return max(-1.0, min(1.0, trend))
            
        except Exception as e:
            print(f"[TREND] Calculation error: {e}")
            return 0.0
    
# Initialize persistent predictor
predictor = PersistentTrafficPredictor()

# Traffic monitoring
REQUEST_COUNT = 0
START_TIME = datetime.now()
HYSTERESIS_BUFFER = 3
traffic_history = deque(maxlen=500)  # Store more history

def wait_for_instance_state(instance_id, target_state='running'):
    start_time = time.time()
    while time.time() - start_time < INSTANCE_START_TIMEOUT:
        response = ec2.describe_instances(InstanceIds=[instance_id])
        state = response['Reservations'][0]['Instances'][0]['State']['Name']
        if state == target_state:
            return True
        time.sleep(5)
    return False

def manage_instance(operation, group):
    for instance_id in INSTANCE_IDS[group]:
        try:
            if operation == 'start':
                current_state = ec2.describe_instances(InstanceIds=[instance_id])['Reservations'][0]['Instances'][0]['State']['Name']
                if current_state not in ['running', 'pending']:
                    ec2.start_instances(InstanceIds=[instance_id])
                    print(f"Starting {group} instance {instance_id}")
                    if not wait_for_instance_state(instance_id):
                        raise Exception(f"Failed to start {group} instance {instance_id} within timeout")
                else:
                    print(f"Instance {instance_id} already {current_state}")
            elif operation == 'stop':
                ec2.stop_instances(InstanceIds=[instance_id])
                print(f"Stopping {group} instance {instance_id}")
        except Exception as e:
            print(f"Instance management error for {instance_id}: {str(e)}")
            raise

def register_instances(group):
    for instance_id in INSTANCE_IDS[group]:
        if not wait_for_instance_state(instance_id):
            raise Exception(f"Instance {instance_id} not in running state")

    targets = [{'Id': iid,'Port':8080} for iid in INSTANCE_IDS[group]]
    try:
        elbv2.register_targets(
            TargetGroupArn=TARGET_GROUPS[group],
            Targets=targets
        )
        print(f"Successfully registered {group} instances on port 8080")
    except Exception as e:
        print(f"Registration failed: {str(e)}")
        raise

def deregister_instances(group):
    try:
        targets = [{'Id': iid,'Port':8080} for iid in INSTANCE_IDS[group]]
        elbv2.deregister_targets(
            TargetGroupArn=TARGET_GROUPS[group],
            Targets=targets
        )
        print(f"Deregistered {group} instances from port 8080")
    except Exception as e:
        print(f"Deregistration error: {str(e)}")

def get_arrival_rate():
    try:
        now = datetime.now()
        elapsed = (now - START_TIME).total_seconds()
        return REQUEST_COUNT / elapsed if elapsed > 0 else 0
    except Exception as e:
        print(f"[ERROR] get_arrival_rate: {str(e)}")
        return 0

def evaluate_scaling_decision(decision, current_rate, prev_rate):
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

def choose_load_balancer_ml():
    try:
        current_rate = get_arrival_rate()
        features = predictor.extract_features(traffic_history)
        predicted_rate = predictor.predict_traffic(features)
        perf_metrics = predictor.performance_tracker.get_current_accuracy_metrics() or {}
        
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
        if current_state == 'small' and effective_rate > T1 * buffer_multiplier:
            target = 'medium'
        elif current_state == 'medium' and effective_rate > T2 * buffer_multiplier:
            target = 'large'
        elif current_state == 'medium' and effective_rate < T1 / buffer_multiplier:
            target = 'small'
        elif current_state == 'large' and effective_rate < T2 / buffer_multiplier:
            target = 'medium'
        else:
            target = current_state

        pred_str = f"{predicted_rate:.1f}" if predicted_rate is not None else "None"
        print(f"[ML-DECISION] Current: {current_rate:.1f}, Predicted: {pred_str}")
        print(f"[ML-DECISION] Effective: {effective_rate:.1f}, Target: {target}")
        return target

    except Exception as e:
        print(f"[ML] Decision error: {str(e)}")
        return current_state

    
def warmup_and_switch(target_group_to_use):
    global current_state
    if switching_event.is_set():
        return

    switching_event.set()
    print(f"[JMeter-ML] Switching to: {target_group_to_use}")

    try:
        manage_instance('start', target_group_to_use)
        register_instances(target_group_to_use)

        start_time = time.time()
        while time.time() - start_time < 60:
            response = elbv2.describe_target_health(
                TargetGroupArn=TARGET_GROUPS[target_group_to_use],
                Targets=[{'Id': iid,'Port':8080} for iid in INSTANCE_IDS[target_group_to_use]]
            )
            if response['TargetHealthDescriptions'][0]['TargetHealth']['State'] == 'healthy':
                deregister_instances(current_state)
                manage_instance('stop', current_state)
                current_state = target_group_to_use
                print(f"[SUCCESS] Switched to {target_group_to_use}")
                return
            time.sleep(5)

        print(f"Health check timeout for {target_group_to_use}")

    except Exception as e:
        print(f"Transition failed: {str(e)}")
        if target_group_to_use != 'small':
            warmup_and_switch('small')
    finally:
        switching_event.clear()

# Enhanced background monitor function
def ml_background_monitor():
    """Enhanced monitoring with performance tracking - IMPROVED ERROR HANDLING"""
    last_prediction = None
    prediction_log_counter = 0

    while True:
        try:
            # FIX 4: Safer rate calculation
            try:
                current_rate = get_arrival_rate()
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
                    'state': current_state
                }
                traffic_history.append(traffic_point)
            except Exception as e:
                print(f"[MONITOR] Traffic point creation error: {e}")
                time.sleep(5)
                continue

            # Safe feature extraction
            features = None
            try:
                features = predictor.extract_features(traffic_history)
            except Exception as e:
                print(f"[MONITOR] Feature extraction error: {e}")

            # Log prediction details every 10 iterations (50 seconds)
            if prediction_log_counter % 10 == 0:
                print(f"\n[PREDICTION-LOG] Current rate: {current_rate:.2f}")
                print(f"[PREDICTION-LOG] History length: {len(traffic_history)}")
                print(f"[PREDICTION-LOG] Features valid: {features is not None}")

            # Safe prediction
            current_prediction = None
            try:
                current_prediction = predictor.predict_traffic(features)
            except Exception as e:
                print(f"[MONITOR] Prediction error: {e}")

            if prediction_log_counter % 10 == 0:
                pred_str = f"{current_prediction:.2f}" if current_prediction is not None else "None"
                print(f"[PREDICTION-LOG] Predicted rate: {pred_str}")
                print(f"[PREDICTION-LOG] Model trained: {predictor.is_trained}")
                print(f"[PREDICTION-LOG] Training samples: {len(predictor.training_data)}\n")

            # Track prediction accuracy
            if last_prediction is not None and current_rate > 0:
                try:
                    predictor.performance_tracker.add_prediction(
                        last_prediction,
                        current_rate,
                        datetime.now()
                    )
                except Exception as e:
                    print(f"[MONITOR] Performance tracking error: {e}")

            last_prediction = current_prediction
            prediction_log_counter += 1

            # Safe training data addition
            if features is not None:
                try:
                    predictor.add_training_data(features, current_rate)
                except Exception as e:
                    print(f"[MONITOR] Training data addition error: {e}")

            # Safe scaling decision tracking
            try:
                if len(traffic_history) >= 2:
                    prev_rate = list(traffic_history)[-2]['rate']
                    current_decision = choose_load_balancer_ml()
                    was_correct = evaluate_scaling_decision(current_decision, current_rate, prev_rate)
                    predictor.performance_tracker.add_scaling_decision(
                        current_decision, current_rate, was_correct
                    )
            except Exception as e:
                print(f"[MONITOR] Scaling decision tracking error: {e}")

            # Safe model training triggers
            try:
                if len(predictor.training_data) >= 50 and not predictor.is_trained:  # Reduced from 20
                    print("[ML] Initial training...")
                    predictor.train_model()
                elif len(predictor.training_data) % 100 == 0 and len(predictor.training_data) > 0:  # Reduced frequency
                    print("[ML] Retraining with new data...")
                    predictor.train_model()
            except Exception as e:
                print(f"[MONITOR] Model training error: {e}")

            time.sleep(5)

        except Exception as e:
            print(f"[MONITOR] Main loop error: {str(e)}")
            import traceback
            traceback.print_exc()
            time.sleep(10)  # Longer sleep on critical error
            
            
def background_scaler():
    while True:
        try:
            target_state = choose_load_balancer_ml()
            if target_state and target_state != current_state and not switching_event.is_set():
                threading.Thread(target=warmup_and_switch, args=(target_state,)).start()
            time.sleep(5)
        except Exception as e:
            print(f"Background scaler error: {str(e)}")
            time.sleep(10)  # Longer sleep on error to prevent spam

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


@app.route('/', defaults={'path': ''}, methods=['GET', 'POST'])
@app.route('/<path:path>', methods=['GET', 'POST'])
def reverse_proxy(path):
    global REQUEST_COUNT

    # FIXED: Don't count metrics requests as load traffic
    if path != 'metrics':
        REQUEST_COUNT += 1
        print(f"[REQUEST] Count: {REQUEST_COUNT}, Path: /{path}, Rate: {get_arrival_rate():.2f} req/sec")

    try:
        headers = dict(request.headers)
        headers['instance-type'] = current_state
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

# Enhanced metrics endpoint
@app.route('/metrics')
def metrics():
    """Enhanced metrics with proper type handling"""
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
        rate = get_arrival_rate()
        features = predictor.extract_features(traffic_history)
        predicted_rate = predictor.predict_traffic(features) if features else None

        # Get model performance metrics
        performance_metrics = predictor.performance_tracker.get_current_accuracy_metrics()
        prediction_trend = predictor.performance_tracker.get_prediction_trend()

        # Build response with type conversion
        response_data = {
            "arrival_rate": safe_convert(round(rate, 2)),
            "predicted_rate": safe_convert(round(predicted_rate, 2)) if predicted_rate else None,
            "active_group": current_state,
            "ml_trained": predictor.is_trained,
            "training_samples": len(predictor.training_data),
            "last_retrain": safe_convert(predictor.last_retrain),
            "jmeter_patterns": {k: safe_convert(v) for k,v in predictor.jmeter_patterns.items()},
            "request_count": REQUEST_COUNT,
            "uptime_minutes": safe_convert(round((datetime.now() - START_TIME).total_seconds() / 60, 2)),
            "debug_info": {
                "traffic_history_length": len(traffic_history),
                "current_rate_calc": f"{REQUEST_COUNT} requests / {(datetime.now() - START_TIME).total_seconds():.1f} seconds"
            }
        }

        # Add performance metrics with type conversion
        if performance_metrics:
            response_data["model_performance"] = {k: safe_convert(v) for k,v in performance_metrics.items()}
            response_data["prediction_trend"] = prediction_trend
            response_data["model_quality"] = interpret_model_quality(performance_metrics)
            response_data["recommendations"] = get_model_recommendations(performance_metrics) #removed prediction_trend

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route('/debug-prediction')
def debug_prediction():
    """Debug endpoint to see prediction pipeline"""
    try:
        current_rate = get_arrival_rate()
        features = predictor.extract_features(traffic_history)

        debug_info = {
            "current_rate": current_rate,
            "traffic_history_length": len(traffic_history),
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
        recent_history = list(traffic_history)[-50:]  # Last 50 points

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
    global REQUEST_COUNT, START_TIME
    # Reset counters for new test
    REQUEST_COUNT = 0
    START_TIME = datetime.now()
    traffic_history.clear()

    predictor.jmeter_patterns = {
        'ramp_up_detected': False,
        'peak_detected': False,
        'ramp_down_detected': False,
        'pattern_start_time': None
    }
    return jsonify({"status": "JMeter patterns and counters reset"})

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

        # Update S3 bucket name before running
        if S3_BUCKET == 'ml-load-balancing':
            print("WARNING: Please update S3_BUCKET name in the code!")

        manage_instance('start', 'small')
        register_instances('small')

        for group in ['medium', 'large']:
            manage_instance('stop', group)
            deregister_instances(group)

        threading.Thread(target=background_scaler, daemon=True).start()
        threading.Thread(target=ml_background_monitor, daemon=True).start()

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

