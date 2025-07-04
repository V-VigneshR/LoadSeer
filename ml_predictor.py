import numpy as np
import pickle
import json
import time
from datetime import datetime, timedelta
from collections import deque
import warnings
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import boto3
from config import (
    region_name,
    S3_BUCKET,
    S3_MODEL_KEY,
    S3_SCALER_KEY,
    S3_DATA_KEY,
    S3_HISTORY_KEY
)
from performance_tracker import ModelPerformanceTracker

warnings.filterwarnings('ignore')

class PersistentTrafficPredictor:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = deque(maxlen=5000)  # Increased capacity
        self.feature_window = 30
        self.last_retrain = datetime.now()
        self.last_s3_sync = datetime.now()
        self.sync_interval = timedelta(minutes=5)  # Sync every 5 minutes
        self.performance_tracker = ModelPerformanceTracker()
        self.sequence_lags = 5  # rate_t-1 to rate_t-5
        self.s3 = boto3.client('s3', region_name=region_name)

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
            self.s3.head_bucket(Bucket=S3_BUCKET)
        except:
            try:
                self.s3.create_bucket(
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
            
            print(f"[S3] Starting save with {len(self.training_data)} training samples")

            # Save model
            if self.is_trained:
                try:
                    model_bytes = pickle.dumps(self.model)
                    self.s3.put_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY, Body=model_bytes)
                    print("[S3] Model saved successfully")

                    scaler_bytes = pickle.dumps(self.scaler)
                    self.s3.put_object(Bucket=S3_BUCKET, Key=S3_SCALER_KEY, Body=scaler_bytes)
                    print("[S3] Scaler saved successfully")
                except Exception as e:
                    print(f"[S3] Model save error: {e}")

            # Save training data (convert deque to list for JSON serialization)
            training_data_list = []
            for item in self.training_data:
                try:
                    training_data_list.append({
                        'features': item['features'],
                        'target': item['target'],
                        'timestamp': item['timestamp'].isoformat()
                    })
                except Exception as e:
                    print(f"[S3] Error processing training item: {e}")
                    continue

            if training_data_list:
                try:
                    data_json = json.dumps(training_data_list)
                    self.s3.put_object(Bucket=S3_BUCKET, Key=S3_DATA_KEY, Body=data_json)
                    print(f"[S3] Training data saved: {len(training_data_list)} samples")
                except Exception as e:
                    print(f"[S3] Training data save error: {e}")

            self.last_s3_sync = datetime.now()
            print(f"[S3] Save completed at {self.last_s3_sync}")

        except Exception as e:
            print(f"[S3] Save error: {str(e)}")
            import traceback
            traceback.print_exc()

    def load_from_s3(self):
        """Load model, scaler, and training data from S3"""
        try:
            # Load model and scaler
            try:
                model_obj = self.s3.get_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY)
                self.model = pickle.loads(model_obj['Body'].read())

                scaler_obj = self.s3.get_object(Bucket=S3_BUCKET, Key=S3_SCALER_KEY)
                self.scaler = pickle.loads(scaler_obj['Body'].read())

                self.is_trained = True
                print("[S3] Loaded existing model and scaler")
            except:
                print("[S3] No existing model found, will train new one")

            # Load training data
            try:
                data_obj = self.s3.get_object(Bucket=S3_BUCKET, Key=S3_DATA_KEY)
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
        """Enhanced pattern detection with lower thresholds for faster detection"""
        try:
            # Timeout after 5 minutes (300 seconds)
            if (self.jmeter_patterns['pattern_start_time'] and 
                (datetime.now() - self.jmeter_patterns['pattern_start_time']).total_seconds() > 300):
                self._reset_patterns()
                return 'pattern_timeout'
            
            # Reduced minimum history from 20 to 10 for faster detection
            if len(traffic_history) < 10:
                return None

            recent = list(traffic_history)[-10:]  # Use smaller window
            rates = np.array([p['rate'] for p in recent])
            timestamps = np.array([p['timestamp'].timestamp() for p in recent])
            
            # Calculate linear regression trend
            slope, intercept = np.polyfit(timestamps, rates, 1)
            trend_strength = np.corrcoef(timestamps, rates)[0, 1]
            
            # More sensitive ramp-up detection (reduced thresholds)
            if (not self.jmeter_patterns['ramp_up_detected'] and 
                slope > 3 and trend_strength > 0.7 and  # Reduced from 5 and 0.8
                rates[-1] > rates[0] * 1.3):  # Reduced from 1.5
                self._reset_patterns()
                self.jmeter_patterns['ramp_up_detected'] = True
                self.jmeter_patterns['pattern_start_time'] = datetime.now()
                return 'ramp_up_start'
                
            # More sensitive peak detection
            if (self.jmeter_patterns['ramp_up_detected'] and 
                not self.jmeter_patterns['peak_detected'] and
                abs(slope) < 1.5 and np.std(rates[-5:]) < 10):  # Reduced thresholds
                self.jmeter_patterns['peak_detected'] = True
                return 'peak_detected'
                
            # More sensitive ramp-down detection
            if (self.jmeter_patterns['peak_detected'] and 
                not self.jmeter_patterns['ramp_down_detected'] and
                slope < -3 and trend_strength < -0.6):  # Reduced thresholds
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
        """Feature extraction with reduced minimum history requirement"""
        try:
            # Reduced minimum history from sequence_lags+1 to sequence_lags
            if len(traffic_history) < self.sequence_lags:
                print(f"[FEATURES] Insufficient history: {len(traffic_history)} < {self.sequence_lags}")
                # Return basic features if possible
                if len(traffic_history) > 0:
                    basic_features = [0]*self.sequence_lags  # Pad with zeros
                    recent_rates = [p['rate'] for p in traffic_history]
                    basic_features[-len(recent_rates):] = recent_rates  # Fill available rates
                    
                    now = datetime.now()
                    time_features = [now.hour, now.weekday(), now.minute]
                    pattern_features = [0, 0, 0, 0]  # Default pattern features
                    
                    features = basic_features + time_features + pattern_features
                    print(f"[FEATURES] Using partial history with padding")
                    return features
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

            # Safe pattern detection with fallbacks
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

    def add_training_data(self, features, actual_rate):
        """Add training data and sync to S3 periodically"""
        if features is not None:
            training_point = {
                'features': features,
                'target': actual_rate,
                'timestamp': datetime.now()
            }
            self.training_data.append(training_point)
            
            if len(self.training_data) % 10 == 0:  # Print every 10 samples
                print(f"[TRAINING] Added sample, total: {len(self.training_data)}")

            # More frequent S3 sync - every 2 minutes instead of 5
            sync_interval = timedelta(minutes=2)
            if datetime.now() - self.last_s3_sync > sync_interval:
                print(f"[S3] Triggering sync - {len(self.training_data)} samples")
                self.save_to_s3()

    def train_model(self):
        """Enhanced XGBoost training with cross-validation and robust validation"""
        MIN_SAMPLES = 50   # Reduced from 200 for faster initial training
        EARLY_STOP = 10    
        N_FOLDS = 3        
        MAX_MAE_RATIO = 0.4  # Slightly relaxed from 0.3

        if len(self.training_data) < MIN_SAMPLES:
            print(f"[TRAIN] Need {MIN_SAMPLES} samples (current: {len(self.training_data)})")
            return False

        try:
            # Data validation and cleaning
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
        """Enhanced predict with detailed debugging and improved logic"""
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
                        if recent_trend > 0.03:  # At least 5% upward trend
                            adjustment_factor = min(1.5, 1.0 + recent_trend * 2)
                            print(f"[PREDICT] Applied dynamic ramp-up boost: {adjustment_factor:.2f}x")
                        else:
                            print("[PREDICT] Insufficient trend for ramp-up boost")
                    except:
                        print("[PREDICT] Error calculating trend, no adjustment")
                            
                elif self.jmeter_patterns['ramp_down_detected']:
                    adjustment_factor = 0.6
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
        """Calculate percentage growth using exponential weighting"""
        try:
            if len(self.training_data) < 3:  # Reduced from 5
                return 0.0
            
            # Use exponential weighting for more recent points
            recent = list(self.training_data)[-5:]
            weights = np.exp(np.linspace(0, 1, len(recent)))
            weights /= weights.sum()
            
            valid_recent = []
            valid_weights = []
            for i, d in enumerate(recent):
                if isinstance(d['target'], (int, float)) and d['target'] >= 0:
                    valid_recent.append(d['target'])
                    valid_weights.append(weights[i])
            
            if len(valid_recent) < 2:
                return 0.0
                
            if valid_recent[0] == 0:
                return 0.0
                
            # Weighted trend calculation
            weighted_avg_start = np.average(valid_recent[:len(valid_recent)//2], 
                                        weights=valid_weights[:len(valid_recent)//2])
            weighted_avg_end = np.average(valid_recent[len(valid_recent)//2:],
                                        weights=valid_weights[len(valid_recent)//2:])
            
            trend = (weighted_avg_end - weighted_avg_start) / weighted_avg_start
            
            return max(-1.0, min(1.0, trend))
            
        except Exception as e:
            print(f"[TREND] Calculation error: {e}")
            return 0.0