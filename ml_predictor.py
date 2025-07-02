# ml_predictor.py
"""Machine Learning predictor for traffic forecasting"""

import json
import pickle
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
import warnings

from config import *
from performance_tracker import ModelPerformanceTracker

warnings.filterwarnings('ignore')


class PersistentTrafficPredictor:
    """ML-based traffic predictor with S3 persistence"""
    
    def __init__(self, s3_client):
        self.model = XGBRegressor(**XGBOOST_PARAMS)
        self.scaler = StandardScaler()
        self.s3 = s3_client
        self.is_trained = False
        self.training_data = deque(maxlen=2000)
        self.feature_window = FEATURE_WINDOW
        self.last_retrain = datetime.now()
        self.last_s3_sync = datetime.now()
        self.sync_interval = timedelta(minutes=SYNC_INTERVAL_MINUTES)
        self.performance_tracker = ModelPerformanceTracker()
        self.sequence_lags = SEQUENCE_LAGS

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
                    CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
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
                self.s3.put_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY, Body=model_bytes)

                scaler_bytes = pickle.dumps(self.scaler)
                self.s3.put_object(Bucket=S3_BUCKET, Key=S3_SCALER_KEY, Body=scaler_bytes)

            # Save training data
            training_data_list = []
            for item in self.training_data:
                training_data_list.append({
                    'features': item['features'],
                    'target': item['target'],
                    'timestamp': item['timestamp'].isoformat()
                })

            data_json = json.dumps(training_data_list)
            self.s3.put_object(Bucket=S3_BUCKET, Key=S3_DATA_KEY, Body=data_json)

            self.last_s3_sync = datetime.now()
            print(f"[S3] Saved model and {len(self.training_data)} training samples")

        except Exception as e:
            print(f"[S3] Save error: {str(e)}")

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
        """Enhanced pattern detection with statistical validation"""
        try:
            # Timeout after 5 minutes
            if (self.jmeter_patterns['pattern_start_time'] and 
                (datetime.now() - self.jmeter_patterns['pattern_start_time']).total_seconds() > 300):
                self._reset_patterns()
                return 'pattern_timeout'
            
            if len(traffic_history) < 20:
                return None

            recent = list(traffic_history)[-20:]
            rates = np.array([p['rate'] for p in recent])
            timestamps = np.array([p['timestamp'].timestamp() for p in recent])
            
            # Calculate linear regression trend
            slope, intercept = np.polyfit(timestamps, rates, 1)
            trend_strength = np.corrcoef(timestamps, rates)[0, 1]
            
            # Pattern detection logic
            if (not self.jmeter_patterns['ramp_up_detected'] and 
                slope > 5 and trend_strength > 0.8 and 
                rates[-1] > rates[0] * 1.5):
                self._reset_patterns()
                self.jmeter_patterns['ramp_up_detected'] = True
                self.jmeter_patterns['pattern_start_time'] = datetime.now()
                return 'ramp_up_start'
                
            if (self.jmeter_patterns['ramp_up_detected'] and 
                not self.jmeter_patterns['peak_detected'] and
                abs(slope) < 2 and np.std(rates[-10:]) < 15):
                self.jmeter_patterns['peak_detected'] = True
                return 'peak_detected'
                
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
        """Feature extraction using lag features + time + JMeter pattern"""
        try:
            if len(traffic_history) < self.sequence_lags + 1:
                print(f"[FEATURES] Insufficient history: {len(traffic_history)} < {self.sequence_lags + 1}")
                return None

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
                
                lag_rates.append(float(rate))

            # Time-based features
            now = datetime.now()
            hour = now.hour
            day_of_week = now.weekday()
            minute = now.minute

            # Pattern detection with fallbacks
            try:
                pattern = self.detect_jmeter_pattern(traffic_history)
            except Exception as e:
                print(f"[FEATURES] Pattern detection failed: {e}")
                pattern = None

            # Create pattern features
            pattern_features = {
                'is_ramp_up': 1 if pattern == 'ramp_up_start' else 0,
                'is_peak': 1 if pattern == 'peak_detected' else 0,
                'is_ramp_down': 1 if pattern == 'ramp_down_start' else 0,
                'test_duration': 0.0
            }

            # Calculate test duration safely
            if self.jmeter_patterns['pattern_start_time']:
                try:
                    duration_seconds = (now - self.jmeter_patterns['pattern_start_time']).total_seconds()
                    pattern_features['test_duration'] = max(0.0, duration_seconds / 60.0)
                except:
                    pattern_features['test_duration'] = 0.0

            # Combine features
            features = lag_rates + [float(hour), float(day_of_week), float(minute)] + list(pattern_features.values())
            
            # Final validation
            if len(features) != (self.sequence_lags + 7):
                print(f"[FEATURES] Unexpected feature count: {len(features)}")
                return None

            for i, feat in enumerate(features):
                if not isinstance(feat, (int, float)) or not np.isfinite(feat):
                    print(f"[FEATURES] Invalid feature at index {i}: {feat}")
                    return None

            print(f"[FEATURES] Successfully extracted {len(features)} features")
            return features

        except Exception as e:
            print(f"[FEATURES] Extract error: {str(e)}")
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
                import threading
                threading.Thread(target=self.save_to_s3).start()

    def train_model(self):
        """Enhanced XGBoost training with cross-validation"""
        if len(self.training_data) < MIN_TRAINING_SAMPLES:
            print(f"[TRAIN] Need {MIN_TRAINING_SAMPLES} samples (current: {len(self.training_data)})")
            return False

        try:
            # Data validation and cleaning
            valid_data = []
            for point in self.training_data:
                try:
                    features = point['features']
                    target = point['target']
                    
                    if (features is None or not isinstance(features, list) or 
                        len(features) != (self.sequence_lags + 7)):
                        continue
                    
                    if (target is None or not isinstance(target, (int, float)) or 
                        target < 0 or not np.isfinite(target)):
                        continue
                    
                    if any(not np.isfinite(f) for f in features):
                        continue
                    
                    valid_data.append({'features': features, 'target': target})
                    
                except Exception as e:
                    continue

            if len(valid_data) < MIN_TRAINING_SAMPLES:
                print(f"[TRAIN] Insufficient valid data: {len(valid_data)} < {MIN_TRAINING_SAMPLES}")
                return False

            print(f"[TRAIN] Using {len(valid_data)} valid samples from {len(self.training_data)} total")

            X = np.array([point['features'] for point in valid_data], dtype=np.float64)
            y = np.array([point['target'] for point in valid_data], dtype=np.float64)
            
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                print("[TRAIN] Found NaN values in training data")
                return False
            
            avg_y = np.mean(y)
            print(f"[TRAIN] Target stats - Mean: {avg_y:.2f}, Range: [{np.min(y):.2f}, {np.max(y):.2f}]")

            # Cross-validated training
            kf = KFold(n_splits=min(N_FOLDS, len(valid_data) // 10))
            val_scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                try:
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    model = XGBRegressor(**XGBOOST_PARAMS)
                    model.fit(X_train, y_train, verbose=False)
                    
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

            avg_mae = np.mean([s['mae'] for s in val_scores])
            avg_r2 = np.mean([s['r2'] for s in val_scores])
            
            print(f"[TRAIN] Cross-val MAE: {avg_mae:.2f}, R²: {avg_r2:.3f}")

            # Train final model
            self.model = XGBRegressor(**XGBOOST_PARAMS)
            self.model.fit(X, y, verbose=False)
            self.is_trained = True
            self.last_retrain = datetime.now()

            # Feature importance
            features = ["lag1", "lag2", "lag3", "lag4", "lag5", 
                       "hour", "weekday", "minute", 
                       "is_ramp", "is_peak", "is_ramp_down", "duration"]
            print("\n[TRAIN] Top Feature Importances:")
            for name, imp in sorted(zip(features, self.model.feature_importances_),
                                   key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {name}: {imp:.3f}")

            return True

        except Exception as e:
            print(f"[TRAIN] Error: {str(e)}")
            return False

    def predict_traffic(self, features):
        """Enhanced prediction with pattern adjustments"""
        if not self.is_trained or features is None:
            return None

        try:
            # Validate features
            if not isinstance(features, list) or len(features) != (self.sequence_lags + 7):
                return None

            for feat in features:
                if not isinstance(feat, (int, float)) or not np.isfinite(feat):
                    return None

            # Get base prediction
            features_array = np.array([features], dtype=np.float64).reshape(1, -1)
            base_prediction = float(self.model.predict(features_array)[0])
            
            if not np.isfinite(base_prediction):
                return None

            print(f"[PREDICT] Base prediction: {base_prediction:.2f}")

            # Apply pattern adjustments
            adjustment_factor = 1.0
            current_time = datetime.now()
            
            try:
                # Check if ramp-up period should expire
                if (self.jmeter_patterns['ramp_up_detected'] and 
                    self.jmeter_patterns['pattern_start_time'] and
                    (current_time - self.jmeter_patterns['pattern_start_time']).total_seconds() > 300):
                    
                    self.jmeter_patterns['ramp_up_detected'] = False
                    self.jmeter_patterns['peak_detected'] = True

                # Apply adjustments
                if self.jmeter_patterns['ramp_up_detected'] and not self.jmeter_patterns['peak_detected']:
                    try:
                        recent_trend = self._calculate_recent_trend()
                        if recent_trend > 0.05:
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
                return base_prediction  # Fallback to base prediction

            print(f"[PREDICT] Final prediction: {final_prediction:.2f}")
            return final_prediction

        except Exception as e:
            print(f"[PREDICT] Prediction error: {str(e)}")
            return None

    def _calculate_recent_trend(self):
        """Calculate percentage growth over last 5 data points"""
        try:
            if len(self.training_data) < 5:
                return 0.0
            
            recent = [d['target'] for d in list(self.training_data)[-5:]]
            valid_recent = [r for r in recent if isinstance(r, (int, float)) and r >= 0]
            
            if len(valid_recent) < 2 or valid_recent[0] == 0:
                return 0.0
                
            trend = (valid_recent[-1] - valid_recent[0]) / valid_recent[0]
            return max(-1.0, min(1.0, trend))
            
        except Exception as e:
            print(f"[TREND] Calculation error: {e}")
            return 0.0

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


                        
