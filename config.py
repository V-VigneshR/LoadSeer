# config.py
"""Configuration settings for the ML Load Balancer"""

# Flask Configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 8080

# Load Balancer Constants
T1 = 125  # Small to Medium threshold
T2 = 225  # Medium to Large threshold
BACKEND_SERVER = "http://load-balancer-700058647.ap-south-1.elb.amazonaws.com"
INSTANCE_START_TIMEOUT = 300
PREDICTION_HORIZON = 300
HYSTERESIS_BUFFER = 3

# AWS Configuration
AWS_REGION = 'ap-south-1'

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
    'small': ['i-0c43a9e5f383352e8'],
    'medium': ['i-0e8b07c25379d30e7'],
    'large': ['i-019ace343c3435424']
}

# ML Model Configuration
MIN_TRAINING_SAMPLES = 50
EARLY_STOP = 10
N_FOLDS = 3
MAX_MAE_RATIO = 0.4
TRAFFIC_HISTORY_SIZE = 500
PREDICTION_HISTORY_SIZE = 1000
ACCURACY_HISTORY_SIZE = 100
SCALING_DECISIONS_SIZE = 200
FEATURE_WINDOW = 30
SEQUENCE_LAGS = 5

# XGBoost Parameters
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.1,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'random_state': 42
}

# Monitoring Configuration
SYNC_INTERVAL_MINUTES = 5
MONITOR_SLEEP_SECONDS = 5
SCALER_SLEEP_SECONDS = 5
ERROR_SLEEP_SECONDS = 10