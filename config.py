# Constants
T1 = 125
T2 = 225
BACKEND_SERVER = "http://load-balancer-1146502231.ap-south-1.elb.amazonaws.com"
INSTANCE_START_TIMEOUT = 300
PREDICTION_HORIZON = 300
HYSTERESIS_BUFFER = 3

# AWS Config
region_name = 'ap-south-1'

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
    'small': ['i-0cf881aeb7c6fd3b2'],
    'medium': ['i-0ef79edd244f81a8e'],
    'large': ['i-013eadffaa6287065']
}