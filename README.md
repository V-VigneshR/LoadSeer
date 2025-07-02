# LoadSeer 
**ML-Enhanced Dynamic Load Balancer for Real-Time Traffic Prediction and Autoscaling**

LoadSeer is an intelligent load balancer powered by XGBoost regression models and adaptive online retraining. It analyzes historical traffic patterns and system metrics to predict future load and adjust system capacity dynamically.

##  Key Features

-  **High Prediction Accuracy**  
  Achieves ~5% MAPE (Mean Absolute Percentage Error), 100% accuracy within ±25% range, and R² score of 0.72+ — ensuring high-confidence autoscaling decisions.

-  **Adaptive Retraining**  
  Automatically retrains the model during runtime based on new traffic samples while avoiding data leakage using delay-window evaluation.

-  **Real-Time Traffic Insights**  
  Provides accurate feature extraction including moving averages, rate of change, hour-of-day/time-based signals, and group activity labels (e.g., `small`, `medium`, `high`).

-  **Lightweight & Cloud-Native**  
  Built with Python and Flask, easily deployable on EC2 instances or containers. No heavyweight ML infra needed.

-  **Model Health Dashboard (Optional)**  
  Logs model metrics like RMSE, R², MAPE, and sample counts — perfect for live monitoring or Grafana ingestion.

##  Tech Stack

- Python 3.x
- XGBoost for regression
- Pandas/Numpy (for historical sequence processing)
- Flask API (for serving predictions)
- JMeter compatible hooks for load testing environments

##  Deployment

LoadSeer is  deployed and tested on:
- AWS EC2 (Amazon Linux 2)

##  Use Cases

- Autoscaling policies in microservice clusters
- Load simulation environments with CI/CD pipelines
- Intelligent traffic shaping for SaaS or high-frequency APIs

---
NOTE: The model is still under improvement for further enhancement 
