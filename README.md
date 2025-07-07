# LoadSeer 
**Proactive Cloud Autoscaling through ML-Based Traffic Prediction**

LoadSeer is an intelligent autoscaling controller powered by XGBoost regression and adaptive online retraining. It analyzes real-time traffic patterns to anticipate demand and dynamically scale EC2 target groups accordingly. While not a traditional load balancer, LoadSeer complements them by optimizing backend capacity based on predictive insights.

##  Key Features

- **High Prediction Accuracy**  
  Achieves <5% MAPE (Mean Absolute Percentage Error), >98% scaling accuracy within ±25% range, and R² scores consistently above 0.72.

- **Adaptive Retraining**  
  Continuously retrains on new traffic samples with safeguards against data leakage using delay-window validation.

- **Real-Time Traffic Intelligence**  
  Extracts engineered features such as rate of change, rolling averages, time-of-day trends, and usage phase labels (`ramp_up`, `peak`, `ramp_down`).

- **Lightweight & Cloud-Native**  
  Python-based (Flask + XGBoost), deployable on EC2 or Docker. No GPU or heavy ML infra required.

- **Model Health Monitoring**  
  Logs metrics such as MAE, R², MAPE, and accuracy — exportable to dashboards like Grafana.

##  Tech Stack

- Python 3.x
- XGBoost (for regression)
- Pandas / NumPy (sequence processing)
- Flask API (prediction service)
- Apache JMeter (traffic simulation integration)

##  Deployment Environment

- Tested and deployed on:  
  **AWS EC2 (Amazon Linux 2)**  
  (Uses Boto3 to manage EC2 scaling based on predicted traffic)

##  Use Cases

- Smart autoscaling for microservice clusters
- Traffic-aware backend provisioning in CI/CD testbeds
- ML-powered control layer over existing load balancers
- Prototype for ML-integrated DevOps workflows

---

>  **Note**: LoadSeer is a predictive autoscaling companion, not a drop-in replacement for conventional load balancers like ALB/NLB.

