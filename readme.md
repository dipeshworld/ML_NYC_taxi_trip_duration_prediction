# 🚖 Taxi Trip Duration Prediction (Production-Ready ML System)

## 📌 Overview

This project builds a **production-ready machine learning system** to predict **taxi trip duration** using historical ride data.

The goal is to help ride-hailing platforms (e.g., Uber, Lyft) improve:

* 🚀 Dispatch efficiency
* 📍 Driver allocation
* ⏱️ ETA predictions

---

## 🎯 Business Problem

To optimize taxi dispatching systems, it is essential to estimate **how long a driver will remain occupied** during a trip.

This enables:

* Smarter ride assignments
* Reduced passenger wait time
* Better fleet utilization

---

## 📂 Dataset Description

The dataset contains taxi trip records with spatial and temporal features:

| Feature                      | Description                  |
| ---------------------------- | ---------------------------- |
| id                           | Unique trip identifier       |
| vendor_id                    | Taxi vendor                  |
| pickup_datetime              | Trip start time              |
| dropoff_datetime             | Trip end time                |
| passenger_count              | Number of passengers         |
| pickup_longitude / latitude  | Pickup location              |
| dropoff_longitude / latitude | Dropoff location             |
| store_and_fwd_flag           | Data transmission flag       |
| trip_duration                | Target variable (in seconds) |

---

## 🧠 Problem Type

* **Supervised Learning**
* **Regression Task** (predict continuous trip duration)

---

## 🚀 Model Used

### 🔥 XGBoost Regressor

Why XGBoost?

* Handles **non-linear relationships**
* Excellent for **tabular data**
* Built-in **regularization (L1 & L2)**
* High performance in real-world ML systems

---

## ⚙️ Feature Engineering

### ⏱️ Time-Based Features

* Hour of day
* Day of week
* Month

### 🌍 Geospatial Features

* **Haversine distance** between pickup and dropoff

### 🔢 Encoding

* store_and_fwd_flag → binary encoding

---

## 🏗️ ML Pipeline Architecture

```id="arch3"
Raw Data
   ↓
Feature Engineering (Time + Distance)
   ↓
Preprocessing (Scaling)
   ↓
XGBoost Model
   ↓
Evaluation (RMSE, R²)

```

---

## 📊 Evaluation Metrics

* **RMSE (Root Mean Squared Error)**
* **R² Score**

---

## 📈 Sample Output

```id="out3"
===== MODEL PERFORMANCE =====
RMSE: 180.25
R2 Score: 0.92

===== SAMPLE PREDICTIONS =====
Actual vs Predicted Trip Duration
```

---

## 📁 Project Structure

```id="struct3"
taxi-duration-prediction/
│
├── data/
│   └── nyc_taxi_trip_duration.csv
├── main.py
├── requirements.txt
└── README.md
```

---

## ▶️ Usage

### 1. Clone the repository

```bash id="cmd7"
git clone https://github.com/your-username/taxi-duration-prediction.git
cd taxi-duration-prediction
```

### 2. Install dependencies

```bash id="cmd8"
pip install -r requirements.txt
```

### 3. Train the model

```bash id="cmd9"
python main.py
```

---



Use it for:

* Batch predictions
* API deployment
* Real-time inference

---

## 🔍 Key Features

* End-to-end ML pipeline
* Feature engineering (time + geospatial)
* Regularized XGBoost model

---

## 💡 Future Improvements

* 📊 Hyperparameter tuning (Optuna / GridSearch)
* 🌐 Deploy using FastAPI
* 📈 Add monitoring & drift detection
* 🗺️ Use map-based clustering (geo features)
* 🚦 Incorporate traffic/weather data

---

## 🧪 Business Impact

* Improved ETA accuracy
* Better driver allocation
* Reduced idle time
* Enhanced customer experience

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* XGBoost community
* Scikit-learn contributors

---

## 📬 Contact

For questions or collaboration, feel free to reach out.

---
