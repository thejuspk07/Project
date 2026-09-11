# LSTM Multi-Sensor Anomaly Detection

An AI-powered predictive-maintenance project that combines **LSTM autoencoders**, multi-sensor motor data, Arduino-based data collection, and Remaining Useful Life (RUL) prediction using the NASA CMAPSS turbofan dataset.

## Overview

This repository contains two related workflows:

1. **Real-time motor anomaly detection** using sensor data collected from an Arduino setup.
2. **NASA CMAPSS RUL prediction** using an LSTM model to estimate remaining engine life.

The motor workflow learns normal sensor behaviour and uses reconstruction error to identify abnormal operating conditions.

## Features

- LSTM autoencoder for motor anomaly detection
- Real-time sensor streaming through Arduino
- MPU6050 and other sensor inputs
- Dataset collection and labelling utilities
- Pre-trained motor anomaly model
- Real-time anomaly detection scripts
- Streamlit dashboard for monitoring
- NASA CMAPSS-based RUL prediction
- Pre-trained LSTM RUL models
- Saved feature scaler for inference

## Project Structure

```text
Project/
├── arduino_stream_test.py       # Test Arduino serial streaming
├── collect_dataset.py           # Collect motor sensor data
├── dashboard.py                 # Streamlit monitoring dashboard
├── g.ipynb                      # Motor anomaly model notebook
├── motor_anomaly_lstm.keras     # Trained motor LSTM model
├── motor_monitor.py             # Motor monitoring script
├── motor_normal_data.csv        # Normal motor data
├── motor_normal_labeled.csv     # Labelled motor data
├── nasa.ipynb                   # NASA CMAPSS RUL notebook
├── nasa_lstm_rul_model.h5       # RUL model (H5)
├── nasa_lstm_rul_model.keras    # RUL model (Keras)
├── real_time_detector.py        # Real-time detector
├── requirement.txt              # Python dependencies
├── scaler.save                   # Saved preprocessing scaler
└── sensor_dataset.csv            # Sensor dataset
```

## How It Works

```text
Arduino Sensors
      │
      ▼
Sensor Data Collection
      │
      ▼
Preprocessing / Scaling
      │
      ▼
LSTM Autoencoder
      │
      ▼
Reconstruction Error
      │
 ┌────┴────┐
 ▼         ▼
Normal   Anomaly
 │         │
 ▼         ▼
Dashboard / Alert
```

For the RUL workflow:

```text
NASA CMAPSS Data
      │
      ▼
Preprocessing
      │
      ▼
Sequence Generation
      │
      ▼
LSTM Model
      │
      ▼
Remaining Useful Life
```

## Installation

Clone the repository:

```bash
git clone https://github.com/thejuspk07/Project.git
cd Project
```

Install dependencies:

```bash
pip install -r requirement.txt
```

## Run the Motor Monitor

Test the Arduino connection first:

```bash
python arduino_stream_test.py
```

Collect sensor data when required:

```bash
python collect_dataset.py
```

Run the real-time detector:

```bash
python real_time_detector.py
```

or:

```bash
python motor_monitor.py
```

## Launch the Dashboard

```bash
streamlit run dashboard.py
```

## Model Training

The notebooks contain the main experimentation and training workflows:

- `g.ipynb` — motor anomaly detection
- `nasa.ipynb` — NASA CMAPSS RUL prediction

Pre-trained models are included for experimentation without retraining from scratch.

## Technologies

- Python
- TensorFlow / Keras
- LSTM Autoencoders
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Arduino
- MPU6050

## Notes

This repository is a research and development project for demonstrating AI-assisted condition monitoring and predictive maintenance. Model performance depends on the quality of the training data, sensor setup, preprocessing, and operating conditions.

## Author

**Thejus P. K.**  
GitHub: https://github.com/thejuspk07
