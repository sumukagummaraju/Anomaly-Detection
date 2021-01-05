# Anomaly-Detection

Anomaly Detection plays the first step in condition monitoring and  predictive maintenance of industrial equipment in process industries. Industrial equipment includes motors, pumps and pipelines. The most predominant method is vibration monitoring. Sensor data from industrial equipment are explored to identify potential anomalies.

The project consists of 2 steps:
1. Data Engineering - Performing ETL on sensor data
2. Anomaly Detection - Training neural networks (unsupervised) to learn the behavior of specific equipment.

Two algorithms used for this purpose are Self-Organizing Maps (SOMs) and Autoencoders (AE).
The resulting models can classify between anomalous and normal data with high confidence.
