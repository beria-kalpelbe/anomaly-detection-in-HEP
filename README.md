# anomaly-detection-in-HEP
Anomaly Detection In High Energy Physics Data

![cover image](https://www.openaccessgovernment.org/wp-content/uploads/2019/07/dreamstime_m_85270302.jpg)

## Overview
This study aims to develop a Machine Learning model to detect anomalous events in HighEnergy Physics data. Despite extensive research in this area, experimental physicists still struggle to manage the huge amount of data collected at the Large Hadron Collider. Statistical tools like Z-scores are still widely used by experimental physicists. Many years ago, ML techniques demonstrated their ability to detect anomalies in various types of data, including HEP data. In this study, both supervised techniques (decision trees and multi-layer perceptron) and unsupervised techniques (autoencoders) are explored to detect anomalies in HEP data. The supervised learning methods are utilized to learn the characteristics that differentiate signal events from background events. On the other hand, unsupervised methods learn the distribution that generated background events in order to detect signals when they appear to deviate from the background distribution.

## 1 - Data Preprocessing <a href="https://colab.research.google.com/drive/1k8eytd5Yd-7VP2uhtrjzd47ZK6AHKjSM" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
In order to prepare the High Energy Physics (HEP) data for anomaly detection, several preprocessing steps are performed. These steps include:
- Data extraction
- Feature scaling
- Data transformations
- Data splitting
- Save the datasets in a .h5 file

## 2 - Supervised anomaly detection
### 2.1 - Decision Trees (DT) <a href="https://colab.research.google.com/drive/1jbiMsVrCthzuic9FspT8FYdy8a8qd7xU" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Decision Trees are a supervised learning algorithm used for classification and regression tasks. In the context of anomaly detection in HEP data, decision trees can be trained to learn the characteristics that differentiate signal events from background events. 

### 2.2 - Multi-Layer Perceptron (MLP)
Multi-Layer Perceptron is a type of artificial neural network that consists of multiple layers of interconnected nodes. MLPs have been widely used for various machine learning tasks, including anomaly detection. 

## 3 - Unsupervised anomaly detection
### 3.1 - Autoencoders (AE)
Autoencoders are unsupervised learning models that are commonly used for dimensionality reduction and anomaly detection. In the context of anomaly detection in HEP data, autoencoders can be trained to learn the distribution that generated background events and detect signals when they deviate from the background distribution. 

### 3.2 - Variational Autoencoders (VAE)
