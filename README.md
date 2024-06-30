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
### 2.1 - Decision Trees <a href="https://colab.research.google.com/drive/1jbiMsVrCthzuic9FspT8FYdy8a8qd7xU" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Decision Trees are a supervised learning algorithm used for classification and regression tasks. In the context of anomaly detection in HEP data, decision trees can be trained to learn the characteristics that differentiate signal events from background events. 

### 2.2 - Multi-Layer Perceptron <a href="https://colab.research.google.com/drive/1MudoIij-ICD5RxRYaYyiEOooi8aqj0eD?authuser=1#scrollTo=F7dRgY3IjtmC" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Multi-Layer Perceptron is a type of artificial neural network that consists of multiple layers of interconnected nodes. MLPs have been widely used for various machine learning tasks, including anomaly detection. 

## 3 - Unsupervised anomaly detection
### 3.1 - Autoencoders <a href="https://colab.research.google.com/drive/1idnyU8BpXibuaV-po4UDstBV_2KRG3-E" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Autoencoders are unsupervised learning models that are commonly used for dimensionality reduction and anomaly detection. In the context of anomaly detection in HEP data, autoencoders can be trained to learn the distribution that generated background events and detect signals when they deviate from the background distribution. 

### 3.2 - Variational Autoencoders <a href="https://colab.research.google.com/drive/1XMF86VdMCQZkLJNQXps-P20JUSo0M9px" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Variational Autoencoders (VAEs) are a type of generative model that extend the basic autoencoder architecture. Unlike traditional autoencoders, which learn a deterministic mapping from input to latent space, VAEs learn a probabilistic distribution in the latent space. The encoder network learns to map the input data to the parameters of a probability distribution (typically a Gaussian distribution), and the decoder network learns to generate new samples from this latent distribution. This allows VAEs to generate new samples that are similar to the training data, making them useful for tasks such as data generation and synthesis. VAEs can also be used for anomaly detection, where samples that deviate significantly from the learned latent distribution are identified as anomalies.