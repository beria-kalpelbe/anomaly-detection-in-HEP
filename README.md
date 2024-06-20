# anomaly-detection-in-HEP
Anomaly Detection In High Energy Physics Data

![cover image](https://www.openaccessgovernment.org/wp-content/uploads/2019/07/dreamstime_m_85270302.jpg)

## Overview
This study aims to develop a Machine Learning model to detect anomalous events in HighEnergy Physics data. Despite extensive research in this area, experimental physicists still struggle to manage the huge amount of data collected at the Large Hadron Collider. Statistical tools like Z-scores are still widely used by experimental physicists. Many years ago, ML techniques demonstrated their ability to detect anomalies in various types of data, including HEP data. In this study, both supervised techniques (decision trees and multi-layer perceptron) and unsupervised techniques (autoencoders) are explored to detect anomalies in HEP data. The supervised learning methods are utilized to learn the characteristics that differentiate signal events from background events. On the other hand, unsupervised methods learn the distribution that generated background events in order to detect signals when they appear to deviate from the background distribution.

## Setting Up
1. Create a virtual environment
```
python3 -m venv
```
2. Install libraries
```
source venv/bin/activate
pip install -r requirements.txt
```
3. Clone the repository
```
git clone https://github.com/beria-kalpelbe/anomaly-detection-in-HEP.git
```