# anomaly-detection-in-HEP
Anomaly Detection In High Energy Physics Data

![cover image](https://www.openaccessgovernment.org/wp-content/uploads/2019/07/dreamstime_m_85270302.jpg)

## Overview
This study aims to develop a Machine Learning (ML) model to detect anomalous events in HighEnergy Physics (HEP) data. Despite extensive research in this area, experimental physicists stillstruggle to manage the huge amount of data collected at the Large Hadron Collider (LHC). Statistical tools like Z-scores are still widely used by experimental physicists. Many years ago, ML techniques demonstrated their ability to detect anomalies in various types of data, including HEP data. In this study, both supervised techniques (decision trees and multi-layer perceptron) and unsupervised techniques (autoencoders) are explored to detect anomalies in HEP data. The supervised learning methods are utilized to learn the characteristics that differentiate signal events from background events. On the other hand, unsupervised methods learn the distribution that generated background events in order to detect signals when they appear to deviate from the background distribution. The Decision Tree (DT) model is found to be the best anomaly detector with an accuracy of 72% and an Area Under Curve (AUC) of 81%. In addition, the most important track parameters in detecting signal events are mainly the transverse momentum ($p_T$), accounting for 58%, and the longitudinal impact (dz ), accounting for 23%. These results can assist experimental physicists in navigating the complexities of the LHC’s massive datasets and identifying rare or unexpected collision events that could lead to groundbreaking discoveries in the search for new physics. On the other hand, despite showing good performance in learning the background distribution, the explored unsupervised learning techniques exhibit very low performance (accuracy of 53% and an AUC of 50%). This could indicate the inadequacy of their architecture for this task. Further research could explore more complex architectures such as Generative Adversarial Network (GAN) and normalizing flows.

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
4. Run the model
```
cd anomaly-detection-in-HEP
python main.py
```