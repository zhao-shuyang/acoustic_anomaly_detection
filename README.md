
# acoustic_anomaly_detection
The repo includes code that is used in DCASE challenge of acoustic anomaly detection.
- It ranks 16/40 in the challenge.
- It is the a sumbission withou using any machine learning.
- Each sound is represented by an Gaussian distribution of MFCCs
- The dissimilarity of each sound pair is measured by KL divergence
- Anomaly score is determined by the most similar normal sound of the same category

# Data
The experiment data and results of submissions are available
https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds

# Usage
1. Download the data.
2. Revise the data root in param.yaml
3. python compute_mfcc_dataset.py
4. python predict_kl.py (or predict_kl2.py, or predict_kl3.py)
5. python evaluate.py
