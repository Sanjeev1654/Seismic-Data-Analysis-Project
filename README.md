🌍 Spectral Acceleration Prediction with Bayesian Neural Network (SDA.py)
 General Idea of the Project

Implements a Bayesian Neural Network (BNN) using PyTorch to predict Spectral Acceleration (SA) values.

Input earthquake parameters include:
Earthquake Magnitude (Mw)
Joyner–Boore distance (Rjb, km) and its logarithm
Shear-wave velocity (Vs30, m/s) and its logarithm
Intra-event vs. Inter-event flag
Hypocenter depth (km)
Target outputs are Spectral Acceleration (SA) values at multiple periods (e.g., 0.01s, 0.02s, … 4.0s).

Key features:
Custom Bayesian Linear Layers for uncertainty-aware predictions
Negative Log Likelihood Loss (NLL Loss) for probabilistic training
Training, validation, and testing pipeline with scaling and stratified splitting
Uncertainty quantification (Epistemic and Aleatory uncertainty)
Sensitivity analysis plots:
SA vs. period for varying magnitudes
SA vs. period for varying distances (Rjb)
SA vs. period for varying site conditions (Vs30)

⚙️ Requirements

Programming Language: Python 3.8+

Libraries & Frameworks:

numpy (numerical computing)

pandas (data handling)

matplotlib (visualization)

scikit-learn (scaling, train/test split, metrics)

torch (PyTorch – deep learning framework)
