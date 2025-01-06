ML For Green Usage of Computing Devices Classifier

This project implements machine learning models to classify the power modes of computing devices to promote green and efficient usage. The models are trained on a dataset (exp1.csv) with features that represent device attributes and three power mode classes.

Features
Dataset: exp1.csv containing features of computing devices and their respective power modes.
Models Implemented:
Support Vector Machine (SVM)
Decision Tree Classifier
Linear Regression (adapted for classification)
Neural Network (NN) using TensorFlow/Keras
Key Functionality:
Preprocessing: Label encoding, standardization, and dataset splitting.
Visualization: Confusion matrices and decision boundaries (where applicable).
Evaluation: Classification reports, confusion matrices, and training history visualization for Neural Networks.
Code Structure
svm_classifier.py: Implements an SVM for power mode classification.
decision_tree_classifier.py: Implements a Decision Tree Classifier for power mode classification.
linear_regression_classifier.py: Uses Linear Regression for class prediction (adapted for multi-class classification).
neural_network_classifier.py: Implements a multi-layer perceptron (NN) for power mode classification.
