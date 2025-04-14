# Iris Classifier

A machine learning project to classify Iris flowers based on their measurements using various classification models.

## Overview

This project implements a simple yet comprehensive machine learning pipeline for classifying Iris flowers into their respective species based on sepal and petal measurements. It demonstrates essential ML concepts including data preprocessing, exploratory data analysis, model training, hyperparameter tuning, and model evaluation.

## Features

- Data loading and preprocessing
- Exploratory data analysis with visualizations
- Implementation of multiple classification models:
  - K-Nearest Neighbors (KNN)
  - Random Forest
- Hyperparameter tuning using GridSearchCV
- Model evaluation with accuracy metrics and confusion matrices
- Visualization of model performance and feature importance

## Project Structure

```
Iris-classifier/
│
├── data/                   # Dataset directory
│   └── Iris.csv            # Iris dataset
│
├── src/                    # Source code
│   ├── data_preprocessing.py # Data loading and preprocessing functions
│   ├── model.py            # Model training and evaluation
│   └── utils.py            # Utility functions for visualization
│
├── results/                # Model evaluation results
│   ├── knn_results.txt     # Results for KNN model
│   ├── tuned_knn_results.txt # Results for tuned KNN model
│   └── rf_results.txt      # Results for Random Forest model
│
├── notebooks/              # Jupyter notebooks
│   └── exploratory_analysis.ipynb # Data exploration
│
├── main.py                 # Main script to run the pipeline
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Andrew51234/Iris-classifier.git
cd Iris-classifier
```

2. Set up a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure the Iris dataset (`Iris.csv`) is in the `data/` directory. You can download it from the following link
https://www.kaggle.com/datasets/uciml/iris

2. Run the main script:
```bash
python main.py
```

3. The script will:
   - Load and preprocess the data
   - Train KNN and Random Forest models
   - Tune hyperparameters for the KNN model
   - Evaluate all models and save results

4. Check the `results/` directory for model performance metrics.

## Models

- **K-Nearest Neighbors (KNN)**: A simple, instance-based learning algorithm.
- **Random Forest**: An ensemble learning method based on decision trees.

## Results

The project evaluates models using:
- Classification reports (precision, recall, F1-score)
- Confusion matrices
- Feature importance (for Random Forest)

## Future Improvements

- Add more classification algorithms (SVM, Neural Networks)
- Implement cross-validation strategies
- Create a web interface for real-time classification
- Add more sophisticated feature engineering techniques

## License

MIT

## Acknowledgments

- UCI Machine Learning Repository for the Iris dataset
- scikit-learn documentation and tutorials
