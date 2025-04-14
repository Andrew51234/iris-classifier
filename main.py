import os

from src.data_preprocessing import load_data, explore_data, preprocess_data, split_data, plot_data
from src.model import train_knn_model, train_random_forest_model, tune_knn_hyperparameters, evaluate_model
from src.utils import plot_confusion_matrix, plot_feature_importance, save_results

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    dataset_path = os.path.join("data", "Iris.csv")
    
    try:
        df = load_data(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        print("Please make sure the Iris dataset CSV file is in the data directory.")
        return
    
    # Step 2: Explore data
    print("Step 2: Exploring data...")
    explore_data(df)
    
    # Step 3: Visualize data
    print("Step 3: Visualizing data...")
    plot_data(df)
    
    # Step 4: Preprocess data
    print("Step 4: Preprocessing data...")
    X, y = preprocess_data(df)
    
    # Step 5: Split data
    print("Step 5: Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=42)
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Step 6: Train models
    print("Step 6: Training models...")
    
    # KNN model
    print("Training KNN model...")
    knn_model = train_knn_model(X_train, y_train)
    knn_best_params = {"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"}
    
    # Random Forest model
    print("Training Random Forest model...")
    rf_model = train_random_forest_model(X_train, y_train)
    
    # Step 7: Hyperparameter tuning
    print("\nStep 7: Tuning hyperparameters for KNN...")
    best_knn, best_params = tune_knn_hyperparameters(X_train, y_train)
    print(f"Best parameters: {best_params}")
    
    # Step 8: Evaluate models
    print("\nStep 8: Evaluating models...")
    
    # Evaluate KNN model
    print("\nEvaluating K-Nearest Neighbors model:")
    _, knn_report, knn_conf_matrix = evaluate_model(knn_model, X_test, y_test)
    print(knn_report)
    plot_confusion_matrix(knn_conf_matrix, class_names=list(set(y)))
    
    # Evaluate tuned KNN model
    print("\nEvaluating tuned K-Nearest Neighbors model:")
    _, best_knn_report, best_knn_conf_matrix = evaluate_model(best_knn, X_test, y_test)
    print(best_knn_report)
    plot_confusion_matrix(best_knn_conf_matrix, class_names=list(set(y)))
    
    # Evaluate Random Forest model
    print("\nEvaluating Random Forest model:")
    _, rf_report, rf_conf_matrix = evaluate_model(rf_model, X_test, y_test)
    print(rf_report)
    plot_confusion_matrix(rf_conf_matrix, class_names=list(set(y)))
    
    # Plot feature importance for Random Forest
    plot_feature_importance(rf_model, X.columns.tolist())
    
    # Step 9: Save results
    print("\nStep 9: Saving results...")
    save_results(knn_model, knn_best_params, knn_report, "results/knn_results.txt")
    save_results(best_knn, best_params, best_knn_report, "results/tuned_knn_results.txt")
    save_results(rf_model, {}, rf_report, "results/rf_results.txt")
    
    print("\nAll results have been saved successfully.")
    
if __name__ == "__main__":
    main()