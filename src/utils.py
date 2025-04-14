import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(conf_matrix, class_names=None, cmap="Blues"):
    """
    Plot confusion matrix using seaborn heatmap.
    
    Parameters:
    -----------
    conf_matrix : numpy.ndarray
        Confusion matrix to plot
    class_names : list, default=None
        List of class names for axis labels
    cmap : str, default="Blues"
        Color map for the heatmap
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if class_names:
        plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45)
        plt.yticks(ticks=range(len(class_names)), labels=class_names, rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("Model does not have feature_importances_ attribute")

def plot_pairplot(df, hue_column):
    """
    Create a pairplot to visualize relationships between features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to plot
    hue_column : str
        Column name to use for color-coding the points
    """
    sns.pairplot(df, hue=hue_column)
    plt.tight_layout()
    plt.show()

def save_results(model, best_params, report, file_path="results.txt"):
    """
    Save model results to a text file.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    best_params : dict
        Best hyperparameters (if tuned)
    report : str
        Classification report
    file_path : str, default="results.txt"
        Path to save the results
    """
    with open(file_path, 'w') as f:
        f.write(f"Model: {type(model).__name__}\n\n")
        f.write(f"Best Parameters: {best_params}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n")
