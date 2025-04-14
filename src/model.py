from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def train_knn_model(X_train, y_train, n_neighbors=5, weights='uniform', metric='euclidean'):
    """
    Train a K-Nearest Neighbors model with specified parameters.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    n_neighbors : int, default=5
        Number of neighbors to use
    weights : str, default='uniform'
        Weight function used in prediction
    metric : str, default='euclidean'
        Distance metric to use
        
    Returns:
    --------
    model : KNeighborsClassifier
        Trained KNN model
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
    model.fit(X_train, y_train)
    return model

def train_random_forest_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest Classifier model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    n_estimators : int, default=100
        Number of trees in the forest
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    model : RandomForestClassifier
        Trained Random Forest model
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def tune_knn_hyperparameters(X_train, y_train, cv=5):
    """
    Tune KNN hyperparameters using GridSearchCV.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    best_model : KNeighborsClassifier
        Best KNN model after hyperparameter tuning
    best_params : dict
        Best hyperparameters found
    """
    knn = KNeighborsClassifier()
    
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }
    
    grid_search = GridSearchCV(
        estimator=knn, 
        param_grid=param_grid, 
        scoring='accuracy', 
        cv=cv,
        verbose=1, 
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model to evaluate
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
        
    Returns:
    --------
    y_pred : numpy.ndarray
        Predicted labels
    report : str
        Classification report
    conf_matrix : numpy.ndarray
        Confusion matrix
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return y_pred, report, conf_matrix
