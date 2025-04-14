import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def load_data(file_path="data/Iris.csv"):
    """
    Load the Iris dataset from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the Iris dataset
        
    Returns:
    --------
    pandas.DataFrame
        The loaded dataset
    """
    return pd.read_csv(file_path)

def explore_data(df):
    """
    Print basic exploratory analysis of the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to explore
    """
    print(df.head())
    print(df.info())
    print(df.describe())
    
    # Check for missing values
    print(df.isnull().sum())
    
def preprocess_data(df):
    """
    Preprocess the dataset by splitting into features and target.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to preprocess
        
    Returns:
    --------
    X : pandas.DataFrame
        Features matrix
    y : pandas.Series
        Target variable
    """
    X = df.drop("Species", axis=1)
    y = df["Species"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features matrix
    y : pandas.Series
        Target variable
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Testing features
    y_train : pandas.Series
        Training target
    y_test : pandas.Series
        Testing target
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def plot_data(df):
    """
    Plot the Iris dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to plot
    """
    sns.pairplot(df, hue="Species")
    plt.show()
    
