import pandas as pd

def find_duplicate_columns(df: pd.DataFrame) -> list:
    """
    Identify and return the names of duplicate columns in a DataFrame.
    
    Args:
    df (pd.DataFrame): The DataFrame to check for duplicate columns.
    
    Returns:
    list or str: A list of column names that are duplicates or a string message if no duplicates are found.
    """
    duplicates = []
    column_dict = {}
    
    # Loop through each column in the DataFrame
    for col in df.columns:
        # Convert the column to a tuple of values (to make it hashable)
        column_tuple = tuple(df[col])
        
        # Check if the tuple already exists in the dictionary
        if column_tuple in column_dict:
            # If it exists, add the column name to duplicates list
            duplicates.append(col)
        else:
            # If it doesn't exist, add the tuple to the dictionary
            column_dict[column_tuple] = col
    
    # Check if any duplicates were found
    if not duplicates:
        return "No duplicate columns found"
    else:
        return duplicates

# Example usage:
# df = pd.DataFrame({
#     'A': [1, 2, 3],
#     'B': [1, 2, 3],
#     'C': [4, 5, 6],
#     'D': [1, 2, 3]
# })
# duplicate_columns = find_duplicate_columns(df)
# print(duplicate_columns)  # Output: ['B', 'D']






import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def detect_outliers(df: pd.DataFrame, column_name: str, lower_quantile: float = 0.05, upper_quantile: float = 0.95) -> pd.Series:
    """
    Detect outliers in a specified column of a DataFrame using the IQR method and visualize them.

    Args:
    df (pd.DataFrame): The DataFrame to analyze.
    column_name (str): The name of the column to check for outliers.
    lower_quantile (float): Lower quantile to calculate IQR (default is 0.05).
    upper_quantile (float): Upper quantile to calculate IQR (default is 0.95).

    Returns:
    pd.Series: A series containing the outliers.
    """
    
    selected_data = df[column_name]

    def detect_outliers_iqr(data):
        Q1 = data.quantile(lower_quantile)
        Q3 = data.quantile(upper_quantile)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return outliers

    # Detect outliers using IQR method
    outliers_iqr = detect_outliers_iqr(selected_data)

    # Print outliers
    print(f"\nOutliers detected using IQR method in column '{column_name}':")
    print(outliers_iqr)

    # Visualize outliers using a box plot
    plt.figure(figsize=(10, 5))
    sns.boxplot(selected_data)
    plt.title(f'Box plot of {column_name}')
    plt.show()

    return outliers_iqr

# Example usage:
# df = pd.DataFrame({
#     'Age': [23, 45, 12, 67, 34, 89, 23, 45, 67, 23, 45, 12, 120]
# })
# outliers = detect_outliers(df, 'Age')
# print(outliers)




def count_missing_values(df):
    """
    Count the number of missing values for each column in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
    pd.DataFrame: A DataFrame with column names and the count of missing values.
    """
    # Initialize an empty list to store the results
    missing_values_list = []
    
    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Count the number of missing values in the column
        missing_count = df[column].isnull().sum()
        
        # Append the column name and missing count to the list
        missing_values_list.append({'Column': column, 'Missing Values': missing_count})
    
    # Convert the list of dictionaries to a DataFrame
    missing_values_df = pd.DataFrame(missing_values_list)
    
    return missing_values_df