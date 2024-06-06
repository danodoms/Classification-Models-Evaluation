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










# # Define a function to calculate lower and upper limits for outliers based on percentiles
# def calculate_outlier_limits(df, columns, lower_percentile=0.05, upper_percentile=0.95):
#     outlier_limits = {}
#     for column in columns:
#         lower_limit = df[column].quantile(lower_percentile)
#         upper_limit = df[column].quantile(upper_percentile)
#         outlier_limits[column] = (lower_limit, upper_limit)
#     return outlier_limits


# age_index = df.columns.get_loc("age")
# agl_index = df.columns.get_loc("avg_glucose_level")
# bmi_index = df.columns.get_loc("bmi")

# # Store the indices in a variable
# #bmi_indices = [bmi_index]

# # Get the column names of Age using their indices
# columns_to_process = [df.columns[age_index],df.columns[agl_index], df.columns[bmi_index]]


# # Calculate outlier limits for Age column
# outlier_limits = calculate_outlier_limits(df, columns_to_process)

# # Detect and store outliers in a separate variable, retaining the original index
# outliers = pd.DataFrame()
# for column in columns_to_process:
#     lower_limit, upper_limit = outlier_limits[column]
#     column_outliers = df[(df[column] < lower_limit) | (df[column] > upper_limit)]
#     outliers = pd.concat([outliers, column_outliers])

# # Retain the original index of the outliers DataFrame
# print(outliers)

# #  original indices of the outliers
# outlier_indices = outliers.index
# print("Indices of outliers:", outlier_indices)




def detect_outliers_and_plot(df, column_name, lower_limit=0.05, upper_limit=0.95):
    """
    Detect outliers in a specified column of a DataFrame and plot a boxplot.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to check for outliers.
    lower_limit (float, optional): The lower bound for outliers. If None, it will be calculated using IQR.
    upper_limit (float, optional): The upper bound for outliers. If None, it will be calculated using IQR.

    Returns:
    pd.Series: A boolean Series indicating whether each element in the column is an outlier.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    column_data = df[column_name]
    
    if lower_limit is None or upper_limit is None:
        Q1 = column_data.quantile(0.25)
        Q3 = column_data.quantile(0.75)
        IQR = Q3 - Q1
        if lower_limit is None:
            lower_limit = Q1 - 1.5 * IQR
        if upper_limit is None:
            upper_limit = Q3 + 1.5 * IQR

    # Detect outliers
    outliers = (column_data < lower_limit) | (column_data > upper_limit)

    # Print row numbers of detected outliers
    outlier_indices = df.index[outliers].tolist()
    print(f"Outliers detected at row numbers: {outlier_indices}")

    # Plotting the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=column_data)
    plt.axhline(lower_limit, color='r', linestyle='--', label=f'Lower Limit ({lower_limit})')
    plt.axhline(upper_limit, color='g', linestyle='--', label=f'Upper Limit ({upper_limit})')
    plt.title(f'Boxplot of {column_name}')
    plt.legend()
    plt.show()

    return outliers







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





def check_duplicate_rows(df):
    """
    Check for duplicate rows in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame containing the duplicate rows, if any.
    """
    # Find duplicate rows
    duplicates = df[df.duplicated()]

    return duplicates


def get_min_max(df, column_name):
    """
    Calculate the minimum and maximum values of a specified column and return them in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column for which to calculate the min and max values.

    Returns:
    pd.DataFrame: A DataFrame containing the minimum and maximum values of the specified column.
    """
    min_value = df[column_name].min()
    max_value = df[column_name].max()

    result_df = pd.DataFrame({
        'Statistic': ['Minimum', 'Maximum'],
        'Value': [min_value, max_value]
    })

    return result_df