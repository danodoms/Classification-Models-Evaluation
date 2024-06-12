import pandas as pd

def get_column_info(df):
    """
    Get column names, data types, non-null counts, and unique counts for the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to inspect.

    Returns:
    pandas.DataFrame: A DataFrame containing the column metadata.
    """
    column_info = {
        "Column Name": df.columns,
        "Data Type": [df[col].dtype for col in df.columns],
        "Non-Null Count": [df[col].count() for col in df.columns],
        "Unique Count": [df[col].nunique() for col in df.columns]
    }

    return pd.DataFrame(column_info)