import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

def encode_dataframe(df):
    """
    Encode categorical columns in a DataFrame using LabelEncoder.

    Parameters:
    df (DataFrame): The input DataFrame containing categorical columns to be encoded.

    Returns:
    encoded_df (DataFrame): A new DataFrame with categorical columns encoded.
    """
    encoded_df = df.copy()  # Make a copy of the original DataFrame to avoid modifying it
    
    # Iterate through each column in the DataFrame
    for column in encoded_df.columns:
        if encoded_df[column].dtype == 'object':  # Check if the column is categorical
            label_encoder = LabelEncoder()  # Initialize LabelEncoder
            encoded_df[column] = label_encoder.fit_transform(encoded_df[column])  # Encode the column
    
    return encoded_df

def get_nan_column(df):
    """
    Get the first column with NaN values.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    column (str or bool): The name of the first column with NaN values, or False if no such column exists.
    """
    for column in df.columns:
        if df[column].isna().any():
            return column
    return False

def fill_missing_values(df):
    """
    Fill missing values in a DataFrame using linear regression.

    Parameters:
    df (DataFrame): The input DataFrame with missing values.

    Returns:
    df (DataFrame): The DataFrame with missing values filled.
    """
    df = encode_dataframe(df)  # Encode categorical columns
    
    while get_nan_column(df):  # Continue until there are no more NaN columns
        column_to_predict = get_nan_column(df)
        
        # Use only columns with no NaNs or just one column with NaNs for the model
        df_for_model = df.dropna(axis=1, how='any').join(df[[column_to_predict]])

        # Separate to train and test
        train_df = df_for_model.dropna(axis=0)
        test_df = df_for_model[df_for_model[column_to_predict].isnull()]

        # Create x and y train
        x_train = train_df.drop(column_to_predict, axis=1)
        y_train = train_df[column_to_predict]

        # Create x test
        x_test = test_df.drop(column_to_predict, axis=1)

        # Create the model
        lr = LinearRegression()
        lr.fit(x_train, y_train)

        # Apply model
        y_pred = lr.predict(x_test)

        # Assign predicted values to the original DataFrame
        df.loc[df[column_to_predict].isnull(), column_to_predict] = y_pred
    
    return df

# # Example usage
# selected_dataset = pd.read_csv('banana_disease_data_numerical.csv')
# processed_df = fill_missing_values(selected_dataset)
# processed_df.to_csv('processed_banana_disease_data.csv', index=False)