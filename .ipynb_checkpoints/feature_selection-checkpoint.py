import pandas as pd
from scipy import stats


def calculate_anova(df, target):
    # Define the target variable (response) and get unique levels
    levels = df.columns[df.columns != target]

    # Calculate ANOVA for each feature
    results = []
    for factor in levels:
        # Calculate the mean for each level of the factor
        means = {level: df[df[factor] == level][target].mean() for level in df[factor].unique()}
        
        # Calculate degrees of freedom
        DFB = len(df[factor].unique()) - 1
        DFW = len(df) - len(df[factor].unique())
        
        # Calculate sum of squares within groups (SSW)
        SSW = sum((df[df[factor] == level][target] - means[level]).pow(2).sum() for level in df[factor].unique())
        
        # Calculate sum of squares between groups (SSB)
        SSB = sum([(means[level] - df[target].mean())**2 * len(df[df[factor] == level]) for level in df[factor].unique()])
        
        # Calculate between-group variance and within-group variance
        BG_Variance = SSB / DFB
        WG_Variance = SSW / DFW
        
        # Calculate F-value
        F_value = BG_Variance / WG_Variance
        
        # Calculate p-value
        p_value = 1 - (stats.f.cdf(F_value, DFB, DFW))
        
        # Determine if the result is significant
        remark = "Significant" if p_value < 0.05 else "Not Significant"
        
        # Store results
        results.append({
            'Feature': factor,
            'F-Value': F_value,
            'p-value': "%f" % p_value,
            'Remark': remark
        })
    
    # Create a dataframe to store the results
    anova_results = pd.DataFrame(results)
    
    return anova_results



from scipy.stats import f_oneway

def anova_using_scipy(data, target):
    """
    Performs one-way ANOVA for each feature in the dataset with respect to the target variable.

    Parameters:
    data (DataFrame): The input data where rows are samples and columns are features.
    target (array-like): The target variable.

    Returns:
    dict: A dictionary containing feature names as keys and corresponding p-values as values.
    """
    p_values = {}
    for feature in data.columns:
        groups = [data[data[feature] == category][target] for category in data[feature].unique()]
        _, p_value = f_oneway(*groups)
        p_values[feature] = p_value
    return p_values



import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

def kendall_heatmap(df):
    """
    Create a heatmap of Kendall's Tau correlation coefficients for a DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame.

    Returns:
    None
    """
    # Calculate Kendall's Tau correlation coefficients
    corr = df.corr(method='kendall')

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Kendall's Tau Correlation Heatmap")
    plt.show()
    
    return corr



from sklearn.preprocessing import LabelEncoder

def encode(df):
    """
    Apply label encoding to categorical columns in a DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame.

    Returns:
    DataFrame: DataFrame with categorical columns label encoded.
    """
    label_encoder = LabelEncoder()
    df_encoded = df.copy()  # Create a copy to avoid modifying the original DataFrame
    
    # Iterate over each column
    for column in df_encoded.columns:
        if df_encoded[column].dtype == 'object':  # Check if the column is categorical
            df_encoded[column] = label_encoder.fit_transform(df_encoded[column])  # Apply label encoding
    
    return df_encoded


def collinearity_kendall(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    # corr_matrix = dataset.corr(method=method)
    
    for i in range(len(dataset.columns)):
        for j in range(i):
            if abs(dataset.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = dataset.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr



def analyze(df, target):
    # encode df for kendall and heatmap
    encode_df = encode(df)
    
    """
    X to Y feature selection
    """
    print(calculate_anova(df.dropna(), target))
    
    # compute kendall coefficients then drop y column
    correlation_coefficients_kendall = encode_df.corr(method='kendall')[target].drop(target)
    
    
    """
    X to X feature selection
    """
    # generate heatmap to visualize collinear x features
    heatmap = kendall_heatmap(encode_df.drop(target, axis=1)) 
    
    # identify collinear x features
    corr_features = collinearity_kendall(heatmap, 0.49)


    return correlation_coefficients_kendall, corr_features



# Datasets, analyze dataset by uncommenting the line


kendall_coeff, collinear_x = analyze(pd.read_csv('data/onlinefoods_encoded.csv'), 'Feedback')

# kendall_coeff, collinear_x = analyze(pd.read_csv('data/TravelInsurancePrediction_encoded.csv'), 'TravelInsurance')



# Verification for ANOVA p value
# p_values = anova_using_scipy(pd.read_csv('shoes_price.csv'), 'Price (USD)')

# # Print p-values for each feature
# for feature, p_value in p_values.items():
#     print(f"P-value for {feature}: {p_value:.6f}")  # Adjust the number of decimal places as needed


import pandas as pd
from scipy.stats import chi2_contingency


# Convert data to DataFrame
df = pd.DataFrame(pd.read_csv('data/onlinefoods_encoded.csv'))

# Perform chi-square test
chi2, p, dof, expected = chi2_contingency(df)

# Output results
print("Chi-square statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:")
print(expected)


