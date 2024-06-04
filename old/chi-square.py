import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

# Load CSV

df_encoded = pd.read_csv("data/onlinefoods_encoded.csv")

isnull = df_encoded.isnull().sum()

X = df_encoded.drop('Feedback', axis=1)
Y = df_encoded['Feedback']

# Perform chi-square test for each feature
feature_significance = {}
for col in X:
    contingency_table = pd.crosstab(X[col], Y)
    chi2, p_val, _, _ = chi2_contingency(contingency_table)
    feature_significance[col] = {"Significance": "Significant" if p_val < 0.05 else "Not Significant",
                                  "p-value": p_val}

# Print significance and p-values of each feature
print("\nFeature Significance and p-values:")
for feature, values in feature_significance.items():
    print(f"{feature}: {values['Significance']}, p-value: {values['p-value']}")

# Create DataFrame to store chi-square test results
chi2_df = pd.DataFrame(feature_significance).T
chi2_df.index.name = 'Feature'
chi2_df.reset_index(inplace=True)

# Display chi2_df in the variable explorer
chi2_df

# Sort the DataFrame by "Significance" column
chi2_df_sorted = chi2_df.sort_values(by='p-value', ascending=True)

# Drop not significant features
significant_features = [feat for feat, values in feature_significance.items() if values["Significance"] == "Significant"]
df_filtered = df_encoded[significant_features + ['Feedback']]

# Save filtered data to CSV
df_filtered.to_csv('filtered_onlinefoods.csv', index=False)
