import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

# Load your data
df_encoded = pd.read_csv("data/onlinefoods_encoded.csv")

# Define a function to calculate Cramér's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

# Initialize a DataFrame to store Cramér's V values
cramers_v_matrix = pd.DataFrame(index=df_encoded.columns, columns=df_encoded.columns)

# Calculate Cramér's V for each pair of features
for col1 in df_encoded.columns:
    for col2 in df_encoded.columns:
        if col1 != 'Feedback' and col2 != 'Feedback':
            cramers_v_matrix.loc[col1, col2] = cramers_v(df_encoded[col1], df_encoded[col2])

# Convert the matrix to float
cramers_v_matrix = cramers_v_matrix.astype(float)

# Plot the Cramér's V heatmap
# plt.figure(figsize=(12, 12))
sns.heatmap(cramers_v_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Cramér's V Heatmap")
plt.show()

# Optional: Drop one feature from pairs of features with high Cramér's V
threshold = 0.25  # Set a threshold for high association
to_drop = set()
for col1 in cramers_v_matrix.columns:
    for col2 in cramers_v_matrix.columns:
        if col1 != col2 and cramers_v_matrix.loc[col1, col2] > threshold:
            to_drop.add(col2)

X_final = df_encoded.drop(columns=to_drop)
X_final = X_final.drop(columns='Feedback')

print("Features dropped due to high Cramér's V:", to_drop)
print("Remaining features:", X_final.columns)
