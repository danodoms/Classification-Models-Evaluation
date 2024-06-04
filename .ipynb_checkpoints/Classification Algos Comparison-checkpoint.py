#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:08:06 2024

@author: danodoms
"""

# %% Load Data
import pandas as pd
import numpy as np
from scipy.stats import zscore

# Load your data
df = pd.read_csv('data/onlinefoods.csv')

# Select the column you want to analyze
column_name = 'Age'  # replace with the actual column name
data = df[column_name]

# Z-score method
def detect_outliers_zscore(data, threshold=3):
    z_scores = zscore(data.dropna())
    outliers = data[np.abs(z_scores) > threshold]
    return outliers

# IQR method
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.05)
    Q3 = data.quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

# Detect outliers using both methods
outliers_zscore = detect_outliers_zscore(data)
outliers_iqr = detect_outliers_iqr(data)

# Print outliers
print("Outliers detected using Z-score method:")
print(outliers_zscore)

print("\nOutliers detected using IQR method:")
print(outliers_iqr)

# Visualize outliers using a box plot
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.boxplot(data)
plt.title(f'Box plot of {column_name}')
plt.show()
