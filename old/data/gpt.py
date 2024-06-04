import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# %% Section1 

# Step 1: Load the Data
data = pd.read_csv('onlinefoods.csv')

# Step 2: Exploratory Data Analysis (EDA)
print(data.info())
print(data.describe())

# Step 3: Outlier Detection and Removal
def detect_outliers(df, n, features):
    outlier_indices = []
    
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    
    return multiple_outliers

# %% Section2


# Assume 'Feedback' is the Feedback column name; replace with your actual Feedback column
features = data.drop(columns=['Feedback']).columns
outliers_to_drop = detect_outliers(data, 2, features)
data = data.drop(outliers_to_drop, axis=0).reset_index(drop=True)

# Step 4: Handle Missing Values
data = data.fillna(data.mean())

# Step 5: Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop('Feedback', axis=1))
data_scaled = pd.DataFrame(scaled_data, columns=data.columns[:-1])
data_scaled['Feedback'] = data['Feedback']

# Step 6: SMOTE
X = data_scaled.drop('Feedback', axis=1)
y = data_scaled['Feedback']

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Step 7: Model Training and Evaluation
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC()
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model_name} Classification Report")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")

# Step 8: Visualizations
# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Model performance comparison
model_names = list(models.keys())
accuracies = [accuracy_score(y_test, models[model].predict(X_test)) for model in model_names]

plt.figure(figsize=(10,6))
sns.barplot(x=model_names, y=accuracies)
plt.title('Model Performance Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()
