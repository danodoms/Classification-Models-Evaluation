import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv('data/onlinefoods.csv')

# Checking data imbalances
x = df.drop(['Feedback', 'Age'], axis=1)
y = df['Feedback']

# Perform data balancing using over-sampling (SMOTE)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

label_encoders = {}
for col_index in [0, 1, 2, 3, 4, 5, 7]:
    label_encoders[col_index] = LabelEncoder()
    x.iloc[:, col_index] = label_encoders[col_index].fit_transform(x.iloc[:, col_index])

smote = SMOTE(random_state=42)
x_smote, y_smote = smote.fit_resample(x, y)
x_smote = pd.DataFrame(x_smote).round().astype(int)

# Transfer the x_smote and y_smote to inputs and targets
inputs = x_smote
target = y_smote

# # Convert ages to categorical ranges
# bins = [17, 22, 26, 29, 34]
# labels = ['1', '2', '3', '4']
# inputs['Age'] = pd.cut(inputs['Age'], bins=bins, labels=labels, right=False)

# Convert Pin code to categorical ranges
bins_na_to_k = [560000, 560025, 560050, 560075, 560100, 560125]
labels_na_to_k = ['1', '2', '3', '4', '5']
inputs['Pin code'] = pd.cut(inputs['Pin code'], bins=bins_na_to_k, labels=labels_na_to_k, right=False)

# Convert target to numbers
le_Drug = LabelEncoder() 
target = le_Drug.fit_transform(target)

# Build a Support Vector Machine model
model = SVC(kernel='linear', random_state=42)  # You can change the kernel as per your requirement

# Train the model
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Get the accuracy of the model  
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Rating:", accuracy)

# Get the F1 Score of the model
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)

# Get the precision score
precision = precision_score(y_test, y_pred, average='weighted')
print("Precision Score:", precision)

# Show the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Calculate the Mean Squares Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate the Mean Absolute error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Visualization of the decision boundary
def plot_decision_boundary(X, y, model):
    # Select numeric features
    numeric_features = X.select_dtypes(include=[np.number])

    # Check if there are at least two numeric features
    if numeric_features.shape[1] < 2:
        print("Warning: Less than two numeric features found for visualization. Cannot plot decision boundary.")
        return

    # Plot decision boundary
    h = .02  # step size in the mesh
    x_min, x_max = numeric_features.iloc[:, 0].min() - 1, numeric_features.iloc[:, 0].max() + 1
    y_min, y_max = numeric_features.iloc[:, 1].min() - 1, numeric_features.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(numeric_features.iloc[:, 0], numeric_features.iloc[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

# Convert data type of "Age" and "Na_to_K" columns to numeric
# X_train['Age'] = pd.to_numeric(X_train['Age'])
# X_train['Na_to_K'] = pd.to_numeric(X_train['Na_to_K'])

# Plot decision boundary
plt.figure(figsize=(10, 6))
# plot_decision_boundary(X_train[['Age', 'Na_to_K']], y_train, model)
plt.title("Decision Boundary of Support Vector Machine")
# plt.xlabel("Age")
# plt.ylabel("Na_to_K")
plt.show()
