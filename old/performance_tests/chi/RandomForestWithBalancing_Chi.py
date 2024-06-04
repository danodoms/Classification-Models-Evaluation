import pandas as pd

#load csv
df = pd.read_csv('data/onlinefoods.csv')
df.head()

#Checking data imbalances
#split to X(Predictors) and Y(Target)
x = df.drop(['Feedback', 'Pin code', 'Gender', 'Family size'], axis=1)
y = df['Feedback']

#count instances for every category
y.value_counts()

#display chart of Y
y.value_counts().plot.pie(autopct='%.2f')

#Peform data balancing using over-sampling (SMOTE)
#so convert categorical to numeric values
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
# for col_index in [0, 1,2,3,4,5,7]:
for col_index in [1,2,3,4,5]:
    label_encoders[col_index] = LabelEncoder()
    x.iloc[:, col_index] = label_encoders[col_index].fit_transform(x.iloc[:, col_index])

# Perform SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_smote, y_smote = smote.fit_resample(x, y)
x_smote = pd.DataFrame(x_smote).round().astype(int)


# Count instances for every category after SMOTE
y_smote_value_counts = pd.Series(y_smote).value_counts()

# Display chart of Y after SMOTE
import matplotlib.pyplot as plt
y_smote_value_counts.plot.pie(autopct='%.2f')
plt.title('Distribution of Feedback after SMOTE')
plt.show()

#Tranfer the x_smote and y_smote to inputs and targets
inputs = x_smote
target = y_smote

# Convert ages to categorical ranges
# Define age categories
bins = [17, 22, 26, 29, 34]
#labels = ['15-24', '25-34', '35-44', '45-54', '55-64', '65-75']
labels = ['1', '2', '3', '4']

# Convert ages to categories
inputs['Age'] = pd.cut(inputs['Age'], bins=bins, labels=labels, right=False)

# # Convert Na_to_K to categorical ranges
# # Define categories for Na_to_K
# bins_na_to_k = [560000, 560025, 560050, 560075, 560100, 560125]  # Include one additional bin edge for values above 13
# #labels_na_to_k = ['0-9', '10-10.9', '11-11.9', '12-12.9', '13 and above']
# labels_na_to_k = ['1', '2', '3', '4', '5']

# # Convert Na_to_K values to categories
# inputs['Pin code'] = pd.cut(inputs['Pin code'], bins=bins_na_to_k, labels=labels_na_to_k, right=False)

# # Convert latitude to categories
# bins_latitude = [12.8, 12.9, 13.0, 13.1, 13.2]
# labels_latitude = ['1', '2', '3', '4']
# inputs['latitude'] = pd.cut(inputs['latitude'], bins=bins_latitude, labels=labels_latitude, right=False, precision=4)


# # Convert latitude to categories
# bins_longitude = [float('-inf'), 77.4, 77.5, 77.6, 77.7, 77.8, float('inf')]
# labels_longitude = ['1', '2', '3', '4', '5', '6']
# inputs['longitude'] = pd.cut(inputs['longitude'], bins=bins_longitude, labels=labels_longitude, right=False, precision=6)


#Convert target to numbers
from sklearn.preprocessing import LabelEncoder
le_Drug = LabelEncoder() 
target = le_Drug.fit_transform(target)

#Build a model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Build the Random Forest Model
# Create an instance of the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X = inputs
y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Random Forest model to the training data
model.fit(X_train, y_train)

# Step 6: Make Predictions
# Use the trained model to make predictions on the testing data
y_pred = model.predict(X_test)

#Get the accuracy of the model  
# Evaluate the performance of the model using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Rating:", accuracy)

#Get the F1 Score of the model
from sklearn.metrics import f1_score

# Calculate the F1-score
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1-score:", f1)

#Get the precision score
from sklearn.metrics import precision_score

# Calculate the precision score
precision = precision_score(y_test, y_pred, average='weighted')
print("Precision Score:", precision)

#Show the confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

#Calculate the Mean Squares Error
import numpy as np
from sklearn.metrics import mean_squared_error

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

#Calculate the Mean Absolute error
# Import the necessary library
from sklearn.metrics import mean_absolute_error

# Calculate the Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

#visualize the decision tree
from sklearn.tree import export_graphviz
import graphviz

# Convert y to a pandas Series
y_series = pd.Series(y)

# Use the unique() method on the pandas Series
class_names = y_series.unique()

# Choose one of the decision trees from the Random Forest (e.g., the first tree)
tree = model.estimators_[0]

# Convert feature names to strings
feature_names = X.columns.astype(str)

# Convert class names to strings
class_names = class_names.astype(str)

# Export the decision tree as a DOT file
dot_data = export_graphviz(tree, out_file=None, 
                           feature_names=feature_names,  
                           class_names=class_names,  
                           filled=True, rounded=True,  
                           special_characters=True)

tree_index = 0 
#'C:/Program Files/Graphviz/bin/dot.exe'

# import os
from graphviz import Graph, Source

# Set the path to the 'dot' executable
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/dot.exe'

# Visualize the decision tree
graph = Source(dot_data)
graph.render("random_forest_tree_" + str(tree_index))
graph.view()  # Display the visualization


#classification report
from sklearn.metrics import classification_report
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))













