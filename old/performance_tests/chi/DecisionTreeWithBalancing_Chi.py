import pandas as pd

#load csv
df = pd.read_csv('data/onlinefoods.csv')
df.head()

#Checking data imbalances
#split to X(Predictors) and Y(Target)
x = df.drop(['Feedback', 'Age'], axis=1)
y = df['Feedback']

#count instances for every category
y.value_counts()

#display chart of Y
y.value_counts().plot.pie(autopct='%.2f')

#Peform data balancing using over-sampling (SMOTE)
#so convert categorical to numeric values
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
# for col_index in [1, 2, 3, 4, 5, 10]:
# for col_index in [1, 2, 3, 4, 5, 8]:
for col_index in [ 0, 1, 2, 3, 4, 7]:
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

# # Convert ages to categorical ranges
# # Define age categories
# bins = [15, 25, 35, 45, 55, 65, 75]
# #labels = ['15-24', '25-34', '35-44', '45-54', '55-64', '65-75']
# labels = ['1', '2', '3', '4', '5', '6']

# # Convert ages to categories
# inputs['Age'] = pd.cut(inputs['Age'], bins=bins, labels=labels, right=False)

# Convert Na_to_K to categorical ranges
# Define categories for Na_to_K
bins_na_to_k = [560000, 560025, 560050, 560075, 560100, 560125]  # Include one additional bin edge for values above 13
#labels_na_to_k = ['0-9', '10-10.9', '11-11.9', '12-12.9', '13 and above']
labels_na_to_k = ['1', '2', '3', '4', '5']

# Convert Na_to_K values to categories
inputs['Pin code'] = pd.cut(inputs['Pin code'], bins=bins_na_to_k, labels=labels_na_to_k, right=False)

# # Convert latitude to categories
# bins_latitude = [12.8, 12.9, 13.0, 13.1, 13.2]
# labels_latitude = ['1', '2', '3', '4']
# inputs['latitude'] = pd.cut(inputs['latitude'], bins=bins_latitude, labels=labels_latitude, right=False, precision=4)


# # Convert latitude to categories
# bins_longitude = [77.4, 77.5, 77.6, 77.7, 77.8]
# labels_longitude = ['1', '2', '3', '4']
# inputs['longitude'] = pd.cut(inputs['longitude'], bins=bins_longitude, labels=labels_longitude, right=False, precision=4)


#Convert target to numbers
from sklearn.preprocessing import LabelEncoder
le_Drug = LabelEncoder() 
target = le_Drug.fit_transform(target)

#Build a model
from sklearn import tree
model = tree.DecisionTreeClassifier()

# Train the model
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)

model.fit(x_train, y_train)

#Get the accuracy of the model  
# Evaluate the model on the testing set
accuracy = model.score(x_test, y_test)
print("Accuracy Rating:", accuracy)

#Get the F1 Score of the model
from sklearn.metrics import f1_score

# Predict on the testing set
predictions = model.predict(x_test)

# Calculate the F1 score
f1 = f1_score(y_test, predictions, average='weighted')
print("F1 Score:", f1)

#Get the precision score
from sklearn.metrics import precision_score

# Calculate the precision
precision = precision_score(y_test, predictions, average='weighted')
print("Precision:", precision)

#Show the confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

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
# Get the predicted probabilities for each class
probabilities = model.predict_proba(x_test)

# Get the index of the class with the highest probability for each sample
predicted_classes = np.argmax(probabilities, axis=1)

# Calculate the MSE using the predicted classes and true labels
mse = mean_squared_error(y_test, predicted_classes)
print("Mean Squared Error:", mse)

#Calculate the Mean Absolute error
# Import the necessary library
from sklearn.metrics import mean_absolute_error

# Calculate the Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

#visualize the decision tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Train the model
model.fit(x_train, y_train)

# Convert class names to a list
class_names = le_Drug.classes_.tolist()

# Visualize the decision tree
plt.figure(figsize=(200,100))
plot_tree(model, feature_names=inputs.columns.tolist(), class_names=class_names, filled=True)
plt.show()


#classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))













