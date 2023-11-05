# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
url = "C:/Users/Dev/Desktop/Oasis Infobyte/Iris/Iris.csv" 
iris_data = pd.read_csv(url)

# Explore the dataset
print(iris_data.head())
print(iris_data.info())

# Data Visualization
# Pairplot to visualize relationships between features
plt.figure(figsize=(12, 6))
sns.pairplot(iris_data, hue="Species")
plt.show()

# Box plots to visualize feature distributions by species
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.boxplot(x="Species", y="SepalLengthCm", data=iris_data)
plt.subplot(2, 2, 2)
sns.boxplot(x="Species", y="SepalWidthCm", data=iris_data)
plt.subplot(2, 2, 3)
sns.boxplot(x="Species", y="PetalLengthCm", data=iris_data)
plt.subplot(2, 2, 4)
sns.boxplot(x="Species", y="PetalWidthCm", data=iris_data)
plt.show()

# Data Preprocessing
# Encode the target variable (Species) using LabelEncoder
le = LabelEncoder()
iris_data['Species'] = le.fit_transform(iris_data['Species'])

# Split the data into features (X) and target (y)
X = iris_data.drop('Species', axis=1)
y = iris_data['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

print(y_pred)

# Print classification report, confusion matrix, and accuracy
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

