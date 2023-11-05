import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Task 1: Data Preparation
data = pd.read_csv("C:/Users/Dev/Desktop/Oasis Infobyte/Car Data Prediction/cardata.csv")

# Task 2: Data Exploration

# Effect of Number of Owners
# Use a box plot to visualize the effect of the number of owners on car price
plt.figure(figsize=(8, 6))
sns.boxplot(x='Owner', y='Selling_Price', data=data)
plt.title('Selling Price by Number of Owners')
plt.xlabel('Number of Owners')
plt.ylabel('Selling Price')
plt.show()




# Mileage vs. Price
# Use a scatter plot to visualize the relationship between mileage and Selling Price
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Driven_kms', y='Selling_Price', data=data)
plt.title('Selling Price vs. Mileage (Driven_kms)')
plt.xlabel('Mileage (Driven_kms)')
plt.ylabel('Selling Price')
plt.show()

# Year vs. Price
# Use a line chart to show the trend in price over time
plt.figure(figsize=(8, 6))
sns.lineplot(x='Year', y='Selling_Price', data=data)
plt.title('Selling Price by Year of Manufacture')
plt.xlabel('Year of Manufacture')
plt.ylabel('Selling Price')
plt.show()

# Fuel Type Comparison
# Use a grouped bar chart to compare Selling Prices for different fuel types
plt.figure(figsize=(8, 6))
sns.barplot(x='Fuel_Type', y='Selling_Price', data=data, ci=None)
plt.title('Selling Price by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Selling Price')
plt.show()


encoder = OneHotEncoder(sparse=False, drop='first')
data_encoded = encoder.fit_transform(data[['Selling_type', 'Fuel_Type', 'Transmission']])
encoded_cols = encoder.get_feature_names_out(['Selling_type', 'Fuel_Type', 'Transmission'])
data_encoded = pd.DataFrame(data_encoded, columns=encoded_cols)
data = pd.concat([data, data_encoded], axis=1)
data = data.drop(columns=['Selling_type', 'Fuel_Type', 'Transmission'])

# Task 3: Regression Modeling
X = data.drop(columns=['Selling_Price', 'Car_Name'])
y = data['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Task 4: Model Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2}")