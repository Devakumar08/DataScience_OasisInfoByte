import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Data Preparation
data = pd.read_csv("C:/Users/Dev/Desktop/Oasis Infobyte/Sale preds/Advertising.csv")  # Replace 'your_dataset.csv' with the actual file path

# Task 2: Data Exploration
# Distribution of Sales
plt.figure(figsize=(8, 6))
sns.histplot(data['Sales'], bins=20, kde=True)
plt.xlabel('Sales')
plt.title('Distribution of Sales')
plt.show()

# Visualize the relationship between TV advertising and Sales
plt.figure(figsize=(8, 6))
sns.scatterplot(x='TV', y='Sales', data=data)
plt.title('Sales vs. TV Advertising')
plt.xlabel('TV Advertising')
plt.ylabel('Sales')
plt.show()

# Visualize the relationship between Radio advertising and Sales
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Radio', y='Sales', data=data)
plt.title('Sales vs. Radio Advertising')
plt.xlabel('Radio Advertising')
plt.ylabel('Sales')
plt.show()

# Visualize the relationship between Newspaper advertising and Sales
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Newspaper', y='Sales', data=data)
plt.title('Sales vs. Newspaper Advertising')
plt.xlabel('Newspaper Advertising')
plt.ylabel('Sales')
plt.show()

# Task 3: Visualize the relationships between advertising media and Sales using bar plots
advertising_media = ['TV', 'Radio', 'Newspaper']
sales = [data['TV'].sum(), data['Radio'].sum(), data['Newspaper'].sum()]

plt.figure(figsize=(8, 6))
sns.barplot(x=advertising_media, y=sales)
plt.xlabel('Advertising Medium')
plt.ylabel('Total Sales')
plt.title('Total Sales by Advertising Medium')
plt.show()


correlation_sales_tv = data['Sales'].corr(data['TV'])
correlation_sales_radio = data['Sales'].corr(data['Radio'])
correlation_sales_newspaper = data['Sales'].corr(data['Newspaper'])

# Create a grouped bar plot to visualize the correlation between Sales and different types of advertising
plt.figure(figsize=(10, 6))
advertising_types = ['TV Advertising', 'Radio Advertising', 'Newspaper Advertising']
correlations = [correlation_sales_tv, correlation_sales_radio, correlation_sales_newspaper]

sns.barplot(x=advertising_types, y=correlations)
plt.ylabel('Correlation')
plt.title('Correlation between Sales and Different Types of Advertising')
plt.show()