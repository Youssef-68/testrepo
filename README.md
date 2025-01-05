#Import necessary library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#fetech the data
data = pd.read_csv('housing_prices.csv')
print(data.isnull().sum())
data.dropna(inplace=True)

#incoding categurall variables 
data = pd.get_dummies(data, drop_first=True)

#analyze the data
sns.histplot(data['Price'], kde=True)
plt.title('Distribution of Housing Prices')
plt.show()

#correlation matrix
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#build a predicteve model
X = data.drop('Price', axis=1)
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

#liner regression model
model = LinerRegression()
model.fit(X_train, y_train)

#evaluate the mean
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

#plotting the actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
