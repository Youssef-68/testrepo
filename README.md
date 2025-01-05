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
