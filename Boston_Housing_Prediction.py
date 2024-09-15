import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('D:\Boston_Housing_Dataset\DataSet.csv')
print(df)

any_nulls = df.isnull().values.any()
print(any_nulls)  
df.drop('ID', axis=1, inplace=True)
print(df)

X = df.drop('medv', axis=1)
y = df['medv'] 

print(X,y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test) 
print(f"Test R² Score: {test_score:.4f}") 

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score (Test): {r2:.4f}")
