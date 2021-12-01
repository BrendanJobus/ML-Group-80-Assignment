import numpy as np
import pandas as pd

def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()]).T.values.tolist()

def normalize(df):
    listOfMinMax = df.apply(minMax)
    index = ['Beds','Baths','Area','Price','Latitude','Longitude']
    norm = df.copy()
    for i in range(len(listOfMinMax)):
        min = listOfMinMax[i][0]
        max = listOfMinMax[i][1]
        gap = max - min
        norm[index[i]] = (norm[index[i]] - min) / gap
    return norm

data = pd.read_csv("../data/geocodedListings2.csv")
# According to the small data set, testing with removing those cols with most 0
data = data.drop(['Address', 'Construction', 'Parking'],axis=1)
# Sort data
data.sort_values('Price',ascending=False)
# Normalize data
data_norm = normalize(data)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

labels = data_norm['Price']
train = data_norm.drop(['Price'],axis=1)

x_train , x_test , y_train , y_test = train_test_split(train , labels , test_size = 0.2 ,random_state =2)

# Create a Linear Regressor
from sklearn.linear_model import LinearRegression
print('\nRunning Linear Regression:')
model = LinearRegression()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print(score)

y_pred = model.predict(x_test)

rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
rsquared_score = r2_score(y_test, y_pred)
print('RMSE score:', rmse_score)
print('R2 score:', rsquared_score)

from xgboost import XGBRegressor

# Create a XGBoost Regressor
print('\nRunning XGBoost Regression:')
model = XGBRegressor()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print(score)

y_pred = model.predict(x_test)

rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
rsquared_score = r2_score(y_test, y_pred)
print('RMSE score:', rmse_score)
print('R2 score:', rsquared_score)

from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regressor
print('\nRunning Random Forest Regression:')
model = RandomForestRegressor()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print(score)

y_pred = model.predict(x_test)

rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
rsquared_score = r2_score(y_test, y_pred)
print('RMSE score:', rmse_score)
print('R2 score:', rsquared_score)

from sklearn.ensemble import GradientBoostingRegressor

# Create a Gradient Gradient Regressor
print('\nRunning Gradient Gradient Regression:')
model = GradientBoostingRegressor()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print(score)

y_pred = model.predict(x_test)

rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
rsquared_score = r2_score(y_test, y_pred)
print('RMSE score:', rmse_score)
print('R2 score:', rsquared_score)




