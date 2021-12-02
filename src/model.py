import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()]).T.values.tolist()

def normalize(df):
    listOfMinMax = df.apply(minMax)
    index = ['Beds','Baths','Area','Price','Latitude','Longitude']
    norm = df.copy()
    print(listOfMinMax)
    for i in index:
        min = listOfMinMax[i][0]
        max = listOfMinMax[i][1]
        gap = max - min
        norm[i] = (norm[i] - min) / gap
    return norm

#Cross validation for choosing polynomial features
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def kfold_calculation(X, y, k, model):
    kf = KFold(n_splits=k)
    scores=[]
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        scores.append(mean_squared_error(y[test],ypred))
    return scores

def kfold_polynomials(X, y, k):
    mean_error=[]; std_error=[]
    q_range = range(1,11)
    for q in q_range:
        Xpoly = PolynomialFeatures(q).fit_transform(X)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        scores = kfold_calculation(Xpoly, y, k, model)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.title('Polynomial features degree choice')
    plt.errorbar(q_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel('q')
    plt.ylabel('Mean square error')
    plt.show()

def kfold_Lasso(X, y, k, poly):
    kf = KFold(n_splits=5)
    mean_error=[]; std_error=[]
    Xpoly = PolynomialFeatures(poly).fit_transform(X)
    c_range = [0.0001, 0.001, 0.01, 1, 5, 100, 1000, 10000]
    for c in c_range:
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=c)
        scores = kfold_calculation(Xpoly, y, k, model)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.title('Lasso regression alpha choice')
    plt.errorbar(c_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel('c')
    plt.ylabel('Mean square error')
    plt.show()

def kfold_Ridge(X, y, k, poly):
    kf = KFold(n_splits=5)
    mean_error=[]; std_error=[]
    Xpoly = PolynomialFeatures(poly).fit_transform(X)
    c_range = [0.0001, 0.001, 0.01, 1, 5, 100, 1000, 10000]
    for c in c_range:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1/(2*c))
        scores = kfold_calculation(Xpoly, y, k, model)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.title('Ridge regression alpha choice')
    plt.errorbar(c_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel('c')
    plt.ylabel('Mean square error')
    plt.show()

def kfold_XGBRegressor(X, y, k, poly):
    kf = KFold(n_splits=5)
    mean_error=[]; std_error=[]
    Xpoly = PolynomialFeatures(poly).fit_transform(X)
    c_range = [0.0001, 0.001, 0.01, 1, 5, 100, 1000, 10000]
    for c in c_range:
        from xgboost import XGBRegressor
        model = XGBRegressor(alpha=c)
        scores = kfold_calculation(Xpoly, y, k, model)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.title('XGB regression alpha choice')
    plt.errorbar(c_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel('c')
    plt.ylabel('Mean square error')
    plt.show()

def kfold_RandomForestRegressor(X, y, k, poly):
    kf = KFold(n_splits=5)
    mean_error=[]; std_error=[]
    Xpoly = PolynomialFeatures(poly).fit_transform(X)
    n_range = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for n in n_range:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=n)
        scores = kfold_calculation(Xpoly, y, k, model)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.title('Random Forest Regressor regression n_estimators choice')
    plt.errorbar(n_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel('n')
    plt.ylabel('Mean square error')
    plt.show()

def kfold_GradientBoostingRegressor(X, y, k, poly):
    kf = KFold(n_splits=5)
    mean_error=[]; std_error=[]
    Xpoly = PolynomialFeatures(poly).fit_transform(X)
    c_range = [0.0001, 0.005, 0.01, 0.05, 0.1, 0.5]
    for c in c_range:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(alpha=c)
        scores = kfold_calculation(Xpoly, y, k, model)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.title('Gradient Boosting regression alpha choice')
    plt.errorbar(c_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel('c')
    plt.ylabel('Mean square error')
    plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("../data/geocodedListings2.csv")
# According to the small data set, testing with removing those cols with most 0
data = data.drop(['Address', 'Construction', 'Parking'],axis=1)
# Sort data
data.sort_values('Price',ascending=False)
# Normalize data
data_norm = normalize(data)

labels = data_norm['Price']
train = data_norm.drop(['Price'],axis=1)

kfold_polynomials(train, labels, 5)

kfold_Lasso(train, labels, 5, 7)

kfold_Ridge(train, labels, 5, 7)

kfold_XGBRegressor(train, labels, 5, 2)

kfold_RandomForestRegressor(train, labels, 5, 7)

kfold_GradientBoostingRegressor(train, labels, 5, 7)

from sklearn.dummy import DummyRegressor
dummy = DummyRegressor(strategy='mean').fit(train, labels)
ydummy = dummy.predict(train)
print('Dummy RMSE:' + str(np.sqrt(mean_squared_error(labels, ydummy))))

x_train , x_test , y_train , y_test = train_test_split(train , labels , test_size = 0.2 ,random_state =2)

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
print('\nRunning Gradient Boosting Regression:')
model = GradientBoostingRegressor()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print(score)

y_pred = model.predict(x_test)

rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
rsquared_score = r2_score(y_test, y_pred)
print('RMSE score:', rmse_score)
print('R2 score:', rsquared_score)




