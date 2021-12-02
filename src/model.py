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
        min = listOfMinMax[i].iloc[0]
        max = listOfMinMax[i].iloc[1]
        gap = max - min
        norm[i] = (norm[i] - min) / gap
    return norm
#Cross validation for choosing polynomial features
def kfold_polynomials(X, y):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    mean_error=[]; std_error=[]
    q_range = range(1,11)
    for q in q_range:
        from sklearn.preprocessing import PolynomialFeatures
        Xpoly = PolynomialFeatures(q).fit_transform(X)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        temp=[]
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            from sklearn.metrics import mean_squared_error
            temp.append(mean_squared_error(y[test],ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.title('Polynomial features degree choice')
    plt.errorbar(q_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel('q')
    plt.ylabel('Mean square error')
    plt.show()

def kfold_Lasso(X, y, poly):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    mean_error=[]; std_error=[]
    from sklearn.preprocessing import PolynomialFeatures
    Xpoly = PolynomialFeatures(poly).fit_transform(X)
    c_range = [0.0001, 0.001, 0.01, 1, 5, 100, 1000, 10000]
    for c in c_range:
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=c)
        temp=[]
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            from sklearn.metrics import mean_squared_error
            temp.append(mean_squared_error(y[test],ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.title('Lasso regression alpha choice')
    plt.errorbar(c_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel('c')
    plt.ylabel('Mean square error')
    plt.show()

def kfold_Ridge(X, y, poly):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    mean_error=[]; std_error=[]
    from sklearn.preprocessing import PolynomialFeatures
    Xpoly = PolynomialFeatures(poly).fit_transform(X)
    c_range = [0.0001, 0.001, 0.01, 1, 5, 100, 1000, 10000]
    for c in c_range:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1/(2*c))
        temp=[]
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            from sklearn.metrics import mean_squared_error
            temp.append(mean_squared_error(y[test],ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.title('Ridge regression alpha choice')
    plt.errorbar(c_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel('c')
    plt.ylabel('Mean square error')
    plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



data = pd.read_csv("data/geocodedListings2.csv")
# According to the small data set, testing with removing those cols with most 0
data = data.drop(['Address', 'Construction', 'Parking'],axis=1)
# Sort data
data.sort_values('Price',ascending=False)
# Normalize data
data_norm = normalize(data)

labels = data_norm['Price']
train = data_norm.drop(['Price'],axis=1)

kfold_polynomials(train, labels)

kfold_Lasso(train, labels, 7)

kfold_Ridge(train, labels, 7)

from sklearn.dummy import DummyRegressor
dummy = DummyRegressor(strategy='mean').fit(train, labels)
ydummy = dummy.predict(train)
print('Dummy RMSE:' + str(np.sqrt(mean_squared_error(labels, ydummy))))

x_train , x_test , y_train , y_test = train_test_split(train , labels , test_size = 0.2 ,random_state =2)

# Create a Linear Regressor

print('\nRunning Linear Regression:')
from sklearn.linear_model import LinearRegression
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




