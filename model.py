import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()]).T.values.tolist()

def normalize(df):
    listOfMinMax = df.apply(minMax)
    index = ['Beds','Baths','Area','Price'] 
    norm = df.copy()
    for i in index:
        min = listOfMinMax[i][0]
        max = listOfMinMax[i][1]
        gap = max - min
        norm[i] = (norm[i] - min) / gap
    return norm

def poly_cluster(X, q):
    cluster = pd.get_dummies(X['cluster_label'], prefix='cluster')
    X = X.drop(['cluster_label'],axis=1)
    Xpoly = PolynomialFeatures(q).fit_transform(X)
    Xpoly = np.column_stack((Xpoly,cluster.to_numpy()))
    return Xpoly

def kfold_calculation(X, y, model):
    kf = KFold(n_splits=5)
    scores=[]
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        scores.append(mean_squared_error(y[test],ypred))
    return scores

def kfold_polynomials(X, y, model, Xpolys):
    mean_error=[]; std_error=[]
    for Xpoly in Xpolys:
        scores = kfold_calculation(Xpoly, y, model)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    return mean_error, std_error

def model_polynomials(X, y):
    res = []
    models = [LinearRegression(), XGBRegressor(), RandomForestRegressor(), DummyRegressor(strategy='mean')]
    Xpolys = []
    q_range = range(1,8)
    for q in q_range:
        tmp_poly = poly_cluster(X, q)
        Xpolys.append(tmp_poly)
    for m in models:
        m_err, s_err = kfold_polynomials(X, y, m, Xpolys)
        res.append([m_err, s_err])
    colors = ['r', 'b', 'g', 'y']
    labels = ['Linear', 'XGB', 'RF', 'Dummy']
    plt.title('Polynomial features degree choice')
    for i in range(len(res)):
        plt.errorbar(q_range,res[i][0],yerr=res[i][1],linewidth=3,color=colors[i])
        plt.legend(labels[i], numpoints=1)
        plt.xlabel('q')
        plt.ylabel('Mean square error')
        plt.show()

def kfold_Linear(X, y):
    mean_error=[]; std_error=[]
    q_range = range(1,3)
    for q in q_range:
        Xpoly = PolynomialFeatures(q).fit_transform(X)
        model = LinearRegression()
        scores = kfold_calculation(Xpoly, y, model)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.title('Polynomial features degree choice')
    plt.errorbar(q_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel('q')
    plt.ylabel('Mean square error'); plt.title('linear regression')
    plt.show()

def kfold_Lasso(X, y, poly):
    kf = KFold(n_splits=5)
    mean_error=[]; std_error=[]
    Xpoly = PolynomialFeatures(poly).fit_transform(X)
    c_range = [0.0001, 0.001, 0.01, 1, 5, 10, 20, 50]
    for c in c_range:
        model = Lasso(alpha=c)
        scores = kfold_calculation(Xpoly, y, model)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.title('Lasso regression alpha choice')
    plt.errorbar(c_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel('c')
    plt.ylabel('Mean square error'); plt.title('lasso')
    plt.show()

def kfold_Ridge(X, y, poly):
    kf = KFold(n_splits=5)
    mean_error=[]; std_error=[]
    Xpoly = PolynomialFeatures(poly).fit_transform(X)
    c_range = [0.0001, 0.001, 0.01, 1, 5, 10, 20, 50]
    for c in c_range:
        model = Ridge(alpha=1/(2*c))
        scores = kfold_calculation(Xpoly, y, model)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.title('Ridge regression alpha choice')
    plt.errorbar(c_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel('c')
    plt.ylabel('Mean square error'); plt.title('ridge regression')
    plt.show()

def kfold_XGBRegressor(X, y, poly):
    kf = KFold(n_splits=5)
    mean_error=[]; std_error=[]
    Xpoly = poly_cluster(X, poly)
    c_range = [0.0001, 0.001, 0.01, 1, 5, 10, 20, 50]
    for c in c_range:
        model = XGBRegressor(alpha=c)
        scores = kfold_calculation(Xpoly, y, model)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.title('XGB regression alpha choice')
    plt.errorbar(c_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel('c')
    plt.ylabel('Mean square error'); plt.title('XGB')
    plt.show()

def kfold_RandomForestRegressor(X, y, poly):
    kf = KFold(n_splits=5)
    mean_error=[]; std_error=[]
    Xpoly = poly_cluster(X, poly)
    n_range = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for n in n_range:
        model = RandomForestRegressor(n_estimators=n)
        scores = kfold_calculation(Xpoly, y, model)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.title('Random Forest Regressor regression n_estimators choice')
    plt.errorbar(n_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel('n')
    plt.ylabel('Mean square error'); plt.title('random forest')
    plt.show()

# Going to use this to find the best 
def findBestK(coords):
    # This is the method discussed in class about finding the best k with regards to the cost function
    K = range(5, 50, 5)
    mean, std = [], []
    for k in K:
        expCoords = coords.copy()

        kmeans = KMeans(n_clusters=k, init="k-means++")
        kmeans.fit(expCoords[expCoords.columns[0:2]])
        expCoords['cluster_label'] = kmeans.fit_predict(expCoords[expCoords.columns[0:2]])

        clusterMeans = []; clusterStd = []
        for cluster in range(k):
            allClusters = expCoords.loc[expCoords['cluster_label'] == cluster]
            clusterMeans.append([cluster, allClusters['Price'].mean()])
            clusterStd.append(allClusters['Price'].std())
        std.append(sum(clusterStd)/len(clusterStd))
    plt.bar(K, std)
    plt.show()
            

    # for finding the best k, get the mean for each cluster and the std
    # std = sqrt( E(datapoint - mean) / N )

def clusterModel(features, targets):
    # Setting up the dataset to cluster the latitude and longitude
    ids = pd.Series([x for x in range(0, len(features.index))])
    features['id'] = ids.values
    coords = features.loc[:,['Latitude', 'Longitude', 'id']]
    coords['Price'] = targets
    findBestK(coords)

    kmeans = KMeans(n_clusters = 15, init = 'k-means++')
    kmeans.fit(coords[coords.columns[0:2]])
    coords['cluster_label'] = kmeans.fit_predict(coords[coords.columns[0:2]])
    centers = kmeans.cluster_centers_
    
    coords.plot.scatter(x = 'Latitude', y='Longitude', c=coords['cluster_label'], s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s = 200, alpha=0.5)
    plt.show()

    coords = coords.drop(['Latitude', 'Longitude'], axis=1)
    features = features.merge(coords, how='inner', on='id')
    features = features.drop(['id', 'Latitude', 'Longitude'], axis=1)
    features.to_csv('data/clusteredData.csv', index=None)
    return features

def model_cluster_numbers(coords, features, y):
    res = []
    models = [LinearRegression(), XGBRegressor(), RandomForestRegressor(), DummyRegressor(strategy='mean')]
    Xclustered = []
    K = range(5, 35, 5)
    for k in K:
        expFeatures = features.copy()
        expCoords = coords.copy()
        kmeans = KMeans(n_clusters=k, init="k-means++")
        kmeans.fit(expCoords[expCoords.columns[0:2]])
        expCoords['cluster_label'] = kmeans.fit_predict(expCoords[expCoords.columns[0:2]])
        expCoords = expCoords.drop(['Latitude', 'Longitude'], axis=1)
        expFeatures = expFeatures.merge(expCoords, how='inner', on='id')
        expFeatures = expFeatures.drop(['id', 'Latitude', 'Longitude'], axis=1)
        expFeatures = pd.concat([expFeatures, pd.get_dummies(expFeatures['cluster_label'], prefix='cluster')], axis=1)
        expFeatures = expFeatures.drop(['cluster_label'], axis=1)
        Xclustered.append(expFeatures.to_numpy())
    for m in models:
        m_err, s_err = kfold_polynomials(features, y, m, Xclustered)
        res.append([m_err, s_err])
    colors = ['r', 'b', 'g', 'y']
    labels = ['Linear', 'XGB', 'RF', 'Dummy']
    plt.title('Errors for different numbers of clusters')
    for i in range(len(res)):
        plt.errorbar(K,res[i][0],yerr=res[i][1],linewidth=3,color=colors[i])
        plt.legend(labels[i], numpoints=1)
        plt.xlabel('Number of clusters')
        plt.ylabel('Mean square error')
        plt.show()

def model_performance(data, labels, model):
    Xpoly = poly_cluster(data, 1)
    x_train , x_test , y_train , y_test = train_test_split(Xpoly , labels , test_size = 0.2 , random_state =42)

    model.fit(x_train,y_train)

    test_pred = model.predict(x_test)
    train_pred = model.predict(x_train)

    test_r2 = r2_score(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    print('Test R2 score:', test_r2)
    print('Train R2 score:', train_r2)

    test_mae = mean_absolute_error(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    print('Test MAE score:', test_mae)
    print('Train MAE score:', train_mae)

    test_mse = mean_squared_error(y_test, test_pred)
    train_mse = mean_squared_error(y_train, train_pred)
    print('Test MSE score:', test_mse)
    print('Train MSE score:', train_mse)

df = pd.read_csv("../data/houseListings.csv")
cluster_df = pd.read_csv("../data/clusteredData.csv")
features = cluster_df

df = df.drop(['Address', 'Construction', 'Parking'],axis=1)
# Sort data
df.sort_values('Price',ascending=False)
# Normalize data
data_norm = normalize(df)

labels = data_norm['Price']
train = features.drop(['Address', 'Construction', 'Parking'],axis=1)

print('\nRunning Linear Regression:')
model = LinearRegression()
model_performance(train, labels, model)

print('\nRunning XGBoost Regression:')
model = XGBRegressor()
model_performance(train, labels, model)

print('\nRunning Random Forest Regression:')
model = RandomForestRegressor()
model_performance(train, labels, model)