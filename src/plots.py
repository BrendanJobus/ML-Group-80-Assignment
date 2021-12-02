import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data = pd.read_csv('data/listing.csv')

data.plot(x='Beds', y='Price', kind='scatter')
plt.show()

data.plot(x='Baths', y='Price', kind='scatter')
plt.show()

dataAreas = data.drop(data[data.Area>10000].index)
#dataCoords = data.drop(data[data.Latitude > 42].index)
#dataCoords = dataCoords.drop(data[data.Longitude > -73].index)
dataAreas.plot(x = 'Area', y = 'Price', kind='scatter')
dataConstruction = data.drop(data[data.Construction == 0].index)
plt.show()

dataConstruction.plot(x='Construction', y='Price', kind='scatter')
plt.show()

#plt.scatter(x=dataCoords['Latitude'], y=dataCoords['Longitude'], c=dataCoords['Price'], cmap='inferno'), 
    
#plt.show()