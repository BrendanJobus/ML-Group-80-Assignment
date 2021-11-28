import pandas as pd
import requests
import json

def geocode():
    apiKey = '5244c5ea8013012013c7f74e40ae42ff8e5899c6'
    data = pd.read_csv('data/listings.csv')
    latitudes = []
    longitudes = []
    addresses = data.iloc[:,0]
    for address in addresses:
        reqUrl = f'https://api.geocodify.com/v2/geocode?api_key={apiKey}&q={address}'
        res = requests.get(url=reqUrl)
        if(res.status_code == 200):
            resData = json.loads(res.content)
            coords = resData['response']['features'][0]['geometry']['coordinates']
            lat = coords[0]
            long = coords[1]
            latitudes.append(lat)
            longitudes.append(long)
        else:
            latitudes.append(0.0)
            longitudes.append(0.0)
    data['Latitude'] = latitudes
    data['Longitude'] = longitudes
    data.to_csv('geocodedListings.csv', index=False, encoding='utf-8')