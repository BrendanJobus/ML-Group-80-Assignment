import pandas as pd

def merge():
    neighbourhoods = ['Manhattan', 'Brooklyn', 'Bronx', 'Staten-Island', 'Queens']
    priceRange = [[x, x + 100000] for x in range(10000, 1000000) if x % 100000 == 0]

    df = pd.DataFrame({'Address': [], 'Beds': [], 'Baths': [], 'Area': [], 'Construction': [], 'Parking': [], 'Price': []})
    df.to_csv('data/listing.csv', index=False, encoding='utf-8')
    for neighbourhood in neighbourhoods:
        for pRange in priceRange:
            data = pd.read_csv(f'data/{neighbourhood}-listings-{pRange[0]}-{pRange[1]}.csv')
            data.to_csv('data/listings.csv', mode='a', index=False, header=False, encoding='utf-8')

merge()