import pandas as pd

def merge():
    neighbourhoods = ['Manhattan', 'Staten-Island', 'Queens']#'Brooklyn', 'Bronx', 'Staten-Island', 'Queens']
    priceRange = [[x, x + 100000] for x in range(10000, 1000000) if x % 100000 == 0]

    df = pd.DataFrame({'Address': [], 'Beds': [], 'Baths': [], 'Area': [], 'Construction': [], 'Parking': [], 'Price': []})
    df.to_csv('data/uncleanedListing.csv', index=False, encoding='utf-8')
    for neighbourhood in neighbourhoods:
        for pRange in priceRange:
            data = pd.read_csv(f'data/{neighbourhood}-listings-{pRange[0]}-{pRange[1]}.csv')
            data.to_csv('data/uncleanedListings.csv', mode='a', index=False, header=False, encoding='utf-8')

def merge2():
    neighbourhoods = ['Bronx', 'Brooklyn']
    priceRange = [[x, x + 100000] for x in range(10000, 1000000) if x % 100000 == 0]
    for neighbourhood in neighbourhoods:
        for pRange in priceRange:
            data = pd.read_csv(f'data/{neighbourhood}-listings-{pRange[0]}-{pRange[1]}.csv')
            data.to_csv('data/uncleanedListings.csv', mode='a', index=False, header=False, encoding='utf-8')


merge()
#merge2()