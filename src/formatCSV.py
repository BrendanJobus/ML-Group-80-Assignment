import pandas as pd

def format():
    data = pd.read_csv('data/listing.csv')
    data['Price'] = data['Price'].str.replace(r'(,|\$)', '', regex=True)
    data.to_csv('data/formattedListing.csv', index=False, encoding='utf-8')