from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

driver = webdriver.Chrome('chromedriver')
driver.get('https://www.realtor.com/realestateandhomes-search/New-York_NY')
content = driver.page_source
soup = BeautifulSoup(content, features='html.parser')
prices = []
beds = []
baths = []
sizes = []
addresses = []
for element in soup.findAll('li', attrs={'class': 'component_property-card'}):
    try :
        price = element.find('span', attrs={'data-label': 'pc-price'}).get_text()
        bed = element.find('li', attrs={'data-label': 'pc-meta-beds'}).get_text()
        bath = element.find('li', attrs={'data-label': 'pc-meta-baths'}).get_text()
        size = element.find('li', attrs={'data-label': 'pc-meta-sqft'}).get_text()
        address = element.find('div', attrs={'data-label': 'pc-address'}).get_text()
        prices.append(price)
        beds.append(bed)
        baths.append(bath)
        sizes.append(size)
        addresses.append(address)
    except :
        print('error')

df = pd.DataFrame({'Address': addresses, 'Price': prices, 'Beds': beds, 'Baths': baths, 'Sizes': sizes})
df.to_csv('listings.csv', index=False, encoding='utf-8')