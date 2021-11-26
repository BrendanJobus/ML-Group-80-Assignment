from bs4 import BeautifulSoup as soup
import pandas as pd
import re
import requests 
import time

def scrape():
    header = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    'referer': 'https://www.zillow.com/brooklyn-new-york-ny/?searchQueryState=%7B%22pagination'
    }

    prices = []
    beds = []
    baths = []
    addresses = []
    areas = []

    listOfNeighborhoods = ['Manhattan', 'Brooklyn', 'Bronx', 'Staten-Island', 'Queens']

    startTime = time.perf_counter()
    timeSinceLastHibernate = 0

    for neighborhood in listOfNeighborhoods:
        for page in range (1,2) :
            if time.perf_counter() - startTime >= 20:
                print("sleep")
                time.sleep(20)
                startTime = time.perf_counter()
                timeSinceLastHibernate += 20
            elif timeSinceLastHibernate == 180:
                time.sleep(120)
                timeSinceLastHibernate = 0
            url = f'https://www.zillow.com/{neighborhood}-new-york-ny/{page}_p/'
            html = requests.get(url=url, headers=header)
            print(html.content)
            if html.status_code != 200:
                break

            bsobj = soup(html.content, 'html.parser')
            for element in bsobj.findAll('div', {'class': 'list-card-info'}):
                if element.find('address', {'class': 'list-card-addr'}):
                    address = element.find('address', {'class': 'list-card-addr'}).get_text()
                    price = element.find('div', {'class': 'list-card-price'}).get_text().replace('$','')
                    element.find('div', {'class': 'list-card-price'}).get_text().replace('$', '')
                    otherInfo = element.find('ul')
                    areaInfo = otherInfo.find('abbr', string=re.compile(r'\s*sqft'))
                    bedInfo = otherInfo.find('abbr', string=re.compile(r'\s*bd.?'))
                    bathInfo = otherInfo.find('abbr', string=re.compile(r'\s*ba'))

                    print(areaInfo)

                    bed = 0 if bedInfo == None else bedInfo.parent.get_text().split(' ')[0]
                    bath = 0 if bathInfo == None else bathInfo.parent.get_text().split(' ')[0]
                    area = 0 if areaInfo == None else areaInfo.parent.get_text().split(' ')[0]
                    prices.append(price)
                    beds.append(bed)
                    baths.append(bath)
                    addresses.append(address)
                    areas.append(area)
                    print(address, price, bed, bath, area)
                    #print(address, price)         

    df = pd.DataFrame({'Address': addresses, 'Price': prices, 'Beds': beds, 'Baths': baths, 'Area': areas})
    #df = pd.DataFrame({'Address': addresses, 'Price': prices})
    df.to_csv('data/listings.csv', index=False, encoding='utf-8')
    print(df.size)

def exists():
    print("hi")

scrape()