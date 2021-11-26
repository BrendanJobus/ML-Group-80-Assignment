from bs4 import BeautifulSoup as soup
import pandas as pd
import random
import requests 

def readProxies():
    with open("utils/http_proxies.txt") as f:
        lines = f.read().splitlines()
        for _ in range(1, 100):
            ip_addresses.append(random.choice(lines))

def scrapeData():
    header = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    'referer': 'https://www.zillow.com/brooklyn-new-york-ny/?searchQueryState=%7B%22pagination'
    }

    prices = []
    beds = []
    baths = []
    addresses = []
    areas = []

    listOfNeighborhoods = ['Manhattan', 'Brooklyn', 'Bronx', 'Staten-Island', 'Queens']

    for neighborhood in listOfNeighborhoods:
        for page in range (1,100) :
            try:
                proxy_index = random.randint(0, len(ip_addresses) - 1)
                proxy = {"http": "http://" + ip_addresses[proxy_index], 
                "https": "http://" + ip_addresses[proxy_index]}

                url = f'https://www.zillow.com/{neighborhood}-new-york-ny/{page}_p/'
                html = requests.get(url=url, proxies=proxy, headers=header)
                if html.status_code != 200:
                    break

                bsobj = soup(html.content, 'html.parser')

                for element in bsobj.findAll('div', {'class': 'list-card-info'}):
                    if element.find('address', {'class': 'list-card-addr'}):
                        address = element.find('address', {'class': 'list-card-addr'}).get_text()
                        price = element.find('div', {'class': 'list-card-price'}).get_text().replace('$','')
                        otherInfo = element.findAll('li')
                        otherer = element.find('ul')
                        #print(otherInfo)
                        #bed = otherInfo[0].get_text().split(' ')[0]
                        #bath = otherInfo[1].get_text().split(' ')[0]
                        #area = otherInfo[2].get_text().split(' ')[0]
                        prices.append(price)
                        #beds.append(bed)
                        #baths.append(bath)
                        addresses.append(address)
                        #areas.append(area)
                        #print(address, price, bed, bath, area)
                        print(address, price)
            except Exception as e:
                print(e)

    #df = pd.DataFrame({'Address': addresses, 'Price': prices, 'Beds': beds, 'Baths': baths, 'Area': areas})
    df = pd.DataFrame({'Address': addresses, 'Price': prices})
    df.to_csv('data/listings.csv', index=False, encoding='utf-8')
    print(df.size)

ip_addresses = []
readProxies()
scrapeData()