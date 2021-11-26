from bs4 import BeautifulSoup as soup
import pandas as pd
import requests 
import time

def parseData(data, address):
	priceAndBed, rest = data.split("bd")
	bath, rest = rest.split("ba")
	area, _ = rest.split("sqft")
	startOfBeds = priceAndBed.rindex(',') + 4
	price = priceAndBed[:startOfBeds].replace("$", "")
	bed = priceAndBed[startOfBeds:]
	price = price.replace(",", "")
	if bed.find("--") != -1:
		bed = 0
	else:
		bed = int(bed)
	if bath.find("--") != -1:
		bath = 0
	else:
		bath = int(bath)
	if area.find("--") != -1:
		area = 0
	else:
		area = int(area.replace(",", ""))
	prices.append(price); beds.append(bed); baths.append(bath); areas.append(area); addresses.append(address)
	print(price, bed, bath, area, address)

header = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
  'referer': 'https://www.zillow.com/brooklyn-new-york-ny/?searchQueryState=%7B%22pagination'
}

prices = []
beds = []
baths = []
addresses = []
areas = []
yearOfConstruction = []

listOfNeighborhoods = ['Manhattan', 'Brooklyn', 'Bronx', 'Staten-Island', 'Queens']

startTime = time.perf_counter()
timeSinceLastHibernate = 0

for neighborhood in listOfNeighborhoods:
	for page in range (1,100) :
		if timeSinceLastHibernate == 180:
			print("hibernating")
			time.sleep(120)
			timeSinceLastHibernate = 0
		elif time.perf_counter() - startTime >= 20:
			print("sleeping")
			time.sleep(20)
			startTime = time.perf_counter()
			timeSinceLastHibernate += 20

		url = f'https://www.zillow.com/{neighborhood}-new-york-ny/{page}_p/'
		html = requests.get(url=url, headers=header)
		if html.status_code != 200:
			break

		bsobj = soup(html.content, 'html.parser')
		for element in bsobj.findAll('div', {'class': 'list-card-info'}):
			for a in element.find_all('a', href=True):
				print("Found the URL:", a['href'])
				getSummary = True
				getDetails = True
				while(getSummary or getDetails):
					href_html = requests.get(url=a['href'], headers=header)
					if href_html.status_code != 200:
						break

					href_bsobj = soup(href_html.content, 'html.parser')
					if href_bsobj.find('div', {'class': 'ds-summary-row-container'}) and getSummary:
						data = href_bsobj.find('div', {'class': 'ds-summary-row-container'}).get_text()
						address = href_bsobj.find('h1', {'id': 'ds-chip-property-address'}).get_text()
						getSummary = False
						parseData(data, address)
					if href_bsobj.find('ul', {'class': 'hdp__sc-1m6tl3b-0 gpsjXQ'}) and getDetails:
						#details = href_bsobj.find('ul', {'class': 'hdp__sc-1m6tl3b-0 gpsjXQ'}).get_text()
						# Just going to get the year built as I don't think any of the others are going to be very useful
						yearList = (href_bsobj.find('li', {'class', 'hdp__sc-1esuh59-0 dmmleM'}))
						yearBuilt = href_bsobj.findAll('span', {'class': 'Text-c11n-8-53-2__sc-aiai24-0 hdp__sc-1esuh59-3 cvftlt hjZqSR'})[3].get_text()
						yearOfConstruction.append(yearBuilt)
						getDetails = False
					else:
						time.sleep(1)

df = pd.DataFrame({'Address': addresses, 'Price': prices, 'Beds': beds, 'Baths': baths, 'Area': areas, 'Construction': yearOfConstruction})
df.to_csv('listings.csv', index=False, encoding='utf-8')
print(df.size)