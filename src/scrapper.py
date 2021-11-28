from bs4 import BeautifulSoup as soup
import pandas as pd
import requests 
import time

def parseAndCleanData(data, address):
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

def parseAndCleanDetails(details):
	yearBuilt, parkingSpots = 0, 0
	for detail in details:
		if detail.get_text().find("Built") != -1:
			yearBuilt = detail.get_text()
			yearBuilt = int(yearBuilt[9:])
		elif detail.get_text().find("Parking space") != -1:
			parkingSpots = detail.get_text()
			parkingSpots = int(parkingSpots.replace("Parking space", ""))
		elif detail.get_text().find("Parking spaces") != -1:
			parkingSpots = detail.get_text()
			parkingSpots = int(parkingSpots.replace("Parking spaces", ""))
	yearOfConstruction.append(yearBuilt)
	parkingSpaces.append(parkingSpots)
	print(yearBuilt, parkingSpots)

def sleepSchedule(startTime, timeSinceLastHibernate):
	if timeSinceLastHibernate == 180:
		print("hibernating")
		time.sleep(120)
		timeSinceLastHibernate = 0
	elif time.perf_counter() - startTime >= 20:
		print("sleeping")
		time.sleep(20)
		startTime = time.perf_counter()
		timeSinceLastHibernate += 20

def extractDataFromHTML(htmlDoc):
	for listing in htmlDoc.findAll('div', {'class': 'list-card-info'}):
		for listingPage in listing.find_all('a', href=True):
			print("Found the URL:", listingPage['href'])
			gotSummary, gotDetails = False, False
			while(not gotSummary or not gotDetails):
				listingHtml = requests.get(url=listingPage['href'], headers=header)
				if listingHtml.status_code != 200:
					break

				listingHtmlDoc = soup(listingHtml.content, 'html.parser')
				if listingHtmlDoc.find('div', {'class': 'ds-summary-row-container'}) and not gotSummary:
					data = listingHtmlDoc.find('div', {'class': 'ds-summary-row-container'}).get_text()
					address = listingHtmlDoc.find('h1', {'id': 'ds-chip-property-address'}).get_text()
					gotSummary = True
					parseAndCleanData(data, address)
				if listingHtmlDoc.find('ul', {'class': 'hdp__sc-1m6tl3b-0 gpsjXQ'}) and not gotDetails:
					deets = listingHtmlDoc.findAll('span', {'class': 'Text-c11n-8-53-2__sc-aiai24-0 hdp__sc-1esuh59-3 cvftlt hjZqSR'})
					parseAndCleanDetails(deets)
					gotDetails = True

def writeData():
	df = pd.DataFrame({'Address': addresses, 'Beds': beds, 'Baths': baths, 'Area': areas, 'Construction': yearOfConstruction, 'Parking': parkingSpaces, 'Price': prices})
	df.to_csv('listings.csv', index=False, encoding='utf-8')
	print(df.size)

prices, beds, baths, addresses, areas, yearOfConstruction, parkingSpaces = [], [], [], [], [], [], []

listOfNeighborhoods = ['Manhattan', 'Brooklyn', 'Bronx', 'Staten-Island', 'Queens']

header = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
  'referer': 'https://www.zillow.com/brooklyn-new-york-ny/?searchQueryState=%7B%22pagination'
}

startTime = time.perf_counter()
timeSinceLastHibernate = 0

for neighborhood in listOfNeighborhoods:
	for page in range (1,100):
		sleepSchedule(startTime, timeSinceLastHibernate)
		url = f'https://www.zillow.com/{neighborhood}-new-york-ny/{page}_p/'
		print(url)
		html = requests.get(url=url, headers=header)
		if html.status_code != 200:
			break

		htmlDoc = soup(html.content, 'html.parser')
		extractDataFromHTML(htmlDoc)

writeData()

# TODO: fix current issue of getting duplicate data, current idea, check to see if the first 5 datapoints of every page are the same as the first page