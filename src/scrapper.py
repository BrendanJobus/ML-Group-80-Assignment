from bs4 import BeautifulSoup as soup
import math
import pandas as pd
import requests
from seleniumwire import webdriver
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

def checkForDuplicates(address):
	if address in duplicateChecks: 
		return True
	else: 
		return False

def extractDataFromHtml(htmlDoc, duplicateChecks, currentNumOfListings, maxListings):
	addToDuplicateChecks = False
	duplicates = 0
	if not duplicateChecks:
		print("first page")
		addToDuplicateChecks = True

	for listing in htmlDoc.findAll('a', {'class': 'list-card-link list-card-link-top-margin list-card-img'}, href=True):
		if currentNumOfListings >= maxListings:
			return True
		print(f"Found URL: {listing['href']}")
		if addToDuplicateChecks:
			duplicateChecks.append(listing['href'])
		elif checkForDuplicates(listing['href']):
			duplicates += 1
			print("Link is a duplicate")
			if duplicates >= 5:
				print("Too many duplicates found, going to next borough")
				return True
			else:
				continue
		gotSummary, gotDetails = False, False
		while(not gotSummary or not gotDetails):
			listingHtml = requests.get(url=listing['href'], headers=header)
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
			currentNumOfListings += 1
	return False

def writeData():
	df = pd.DataFrame({'Address': addresses, 'Beds': beds, 'Baths': baths, 'Area': areas, 'Construction': yearOfConstruction, 'Parking': parkingSpaces, 'Price': prices})
	df.to_csv('data/listings.csv', mode='a', header=False, index=False, encoding='utf-8')
	print(df.size)

def createCSV():
	df = pd.DataFrame({'Address', 'Beds', 'Baths', 'Area', 'Construction', 'Parking', 'Price'})
	df.to_csv('data/listing.csv', index=False, encoding='utf-8')

def interceptor(request):
	del request.headers['user-agent']
	request.headers['user-agent'] = 'Mozilla/5.0 (X11; Linux x86_64; rv:94.0) Gecko/20100101 Firefox/94.0'
	del request.headers['Referer']
	request.headers['Referer'] = 'https://www.zillow.com/brooklyn-new-york-ny/?searchQueryState=%7B%22pagination'

# Go through all of the webpages and get the number of listings per page, and then work out how many pages each price range should get in relation to the max
def getMaxListingsPerPriceRange():
	mostListings = 0
	for neighbourhood, info in neighbourhoodInfo.items():
		for priceRange in info[0]:
			lowerPrice, upperPrice = str(priceRange[0]), str(priceRange[1])
			# Get the number of listings for each price range and find the max of these
			url = 'https://www.zillow.com/' + neighbourhood + '-new-york-ny/sold/?searchQueryState=' + info[1] + lowerPrice + info[2] + upperPrice + info[3]
			res = requests.get(url=url, headers=header)
			htmlDoc = soup(res.content, 'html.parser')
			count = int(htmlDoc.find('span', {'class': 'result-count'}).get_text().replace("results", "").replace(",", "").strip())
			if count > mostListings:
				mostListings = count
			priceRange.append(count)

	for _, info in neighbourhoodInfo.items():
		for data in info[0]:
			maxListings = data[2]
			ratioToSixHundred = maxListings / mostListings
			maxListings = 600 * ratioToSixHundred
			data[2] = math.ceil(maxListings)
	
neighbourhoodInfo = {'Manhattan': ['{"pagination"%3A{}%2C"usersSearchTerm"%3A"Manhattan%2C New York%2C NY"%2C"mapBounds"%3A{"west"%3A-74.040174%2C"east"%3A-73.906999%2C"south"%3A40.680598%2C"north"%3A40.879278}%2C"regionSelection"%3A[{"regionId"%3A12530%2C"regionType"%3A17}]%2C"isMapVisible"%3Afalse%2C"filterState"%3A{"price"%3A{"min"%3A', '%2C"max"%3A', '}%2C"mp"%3A{"min"%3A331%2C"max"%3A993}%2C"sort"%3A{"value"%3A"globalrelevanceex"}%2C"fsba"%3A{"value"%3Afalse}%2C"fsbo"%3A{"value"%3Afalse}%2C"nc"%3A{"value"%3Afalse}%2C"fore"%3A{"value"%3Afalse}%2C"cmsn"%3A{"value"%3Afalse}%2C"auc"%3A{"value"%3Afalse}%2C"rs"%3A{"value"%3Atrue}%2C"ah"%3A{"value"%3Atrue}}%2C"isListVisible"%3Atrue}'],
						'Brooklyn': ['{"pagination"%3A{}%2C"usersSearchTerm"%3A"Brooklyn%2C New York%2C NY"%2C"mapBounds"%3A{"west"%3A-74.041603%2C"east"%3A-73.833646%2C"south"%3A40.570841%2C"north"%3A40.739446}%2C"regionSelection"%3A[{"regionId"%3A37607%2C"regionType"%3A17}]%2C"isMapVisible"%3Afalse%2C"filterState"%3A{"sort"%3A{"value"%3A"globalrelevanceex"}%2C"fsba"%3A{"value"%3Afalse}%2C"fsbo"%3A{"value"%3Afalse}%2C"nc"%3A{"value"%3Afalse}%2C"fore"%3A{"value"%3Afalse}%2C"cmsn"%3A{"value"%3Afalse}%2C"auc"%3A{"value"%3Afalse}%2C"rs"%3A{"value"%3Atrue}%2C"ah"%3A{"value"%3Atrue}%2C"price"%3A{"min"%3A', '%2C"max"%3A', '}%2C"mp"%3A{"min"%3A331%2C"max"%3A662}}%2C"isListVisible"%3Atrue}'],
						'Bronx': ['{"pagination"%3A{}%2C"usersSearchTerm"%3A"Bronx%2C New York%2C NY"%2C"mapBounds"%3A{"west"%3A-73.933405%2C"east"%3A-73.765273%2C"south"%3A40.785743%2C"north"%3A40.915266}%2C"regionSelection"%3A[{"regionId"%3A17182%2C"regionType"%3A17}]%2C"isMapVisible"%3Afalse%2C"filterState"%3A{"price"%3A{"min"%3A', '%2C"max"%3A', '}%2C"mp"%3A{"min"%3A331%2C"max"%3A662}%2C"sort"%3A{"value"%3A"globalrelevanceex"}%2C"fsba"%3A{"value"%3Afalse}%2C"fsbo"%3A{"value"%3Afalse}%2C"nc"%3A{"value"%3Afalse}%2C"fore"%3A{"value"%3Afalse}%2C"cmsn"%3A{"value"%3Afalse}%2C"auc"%3A{"value"%3Afalse}%2C"rs"%3A{"value"%3Atrue}%2C"ah"%3A{"value"%3Atrue}}%2C"isListVisible"%3Atrue}'],
						'Staten-Island': ['{"pagination"%3A{}%2C"usersSearchTerm"%3A"Staten Island%2C New York%2C NY"%2C"mapBounds"%3A{"west"%3A-74.255586%2C"east"%3A-74.052267%2C"south"%3A40.496432%2C"north"%3A40.648857}%2C"regionSelection"%3A[{"regionId"%3A27252%2C"regionType"%3A17}]%2C"isMapVisible"%3Afalse%2C"filterState"%3A{"price"%3A{"min"%3A', '%2C"max"%3A', '}%2C"mp"%3A{"min"%3A331%2C"max"%3A993}%2C"sort"%3A{"value"%3A"globalrelevanceex"}%2C"fsba"%3A{"value"%3Afalse}%2C"fsbo"%3A{"value"%3Afalse}%2C"nc"%3A{"value"%3Afalse}%2C"fore"%3A{"value"%3Afalse}%2C"cmsn"%3A{"value"%3Afalse}%2C"auc"%3A{"value"%3Afalse}%2C"rs"%3A{"value"%3Atrue}%2C"ah"%3A{"value"%3Atrue}}%2C"isListVisible"%3Atrue}'],
						'Queens': ['{"pagination"%3A{}%2C"usersSearchTerm"%3A"Queens%2C New York%2C NY"%2C"mapBounds"%3A{"west"%3A-73.962445%2C"east"%3A-73.700271%2C"south"%3A40.541745%2C"north"%3A40.800709}%2C"regionSelection"%3A[{"regionId"%3A270915%2C"regionType"%3A17}]%2C"isMapVisible"%3Afalse%2C"filterState"%3A{"price"%3A{"min"%3A', '%2C"max"%3A', '}%2C"mp"%3A{"min"%3A331%2C"max"%3A662}%2C"sort"%3A{"value"%3A"globalrelevanceex"}%2C"fsba"%3A{"value"%3Afalse}%2C"fsbo"%3A{"value"%3Afalse}%2C"nc"%3A{"value"%3Afalse}%2C"fore"%3A{"value"%3Afalse}%2C"cmsn"%3A{"value"%3Afalse}%2C"auc"%3A{"value"%3Afalse}%2C"rs"%3A{"value"%3Atrue}%2C"ah"%3A{"value"%3Atrue}}%2C"isListVisible"%3Atrue}']
					}
for _, borough in neighbourhoodInfo.items():
	priceRange = [[x, x + 100000] for x in range(10000, 1000000) if x % 100000 == 0]
	borough.insert(0, priceRange)

header = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:94.0) Gecko/20100101 Firefox/94.0',
  'referer': 'https://www.zillow.com/brooklyn-new-york-ny/?searchQueryState=%7B%22pagination'
}

getMaxListingsPerPriceRange()
createCSV()

for neighbourhood, info in neighbourhoodInfo.items():
	for priceRange in info[0]:
		prices, beds, baths, addresses, areas, yearOfConstruction, parkingSpaces = [], [], [], [], [], [], []
		lowerPrice, upperPrice, maxListingsForCurrentRange = str(priceRange[0]), str(priceRange[1]), priceRange[2]
		listingsFromCurrentRange = 0
		duplicateChecks = []
		for page in range (1,20):
			driver = webdriver.Chrome()
			if page > 1:
				url = 'https://www.zillow.com/' + neighbourhood + '-new-york-ny/sold/' + str(page) + '_p/?searchQueryState=' +  info[1] + lowerPrice + info[2] + upperPrice + info[3]
			else:
				url = 'https://www.zillow.com/' + neighbourhood + '-new-york-ny/sold/?searchQueryState=' + info[1] + lowerPrice + info[2] + upperPrice + info[3]
			driver.request_interceptor = interceptor
			driver.get(url)
			time.sleep(1)
			driver.execute_script("window.scrollTo({top: document.body.scrollHeight, left: 0, behavior: 'smooth'});")
			time.sleep(5)
			html = driver.page_source
			htmlDoc = soup(html, 'html.parser')
			goToNextRange = extractDataFromHtml(htmlDoc, duplicateChecks, listingsFromCurrentRange, maxListingsForCurrentRange)
			listingsFromCurrentRange = len(prices)
			if goToNextRange:
				print(f"Datapoints from this range: {listingsFromCurrentRange} out of {maxListingsForCurrentRange}")
				break
			driver.close()
		# write here to save data
		writeData()

#writeData()
# TODO: setup price ranging and then save per page