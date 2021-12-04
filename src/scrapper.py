from bs4 import BeautifulSoup as soup
import math
from os.path import exists
import pandas as pd
import pickle
import requests
from seleniumwire import webdriver
import time

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#', printEnd = "\r"):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration	- Required	: current iteration (Int)
		total		- Required	: total iterations (Int)
		prefix		- Optional	: prefix string (Str)
		suffix		- Optional	: suffix string (Str)
		decimals	- Optional	: positive number of decimals in percent complete (Int)
		length		- Optional	: character length of bar (Int)
		fill		- Optional	: bar fill character (Str)
		printEnd	- Optional	: end character (e.g. "\r", "\r\n") (Str)
	"""
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
	if iteration == total:
		print()

def parseAndCleanData(data, address):
	priceAndBed, rest = data.split("bd")
	bath, rest = rest.split("ba")
	area, _ = rest.split("sqft")
	startOfBeds = priceAndBed.rindex(',') + 4
	price = priceAndBed[:startOfBeds].replace("$", "")
	bed = priceAndBed[startOfBeds:]
	price = price.replace(",", "")
	if area.find("--") != -1:
		return False
	else:
		area = int(area.replace(",", ""))
	if bed.find("--") != -1:
		bed = 0
	else:
		bed = int(bed)
	if bath.find("--") != -1:
		bath = 0
	else:
		bath = int(bath)
	prices.append(price); beds.append(bed); baths.append(bath); areas.append(area); addresses.append(address)
	return True

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
	# addToDuplicateChecks = False
	# duplicates = 0
	# if not duplicateChecks:
	# 	addToDuplicateChecks = True

	for listing in htmlDoc.findAll('a', {'class': 'list-card-link list-card-link-top-margin list-card-img'}, href=True):
		#if currentNumOfListings >= maxListings:
		#	return True
		#print(f"Found URL: {listing['href']}")
		if listing['href'] in duplicateChecks:
			continue
		else:
			duplicateChecks.append(listing['href'])

		# if addToDuplicateChecks:
		# 	duplicateChecks.append(listing['href'])
		# elif checkForDuplicates(listing['href']):
		# 	duplicates += 1
		# 	print("Link is a duplicate")
		# 	if duplicates >= 5:
		# 		print("Too many duplicates found, going to next borough")
		# 		return True
		# 	else:
		# 		continue
		# while(not gotDetails):
		listingHtml = requests.get(url=listing['href'], headers=header)
		if listingHtml.status_code != 200:
			break

		listingHtmlDoc = soup(listingHtml.content, 'html.parser')
		if listingHtmlDoc.find('div', {'class': 'hdp__sc-1j01zad-0 cRaELx'}):
			deets = listingHtmlDoc.findAll('span', {'class': 'Text-c11n-8-53-2__sc-aiai24-0 hdp__sc-1esuh59-3 cvftlt hjZqSR'})
			parseAndCleanDetails(deets)
			area = 0
			bed = 0
			bath = 0
			for interiorDetails in listingHtmlDoc.findAll('span', {'class': 'Text-c11n-8-53-2__sc-aiai24-0 cvftlt'}):
				interiorDetails = interiorDetails.get_text()
				if 'Total interior livable area' in interiorDetails and area == 0:
					area = int(interiorDetails.replace("Total interior livable area:", "").replace("sqft", "").replace(",", ""))
				elif 'Bedrooms' in interiorDetails and bed == 0:
					bed = int(interiorDetails.replace('Bedrooms:', ""))
				elif 'Bathrooms' in interiorDetails and bath == 0:
					bath = int(interiorDetails.replace("Bathrooms:", ""))
			price = listingHtmlDoc.find('span', {'class', 'Text-c11n-8-53-2__sc-aiai24-0 hdp__sc-b5iact-0 haTioN ezoFMW'}).get_text()
			address = listingHtmlDoc.find('h1', {'id': 'ds-chip-property-address'}).get_text()
			#if area != 0:
			prices.append(price); addresses.append(address); beds.append(bed); baths.append(bath); areas.append(area)
				#gotDetails = True
			# else:
			# 	yearOfConstruction.pop(); parkingSpaces.pop()
			# 	currentNumOfListings -= 1
			# 	break
		currentNumOfListings += 1
	return False

def writeData(neighbourhood, lowerPrice, upperPrice):
	filename = f'data/{neighbourhood}-listings-{lowerPrice}-{upperPrice}.csv'
	df = pd.DataFrame({'Address': addresses, 'Beds': beds, 'Baths': baths, 'Area': areas, 'Construction': yearOfConstruction, 'Parking': parkingSpaces, 'Price': prices})
	df.to_csv(filename, index=False, encoding='utf-8')

def interceptor(request):
	del request.headers['user-agent']
	request.headers['user-agent'] = 'Mozilla/5.0 (X11; Linux x86_64; rv:94.0) Gecko/20100101 Firefox/94.0'
	del request.headers['Referer']
	request.headers['Referer'] = 'https://www.zillow.com/brooklyn-new-york-ny/?searchQueryState=%7B%22pagination'

# Go through all of the webpages and get the number of listings per page, and then work out how many pages each price range should get in relation to the max
def getNeighbourhoodInfo():
	print("Getting prerequisite info on neighbourhoods: ")
	if exists('data/neighbourhoodInfo.pkl'):
		print('Reading in data from data/neighbourhoodInfo.pkl')
		with open('data/neighbourhoodInfo.pkl', 'rb') as f:
			neighbourhoodInfo = pickle.load(f)
			print('Got data')
			return neighbourhoodInfo
	else:
		print('Creating base data')
		neighbourhoodInfo = { 
			#'Manhattan': ['{"pagination"%3A{}%2C"usersSearchTerm"%3A"Manhattan%2C New York%2C NY"%2C"mapBounds"%3A{"west"%3A-74.040174%2C"east"%3A-73.906999%2C"south"%3A40.680598%2C"north"%3A40.879278}%2C"regionSelection"%3A[{"regionId"%3A12530%2C"regionType"%3A17}]%2C"isMapVisible"%3Afalse%2C"filterState"%3A{"ah"%3A{"value"%3Atrue}%2C"price"%3A{"min"%3A', '%2C"max"%3A', '}%2C"mp"%3A{"min"%3A331%2C"max"%3A663}}%2C"isListVisible"%3Atrue}'],
								'Brooklyn': ['{"pagination"%3A{}%2C"usersSearchTerm"%3A"Brooklyn%2C New York%2C NY"%2C"mapBounds"%3A{"west"%3A-74.041603%2C"east"%3A-73.833646%2C"south"%3A40.570841%2C"north"%3A40.739446}%2C"regionSelection"%3A[{"regionId"%3A37607%2C"regionType"%3A17}]%2C"isMapVisible"%3Afalse%2C"filterState"%3A{"sort"%3A{"value"%3A"globalrelevanceex"}%2C"ah"%3A{"value"%3Atrue}%2C"price"%3A{"min"%3A', '%2C"max"%3A', '}%2C"mp"%3A{"min"%3A331%2C"max"%3A663}}%2C"isListVisible"%3Atrue}'],
								'Bronx': ['{"pagination"%3A{}%2C"usersSearchTerm"%3A"Bronx%2C New York%2C NY"%2C"mapBounds"%3A{"west"%3A-73.933405%2C"east"%3A-73.765273%2C"south"%3A40.785743%2C"north"%3A40.915266}%2C"regionSelection"%3A[{"regionId"%3A17182%2C"regionType"%3A17}]%2C"isMapVisible"%3Afalse%2C"filterState"%3A{"price"%3A{"min"%3A', '%2C"max"%3A', '}%2C"mp"%3A{"min"%3A331%2C"max"%3A994}%2C"sort"%3A{"value"%3A"globalrelevanceex"}%2C"ah"%3A{"value"%3Atrue}}%2C"isListVisible"%3Atrue}'],
								#'Staten-Island': ['{"pagination"%3A{}%2C"usersSearchTerm"%3A"Staten Island%2C New York%2C NY"%2C"mapBounds"%3A{"west"%3A-74.255586%2C"east"%3A-74.052267%2C"south"%3A40.496432%2C"north"%3A40.648857}%2C"regionSelection"%3A[{"regionId"%3A27252%2C"regionType"%3A17}]%2C"isMapVisible"%3Afalse%2C"filterState"%3A{"price"%3A{"min"%3A', '%2C"max"%3A', '}%2C"mp"%3A{"min"%3A331%2C"max"%3A663}%2C"sort"%3A{"value"%3A"globalrelevanceex"}%2C"ah"%3A{"value"%3Atrue}}%2C"isListVisible"%3Atrue}'],
								#'Queens': ['{"pagination"%3A{}%2C"usersSearchTerm"%3A"Queens%2C New York%2C NY"%2C"mapBounds"%3A{"west"%3A-73.962445%2C"east"%3A-73.700271%2C"south"%3A40.541745%2C"north"%3A40.800709}%2C"regionSelection"%3A[{"regionId"%3A270915%2C"regionType"%3A17}]%2C"isMapVisible"%3Afalse%2C"filterState"%3A{"price"%3A{"min"%3A', '%2C"max"%3A', '}%2C"mp"%3A{"min"%3A331%2C"max"%3A994}%2C"sort"%3A{"value"%3A"globalrelevanceex"}%2C"ah"%3A{"value"%3Atrue}}%2C"isListVisible"%3Atrue}']
							}
		for _, borough in neighbourhoodInfo.items():
			priceRange = [[x, x + 100000] for x in range(100000, 1000000) if x % 100000 == 0]
			borough.insert(0, priceRange)

		print('Done\nGetting total listings for each price range')
		mostListings = 0
		for i, (neighbourhood, info) in enumerate(neighbourhoodInfo.items()):
			printProgressBar(i * 10, (len(neighbourhoodInfo) + 1) * 10, prefix = 'Progress:', suffix = 'Complete', length = 50)
			for j, priceRange in enumerate(info[0]):
				lowerPrice, upperPrice = str(priceRange[0]), str(priceRange[1])
				# Get the number of listings for each price range and find the max of these
				url = 'https://www.zillow.com/' + neighbourhood + '-new-york-ny/?searchQueryState=' + info[1] + lowerPrice + info[2] + upperPrice + info[3]
				res = requests.get(url=url, headers=header)
				htmlDoc = soup(res.content, 'html.parser')
				count = int(htmlDoc.find('div', {'class': 'total-text'}).get_text().replace(",", "").strip())
				if count > mostListings:
					mostListings = count
				priceRange.append(count)
				printProgressBar(i * 10 + j, (len(neighbourhoodInfo) + 1) * 10, prefix = 'Progress:', suffix = 'Complete', length = 50)

		print("\rNormalizing: \r")
		for _, info in neighbourhoodInfo.items():
			for data in info[0]:
				maxListings = data[2]
				ratioToSixHundred = maxListings / mostListings
				maxListings = 600 * ratioToSixHundred
				data[2] = math.ceil(maxListings)
		print("\rDone\n\rWriting data to file: ")
		with open('data/neighbourhoodInfo.pkl', 'wb+') as f:
			pickle.dump(neighbourhoodInfo, f)
			print("\rDone\n")
			return neighbourhoodInfo

chromeOptions = webdriver.ChromeOptions()
#chromeOptions.add_argument('--headless')

header = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:78.0) Gecko/20100101 Firefox/78.0',
  'referer': 'https://www.zillow.com/brooklyn-new-york-ny/?searchQueryState=%7B%22pagination'
}

neighbourhoodInfo = getNeighbourhoodInfo()
#numOfPriceRanges = len(neighbourhoodInfo.get('Queens')[0])

for i, (neighbourhood, info) in enumerate(neighbourhoodInfo.items()):
	for j, priceRange in enumerate(info[0]):
		print(neighbourhood + ": " + str(priceRange[0]) + " to " + str(priceRange[1]))
		prices, beds, baths, addresses, areas, yearOfConstruction, parkingSpaces = [], [], [], [], [], [], []
		lowerPrice, upperPrice, maxListingsForCurrentRange = str(priceRange[0]), str(priceRange[1]), priceRange[2]
		listingsFromCurrentRange = 0
		duplicateChecks = []
		for page in range(1,20):
			try:
				print("page: " + str(page))
				driver = webdriver.Chrome("utils/chromeDriverLinux", chrome_options=chromeOptions)
				if page > 1:
					url = 'https://www.zillow.com/' + neighbourhood + '-new-york-ny/sold/' + str(page) + '_p/?searchQueryState=' +  info[1] + lowerPrice + info[2] + upperPrice + info[3]
				else:
					url = 'https://www.zillow.com/' + neighbourhood + '-new-york-ny/sold/?searchQueryState=' + info[1] + lowerPrice + info[2] + upperPrice + info[3]
				driver.request_interceptor = interceptor
				driver.get(url)
				time.sleep(1)
				driver.execute_script("window.scrollTo({top: document.body.scrollHeight, left: 0, behavior: 'smooth'});")
				time.sleep(3)
				html = driver.page_source
				htmlDoc = soup(html, 'html.parser')
				goToNextRange = extractDataFromHtml(htmlDoc, duplicateChecks, listingsFromCurrentRange, maxListingsForCurrentRange)
				listingsFromCurrentRange = len(prices)
				if goToNextRange:
					break
				driver.close()
			except Exception as e:
				if e != KeyboardInterrupt:
					break
				else:
					print(e)
		writeData(neighbourhood, lowerPrice, upperPrice)