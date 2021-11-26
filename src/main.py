import numpy as np
import os.path
import pandas as pd
import scrapper
import sys

def extractAndFormatData():
    data = pd.read_csv('data/listings.csv')
    address = data.iloc[:,0]
    price = data.iloc[:,1]
    x = np.column_stack(address)
    y = price
    print(x)

def main():
    print("maining")
    if not os.path.exists("data/listings.csv"):
        answer = ""
        while answer == "":
            answer = input("Data not available\nDownload data(y/n): ")
            if answer == 'y':
                continue
            elif answer == 'n':
                sys.exit()
            else:
                answer = ""
        scrapper.scrape()
    extractAndFormatData()


if __name__ == "__main__":
    main()

# TODO: figure out what we're going to use as features