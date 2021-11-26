PY = python

SCRAPE = src/scrapper.py

MAIN = src/main.py

scrape:
	$(PY) $(SCRAPE)

run:
	$(PY) $(MAIN)