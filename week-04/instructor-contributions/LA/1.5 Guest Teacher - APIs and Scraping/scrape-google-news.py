import requests
from bs4 import BeautifulSoup

GOOGLE_URL = 'https://news.google.com'

r = requests.get(GOOGLE_URL)
bs = BeautifulSoup(r.text)

titles = [tag.text for tag in bs.select('span.titletext')]

for title in titles:
    print(title)