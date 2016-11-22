import requests
from bs4 import BeautifulSoup


YELP_URL = 'http://www.yelp.com/biz/wills-jamaican-cuisine-inglewood?hrid=lFvP6CD1LgMYInElXwpsEQ'

r = requests.get(YELP_URL)
bs = BeautifulSoup(r.text)

# <h1 class="biz-page-title">Business Name</h1>
name = bs.select('h1.biz-page-title')[0].text.strip()

# <address>1253 2nd St; Santa Monica, CA</address>
address = bs.select('address')[0].text.strip()

# <p itemprop="description">Review Here</p>
reviews = [review.text for review in bs.select('p[itemprop="description"]')]

# <meta itemprop="ratingValue" content="3.0" />
rating_tags = bs.select('meta[itemprop="ratingValue"]')
ratings = [float(tag.attrs['content']) for tag in rating_tags]   

# <span class="hour-range"><span>8:30am</span> - <span>5:00pm</span></span>
hours_open_today = bs.select('span.hour-range')[0].get_text()

# <div class="short-def-list"><dl> ... </dl><dl> ... </dl></div>
biz_info_tags = bs.select('div.short-def-list > dl')

# <dl> <dt>key</dt>  <dd>value</dd> </dl>
biz_info = {}
for tag in biz_info_tags:
	key = tag.select('dt')[0].text.strip()
	value = tag.select('dd')[0].text.strip()
	biz_info[key] = value


print('Business Name: ', name)
print('Address: ', address)
print('Reviews (list): ', reviews)
print('Ratings (list): ', ratings)
print('Hours open today: ', hours_open_today)
print('Business info (dictionary): ', biz_info)
