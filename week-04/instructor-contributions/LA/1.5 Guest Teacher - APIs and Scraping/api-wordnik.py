import requests
import sys

WORDNIK_URL = 'http://api.wordnik.com/v4/word.json/'
WORDNIK_PARAMS = '/definitions?limit=100&' \
                 'includeRelated=true&' \
                 'useCanonical=false&' \
                 'includeTags=false&' \
                 'api_key=a2a73e7b926c924fad7001ca3111acd55af2ffabf50eb4ae5'

if len(sys.argv) > 1:
    word = sys.argv[1]
else:
    print('Please enter the word to lookup after the script name.')
    sys.exit(0)


 # WORDNIK_URL + word + WORDNIK_PARAMS
 # '{}{}{}'.format(WORDNIK_URL, word, WORDNIK_PARAMS)
 # '%s%s%s' % (WORDNIK_URL, word, WORDNIK_PARAMS)

r = requests.get(WORDNIK_URL + word + WORDNIK_PARAMS)

definitions = r.json()

for definition in definitions:
    print(definition['text'])

