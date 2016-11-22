"""
A quick and dirty Mashable link scraper.
This pulls links from the "Must Reads" page.

This is to demo the awesome power of BeautifulSoup!

Thanks to Joe Carli @ GA NYC! (May 2014)
 - Updated to use requests, removed classes, Python 3. (Dan Wilhelm @ GA LA, June 2015)
"""

import sys

import requests
from bs4 import BeautifulSoup

def get_page(url):
    """
    Downloads and returns the page contents at url.
    Returns an empty string if an error is encountered.
    """

    try:
        page = requests.get(url).content
    except:
        return ""
    
    return page


def find_links(document):
    """
    Given a document, returns a list of links.
    The returned item is a list, where each element is a RedditLink object.
    """

    # Create an empty list that we will populate
    links = []

    # Instantiate a BeautifulSoup object from the document
    soup = BeautifulSoup(document)
    sections_of_interest = soup.find_all('h1', {'class':'article-title'})
    if sections_of_interest == None:
        print("Unable to find the expected headers.")
        return links

    print("Found {} likely links".format(len(sections_of_interest)))
    for h1 in sections_of_interest:
        anchor = h1.find('a')
        if anchor == None:
            sys.stderr.write('Encountered a bad anchor tag.\n')
            continue
           
        if len(anchor.contents) > 0 and anchor.has_attr('href'):
            new_link = anchor['href'].strip() + "\n\t" + anchor.contents[0]
            links.append(new_link)

    print("Returning {} links".format(len(links)))
    return links
# end of find_links()


# Program execution starts here if run from the command line
if __name__=="__main__":

    url = 'http://mashable.com/category/must-reads'
    page = get_page(url)
    
    print("Read {} bytes from {}".format(len(page), url))
    
    links = find_links(page)
    
    print("")
    for l in links:
        print(l)
    print("")
