from add_sqlalchemy_table import add_records
from sqlalchemy_schema import Restaurants

from datetime import datetime as dt

from bs4 import BeautifulSoup
import requests
import json



if __name__ == '__main__':
    r = requests.get("https://www.kayak.com/New-York.15830.restaurantlist.ksp")
    data = r.content
    soup = BeautifulSoup(data, 'html.parser')
    restaurants = soup.find("div", {"class": "keel-container"})
    hrefs = []
    names = []
    for restaurant in restaurants.find_all("a"):
        href = restaurant.attrs['href']
        name = restaurant.text
        hrefs.append(href)
        names.append(name)
    # request individual page
    cuisine_types = []
    descriptions = []
    price_ranges = []
    aggregate_ratings = []
    for href in hrefs:
        r = requests.get(f"https://www.kayak.com{href}")
        data = r.content
        soup = BeautifulSoup(data, 'html.parser')
        second = soup.find_all("script", type="application/ld+json")[1]
        s = json.loads((second.contents[0]))
        if 'servesCuisine' in s:
            temp = ', '.join(s['servesCuisine'])
            cuisine_types.append(temp)
        else:
            cuisine_types.append(None)
        if 'descriptions' in s:
            descriptions.append(s['descriptions'])
        else:
            descriptions.append(None)
        if 'priceRange' in s:
            price_ranges.append(s['priceRange'])
        else:
            price_ranges.append(None)
        if 'aggregateRating' in s:
            temp = json.dumps(s['aggregateRating'])
            aggregate_ratings.append(temp)
        else:
            aggregate_ratings.append(None)

    for href, name, cuisine_type, description, price_range, aggregate_rating in zip(hrefs, names, cuisine_types, descriptions, price_ranges, aggregate_ratings):
        now = dt.now()
        row = {'restaurant': name, 'cuisine_types': cuisine_type, 'description': description, 'price_ranges': price_range, 'aggregate_ratings': aggregate_rating, 'added_timestamp': now, 'source': 'kayak.com'}
        print(row)
        add_records(Restaurants, row)

