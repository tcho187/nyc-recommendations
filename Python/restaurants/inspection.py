import pandas as pd

from bs4 import BeautifulSoup
import requests
import json

from add_sqlalchemy_table import add_records
from sqlalchemy_schema import Restaurants

from datetime import datetime as dt

def read_inspection(source):
    df = pd.read_csv(source)
    restaurants = df.DBA.unique()

    for restaurant in restaurants:
        try:
            # google
            r = requests.get(f'https://google.com/search?q={restaurant} nyc')
            data = r.content
            soup = BeautifulSoup(data, 'html.parser')
            # print(soup.prettify())
            side_bar = soup.find("h3", {"class": "zBAuLc"})
            place = side_bar.find("div", {"class": "BNeawe deIvCb AP7Wnd"}).text
        except Exception as e:
            place = restaurant
        now = dt.now()
        row = {'restaurant': place, 'cuisine_types': None, 'description': None, 'price_ranges': None, 'aggregate_ratings': None, 'added_timestamp': now, 'source': 'restaurant data set 2'}
        print(row)
        add_records(Restaurants, row)
    return place

if __name__ == '__main__':
    file = '/Users/thomascho/Downloads/restaurant_data_set_2.csv'
    read_inspection(source=file)
