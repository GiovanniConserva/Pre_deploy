import pandas as pd
from django.db import models
from models import Listing
import json


def import_listings():
    listings = pd.read_csv('../../data/listings/nyc_listings.csv')
    failed_import = []
    for idx, listing in listings.iterrows():
        db_item = Listing(airBnbId = listing['id'], url = listing['listing_url'], name = listing['name']
                        , picture_url = listing['picture_url'], host_url = listing['host_url'],
                        host_name = listing['host_name'], latitude = listing['latitude'], longitude = listing['longitude'],
                        room_type = listing['room_type'], price = listing['price'][1:], minimum_nights = listing['minimum_nights'],
                        number_of_reviews = listing['number_of_reviews'], review_scores_rating = listing['review_scores_rating'],
                        neighbourhood= listing['neighbourhood'])
        try:
            db_item.save()
        except:
            failed_import.append(listing['id'])

    with open('../../data/listings/failed_import.json', 'w') as outfile:
        json.dump(failed_import, outfile)