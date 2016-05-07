import pandas as pd
import numpy as np
from models import Availability, Listing
import json

def parse_calendar(list_calendar):
    date = []
    price_USD = []
    availability = []
    for month in list_calendar['calendar_months']:
        for day in month['days']:
            date.append(day['date'])
            price_USD.append(day['price']['native_price'])
            availability.append(day['available'])
    return pd.DataFrame({'date':date, 'price_USD': price_USD, 'availability':availability})

# returns a list of tuples: [(id, start, end)...]
def find_availability(parsed_calendars):
    availability = []
    for k, v in parsed_calendars.iteritems():
        condition = np.array(v['availability'])
        avai = np.where(np.concatenate(([condition[0]], condition[:-1] != condition[1:], [True])))[0]
        start_idx = list(avai[::2])
        end_idx = sorted(list(set(avai) - set(avai[::2])))
        start = v['date'][start_idx]
        end = v['date'][end_idx]
        avg_price = [np.mean(v['price_USD'][start_idx[i]:end_idx[i]]) for i in range(len(end))]
        availability += zip([k]* len(start), start, end, avg_price)

    return [x for x in availability if x[2]==x[2]]

def import_availability(parsed_calendars):
    availabilities = find_availability(parsed_calendars)
    failed_import = []
    for avai in availabilities:
        try:
            db_item = Availability(property_id = Listing.objects.all().get(airBnbId = int(avai[0])), start_date = avai[1], end_date = avai[2], avg_price = avai[3])
            db_item.save()
        except:
            failed_import.append(avai[0])

    with open('../../data/listings/avai_failed_import.json', 'w') as outfile:
        json.dump(failed_import, outfile)

def main():
    #Read in raw files
    cal = pd.read_json('../../data/calendars/nyc_cal.json')
    # cal2 = pd.read_json('../../data/calendars/nyc_cal_2.json')
    # cal3 = pd.read_json('../../data/calendars/nyc_cal_3.json')
    # cal4 = pd.read_json('../../data/calendars/nyc_cal_4.json')

    #Parse calendars
    parsed_calendars = {}
    for i in cal.keys():
        parsed_calendars[i] = parse_calendar(cal[i])
    # for i in cal2.keys():
    #     parsed_calendars[i] = parse_calendar(cal2[i])
    # for i in cal3.keys():
    #     parsed_calendars[i] = parse_calendar(cal3[i])
    # for i in cal4.keys():
    #     parsed_calendars[i] = parse_calendar(cal4[i])

    import_availability(parsed_calendars)