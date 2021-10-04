from TikTokApi import TikTokApi

from pathlib import Path

from sqlalchemy_schema import Stickers
from database import DBSession

from datetime import datetime as dt

from main import main
from add_sqlalchemy_table import add_records
from sqlalchemy_schema import TikTokers, Stickers

import json


def user_liked_by_username(user):
    verifyFp = ""
    api = TikTokApi.get_instance(custop_verifyFp=verifyFp, use_test_endpoints=True)
    results = 10
    did = api.generate_did()
    tiktoks = api.user_liked_by_username(username=user, count=1000, offset=0, custom_did=did)
    final_list = []
    for index, tiktok in enumerate(tiktoks):
        try:
            playaddr = tiktok['video']['playAddr']
            final_list.append(playaddr)
        except Exception as e:
            pass
    return final_list

if __name__ == '__main__':
    username = 'tcho1870'
    video_urls = user_liked_by_username(user=username)
    dummy_dict = {'1': video_urls}
    json_string = json.dumps(dummy_dict)
    with open('/Users/thomascho/PycharmProjects/Hachi&Thomas/data.txt', 'w') as outfile:
        json.dump(dummy_dict, outfile)
    print (video_urls)