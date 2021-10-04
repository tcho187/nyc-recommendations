from TikTokApi import TikTokApi
from tqdm import tqdm


from pathlib import Path

from sqlalchemy_schema import Stickers
from database import DBSession

from datetime import datetime as dt

from main import main
from add_sqlalchemy_table import add_records
from sqlalchemy_schema import TikTokers, Stickers



def by_tag(hashtag):
    verifyFp = ""
    api = TikTokApi.get_instance(custop_verifyFp=verifyFp, use_test_endpoints=True)
    results = 10


    did = api.generate_did()
    hashtag = api.by_hashtag(hashtag, count=100, offset=0, custom_did=did)

    insert = []

    authors = []

    for x in tqdm(hashtag):
        if 'stickersOnItem' in x:
            for stickers in x['stickersOnItem']:
                for index, sticker in enumerate(stickers['stickerText']):
                    timestamp = dt.fromtimestamp(x['createTime'])
                    record = {'nickname': x['author']['uniqueId'], 'video_id': x['id'], 'sticker_id': index, 'sticker': sticker,
                              'create_time': timestamp}
                    if x['author']['uniqueId'] not in authors:
                        authors.append(x['author']['uniqueId'])
                    insert.append(record)
    for tiktoker in authors:
        tiktok_user = tiktoker
        now = dt.now()
        row = {'nickname': tiktok_user, 'added_timestamp': now}
        add_records(TikTokers, 'nickname', row)
        main(user=tiktok_user, table=Stickers)

if __name__ == '__main__':
    tag = 'nyceats'
    by_tag(hashtag=tag)
