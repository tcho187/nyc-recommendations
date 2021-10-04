# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from TikTokApi import TikTokApi

from pathlib import Path

from sqlalchemy_schema import Stickers
from database import DBSession

from datetime import datetime as dt


def main(user, table):
    api = TikTokApi.get_instance()
    results = 10

    verifyFp = ""

    trending = api.trending(count=results, custop_verifyFp=verifyFp)

    pager = api.get_user_pager(user, page_size=30, cursor=0)

    # playAddr = (p[1][0]['video']['playAddr'])
    did = api.generate_did()
    for page in pager:
        try:
            id = page[0]['id']
            get_tiktok_by_id = api.get_tiktok_by_id(id=id)
            by_username = api.by_username(username=user, custom_did=did, custop_verifyFp=verifyFp)
            vid_id = (by_username[0]['id'])

            insert = []
            for x in range(len(by_username)):
                video = by_username[x]
                timestamp = dt.fromtimestamp(video['createTime'])
                if 'stickersOnItem' in video:
                    for stickers in video['stickersOnItem']:
                        for index, sticker in enumerate(stickers['stickerText']):
                            record = {'nickname': user, 'video_id': video['id'], 'sticker_id': index, 'sticker': sticker, 'create_time': timestamp}
                            insert.append(record)
            sess = DBSession()
            try:
                if len(insert)>0:
                    sess.bulk_insert_mappings(table, insert)
                    sess.commit()
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)

    #
    # Path(f"/Users/thomascho/code/tiktokvideos/{user}").mkdir(parents=True, exist_ok=True)
    #
    # for x in range(len(by_username)):
    #     b = api.get_Video_By_TikTok(by_username[x], custom_did=did)
    #     vid_id = (by_username[x]['id'])
    #     Path(f"/Users/thomascho/code/tiktokvideos/{user}/{vid_id}").mkdir(parents=True, exist_ok=True)
    #     with open(f"/Users/thomascho/code/tiktokvideos/{user}/{vid_id}/{vid_id}.mp4", 'wb') as output:
    #         output.write(b)
    #     print(f"video {x} is done.")

    # get_video_by_tiktok = api.get_video_by_tiktok(data=by_username[0], did=did)
    # print(get_video_by_tiktok)
    # get_video_by_download_url = api.get_video_by_download_url(download_url = playAddr)
    # print (get_video_by_download_url)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tiktok_user = 'stephtravels_nyc'
    main(user=tiktok_user, table=Stickers)
