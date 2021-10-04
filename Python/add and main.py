from main import main
from add_sqlalchemy_table import add_records
from sqlalchemy_schema import TikTokers, Stickers
from datetime import datetime as dt

if __name__ == '__main__':
    tiktok_user = 'amorraytravels'
    now = dt.now()
    row = {'nickname': tiktok_user, 'added_timestamp': now}
    add_records(TikTokers, 'nickname', row)

    main(user=tiktok_user, table=Stickers)
