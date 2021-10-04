import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists
from sqlalchemy_schema import Stickers, TikTokers, Restaurants

from datetime import datetime as dt


def add_records(table, column, record):
    Base = declarative_base()
    db_URI = ''
    engine = create_engine(db_URI)

    Base.metadata.bind = engine

    DBSession = sessionmaker()

    DBSession.bind = engine

    session = DBSession()

    # record = {'id': 1, 'user': 'testing', 'video_id': 'testing', 'frame_id': 1, 'text': 'testing'}
    row = table(**record)
    try:
        record_exist = session.query(table).filter(getattr(table, column) == record[column]).scalar()
        if not record_exist:
            session.add(row)
            session.commit()
        return True
    except Exception as e:
        print(e)
        return False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    now = dt.now()
    row = {'nickname': 'stephtravels_nyc', 'added_timestamp': now}
    add_records(TikTokers, row)
