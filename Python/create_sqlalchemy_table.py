import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_schema import Stickers, TikTokers, Restaurants

def create_table(table):
    Base = declarative_base()
    db_URI = ''
    engine = create_engine(db_URI)

    Base.metadata.bind = engine

    table.__table__.create(bind=engine, checkfirst=True)


if __name__ == '__main__':

    create_table(Restaurants)