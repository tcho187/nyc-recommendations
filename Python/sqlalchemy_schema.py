import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base

from database import Base


class Stickers(Base):
    __tablename__ = 'stickers'
    id = Column(Integer, autoincrement=True, primary_key=True)
    nickname = Column(String, nullable=False)
    video_id = Column(String, nullable=False)
    sticker_id = Column(Integer, nullable=False)
    sticker = Column(String, nullable=True)
    create_time = Column(DateTime, nullable=True)


class TikTokers(Base):
    __tablename__ = 'tiktokers'
    nickname = Column(String, nullable=False, primary_key=True)
    added_timestamp = Column(DateTime, nullable=True)


class Restaurants(Base):
    __tablename__ = 'restaurants'
    restaurant = Column(String, nullable=False, primary_key=True)
    cuisine_types = Column(String, nullable=True)
    description = Column(String, nullable=True)
    price_ranges = Column(String, nullable=True)
    aggregate_ratings = Column(String, nullable=True)
    source = Column(String, nullable=True)
    use_analysis = Column(Boolean, nullable=True)
    added_timestamp = Column(DateTime, nullable=True)



