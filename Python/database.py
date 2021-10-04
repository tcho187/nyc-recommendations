import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
db_URI = ''
engine = create_engine(db_URI)

Base.metadata.bind = engine

DBSession = sessionmaker()

DBSession.bind = engine

session = DBSession()

