import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Main(Base):
    __tablename__ = 'main'
    id = Column(Integer, primary_key=True)
    user = Column(String, nullable=False)
    video_id = Column(String, nullable=False)
    frame_id = Column(Integer, nullable=False)
    text = Column(String, nullable=True)





# Create an engine that stores data in the local directory's
# sqlalchemy_example.db file.

def main():
    db_URI = ''
    engine = create_engine(db_URI)

    Base.metadata.bind = engine

    DBSession = sessionmaker()

    DBSession.bind = engine

    session = DBSession()

    # Main.__table__.create(bind=engine, checkfirst=True)

    record = {'id': 1, 'user': 'testing', 'video_id': 'testing', 'frame_id': 1, 'text': 'testing'}
    row = Main(**record)

    session.add(row)
    session.commit()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
