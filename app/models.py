from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.sql import func

from .db import Base


class Event(Base):
    __tablename__ = "events"

    id = Column(String, primary_key=True, index=True)
    ts = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    type = Column(String, nullable=False)
    service = Column(String, nullable=False)
    env = Column(String, nullable=False)

    message = Column(String, nullable=True)
    metadata_json = Column(Text, nullable=False, default="{}")
