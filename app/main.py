from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4
import json

from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import SessionLocal, engine
from app.models import Event

# -------------------------------------------------------------------
# App setup
# -------------------------------------------------------------------
app = FastAPI(
    title="Cloud Intelligence Operations Platform",
    version="0.1.0",
)

# Create DB tables
Event.metadata.create_all(bind=engine)

# -------------------------------------------------------------------
# DB dependency
# -------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------
EventType = Literal["deployment", "metric", "cost"]

class EventIn(BaseModel):
    type: EventType
    service: str = Field(..., examples=["api", "worker"])
    env: str = Field("dev", examples=["dev", "staging", "prod"])
    message: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

class EventOut(EventIn):
    id: str
    ts: datetime

# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------------------------------------------
# Create event (persisted)
# -------------------------------------------------------------------
@app.post("/events", response_model=dict)
def create_event(event: EventIn, db: Session = Depends(get_db)):
    event_id = str(uuid4())

    db_event = Event(
        id=event_id,
        type=event.type,
        service=event.service,
        env=event.env,
        message=event.message,
        metadata_json=json.dumps(event.metadata),
    )

    db.add(db_event)
    db.commit()

    return {"id": event_id}

# -------------------------------------------------------------------
# List events (timeline)
# -------------------------------------------------------------------
@app.get("/events", response_model=list[EventOut])
def list_events(
    service: Optional[str] = None,
    type: Optional[EventType] = None,
    env: Optional[str] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Event)

    if service:
        query = query.filter(Event.service == service)
    if type:
        query = query.filter(Event.type == type)
    if env:
        query = query.filter(Event.env == env)

    results = query.order_by(Event.ts.desc()).all()

    return [
        EventOut(
            id=e.id,
            ts=e.ts.replace(tzinfo=timezone.utc),
            type=e.type,
            service=e.service,
            env=e.env,
            message=e.message,
            metadata=json.loads(e.metadata_json),
        )
        for e in results
    ]

