# app/main.py
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta, date
from typing import Optional, Literal, Any, Dict, List, Tuple
from uuid import uuid4

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session


# ------------------------------------------------------------------------------
# DB (SQLite)
# ------------------------------------------------------------------------------
DATABASE_URL = "sqlite:///./ciop.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class EventORM(Base):
    __tablename__ = "events"

    id = Column(String, primary_key=True, index=True)
    ts = Column(DateTime(timezone=True), index=True, nullable=False)
    type = Column(String, index=True, nullable=False)
    service = Column(String, index=True, nullable=False)
    env = Column(String, index=True, nullable=False)
    message = Column(String, nullable=False)
    metadata_json = Column(Text, nullable=False)  # store as JSON string


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
EventType = Literal["deployment", "cost", "metric"]


class EventIn(BaseModel):
    type: EventType
    service: str = Field(min_length=1, max_length=80)
    env: str = Field(min_length=1, max_length=40)
    message: str = Field(min_length=1, max_length=240)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EventOut(BaseModel):
    id: str
    ts: str
    type: EventType
    service: str
    env: str
    message: str
    metadata: Dict[str, Any]


class EventsPage(BaseModel):
    items: List[EventOut]
    total: int
    limit: int
    offset: int
    service: Optional[str]
    type: Optional[str]
    env: Optional[str]


# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Cloud Intelligence Operations Platform",
    version="0.2.0",
)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_json_safely(s: str) -> Dict[str, Any]:
    try:
        val = json.loads(s)
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}


def fmt_ts(dt: datetime) -> str:
    # keep similar to your UI: "2026-01-05 15:43:43Z"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def coerce_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def build_query_params_banner(service: Optional[str], type_: Optional[str], env: Optional[str], limit: int, offset: int) -> str:
    def v(x: Optional[str]) -> str:
        return x if x else "*"
    return f"service={v(service)} | type={v(type_)} | env={v(env)} | limit={limit} | offset={offset}"


def get_events_filtered(
    db: Session,
    service: Optional[str],
    type_: Optional[str],
    env: Optional[str],
    limit: int,
    offset: int,
) -> Tuple[List[EventORM], int]:
    q = db.query(EventORM)

    if service:
        q = q.filter(EventORM.service == service)
    if type_:
        q = q.filter(EventORM.type == type_)
    if env:
        q = q.filter(EventORM.env == env)

    total = q.count()
    items = (
        q.order_by(EventORM.ts.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return items, total


# ------------------------------------------------------------------------------
# API
# ------------------------------------------------------------------------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK. Go to /events/html"


@app.post("/events", response_model=EventOut)
def create_event(payload: EventIn, db: Session = Depends(get_db)):
    e = EventORM(
        id=str(uuid4()),
        ts=now_utc(),
        type=payload.type,
        service=payload.service.strip(),
        env=payload.env.strip(),
        message=payload.message.strip(),
        metadata_json=json.dumps(payload.metadata, ensure_ascii=False),
    )
    db.add(e)
    db.commit()
    db.refresh(e)

    return EventOut(
        id=e.id,
        ts=fmt_ts(e.ts),
        type=e.type,  # type: ignore
        service=e.service,
        env=e.env,
        message=e.message,
        metadata=parse_json_safely(e.metadata_json),
    )


@app.get("/events", response_model=EventsPage)
def list_events(
    service: Optional[str] = Query(default=None),
    type: Optional[str] = Query(default=None),
    env: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    items, total = get_events_filtered(db, service, type, env, limit, offset)

    out = []
    for e in items:
        out.append(
            EventOut(
                id=e.id,
                ts=fmt_ts(e.ts),
                type=e.type,  # type: ignore
                service=e.service,
                env=e.env,
                message=e.message,
                metadata=parse_json_safely(e.metadata_json),
            )
        )

    return EventsPage(
        items=out,
        total=total,
        limit=limit,
        offset=offset,
        service=service,
        type=type,
        env=env,
    )


@app.get("/events/csv")
def export_csv(
    service: Optional[str] = Query(default=None),
    type: Optional[str] = Query(default=None),
    env: Optional[str] = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    items, total = get_events_filtered(db, service, type, env, limit, offset)

    # CSV (simple)
    rows = ["id,ts,type,service,env,message,metadata_json"]
    for e in items:
        safe_msg = e.message.replace('"', '""')
        safe_meta = e.metadata_json.replace('"', '""')
        rows.append(f'"{e.id}","{fmt_ts(e.ts)}","{e.type}","{e.service}","{e.env}","{safe_msg}","{safe_meta}"')

    body = "\n".join(rows)
    headers = {
        "Content-Disposition": 'attachment; filename="events.csv"'
    }
    return Response(content=body, media_type="text/csv", headers=headers)


@app.post("/events/seed", response_class=PlainTextResponse)
def seed_demo_data(db: Session = Depends(get_db)):
    # Minimal seed so dashboard looks alive
    base_day = date.today()
    today = datetime.combine(base_day, datetime.min.time()).replace(tzinfo=timezone.utc)
    yesterday = today - timedelta(days=1)

    demo = [
        # yesterday
        EventORM(
            id=str(uuid4()),
            ts=yesterday + timedelta(hours=10, minutes=22),
            type="deployment",
            service="ciop-api",
            env="dev",
            message="CI push deployment from GitHub Actions",
            metadata_json=json.dumps(
                {"repo": "shivpatil999/ciop", "ref": "refs/heads/main", "actor": "shivpatil999"},
                ensure_ascii=False,
            ),
        ),
        EventORM(
            id=str(uuid4()),
            ts=yesterday + timedelta(hours=15, minutes=43),
            type="cost",
            service="ciop-api",
            env="dev",
            message="Daily service cost snapshot",
            metadata_json=json.dumps(
                {"amount": 10.10, "currency": "AUD", "period": (yesterday.date().isoformat()), "source": "mock"},
                ensure_ascii=False,
            ),
        ),
        # today
        EventORM(
            id=str(uuid4()),
            ts=today + timedelta(hours=12, minutes=36),
            type="deployment",
            service="ciop-api",
            env="dev",
            message="CI push deployment from GitHub Actions",
            metadata_json=json.dumps(
                {"repo": "shivpatil999/ciop", "ref": "refs/heads/main", "actor": "shivpatil999"},
                ensure_ascii=False,
            ),
        ),
        EventORM(
            id=str(uuid4()),
            ts=today + timedelta(hours=15, minutes=43),
            type="cost",
            service="ciop-api",
            env="dev",
            message="Daily service cost snapshot",
            metadata_json=json.dumps(
                {"amount": 12.45, "currency": "AUD", "period": (today.date().isoformat()), "source": "mock"},
                ensure_ascii=False,
            ),
        ),
        EventORM(
            id=str(uuid4()),
            ts=today + timedelta(hours=16, minutes=5),
            type="cost",
            service="web",
            env="dev",
            message="Static hosting cost snapshot",
            metadata_json=json.dumps(
                {"amount": 1.80, "currency": "AUD", "period": (today.date().isoformat()), "source": "mock"},
                ensure_ascii=False,
            ),
        ),
        EventORM(
            id=str(uuid4()),
            ts=today + timedelta(hours=16, minutes=25),
            type="metric",
            service="ciop-api",
            env="dev",
            message="Latency p95 updated",
            metadata_json=json.dumps({"p95_ms": 180, "window": "5m"}, ensure_ascii=False),
        ),
    ]

    for d in demo:
        exists = db.query(EventORM).filter(EventORM.message == d.message, EventORM.ts == d.ts).first()
        if not exists:
            db.add(d)

    db.commit()
    return "Seeded demo data."


# ------------------------------------------------------------------------------
# Intelligence Layer (backend aggregation)
# ------------------------------------------------------------------------------
def compute_cost_intelligence(
    db: Session,
    service: Optional[str],
    env: Optional[str],
) -> Dict[str, Any]:
    """
    Computes:
    - today_total (AUD) (cost events only)
    - yesterday_total
    - delta, delta_pct
    - per_service breakdown for today (or filtered service only)
    - cost-by-day bar series (last 7 days)
    - correlation flags: cost events that have deployment within ±24h same service/env
    """
    # Pull recent window for aggregations (7 days)
    t_now = now_utc()
    t_start = t_now - timedelta(days=7)

    q = db.query(EventORM).filter(EventORM.ts >= t_start)

    if env:
        q = q.filter(EventORM.env == env)
    if service:
        q = q.filter(EventORM.service == service)

    # We need cost + deployment for correlation, and cost for totals
    rows = q.order_by(EventORM.ts.desc()).all()

    # Build lists
    costs: List[Tuple[EventORM, Dict[str, Any], Optional[float], str]] = []
    deployments: List[Tuple[EventORM, Dict[str, Any]]] = []

    for r in rows:
        meta = parse_json_safely(r.metadata_json)
        if r.type == "cost":
            amt = coerce_float(meta.get("amount"))
            currency = str(meta.get("currency") or "AUD")
            costs.append((r, meta, amt, currency))
        elif r.type == "deployment":
            deployments.append((r, meta))

    # Today / yesterday boundaries in UTC
    today_date = datetime.now(timezone.utc).date()
    yesterday_date = today_date - timedelta(days=1)

    def is_day(dt: datetime, d: date) -> bool:
        dtu = dt.astimezone(timezone.utc)
        return dtu.date() == d

    # Totals (AUD only; if mixed currencies, we still surface currency label but skip non-AUD)
    today_total = 0.0
    yesterday_total = 0.0

    for (r, meta, amt, currency) in costs:
        if amt is None:
            continue
        if str(currency).upper() != "AUD":
            continue
        if is_day(r.ts, today_date):
            today_total += amt
        if is_day(r.ts, yesterday_date):
            yesterday_total += amt

    delta = today_total - yesterday_total
    delta_pct = None
    if yesterday_total > 0:
        delta_pct = (delta / yesterday_total) * 100.0

    # Per-service breakdown for today (AUD)
    per_service_today: Dict[str, float] = {}
    currency_label = "AUD"
    for (r, meta, amt, currency) in costs:
        if amt is None:
            continue
        currency_label = str(currency or "AUD")
        if str(currency).upper() != "AUD":
            continue
        if is_day(r.ts, today_date):
            per_service_today[r.service] = per_service_today.get(r.service, 0.0) + amt

    # Cost-by-day for last 7 days (AUD)
    day_buckets: Dict[str, float] = {}
    for i in range(0, 7):
        d = (today_date - timedelta(days=i)).isoformat()
        day_buckets[d] = 0.0

    for (r, meta, amt, currency) in costs:
        if amt is None:
            continue
        if str(currency).upper() != "AUD":
            continue
        d = r.ts.astimezone(timezone.utc).date().isoformat()
        if d in day_buckets:
            day_buckets[d] += amt

    # Order ascending for chart
    days_sorted = sorted(day_buckets.keys())
    series_days = days_sorted
    series_costs = [round(day_buckets[d], 2) for d in days_sorted]

    # Correlation: for each cost event, find deployments within ±24h same service/env
    # We’ll return a map cost_event_id -> list of deployments (summary)
    corr: Dict[str, List[Dict[str, Any]]] = {}
    window = timedelta(hours=24)

    # index deployments by (service, env)
    dep_index: Dict[Tuple[str, str], List[EventORM]] = {}
    for (d, dmeta) in deployments:
        key = (d.service, d.env)
        dep_index.setdefault(key, []).append(d)

    for (c, cmeta, amt, currency) in costs:
        key = (c.service, c.env)
        rel = []
        for d in dep_index.get(key, []):
            if abs((c.ts - d.ts).total_seconds()) <= window.total_seconds():
                rel.append({
                    "id": d.id,
                    "ts": fmt_ts(d.ts),
                    "message": d.message,
                })
        if rel:
            # keep most recent 3
            rel_sorted = sorted(rel, key=lambda x: x["ts"], reverse=True)[:3]
            corr[c.id] = rel_sorted

    return {
        "today_date": today_date.isoformat(),
        "yesterday_date": yesterday_date.isoformat(),
        "today_total": round(today_total, 2),
        "yesterday_total": round(yesterday_total, 2),
        "delta": round(delta, 2),
        "delta_pct": (round(delta_pct, 2) if delta_pct is not None else None),
        "currency": currency_label,
        "per_service_today": {k: round(v, 2) for k, v in sorted(per_service_today.items(), key=lambda x: x[1], reverse=True)},
        "cost_by_day": {
            "labels": series_days,
            "values": series_costs,
        },
        "correlations": corr,
    }


# ------------------------------------------------------------------------------
# HTML Dashboard
# ------------------------------------------------------------------------------
@app.get("/events/html", response_class=HTMLResponse)
def events_html(
    service: Optional[str] = Query(default=None),
    type: Optional[str] = Query(default=None),
    env: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    items, total = get_events_filtered(db, service, type, env, limit, offset)
    banner = build_query_params_banner(service, type, env, limit, offset)

    # Intelligence layer (today vs yesterday + breakdown + correlations)
    intel = compute_cost_intelligence(db, service=service, env=env)

    # Convert table items to dict for HTML
    view_items: List[Dict[str, Any]] = []
    for e in items:
        meta = parse_json_safely(e.metadata_json)
        view_items.append({
            "id": e.id,
            "ts": fmt_ts(e.ts),
            "type": e.type,
            "service": e.service,
            "env": e.env,
            "message": e.message,
            "metadata": meta,
        })

    # Event mix counts (service/env filtered but ignores type filter to show overall mix)
    mix_q = db.query(EventORM)
    if service:
        mix_q = mix_q.filter(EventORM.service == service)
    if env:
        mix_q = mix_q.filter(EventORM.env == env)

    mix_rows = mix_q.all()
    mix_counts = {"deployment": 0, "cost": 0, "metric": 0}
    for r in mix_rows:
        if r.type in mix_counts:
            mix_counts[r.type] += 1

    # total cost (within current filter window) - cost events only (AUD)
    # We compute from intel as last 7 days; but your UI expects "total cost filtered".
    # For simplicity and correctness: compute from all matching cost events in DB (AUD).
    cost_q = db.query(EventORM).filter(EventORM.type == "cost")
    if service:
        cost_q = cost_q.filter(EventORM.service == service)
    if env:
        cost_q = cost_q.filter(EventORM.env == env)
    cost_rows = cost_q.all()

    total_cost = 0.0
    for r in cost_rows:
        m = parse_json_safely(r.metadata_json)
        amt = coerce_float(m.get("amount"))
        cur = str(m.get("currency") or "AUD")
        if amt is None:
            continue
        if cur.upper() != "AUD":
            continue
        total_cost += amt
    total_cost = round(total_cost, 2)

    # Decide delta badge style
    delta = intel["delta"]
    delta_pct = intel["delta_pct"]
    delta_dir = "flat"
    if delta > 0:
        delta_dir = "up"
    elif delta < 0:
        delta_dir = "down"

    # Pass JSON blobs to JS
    js_intel = json.dumps(intel, ensure_ascii=False)
    js_items = json.dumps(view_items, ensure_ascii=False)
    js_mix = json.dumps(mix_counts, ensure_ascii=False)

    # Basic prev/next
    prev_offset = max(0, offset - limit)
    next_offset = offset + limit if (offset + limit) < total else offset

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Cloud Intelligence Operations Platform</title>

  <!-- Chart.js -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

  <style>
    :root {{
      --bg0:#070b14;
      --bg1:#0a1020;
      --panel: rgba(255,255,255,0.06);
      --panel2: rgba(255,255,255,0.04);
      --stroke: rgba(255,255,255,0.10);
      --text:#eaf0ff;
      --muted:rgba(234,240,255,0.65);
      --muted2:rgba(234,240,255,0.45);
      --accent:#7c5cff;
      --accent2:#22c55e;
      --warn:#f59e0b;
      --bad:#ef4444;
      --good:#22c55e;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      --radius:18px;
      --shadow: 0 20px 60px rgba(0,0,0,.55);
    }}
    *{{box-sizing:border-box}}
    body {{
      margin:0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(1200px 600px at 20% 10%, rgba(124,92,255,.25), transparent 60%),
        radial-gradient(900px 700px at 70% 20%, rgba(34,197,94,.10), transparent 60%),
        radial-gradient(1000px 800px at 50% 100%, rgba(59,130,246,.12), transparent 60%),
        linear-gradient(180deg, var(--bg0), var(--bg1));
      min-height:100vh;
    }}
    .topbar {{
      display:flex;
      align-items:center;
      justify-content:space-between;
      padding:22px 26px;
      border-bottom:1px solid rgba(255,255,255,.06);
      background: linear-gradient(180deg, rgba(255,255,255,0.04), transparent);
      position:sticky;
      top:0;
      backdrop-filter: blur(10px);
      z-index:10;
    }}
    .brand {{
      display:flex;
      gap:14px;
      align-items:center;
    }}
    .logo {{
      width:44px; height:44px; border-radius:14px;
      background: linear-gradient(135deg, rgba(124,92,255,.95), rgba(59,130,246,.55));
      box-shadow: 0 12px 40px rgba(124,92,255,.25);
    }}
    .brand h1 {{
      font-size:18px;
      margin:0;
      letter-spacing:.2px;
    }}
    .brand p {{
      margin:2px 0 0;
      font-size:12px;
      color: var(--muted);
    }}
    .pill {{
      font-family: var(--mono);
      font-size:12px;
      padding:10px 14px;
      border:1px solid var(--stroke);
      background: rgba(0,0,0,.20);
      border-radius:999px;
      color: rgba(234,240,255,.85);
    }}
    .wrap {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 18px 20px 40px;
    }}
    .grid2 {{
      display:grid;
      grid-template-columns: 1.6fr .9fr;
      gap: 18px;
      margin-top: 16px;
    }}
    .card {{
      background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
      border:1px solid rgba(255,255,255,0.10);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow:hidden;
    }}
    .card .head {{
      padding:16px 18px 0;
    }}
    .card h2 {{
      margin:0;
      font-size:14px;
      color: rgba(234,240,255,0.85);
      letter-spacing:.2px;
    }}
    .sub {{
      margin:6px 0 0;
      font-size:12px;
      color: var(--muted2);
    }}
    .content {{
      padding: 14px 18px 18px;
    }}
    .kpis {{
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 10px;
    }}
    .kpi {{
      background: rgba(0,0,0,.18);
      border:1px solid rgba(255,255,255,.09);
      border-radius: 14px;
      padding: 14px;
    }}
    .kpi .label {{
      font-size:12px;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .kpi .value {{
      font-size:28px;
      font-weight:800;
      letter-spacing:.2px;
    }}
    .kpi .hint {{
      font-size:12px;
      color: var(--muted2);
      margin-top: 8px;
      display:flex;
      gap:8px;
      align-items:center;
    }}
    .badge {{
      font-family: var(--mono);
      font-size:11px;
      padding:4px 8px;
      border-radius:999px;
      border:1px solid rgba(255,255,255,.12);
      background: rgba(0,0,0,.22);
      color: rgba(234,240,255,.85);
    }}
    .badge.good {{
      border-color: rgba(34,197,94,.35);
      background: rgba(34,197,94,.10);
      color: rgba(167,243,208,.95);
    }}
    .badge.bad {{
      border-color: rgba(239,68,68,.35);
      background: rgba(239,68,68,.10);
      color: rgba(254,202,202,.95);
    }}
    .badge.flat {{
      border-color: rgba(245,158,11,.35);
      background: rgba(245,158,11,.10);
      color: rgba(253,230,138,.95);
    }}
    .chartbox {{
      margin-top: 12px;
      background: rgba(0,0,0,.18);
      border:1px solid rgba(255,255,255,.09);
      border-radius: 14px;
      padding: 12px;
    }}
    .btnrow {{
      display:flex;
      gap:10px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    .btn {{
      display:inline-flex;
      gap:10px;
      align-items:center;
      border:1px solid rgba(255,255,255,.12);
      background: rgba(0,0,0,.20);
      color: rgba(234,240,255,.90);
      padding: 10px 12px;
      border-radius: 12px;
      text-decoration:none;
      font-weight:650;
      font-size:13px;
    }}
    .btn.primary {{
      background: linear-gradient(135deg, rgba(124,92,255,.90), rgba(124,92,255,.55));
      border-color: rgba(124,92,255,.55);
    }}
    .filters {{
      margin-top: 18px;
      padding: 16px 18px 18px;
      background: rgba(0,0,0,.16);
      border-top:1px solid rgba(255,255,255,.08);
    }}
    .filters h3 {{
      margin: 0 0 12px;
      font-size: 14px;
      letter-spacing:.2px;
      color: rgba(234,240,255,.88);
    }}
    .row {{
      display:grid;
      grid-template-columns: 1fr 1fr 1fr .6fr .6fr auto;
      gap: 10px;
      align-items:end;
    }}
    .field label {{
      display:block;
      font-size:12px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .field input {{
      width:100%;
      padding: 11px 12px;
      border-radius: 12px;
      border:1px solid rgba(255,255,255,.12);
      background: rgba(0,0,0,.20);
      color: rgba(234,240,255,.92);
      outline:none;
      font-family: var(--sans);
    }}
    .field input::placeholder {{ color: rgba(234,240,255,.35); }}
    .tip {{
      font-family: var(--mono);
      font-size: 12px;
      color: rgba(234,240,255,.70);
      padding: 10px 12px;
      border: 1px dashed rgba(255,255,255,.16);
      border-radius: 12px;
      background: rgba(0,0,0,.18);
    }}

    .tablewrap {{
      margin-top: 18px;
      padding: 0;
    }}
    table {{
      width:100%;
      border-collapse: collapse;
      overflow:hidden;
    }}
    thead th {{
      font-size:12px;
      text-align:left;
      padding: 14px 16px;
      color: rgba(234,240,255,.60);
      border-bottom: 1px solid rgba(255,255,255,.08);
      letter-spacing:.6px;
    }}
    tbody td {{
      padding: 16px;
      border-bottom: 1px solid rgba(255,255,255,.06);
      vertical-align: top;
    }}
    .mono {{ font-family: var(--mono); }}
    .typepill {{
      font-family: var(--mono);
      font-size: 12px;
      padding: 6px 10px;
      border-radius: 999px;
      display:inline-block;
      border:1px solid rgba(255,255,255,.12);
      background: rgba(0,0,0,.20);
    }}
    .typepill.cost {{ border-color: rgba(34,197,94,.35); background: rgba(34,197,94,.10); }}
    .typepill.deployment {{ border-color: rgba(59,130,246,.35); background: rgba(59,130,246,.10); }}
    .typepill.metric {{ border-color: rgba(124,92,255,.35); background: rgba(124,92,255,.10); }}
    .actionbtn {{
      border:1px solid rgba(255,255,255,.12);
      background: rgba(0,0,0,.20);
      padding: 9px 10px;
      border-radius: 12px;
      color: rgba(234,240,255,.88);
      font-weight:700;
      cursor:pointer;
    }}
    .small {{
      font-size:12px;
      color: rgba(234,240,255,.65);
    }}
    .corr {{
      margin-top: 8px;
      padding: 10px 10px;
      border-radius: 12px;
      border:1px solid rgba(255,255,255,.10);
      background: rgba(0,0,0,.18);
    }}
    .corr strong {{
      font-size:12px;
      color: rgba(234,240,255,.90);
    }}
    .corr ul {{
      margin: 8px 0 0;
      padding-left: 18px;
    }}
    .corr li {{
      margin: 4px 0;
      font-size: 12px;
      color: rgba(234,240,255,.75);
    }}
    .mixrow {{
      display:flex;
      justify-content:space-between;
      align-items:center;
      margin: 10px 0;
      gap: 10px;
    }}
    .bar {{
      height:10px;
      border-radius:999px;
      background: rgba(255,255,255,.08);
      border:1px solid rgba(255,255,255,.10);
      overflow:hidden;
      flex:1;
    }}
    .bar > div {{
      height:100%;
      border-radius:999px;
      background: linear-gradient(90deg, rgba(124,92,255,.85), rgba(34,197,94,.60));
      width:0%;
    }}
  </style>
</head>

<body>
  <div class="topbar">
    <div class="brand">
      <div class="logo"></div>
      <div>
        <h1>Cloud Intelligence Operations Platform</h1>
        <p>Events timeline • filters • costs • deployments • metrics</p>
      </div>
    </div>
    <div class="pill">{banner}</div>
  </div>

  <div class="wrap">
    <div class="grid2">
      <div class="card">
        <div class="head">
          <h2>Cost overview</h2>
          <div class="sub">Totals are based on cost events (AUD only).</div>
        </div>
        <div class="content">
          <div class="kpis">
            <div class="kpi">
              <div class="label">Total cost (filtered by service/env)</div>
              <div class="value" id="totalCost">{total_cost:.2f} AUD</div>
              <div class="hint">
                <span class="badge">Uses cost events only</span>
              </div>
            </div>
            <div class="kpi">
              <div class="label">Today cost (UTC)</div>
              <div class="value" id="todayCost">{intel["today_total"]:.2f} {intel["currency"]}</div>
              <div class="hint">
                <span class="badge {delta_dir if delta_dir in ("good","bad","flat") else "badge"}" id="deltaBadge"></span>
                <span class="small">vs {intel["yesterday_date"]} (UTC)</span>
              </div>
            </div>
          </div>

          <div class="chartbox">
            <div class="small" style="margin-bottom:8px;">Cost by day (last 7 days)</div>
            <canvas id="costByDay" height="92"></canvas>
          </div>

          <div class="btnrow">
            <a class="btn" href="/events/csv?service={service or ""}&type={type or ""}&env={env or ""}&limit={limit}&offset={offset}">⬇ Export CSV</a>
            <a class="btn" href="/events?service={service or ""}&type={type or ""}&env={env or ""}&limit={limit}&offset={offset}">📄 Text view</a>
            <a class="btn" href="/docs">🛠 Swagger</a>
            <a class="btn primary" href="/events/seed" onclick="return seedDemo(event)">🌱 Seed demo data</a>
          </div>

          <div class="chartbox" style="margin-top:12px;">
            <div class="small" style="margin-bottom:8px;">Today cost breakdown by service</div>
            <canvas id="costByService" height="110"></canvas>
          </div>
        </div>

        <div class="filters">
          <h3>Filters</h3>
          <form method="get" action="/events/html">
            <div class="row">
              <div class="field">
                <label>service</label>
                <input name="service" placeholder="e.g. ciop-api" value="{service or ""}">
              </div>
              <div class="field">
                <label>type</label>
                <input name="type" placeholder="deployment | cost | metric" value="{type or ""}">
              </div>
              <div class="field">
                <label>env</label>
                <input name="env" placeholder="dev | staging | prod" value="{env or ""}">
              </div>
              <div class="field">
                <label>limit</label>
                <input name="limit" placeholder="50" value="{limit}">
              </div>
              <div class="field">
                <label>offset</label>
                <input name="offset" placeholder="0" value="{offset}">
              </div>
              <button class="btn primary" type="submit">Apply</button>
            </div>
            <div style="margin-top:10px; display:flex; gap:10px; justify-content:space-between; align-items:center;">
              <a class="btn" href="/events/html">Clear</a>
              <div class="tip">Tip: /events/html?type=cost&service=ciop-api</div>
            </div>
          </form>
        </div>
      </div>

      <div class="card">
        <div class="head">
          <h2>Event mix (service/env)</h2>
          <div class="sub">Counts ignore the type filter so you see the overall mix.</div>
        </div>
        <div class="content" id="mixBox"></div>
      </div>
    </div>

    <div class="card tablewrap">
      <div class="head" style="padding:16px 18px 0; display:flex; justify-content:space-between; align-items:center;">
        <h2>Events table</h2>
        <div style="display:flex; gap:10px;">
          <a class="btn" href="/events/html?service={service or ""}&type={type or ""}&env={env or ""}&limit={limit}&offset={prev_offset}">◀ Previous</a>
          <a class="btn primary" href="/events/html?service={service or ""}&type={type or ""}&env={env or ""}&limit={limit}&offset={next_offset}">Next ▶</a>
        </div>
      </div>

      <div class="content">
        <table>
          <thead>
            <tr>
              <th class="mono">TIME (UTC)</th>
              <th>TYPE</th>
              <th>SERVICE</th>
              <th>ENV</th>
              <th>MESSAGE</th>
              <th class="mono">METADATA</th>
              <th>ACTIONS</th>
            </tr>
          </thead>
          <tbody id="tbody"></tbody>
        </table>
        <div class="small" style="margin-top:12px;">
          Showing <span class="mono">{min(limit, len(view_items))}</span> of <span class="mono">{total}</span> (limit={limit}, offset={offset})
        </div>
      </div>
    </div>
  </div>

<script>
  const INTEL = {js_intel};
  const ITEMS = {js_items};
  const MIX = {js_mix};

  function formatDelta() {{
    const delta = INTEL.delta;
    const pct = INTEL.delta_pct;
    const badge = document.getElementById("deltaBadge");

    let cls = "badge flat";
    let arrow = "→";
    if (delta > 0) {{ cls = "badge bad"; arrow = "↑"; }}
    if (delta < 0) {{ cls = "badge good"; arrow = "↓"; }}

    badge.className = cls;
    const absDelta = Math.abs(delta).toFixed(2);
    let txt = `${{arrow}} ${{absDelta}}`;
    if (pct !== null && pct !== undefined) {{
      txt += ` (${{Math.abs(pct).toFixed(2)}}%)`;
    }}
    txt += " vs yesterday";
    badge.textContent = txt;
  }}

  function seedDemo(ev) {{
    ev.preventDefault();
    fetch("/events/seed", {{ method:"POST" }})
      .then(r => r.text())
      .then(() => window.location.reload())
      .catch(() => window.location.reload());
    return false;
  }}

  function renderMix() {{
    const box = document.getElementById("mixBox");
    const total = (MIX.deployment + MIX.cost + MIX.metric) || 1;
    const rows = [
      ["deployment", MIX.deployment],
      ["cost", MIX.cost],
      ["metric", MIX.metric],
    ];

    box.innerHTML = rows.map(([name, count]) => {{
      const pct = Math.round((count / total) * 100);
      return `
        <div class="mixrow">
          <div style="width:110px; font-weight:700; color: rgba(234,240,255,.85)">${{name}}</div>
          <div class="bar"><div style="width:${{pct}}%"></div></div>
          <div class="mono" style="width:30px; text-align:right;">${{count}}</div>
        </div>
      `;
    }}).join("");
  }}

  function copyJson(obj) {{
    const s = JSON.stringify(obj, null, 2);
    navigator.clipboard.writeText(s);
  }}

  function typeClass(t) {{
    if (t === "cost") return "typepill cost";
    if (t === "deployment") return "typepill deployment";
    return "typepill metric";
  }}

  function renderTable() {{
  const tb = document.getElementById("tbody");
  tb.innerHTML = "";

  for (const e of ITEMS) {{
    const metaStr = JSON.stringify(e.metadata, null, 2);
    const corr = INTEL.correlations && INTEL.correlations[e.id] ? INTEL.correlations[e.id] : null;

    const corrHtml = corr ? `
      <div class="corr">
        <strong>Correlated deployments (±24h)</strong>
        <ul>
          ${{corr.map(d => `<li><span class="mono">${{d.ts}}</span> — ${{d.message}}</li>`).join("")}}
        </ul>
      </div>
    ` : "";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${{e.ts}}</td>
      <td><span class="${{typeClass(e.type)}}">${{e.type}}</span></td>
      <td style="font-weight:750;">${{e.service}}</td>
      <td class="mono">${{e.env}}</td>
      <td>
        <div style="font-weight:700;">${{e.message}}</div>
        ${{corrHtml}}
      </td>
      <td class="mono"><pre style="margin:0; white-space:pre-wrap; color:rgba(234,240,255,.75)">${{metaStr}}</pre></td>
      <td>
        <button class="actionbtn" onclick='copyJson(${{metaStr.replace(/</g,"\\\\u003c")}})'>Copy JSON</button>
      </td>
    `;
    tb.appendChild(tr);
  }}
}}


  function renderCharts() {{
    // cost by day
    const dayLabels = INTEL.cost_by_day.labels;
    const dayValues = INTEL.cost_by_day.values;

    new Chart(document.getElementById("costByDay"), {{
      type: "bar",
      data: {{
        labels: dayLabels,
        datasets: [{{
          label: "Cost (AUD)",
          data: dayValues,
          borderWidth: 1,
          borderRadius: 10
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{ enabled: true }}
        }},
        scales: {{
          x: {{
            ticks: {{ color: "rgba(234,240,255,.65)" }},
            grid: {{ color: "rgba(255,255,255,.06)" }}
          }},
          y: {{
            ticks: {{ color: "rgba(234,240,255,.65)" }},
            grid: {{ color: "rgba(255,255,255,.06)" }}
          }}
        }}
      }}
    }});

    // per service today
    const svc = INTEL.per_service_today || {{}};
    const svcLabels = Object.keys(svc);
    const svcValues = Object.values(svc);

    new Chart(document.getElementById("costByService"), {{
      type: "bar",
      data: {{
        labels: svcLabels.length ? svcLabels : ["(no cost events today)"],
        datasets: [{{
          label: "Today cost (AUD)",
          data: svcLabels.length ? svcValues : [0],
          borderWidth: 1,
          borderRadius: 10
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{ enabled: true }}
        }},
        scales: {{
          x: {{
            ticks: {{ color: "rgba(234,240,255,.65)" }},
            grid: {{ color: "rgba(255,255,255,.06)" }}
          }},
          y: {{
            ticks: {{ color: "rgba(234,240,255,.65)" }},
            grid: {{ color: "rgba(255,255,255,.06)" }}
          }}
        }}
      }}
    }});
  }}

  formatDelta();
  renderMix();
  renderTable();
  renderCharts();
</script>

</body>
</html>
    """
    return HTMLResponse(html)
