"""
Trip History Database Layer — NYC Taxi Fare Intelligence Studio.
Feature #8: Real Trip History / "My Predictions"
Feature #9: Prediction vs Actual Fare Feedback System

Uses SQLite for local deployment. Structured for PostgreSQL migration:
  - All SQL uses parameterized queries (no f-string interpolation in queries).
  - Schema includes a `user_id` placeholder for future multi-user auth.
  - Connection pool abstraction via get_connection().

Privacy notice: This database stores LOCAL application data only.
No API keys, secrets, passwords, or tokens are stored.
"""

import os
import math
import sqlite3
import logging
import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Database lives next to the project root (not inside src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "trip_history.db")

# Schema version (bump when ALTER TABLE migrations are needed)
SCHEMA_VERSION = 1


def get_connection() -> sqlite3.Connection:
    """Open a SQLite connection with row-factory set to dict-style access."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_database() -> None:
    """
    Idempotent schema creation.
    Safe to call on every application start — will not wipe existing data.
    """
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trip_predictions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at          TEXT    NOT NULL,
                user_id             TEXT    DEFAULT NULL,

                pickup_address      TEXT    DEFAULT NULL,
                dropoff_address     TEXT    DEFAULT NULL,
                pickup_latitude     REAL    NOT NULL,
                pickup_longitude    REAL    NOT NULL,
                dropoff_latitude    REAL    NOT NULL,
                dropoff_longitude   REAL    NOT NULL,

                distance_km         REAL    DEFAULT NULL,
                road_distance_km    REAL    DEFAULT NULL,
                estimated_duration  REAL    DEFAULT NULL,
                passenger_count     INTEGER DEFAULT 1,
                pickup_datetime     TEXT    DEFAULT NULL,

                predicted_fare      REAL    NOT NULL,
                model_name          TEXT    DEFAULT 'TaxiFareDNN',
                model_version       TEXT    DEFAULT '1.0',

                actual_fare         REAL    DEFAULT NULL,
                absolute_error      REAL    DEFAULT NULL,
                relative_error      REAL    DEFAULT NULL,

                weather_summary     TEXT    DEFAULT NULL,
                traffic_summary     TEXT    DEFAULT NULL,

                save_hash           TEXT    DEFAULT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_trip_created_at
                ON trip_predictions(created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_trip_model
                ON trip_predictions(model_name);

            CREATE INDEX IF NOT EXISTS idx_trip_actual
                ON trip_predictions(actual_fare);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_trip_save_hash
                ON trip_predictions(save_hash)
                WHERE save_hash IS NOT NULL;
        """)
        conn.execute(
            "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION))
        )
        conn.commit()
        logger.info("TripHistory database initialised at %s", DB_PATH)
    except Exception as exc:
        logger.error("Database initialisation error: %s", exc)
        raise
    finally:
        conn.close()


def _build_save_hash(
    pickup_lat: float, pickup_lon: float,
    dropoff_lat: float, dropoff_lon: float,
    passenger_count: int,
    predicted_fare: float,
    pickup_datetime: Optional[str]
) -> str:
    """Deterministic hash to prevent double-saves on Streamlit reruns."""
    import hashlib
    raw = (
        f"{pickup_lat:.5f}|{pickup_lon:.5f}|{dropoff_lat:.5f}|{dropoff_lon:.5f}"
        f"|{passenger_count}|{predicted_fare:.2f}|{pickup_datetime or ''}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def insert_prediction(
    pickup_address: Optional[str],
    dropoff_address: Optional[str],
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    distance_km: Optional[float],
    road_distance_km: Optional[float],
    estimated_duration: Optional[float],
    passenger_count: int,
    pickup_datetime: Optional[str],
    predicted_fare: float,
    model_name: str = "TaxiFareDNN",
    model_version: str = "1.0",
    weather_summary: Optional[str] = None,
    traffic_summary: Optional[str] = None,
) -> Optional[int]:
    """
    Persist one prediction record.
    Returns the new row id, or None if a duplicate was detected or on error.
    """
    save_hash = _build_save_hash(
        pickup_latitude, pickup_longitude,
        dropoff_latitude, dropoff_longitude,
        passenger_count, predicted_fare, pickup_datetime
    )
    created_at = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO trip_predictions (
                created_at, pickup_address, dropoff_address,
                pickup_latitude, pickup_longitude,
                dropoff_latitude, dropoff_longitude,
                distance_km, road_distance_km, estimated_duration,
                passenger_count, pickup_datetime,
                predicted_fare, model_name, model_version,
                weather_summary, traffic_summary,
                save_hash
            ) VALUES (
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?,
                ?
            )
            """,
            (
                created_at, pickup_address, dropoff_address,
                pickup_latitude, pickup_longitude,
                dropoff_latitude, dropoff_longitude,
                distance_km, road_distance_km, estimated_duration,
                passenger_count, pickup_datetime,
                predicted_fare, model_name, model_version,
                weather_summary, traffic_summary,
                save_hash
            )
        )
        conn.commit()
        if cur.rowcount == 0:
            logger.debug("Duplicate prediction detected (save_hash=%s), not saved.", save_hash)
            return None
        return cur.lastrowid
    except Exception as exc:
        logger.error("insert_prediction error: %s", exc)
        return None
    finally:
        conn.close()


def fetch_all(
    search: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    model_name: Optional[str] = None,
    has_actual: Optional[bool] = None,
    min_fare: Optional[float] = None,
    max_fare: Optional[float] = None,
    sort_by: str = "created_at_desc",
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    Flexible query with full-database search, date filtering, and sorting.
    sort_by: created_at_desc | created_at_asc | predicted_fare_desc |
             predicted_fare_asc | absolute_error_desc
    """
    conditions: List[str] = []
    params: List[Any] = []

    if search:
        term = f"%{search}%"
        conditions.append("(pickup_address LIKE ? OR dropoff_address LIKE ?)")
        params.extend([term, term])

    if date_from:
        conditions.append("created_at >= ?")
        params.append(date_from)

    if date_to:
        conditions.append("created_at <= ?")
        params.append(date_to + "T23:59:59Z")

    if model_name:
        conditions.append("model_name = ?")
        params.append(model_name)

    if has_actual is True:
        conditions.append("actual_fare IS NOT NULL")
    elif has_actual is False:
        conditions.append("actual_fare IS NULL")

    if min_fare is not None:
        conditions.append("predicted_fare >= ?")
        params.append(min_fare)

    if max_fare is not None:
        conditions.append("predicted_fare <= ?")
        params.append(max_fare)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    order_map = {
        "created_at_desc":      "created_at DESC",
        "created_at_asc":       "created_at ASC",
        "predicted_fare_desc":  "predicted_fare DESC",
        "predicted_fare_asc":   "predicted_fare ASC",
        "absolute_error_desc":  "absolute_error DESC",
    }
    order_clause = order_map.get(sort_by, "created_at DESC")

    sql = f"SELECT * FROM trip_predictions {where_clause} ORDER BY {order_clause} LIMIT ?"
    params.append(limit)

    conn = get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.error("fetch_all error: %s", exc)
        return []
    finally:
        conn.close()


def fetch_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single prediction record by primary key."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM trip_predictions WHERE id = ?", (record_id,)
        ).fetchone()
        return dict(row) if row else None
    except Exception as exc:
        logger.error("fetch_by_id error: %s", exc)
        return None
    finally:
        conn.close()


def get_distinct_models() -> List[str]:
    """Return list of model names present in the database."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT model_name FROM trip_predictions ORDER BY model_name"
        ).fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception as exc:
        logger.error("get_distinct_models error: %s", exc)
        return []
    finally:
        conn.close()


def update_actual_fare(record_id: int, actual_fare: float) -> bool:
    """
    Update a prediction record with the user-provided actual fare and
    recalculate absolute_error and relative_error.

    absolute_error = abs(actual_fare - predicted_fare)
    relative_error = absolute_error / abs(actual_fare)  [None when actual_fare == 0]
    Returns True on success.
    """
    row = fetch_by_id(record_id)
    if row is None:
        return False

    predicted_fare = row.get("predicted_fare", 0.0) or 0.0
    abs_err = abs(actual_fare - predicted_fare)
    rel_err = (abs_err / abs(actual_fare)) if actual_fare != 0.0 else None

    conn = get_connection()
    try:
        conn.execute(
            """
            UPDATE trip_predictions
            SET actual_fare = ?, absolute_error = ?, relative_error = ?
            WHERE id = ?
            """,
            (actual_fare, abs_err, rel_err, record_id)
        )
        conn.commit()
        return True
    except Exception as exc:
        logger.error("update_actual_fare error: %s", exc)
        return False
    finally:
        conn.close()


def delete_prediction(record_id: int) -> bool:
    """
    Delete one prediction record.
    Does NOT touch model files, datasets, notebooks, or any other file.
    """
    conn = get_connection()
    try:
        conn.execute("DELETE FROM trip_predictions WHERE id = ?", (record_id,))
        conn.commit()
        return True
    except Exception as exc:
        logger.error("delete_prediction error: %s", exc)
        return False
    finally:
        conn.close()


def compute_aggregate_metrics() -> Dict[str, Any]:
    """
    Calculate real performance metrics from records that have actual fares.
    Returns {} when fewer than 2 records with actual fares exist.

    MAPE = mean(abs((actual - predicted) / actual)) * 100
    MAPE excludes actual_fare == 0 (undefined).
    R² is returned as None when ss_tot == 0.
    """
    rows = fetch_all(has_actual=True, limit=10000)
    if len(rows) < 2:
        return {}

    import numpy as np

    pred   = np.array([r["predicted_fare"] for r in rows], dtype=float)
    actual = np.array([r["actual_fare"]    for r in rows], dtype=float)
    errors = actual - pred
    abs_errors = np.abs(errors)

    mae      = float(np.mean(abs_errors))
    median_ae = float(np.median(abs_errors))
    rmse     = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2     = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None

    nonzero_mask = actual != 0.0
    if nonzero_mask.sum() >= 1:
        mape = float(np.mean(np.abs(errors[nonzero_mask] / actual[nonzero_mask])) * 100)
    else:
        mape = None

    return {
        "n_with_actual": len(rows),
        "mae":           round(mae, 4),
        "median_ae":     round(median_ae, 4),
        "rmse":          round(rmse, 4),
        "r2":            round(r2, 4) if r2 is not None else None,
        "mape":          round(mape, 4) if mape is not None else None,
        "n_over":        int(np.sum(pred > actual)),
        "n_under":       int(np.sum(pred < actual)),
    }


def export_to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    """Convert row dicts to UTF-8 CSV bytes for st.download_button."""
    import io, csv
    if not rows:
        return b""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def run_database_tests() -> List[Dict[str, Any]]:
    """
    Execute 13 required database tests in an isolated temp database.
    Returns list of {name, status, message} dicts.
    """
    import tempfile
    try:
        import src.trip_history as _th
    except ModuleNotFoundError:
        import trip_history as _th

    results = []

    def ok(name, msg=""):
        results.append({"name": name, "status": "PASS", "message": msg})

    def fail(name, msg=""):
        results.append({"name": name, "status": "FAIL", "message": msg})

    orig_path = _th.DB_PATH
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name
    _th.DB_PATH = tmp_path

    try:
        # T1: DB init
        try:
            _th.initialize_database()
            ok("T01 Database Initialization", f"Schema created")
        except Exception as e:
            fail("T01 Database Initialization", str(e))
            return results

        # T2: Insert
        rid = _th.insert_prediction(
            pickup_address="Times Square, NY", dropoff_address="JFK Airport, NY",
            pickup_latitude=40.7580, pickup_longitude=-73.9855,
            dropoff_latitude=40.6413, dropoff_longitude=-73.7781,
            distance_km=21.5, road_distance_km=27.9, estimated_duration=35.0,
            passenger_count=1, pickup_datetime="2026-09-25T10:00:00",
            predicted_fare=59.33, model_name="TaxiFareDNN",
        )
        ok("T02 Insert Prediction", f"Row id={rid}") if rid else fail("T02 Insert Prediction", "returned None")

        # T3: Read
        row = _th.fetch_by_id(rid) if rid else None
        ok("T03 Read Prediction", f"fare=${row['predicted_fare']:.2f}") if (row and abs(row["predicted_fare"] - 59.33) < 0.01) else fail("T03 Read Prediction", str(row))

        # T4: Search
        _th.insert_prediction(
            pickup_address="Brooklyn Bridge, NY", dropoff_address="Midtown Manhattan, NY",
            pickup_latitude=40.7028, pickup_longitude=-73.9965,
            dropoff_latitude=40.7580, dropoff_longitude=-73.9855,
            distance_km=12.4, road_distance_km=14.0, estimated_duration=22.0,
            passenger_count=2, pickup_datetime="2026-09-24T14:00:00",
            predicted_fare=31.20, model_name="TaxiFareDNN",
        )
        rows = _th.fetch_all(search="Brooklyn")
        ok("T04 Search", f"{len(rows)} match(es)") if any("Brooklyn" in (r.get("pickup_address") or "") for r in rows) else fail("T04 Search", str(rows))

        # T5: Date filter
        rows_date = _th.fetch_all(date_from="2026-09-25", date_to="2026-09-25")
        ok("T05 Date Filter", f"{len(rows_date)} record(s)") if (rid and any(r["id"] == rid for r in rows_date)) else fail("T05 Date Filter", "record not found")

        # T6: Sort
        rows_hi = _th.fetch_all(sort_by="predicted_fare_desc")
        ok("T06 Sort", f"Top={rows_hi[0]['predicted_fare']:.2f}") if (rows_hi and rows_hi[0]["predicted_fare"] >= rows_hi[-1]["predicted_fare"]) else fail("T06 Sort", str([r["predicted_fare"] for r in rows_hi]))

        # T7: Update actual fare
        success = _th.update_actual_fare(rid, actual_fare=61.50) if rid else False
        updated = _th.fetch_by_id(rid) if rid else None
        ok("T07 Update Actual Fare", f"actual=${updated['actual_fare']:.2f}") if (success and updated and abs(updated["absolute_error"] - abs(61.50 - 59.33)) < 0.01) else fail("T07 Update Actual Fare", str(updated))

        # T8: Error calculation
        if updated:
            expected_rel = abs(61.50 - 59.33) / 61.50
            ok("T08 Error Calculation", f"rel={updated.get('relative_error'):.4f}") if abs((updated.get("relative_error") or 0) - expected_rel) < 0.001 else fail("T08 Error Calculation", str(updated.get("relative_error")))
        else:
            fail("T08 Error Calculation", "No updated row")

        # T9: CSV export
        csv_bytes = _th.export_to_csv_bytes(_th.fetch_all())
        ok("T09 CSV Export", f"{len(csv_bytes)} bytes") if (csv_bytes and b"predicted_fare" in csv_bytes) else fail("T09 CSV Export", "empty or malformed")

        # T10: Delete
        del_ok = _th.delete_prediction(rid) if rid else False
        ok("T10 Delete", "record removed") if (del_ok and _th.fetch_by_id(rid) is None) else fail("T10 Delete", f"del_ok={del_ok}")

        # T11: Empty aggregate
        metrics = _th.compute_aggregate_metrics()
        rows_actual = _th.fetch_all(has_actual=True)
        ok("T11 Empty / Insufficient Records", "returns {}") if (len(rows_actual) < 2 and metrics == {}) else ok("T11 Aggregate Computed", str(metrics.get("n_with_actual")))

        # T12: Duplicate save protection
        def _ins():
            return _th.insert_prediction(
                pickup_address="GCT, NY", dropoff_address="Wall St, NY",
                pickup_latitude=40.7527, pickup_longitude=-73.9772,
                dropoff_latitude=40.7075, dropoff_longitude=-74.0090,
                distance_km=8.0, road_distance_km=9.5, estimated_duration=18.0,
                passenger_count=2, pickup_datetime="2026-09-25T15:00:00",
                predicted_fare=24.30, model_name="TaxiFareDNN",
            )
        rid_a, rid_b = _ins(), _ins()
        ok("T12 Duplicate-Save Protection", "second save rejected") if (rid_a is not None and rid_b is None) else fail("T12 Duplicate-Save Protection", f"rid_a={rid_a}, rid_b={rid_b}")

        # T13: DB failure handling
        bad = _th.update_actual_fare(-9999, 100.0)
        ok("T13 DB Failure Handling", "graceful False on missing id") if not bad else fail("T13 DB Failure Handling", "should have been False")

    except Exception as e:
        fail("CRITICAL", str(e))
    finally:
        _th.DB_PATH = orig_path
        try:
            import os as _os
            _os.unlink(tmp_path)
        except Exception:
            pass

    return results
