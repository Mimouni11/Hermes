"""The appointment book, kept in a SQLite file.

Three tables, and nothing else:

- ``rooms``         one row per examination room, with the modality it runs
- ``slots``         every bookable window, ``status`` in free / held / booked
- ``appointments``  one row per booking, identified by a 6-character ref code

Plain functions, no LLM and no network. Every write goes through here, so the
booking rules live in one place:

1. the slot must exist,
2. it must be free, or held by *this* call — a hold carries the ``call_id`` that
   took it and expires after :data:`HOLD_TTL_MINUTES`; an expired hold counts as
   free, and a live hold belonging to another call is refused,
3. the room's modality must match the exam's,
4. an exam that needs fasting ("à jeun") must **start** before 12:00 — a window
   that runs past noon is fine, so a 50-minute fasting exam may be booked at
   11:45 and end at 12:35,
5. **an exam longer than one slot takes consecutive slots in the same room** —
   49 of the 173 catalogue exams are longer than their modality's slot length,
   so this is the normal case, not an edge case. Each slot's own length is used,
   never one room's length applied to another, so two rooms of the same modality
   may run different slot lengths,
6. a slot can never be booked twice — the status change is a conditional
   ``UPDATE`` inside a ``BEGIN IMMEDIATE`` transaction, and for a multi-slot exam
   it is one statement over the whole run whose row count must match, so a
   partially booked appointment cannot exist.

Exam metadata (modality, duration, fasting) is read from the exam catalogue CSV,
not duplicated into the database.

Appointments hold no real personal data: ``patient_alias`` is an alias the caller
chooses, and ``phone`` is stored masked (see :func:`mask_phone`).
"""

from __future__ import annotations

import csv
import math
import random
import sqlite3
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator, Literal, Sequence

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "processed" / "agenda.db"
DEFAULT_EXAMS_CSV = _PROJECT_ROOT / "data" / "raw" / "exams.csv"

DB_PATH: Path = DEFAULT_DB_PATH
EXAMS_CSV: Path = DEFAULT_EXAMS_CSV

DT_FORMAT = "%Y-%m-%d %H:%M"
DATE_FORMAT = "%Y-%m-%d"

#: An exam that needs fasting cannot *start* at or after this hour. The rule is
#: on the window's start only: a long exam may run past it.
FASTING_CUTOFF_HOUR = 12

#: How long a hold survives before it counts as free again. A caller who hangs
#: up mid-booking must not take a slot off the market for the rest of the day.
HOLD_TTL_MINUTES = 5

#: Alphabet for ref codes: no I, O, 0 or 1, so codes survive being read aloud.
REF_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
REF_LENGTH = 6

#: Refuse to chain more than this many slots into one appointment.
MAX_SLOTS_PER_APPOINTMENT = 12

SlotStatus = Literal["free", "held", "booked"]
AppointmentStatus = Literal["confirmed", "cancelled"]

MORNING = {"matin", "morning", "am", "matinee", "matinée"}
AFTERNOON = {"apres-midi", "après-midi", "apresmidi", "afternoon", "pm", "soir"}


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class AgendaError(RuntimeError):
    """Base class for every booking rule the agenda refuses to break."""


class UnknownExam(AgendaError):
    """The exam id is not in the catalogue."""


class SlotNotFound(AgendaError):
    """No slot with that id exists."""


class SlotUnavailable(AgendaError):
    """The slot is already booked, or not in a state the operation allows."""


class NotEnoughConsecutiveSlots(SlotUnavailable):
    """The exam is longer than one slot and the room has no long enough run free."""


class HoldBelongsToAnotherCall(SlotUnavailable):
    """The slot is held by a different ``call_id`` and that hold is still live."""


class ModalityMismatch(AgendaError):
    """The room cannot run this exam's modality."""


class FastingRuleViolation(AgendaError):
    """An exam needing fasting was offered an afternoon slot."""


class UnknownRefCode(AgendaError):
    """No live appointment carries that reference code."""


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Room:
    """One examination room."""

    id: str
    modality: str
    name: str


@dataclass(frozen=True)
class Slot:
    """One row of the ``slots`` table."""

    id: str
    room_id: str
    room_name: str
    modality: str
    start: datetime
    end: datetime
    status: SlotStatus
    appointment_id: int | None = None
    call_id: str | None = None
    held_until: datetime | None = None

    @property
    def minutes(self) -> int:
        """How long this slot is."""
        return int((self.end - self.start).total_seconds() // 60)

    @property
    def is_morning(self) -> bool:
        """Whether this slot *starts* before the fasting cut-off.

        Only the start matters: a 50-minute exam beginning at 11:45 is a morning
        appointment even though it ends at 12:35.
        """
        return self.start.hour < FASTING_CUTOFF_HOUR

    def hold_is_live(self, now: datetime | None = None) -> bool:
        """Whether this slot carries a hold that has not yet expired."""
        if self.status != "held":
            return False
        if self.held_until is None:
            return True
        return self.held_until > (now or datetime.now())


@dataclass(frozen=True)
class Offer:
    """A window an exam actually fits in: one slot, or several consecutive ones.

    ``slot_id`` is the first slot of the run and all :func:`confirm_booking`
    needs: the agenda works the rest of the run out from the exam's duration and
    the room's own slot lengths. Pass the whole ``slot_ids`` tuple to
    :func:`hold_slot`, which holds exactly the slots it is given.
    """

    exam_id: str
    slot_id: str
    slot_ids: tuple[str, ...]
    room_id: str
    room_name: str
    modality: str
    start: datetime
    end: datetime

    @property
    def minutes(self) -> int:
        """The length of the whole window."""
        return int((self.end - self.start).total_seconds() // 60)

    @property
    def n_slots(self) -> int:
        """How many slots this appointment occupies."""
        return len(self.slot_ids)

    @property
    def is_morning(self) -> bool:
        """Whether the window *starts* before the fasting cut-off.

        The window's end is not considered, so a fasting exam offered at 11:45
        and running to 12:35 is a legitimate morning offer.
        """
        return self.start.hour < FASTING_CUTOFF_HOUR


@dataclass(frozen=True)
class Exam:
    """The slice of the catalogue the agenda needs."""

    id: str
    modality: str
    name_fr: str
    duration_min: int
    requires_fasting: bool


@dataclass(frozen=True)
class Appointment:
    """One booking."""

    id: int
    ref_code: str
    exam_id: str
    slot_id: str
    patient_alias: str
    phone: str | None
    status: AppointmentStatus
    created_at: datetime
    start: datetime | None = None
    end: datetime | None = None
    room_name: str | None = None
    slot_ids: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def configure(db_path: str | Path | None = None, exams_csv: str | Path | None = None) -> None:
    """Point the module at another database or catalogue, and drop the caches.

    Used by the tests and by the build script; the demo uses the defaults.
    """
    global DB_PATH, EXAMS_CSV
    if db_path is not None:
        DB_PATH = Path(db_path)
    if exams_csv is not None:
        EXAMS_CSV = Path(exams_csv)
    _EXAM_CACHE.clear()


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    """Return ``text`` without diacritics, so "échographie" matches "echographie"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalise_modality(modality: str) -> str:
    """Normalise a modality to a comparable key, accents and case removed."""
    key = strip_accents(modality or "").strip().lower()
    aliases = {
        "mri": "irm",
        "ct": "scanner",
        "tdm": "scanner",
        "scanographie": "scanner",
        "tomodensitometrie": "scanner",
        "radio": "radiographie",
        "rx": "radiographie",
        "echo": "echographie",
        "us": "echographie",
        "mammo": "mammographie",
        "dexa": "densitometrie",
        "osteodensitometrie": "densitometrie",
        "panoramique": "dentaire",
        "cbct": "dentaire",
    }
    return aliases.get(key, key)


def needs_fasting(preparation: str) -> bool:
    """Whether a preparation note tells the patient to come fasting."""
    return "a jeun" in strip_accents(preparation or "").lower()


def mask_phone(phone: str | None) -> str | None:
    """Mask a phone number, keeping only its last two digits.

    The centre's real call-back list lives in its own system; this database is a
    school project and keeps nothing that identifies a caller.
    """
    if phone is None:
        return None
    digits = [ch for ch in phone if ch.isdigit()]
    if not digits:
        return None
    if len(digits) <= 2:
        return "*" * len(digits)
    return "*" * (len(digits) - 2) + "".join(digits[-2:])


# --------------------------------------------------------------------------
# The exam catalogue
# --------------------------------------------------------------------------

_EXAM_CACHE: dict[str, Exam] = {}


def load_exams(path: str | Path | None = None, refresh: bool = False) -> dict[str, Exam]:
    """Read the exam catalogue CSV and return it keyed by exam id."""
    global _EXAM_CACHE
    if _EXAM_CACHE and not refresh and path is None:
        return _EXAM_CACHE

    csv_path = Path(path) if path is not None else EXAMS_CSV
    exams: dict[str, Exam] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            exam_id = (row.get("id") or "").strip()
            if not exam_id:
                continue
            try:
                duration = int(float(row.get("duration_min") or 0))
            except ValueError:
                duration = 0
            exams[exam_id] = Exam(
                id=exam_id,
                modality=normalise_modality(row.get("modality") or ""),
                name_fr=(row.get("name_fr") or "").strip(),
                duration_min=duration,
                requires_fasting=needs_fasting(row.get("preparation_fr") or ""),
            )

    if path is None:
        _EXAM_CACHE = exams
    return exams


def get_exam(exam_id: str) -> Exam:
    """Look one exam up, raising :class:`UnknownExam` if it is not in the catalogue."""
    exams = load_exams()
    try:
        return exams[exam_id.strip()]
    except KeyError as exc:
        raise UnknownExam(f"unknown exam id: {exam_id!r}") from exc


def modal_durations(path: str | Path | None = None) -> dict[str, int]:
    """Return the most common ``duration_min`` per modality, used as slot length."""
    counts: dict[str, dict[int, int]] = {}
    for exam in load_exams(path, refresh=True).values():
        if exam.duration_min <= 0:
            continue
        counts.setdefault(exam.modality, {})
        counts[exam.modality][exam.duration_min] = (
            counts[exam.modality].get(exam.duration_min, 0) + 1
        )
    # Ties break towards the shorter duration, so the agenda stays granular.
    return {
        modality: sorted(by_duration.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for modality, by_duration in counts.items()
    }


def slots_needed(duration_min: int, slot_minutes: int) -> int:
    """How many consecutive slots an exam of ``duration_min`` occupies.

    Rounds up: a 45-minute exam in a 25-minute room takes two slots, and the
    second is not offered to anyone else even though 5 minutes of it go unused.
    """
    if slot_minutes <= 0:
        raise AgendaError(f"slot length must be positive, got {slot_minutes}")
    if duration_min <= 0:
        return 1
    return max(1, math.ceil(duration_min / slot_minutes))


# --------------------------------------------------------------------------
# Database plumbing
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS rooms (
    id       TEXT PRIMARY KEY,
    modality TEXT NOT NULL,
    name     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS slots (
    id             TEXT PRIMARY KEY,
    room_id        TEXT NOT NULL REFERENCES rooms(id),
    start          TEXT NOT NULL,
    "end"          TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'free'
                   CHECK (status IN ('free', 'held', 'booked')),
    -- Which appointment holds this slot, if any. A multi-slot exam puts the
    -- same id on every slot of its run, which is how cancelling knows what to
    -- free. NULL on a free slot, and on the pre-booked slots the build script
    -- creates to stand for bookings taken at the counter.
    appointment_id INTEGER REFERENCES appointments(id),
    -- Which call holds this slot, and until when. Both NULL unless status is
    -- 'held'. A hold whose held_until has passed counts as free everywhere, so
    -- a caller who hangs up mid-booking cannot park a slot indefinitely.
    call_id        TEXT,
    held_until     TEXT,
    UNIQUE (room_id, start)
);

CREATE TABLE IF NOT EXISTS appointments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ref_code      TEXT NOT NULL UNIQUE,
    exam_id       TEXT NOT NULL,
    slot_id       TEXT NOT NULL REFERENCES slots(id),
    patient_alias TEXT NOT NULL,
    phone         TEXT,
    status        TEXT NOT NULL DEFAULT 'confirmed'
                  CHECK (status IN ('confirmed', 'cancelled')),
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_slots_lookup ON slots (status, start);
CREATE INDEX IF NOT EXISTS idx_slots_room   ON slots (room_id, start);
CREATE INDEX IF NOT EXISTS idx_slots_appt   ON slots (appointment_id);
CREATE INDEX IF NOT EXISTS idx_appt_ref     ON appointments (ref_code);
CREATE INDEX IF NOT EXISTS idx_appt_slot    ON appointments (slot_id);
"""


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open the agenda database with explicit transactions and row access by name."""
    path = Path(db_path) if db_path is not None else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block inside ``BEGIN IMMEDIATE``, rolling back on any exception.

    ``IMMEDIATE`` takes the write lock up front, so two processes cannot both
    read a slot as free and then both book it.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _migrate_slots(conn: sqlite3.Connection) -> None:
    """Add the hold-ownership columns to a ``slots`` table that predates them.

    ``CREATE TABLE IF NOT EXISTS`` leaves an existing table alone, so a database
    built before holds had an owner would otherwise be missing both columns.
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(slots)")}
    if not existing:
        return
    for column in ("call_id", "held_until"):
        if column not in existing:
            conn.execute(f"ALTER TABLE slots ADD COLUMN {column} TEXT")


def init_db(db_path: str | Path | None = None, reset: bool = False) -> None:
    """Create the three tables, optionally dropping what is already there."""
    conn = connect(db_path)
    try:
        if reset:
            conn.executescript(
                "DROP TABLE IF EXISTS appointments;"
                "DROP TABLE IF EXISTS slots;"
                "DROP TABLE IF EXISTS rooms;"
            )
        _migrate_slots(conn)
        conn.executescript(SCHEMA)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Row mapping
# --------------------------------------------------------------------------

def _parse_dt(value: str) -> datetime:
    return datetime.strptime(value, DT_FORMAT)


def _get(row: sqlite3.Row, key: str, default: object = None) -> object:
    return row[key] if key in row.keys() else default


def _slot_from_row(row: sqlite3.Row) -> Slot:
    held_until = _get(row, "held_until")
    return Slot(
        id=row["id"],
        room_id=row["room_id"],
        room_name=str(_get(row, "room_name", "") or ""),
        modality=str(_get(row, "modality", "") or ""),
        start=_parse_dt(row["start"]),
        end=_parse_dt(row["end"]),
        status=row["status"],
        appointment_id=_get(row, "appointment_id"),  # type: ignore[arg-type]
        call_id=_get(row, "call_id"),  # type: ignore[arg-type]
        held_until=_parse_dt(str(held_until)) if held_until else None,
    )


def _appointment_from_row(
    row: sqlite3.Row,
    slot_ids: Sequence[str] = (),
) -> Appointment:
    start = _get(row, "start")
    end = _get(row, "end")
    return Appointment(
        id=row["id"],
        ref_code=row["ref_code"],
        exam_id=row["exam_id"],
        slot_id=row["slot_id"],
        patient_alias=row["patient_alias"],
        phone=row["phone"],
        status=row["status"],
        created_at=_parse_dt(row["created_at"]),
        start=_parse_dt(str(start)) if start else None,
        end=_parse_dt(str(end)) if end else None,
        room_name=_get(row, "room_name"),  # type: ignore[arg-type]
        slot_ids=tuple(slot_ids),
    )


_SLOT_SELECT = """
SELECT s.id, s.room_id, s.start, s."end", s.status, s.appointment_id,
       s.call_id, s.held_until, r.name AS room_name, r.modality
FROM slots s JOIN rooms r ON r.id = s.room_id
"""

_APPT_SELECT = """
SELECT a.id, a.ref_code, a.exam_id, a.slot_id, a.patient_alias, a.phone,
       a.status, a.created_at, s.start, s."end", r.name AS room_name
FROM appointments a
LEFT JOIN slots s ON s.id = a.slot_id
LEFT JOIN rooms r ON r.id = s.room_id
"""


def _fetch_slot(conn: sqlite3.Connection, slot_id: str) -> Slot:
    row = conn.execute(_SLOT_SELECT + " WHERE s.id = ?", (slot_id,)).fetchone()
    if row is None:
        raise SlotNotFound(f"no slot with id {slot_id!r}")
    return _slot_from_row(row)


def _appointment_slot_ids(conn: sqlite3.Connection, appointment_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT id FROM slots WHERE appointment_id = ? ORDER BY start",
        (appointment_id,),
    ).fetchall()
    return [row["id"] for row in rows]


def _fetch_appointment(conn: sqlite3.Connection, ref_code: str) -> Appointment:
    row = conn.execute(
        _APPT_SELECT + " WHERE a.ref_code = ? AND a.status = 'confirmed'",
        (ref_code.strip().upper(),),
    ).fetchone()
    if row is None:
        raise UnknownRefCode(f"no confirmed appointment with ref code {ref_code!r}")
    appointment = _appointment_from_row(row, _appointment_slot_ids(conn, row["id"]))
    # The window runs to the end of the last slot, not the first.
    if appointment.slot_ids:
        last = conn.execute(
            'SELECT "end" FROM slots WHERE id = ?', (appointment.slot_ids[-1],)
        ).fetchone()
        if last is not None:
            appointment = replace(appointment, end=_parse_dt(last["end"]))
    return appointment


# --------------------------------------------------------------------------
# Runs of consecutive slots
# --------------------------------------------------------------------------

def _row_minutes(row: sqlite3.Row) -> int:
    """How long one slot row is, from its own start and end."""
    return int((_parse_dt(row["end"]) - _parse_dt(row["start"])).total_seconds() // 60)


def _is_contiguous(rows: Sequence[sqlite3.Row]) -> bool:
    """Whether each row's end is exactly the next row's start, same room."""
    for first, second in zip(rows, rows[1:]):
        if first["room_id"] != second["room_id"]:
            return False
        if first["end"] != second["start"]:
            return False
    return True


def _runs_for_duration(
    rows: Sequence[sqlite3.Row],
    duration_min: int,
) -> list[list[sqlite3.Row]]:
    """Find every run of contiguous slots long enough for ``duration_min``.

    Each run is grown from its own slots' lengths until it covers the exam, so a
    room with 10-minute slots needs five for a 50-minute exam while a room with
    25-minute slots needs two — the two can sit side by side in one modality.

    Only available slots are passed in, so a booked slot in the middle of a day
    breaks the chain by itself: the end/start handshake simply fails.
    """
    target = max(duration_min, 1)

    by_room: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_room.setdefault(row["room_id"], []).append(row)

    runs: list[list[sqlite3.Row]] = []
    for room_rows in by_room.values():
        room_rows.sort(key=lambda r: r["start"])
        for i, first in enumerate(room_rows):
            window = [first]
            covered = _row_minutes(first)
            while covered < target and len(window) < MAX_SLOTS_PER_APPOINTMENT:
                following = room_rows[i + len(window) : i + len(window) + 1]
                if not following or following[0]["start"] != window[-1]["end"]:
                    break
                window.append(following[0])
                covered += _row_minutes(following[0])
            if covered >= target:
                runs.append(window)
    runs.sort(key=lambda w: (w[0]["start"], w[0]["room_id"]))
    return runs


def _check_available(
    part: Slot,
    now: datetime,
    call_id: str | None,
    own_appointment_id: int | None,
) -> None:
    """Raise unless ``part`` can be claimed by this caller.

    Free is available. A hold is available to the call that took it, and to
    anyone once it has expired. A booked slot is available only to the
    appointment that already owns it, so an appointment can be moved onto a
    window overlapping where it already sits.
    """
    if part.status == "free":
        return
    if part.status == "held":
        if not part.hold_is_live(now):
            return
        if call_id is not None and part.call_id == call_id:
            return
        until = f" until {part.held_until:%H:%M}" if part.held_until else ""
        raise HoldBelongsToAnotherCall(
            f"slot {part.id} is held by another call{until}"
        )
    if own_appointment_id is not None and part.appointment_id == own_appointment_id:
        return
    raise NotEnoughConsecutiveSlots(
        f"slot {part.id} is {part.status}, so the run from there is not available"
    )


def _run_starting_at(
    conn: sqlite3.Connection,
    slot: Slot,
    exam: Exam,
    call_id: str | None = None,
    own_appointment_id: int | None = None,
    now: datetime | None = None,
) -> list[Slot]:
    """Return the consecutive slots starting at ``slot`` that cover ``exam``.

    The run is grown from each slot's own length, so the same exam takes two
    slots in a 25-minute room and five in a 10-minute one.
    """
    moment = now or datetime.now()
    target = max(exam.duration_min, 1)

    rows = conn.execute(
        _SLOT_SELECT + " WHERE s.room_id = ? AND s.start >= ? ORDER BY s.start LIMIT ?",
        (
            slot.room_id,
            slot.start.strftime(DT_FORMAT),
            MAX_SLOTS_PER_APPOINTMENT + 1,
        ),
    ).fetchall()

    window: list[sqlite3.Row] = []
    covered = 0
    for row in rows:
        if window and row["start"] != window[-1]["end"]:
            break
        window.append(row)
        covered += _row_minutes(row)
        if covered >= target:
            break

    if covered < target:
        raise NotEnoughConsecutiveSlots(
            f"slot {slot.id} does not start a long enough run in room "
            f"{slot.room_id}: the exam needs {target} min, and only "
            f"{covered} min of consecutive slots follow"
        )
    if len(window) > MAX_SLOTS_PER_APPOINTMENT:
        raise AgendaError(
            f"exam {exam.id} would need {len(window)} slots in room "
            f"{slot.room_id}; the limit is {MAX_SLOTS_PER_APPOINTMENT}"
        )

    run = [_slot_from_row(row) for row in window]
    for part in run:
        _check_available(part, moment, call_id, own_appointment_id)
    return run


# --------------------------------------------------------------------------
# Rule checks
# --------------------------------------------------------------------------

def check_slot_matches_exam(slot: Slot, exam: Exam) -> None:
    """Raise unless the room can run the exam and the fasting rule is respected.

    The fasting rule is tested on the window's **start** only, via
    :attr:`Slot.is_morning`. An exam that begins before 12:00 and runs past it is
    accepted: the patient's fast ends when the exam begins, not when it finishes,
    so a 50-minute fasting exam booked at 11:45 and ending at 12:35 is valid.
    """
    if normalise_modality(slot.modality) != exam.modality:
        raise ModalityMismatch(
            f"exam {exam.id} is {exam.modality}, "
            f"but room {slot.room_id} runs {normalise_modality(slot.modality)}"
        )
    if exam.requires_fasting and not slot.is_morning:
        raise FastingRuleViolation(
            f"exam {exam.id} requires fasting and must start before "
            f"{FASTING_CUTOFF_HOUR}:00, not at {slot.start.strftime('%H:%M')}"
        )


def _require_bookable(slot: Slot) -> None:
    if slot.status == "booked":
        raise SlotUnavailable(f"slot {slot.id} is already booked")
    if slot.status not in ("free", "held"):
        raise SlotUnavailable(f"slot {slot.id} has status {slot.status!r}")


def _claim(
    conn: sqlite3.Connection,
    slot_ids: Sequence[str],
    call_id: str | None = None,
    own_appointment_id: int | None = None,
    now: datetime | None = None,
) -> None:
    """Flip a whole run to booked in one statement, or raise.

    One ``UPDATE`` for the entire run, with the row count checked: either every
    slot was still available and the appointment is whole, or nothing moved.

    The ``WHERE`` clause repeats the rule in :func:`_check_available`, in SQL:
    free, or a hold this call owns, or a hold that has expired, or a slot this
    appointment already occupies. Re-stating it here is what makes the claim
    safe against a second caller slipping in after the check.
    """
    moment = (now or datetime.now()).strftime(DT_FORMAT)
    placeholders = ",".join("?" * len(slot_ids))

    conditions = [
        "status = 'free'",
        "(status = 'held' AND held_until IS NOT NULL AND held_until <= ?)",
    ]
    params: list[object] = [*slot_ids, moment]
    if call_id is not None:
        conditions.append("(status = 'held' AND call_id = ?)")
        params.append(call_id)
    if own_appointment_id is not None:
        conditions.append("appointment_id = ?")
        params.append(own_appointment_id)

    cursor = conn.execute(
        f"UPDATE slots SET status = 'booked', call_id = NULL, held_until = NULL "
        f"WHERE id IN ({placeholders}) AND ({' OR '.join(conditions)})",
        params,
    )
    if cursor.rowcount != len(slot_ids):
        raise SlotUnavailable(
            f"{len(slot_ids) - cursor.rowcount} of {len(slot_ids)} slots were "
            "taken before the booking completed"
        )


def _attach(conn: sqlite3.Connection, slot_ids: Sequence[str], appointment_id: int) -> None:
    placeholders = ",".join("?" * len(slot_ids))
    conn.execute(
        f"UPDATE slots SET appointment_id = ? WHERE id IN ({placeholders})",
        [appointment_id, *slot_ids],
    )


def _free_appointment_slots(
    conn: sqlite3.Connection,
    appointment_id: int,
    keep: Sequence[str] = (),
) -> None:
    sql = (
        "UPDATE slots SET status = 'free', appointment_id = NULL, "
        "call_id = NULL, held_until = NULL WHERE appointment_id = ?"
    )
    params: list[object] = [appointment_id]
    if keep:
        placeholders = ",".join("?" * len(keep))
        sql += f" AND id NOT IN ({placeholders})"
        params.extend(keep)
    conn.execute(sql, params)


# --------------------------------------------------------------------------
# Reference codes
# --------------------------------------------------------------------------

def new_ref_code(rng: random.Random | None = None) -> str:
    """Return a fresh 6-character reference code, derived from nothing personal."""
    source = rng or random.SystemRandom()
    return "".join(source.choice(REF_ALPHABET) for _ in range(REF_LENGTH))


def _unique_ref_code(conn: sqlite3.Connection, attempts: int = 50) -> str:
    for _ in range(attempts):
        code = new_ref_code()
        taken = conn.execute(
            "SELECT 1 FROM appointments WHERE ref_code = ?", (code,)
        ).fetchone()
        if taken is None:
            return code
    raise AgendaError("could not generate an unused reference code")


# --------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------

def _period_bounds(period: str | None) -> tuple[int, int]:
    """Translate a spoken period into an (inclusive, exclusive) hour range."""
    if period is None:
        return 0, 24
    key = strip_accents(period).strip().lower()
    if key in MORNING:
        return 0, FASTING_CUTOFF_HOUR
    if key in AFTERNOON:
        return FASTING_CUTOFF_HOUR, 24
    raise ValueError(f"unknown period {period!r}; expected 'matin' or 'apres-midi'")


def _as_date(day: date | datetime | str) -> date:
    if isinstance(day, datetime):
        return day.date()
    if isinstance(day, date):
        return day
    return datetime.strptime(day.strip(), DATE_FORMAT).date()


def find_slots(
    exam_id: str,
    day: date | datetime | str | None = None,
    period: str | None = None,
    limit: int = 3,
) -> list[Offer]:
    """Return up to ``limit`` windows this exam could actually be booked into.

    Each :class:`Offer` is a window long enough for the whole exam: one slot for
    a short exam, several consecutive slots in the same room for a long one.
    Only rooms of the exam's modality are considered, only windows starting in
    the future, and — for an exam needing fasting — only windows starting before
    12:00. An empty list means there is nothing to offer, which is the caller's
    signal to widen the search or hand over.

    ``day`` accepts a date or an ISO string, ``period`` the spoken "matin" or
    "après-midi". The filters apply to the window's start, so a long exam may
    run past noon — and a fasting exam may legitimately be offered at 11:45.

    Each room's windows are built from that room's own slot lengths, so two
    rooms of one modality may run different lengths without either being
    measured by the other's.
    """
    exam = get_exam(exam_id)

    start_hour, end_hour = _period_bounds(period)
    if exam.requires_fasting:
        end_hour = min(end_hour, FASTING_CUTOFF_HOUR)
    if start_hour >= end_hour or limit <= 0:
        return []

    # Fetch the available slots, then assemble runs — the hour filter is applied
    # to the run's start afterwards, so a window is never truncated at noon. An
    # expired hold is available again, and is offered like a free slot.
    now = datetime.now().strftime(DT_FORMAT)
    sql = _SLOT_SELECT + (
        " WHERE r.modality = ? AND s.start > ?"
        " AND (s.status = 'free'"
        "      OR (s.status = 'held' AND s.held_until IS NOT NULL"
        "          AND s.held_until <= ?))"
    )
    params: list[object] = [exam.modality, now, now]
    if day is not None:
        sql += " AND substr(s.start, 1, 10) = ?"
        params.append(_as_date(day).strftime(DATE_FORMAT))
    sql += " ORDER BY s.room_id, s.start"

    conn = connect()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    if not rows:
        return []

    offers: list[Offer] = []
    for run in _runs_for_duration(rows, exam.duration_min):
        first, last = run[0], run[-1]
        hour = _parse_dt(first["start"]).hour
        if not (start_hour <= hour < end_hour):
            continue
        offers.append(
            Offer(
                exam_id=exam.id,
                slot_id=first["id"],
                slot_ids=tuple(row["id"] for row in run),
                room_id=first["room_id"],
                room_name=first["room_name"],
                modality=first["modality"],
                start=_parse_dt(first["start"]),
                end=_parse_dt(last["end"]),
            )
        )
        if len(offers) >= limit:
            break
    return offers


def get_slot(slot_id: str) -> Slot:
    """Look one slot up by id."""
    conn = connect()
    try:
        return _fetch_slot(conn, slot_id)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Holds
# --------------------------------------------------------------------------

def hold_slot(slot_ids: str | Sequence[str], call_id: str) -> list[Slot]:
    """Reserve a window for one call while the caller decides.

    ``slot_ids`` is the whole window to hold — pass an :class:`Offer`'s
    ``slot_ids``, since holding only the first slot of a 50-minute IRM would
    leave the second free for someone else. A single slot id is also accepted.

    ``call_id`` identifies the call that owns the hold. It is stored on every
    slot in the run, together with ``held_until`` = now + :data:`HOLD_TTL_MINUTES`:

    - only :func:`confirm_booking` called with the same ``call_id`` can book it,
    - once ``held_until`` has passed the slots are on offer again, so a caller
      who hangs up mid-booking cannot park them for the rest of the day,
    - re-holding a window this call already holds extends it.

    A hold is not a booking: it keeps the slots out of :func:`find_slots` until
    it is confirmed, released, or expires.
    """
    wanted = [slot_ids] if isinstance(slot_ids, str) else list(slot_ids)
    if not wanted:
        raise AgendaError("hold_slot needs at least one slot id")
    if not (call_id or "").strip():
        raise AgendaError("call_id is required to hold a slot")

    now = datetime.now()
    held_until = now + timedelta(minutes=HOLD_TTL_MINUTES)

    conn = connect()
    try:
        with transaction(conn):
            for slot_id in wanted:
                slot = _fetch_slot(conn, slot_id)
                if slot.status == "booked":
                    raise SlotUnavailable(f"slot {slot_id} is already booked")
                if slot.hold_is_live(now) and slot.call_id != call_id:
                    raise HoldBelongsToAnotherCall(
                        f"slot {slot_id} is already held by another call"
                    )

            placeholders = ",".join("?" * len(wanted))
            cursor = conn.execute(
                f"UPDATE slots SET status = 'held', call_id = ?, held_until = ? "
                f"WHERE id IN ({placeholders}) AND ("
                "  status = 'free'"
                "  OR (status = 'held' AND call_id = ?)"
                "  OR (status = 'held' AND held_until IS NOT NULL AND held_until <= ?)"
                ")",
                [
                    call_id,
                    held_until.strftime(DT_FORMAT),
                    *wanted,
                    call_id,
                    now.strftime(DT_FORMAT),
                ],
            )
            if cursor.rowcount != len(wanted):
                raise SlotUnavailable(
                    f"could not hold all {len(wanted)} slots for call {call_id}"
                )
        return [_fetch_slot(conn, sid) for sid in wanted]
    finally:
        conn.close()


def release_hold(slot_id: str, call_id: str | None = None) -> int:
    """Put a held window back on offer. Returns how many slots were released.

    Releases ``slot_id`` and any slots immediately following it in the same room
    that are also held, so a multi-slot hold is undone in one call. With
    ``call_id``, the walk stops at the first slot held by a different call, so
    one caller can never release another's hold.
    """
    conn = connect()
    try:
        with transaction(conn):
            slot = _fetch_slot(conn, slot_id)
            if slot.status == "booked":
                raise SlotUnavailable(
                    f"slot {slot_id} is booked; cancel the appointment instead"
                )
            if slot.status != "held":
                return 0
            if call_id is not None and slot.call_id != call_id:
                raise HoldBelongsToAnotherCall(
                    f"slot {slot_id} is held by another call"
                )

            rows = conn.execute(
                _SLOT_SELECT + " WHERE s.room_id = ? AND s.start >= ? ORDER BY s.start",
                (slot.room_id, slot.start.strftime(DT_FORMAT)),
            ).fetchall()

            run_ids: list[str] = []
            previous_end: str | None = None
            for row in rows:
                if row["status"] != "held":
                    break
                if previous_end is not None and row["start"] != previous_end:
                    break
                if call_id is not None and row["call_id"] != call_id:
                    break
                run_ids.append(row["id"])
                previous_end = row["end"]

            placeholders = ",".join("?" * len(run_ids))
            cursor = conn.execute(
                f"UPDATE slots SET status = 'free', call_id = NULL, "
                f"held_until = NULL WHERE id IN ({placeholders}) AND status = 'held'",
                run_ids,
            )
            released = cursor.rowcount
        return released
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Booking
# --------------------------------------------------------------------------

def confirm_booking(
    slot_id: str,
    exam_id: str,
    patient_alias: str,
    phone: str | None = None,
    call_id: str | None = None,
) -> str:
    """Book a window for an exam and return the appointment's reference code.

    ``slot_id`` is the first slot. If the exam is longer than one slot, the
    consecutive slots it needs are claimed too, in the same statement — so a
    50-minute IRM takes two 25-minute slots, or five 10-minute ones in a room
    that runs those, and nobody else is offered the rest of the run.

    ``call_id`` is the call doing the booking. A slot held by a *different* call
    is refused with :class:`HoldBelongsToAnotherCall` while that hold is live;
    once it has expired the slot is bookable by anyone. Passing no ``call_id``
    books only slots that are free or whose holds have expired.

    The whole check-and-claim runs in one ``BEGIN IMMEDIATE`` transaction, so a
    slot cannot be booked twice even if two calls arrive at once.
    """
    exam = get_exam(exam_id)
    alias = (patient_alias or "").strip()
    if not alias:
        raise AgendaError("patient_alias is required")

    now = datetime.now()
    conn = connect()
    try:
        with transaction(conn):
            slot = _fetch_slot(conn, slot_id)
            _require_bookable(slot)
            check_slot_matches_exam(slot, exam)

            run = _run_starting_at(conn, slot, exam, call_id=call_id, now=now)
            run_ids = [part.id for part in run]
            _claim(conn, run_ids, call_id=call_id, now=now)

            ref_code = _unique_ref_code(conn)
            cursor = conn.execute(
                "INSERT INTO appointments "
                "(ref_code, exam_id, slot_id, patient_alias, phone, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'confirmed', ?)",
                (
                    ref_code,
                    exam.id,
                    slot_id,
                    alias,
                    mask_phone(phone),
                    datetime.now().strftime(DT_FORMAT),
                ),
            )
            _attach(conn, run_ids, int(cursor.lastrowid or 0))
        return ref_code
    finally:
        conn.close()


def get_appointment(ref_code: str) -> Appointment:
    """Look a confirmed appointment up by its reference code."""
    conn = connect()
    try:
        return _fetch_appointment(conn, ref_code)
    finally:
        conn.close()


def change_appointment(
    ref_code: str,
    new_slot_id: str,
    call_id: str | None = None,
) -> Appointment:
    """Move an appointment to another window, keeping the same reference code.

    The new window has to pass every rule the original booking did, the exam's
    full length included — and a live hold belonging to another call blocks the
    move just as it blocks a fresh booking. The old slots are only freed once
    the new ones have been claimed, so a failed move leaves the appointment
    exactly as it was.
    """
    now = datetime.now()
    conn = connect()
    try:
        with transaction(conn):
            appointment = _fetch_appointment(conn, ref_code)
            if appointment.slot_id == new_slot_id:
                return appointment

            exam = get_exam(appointment.exam_id)
            new_slot = _fetch_slot(conn, new_slot_id)
            if new_slot.status == "booked" and new_slot.appointment_id != appointment.id:
                raise SlotUnavailable(f"slot {new_slot_id} is already booked")
            check_slot_matches_exam(new_slot, exam)

            run = _run_starting_at(
                conn,
                new_slot,
                exam,
                call_id=call_id,
                own_appointment_id=appointment.id,
                now=now,
            )
            run_ids = [part.id for part in run]

            _claim(
                conn,
                run_ids,
                call_id=call_id,
                own_appointment_id=appointment.id,
                now=now,
            )
            _free_appointment_slots(conn, appointment.id, keep=run_ids)
            _attach(conn, run_ids, appointment.id)
            conn.execute(
                "UPDATE appointments SET slot_id = ? WHERE id = ?",
                (new_slot_id, appointment.id),
            )
        return _fetch_appointment(conn, ref_code)
    finally:
        conn.close()


def cancel_appointment(ref_code: str) -> Appointment:
    """Cancel an appointment and put every slot it held back on offer."""
    conn = connect()
    try:
        with transaction(conn):
            appointment = _fetch_appointment(conn, ref_code)
            conn.execute(
                "UPDATE appointments SET status = 'cancelled' WHERE id = ?",
                (appointment.id,),
            )
            _free_appointment_slots(conn, appointment.id)
            row = conn.execute(
                _APPT_SELECT + " WHERE a.id = ?", (appointment.id,)
            ).fetchone()
        return _appointment_from_row(row, appointment.slot_ids)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Reading back
# --------------------------------------------------------------------------

def list_rooms(modality: str | None = None) -> list[Room]:
    """Return the centre's rooms, optionally only those of one modality."""
    sql = "SELECT id, modality, name FROM rooms"
    params: tuple[object, ...] = ()
    if modality is not None:
        sql += " WHERE modality = ?"
        params = (normalise_modality(modality),)
    sql += " ORDER BY id"
    conn = connect()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [Room(id=r["id"], modality=r["modality"], name=r["name"]) for r in rows]


def day_schedule(day: date | datetime | str, room_id: str | None = None) -> list[Slot]:
    """Return every slot on a day, for the demo's calendar view."""
    sql = _SLOT_SELECT + " WHERE substr(s.start, 1, 10) = ?"
    params: list[object] = [_as_date(day).strftime(DATE_FORMAT)]
    if room_id is not None:
        sql += " AND s.room_id = ?"
        params.append(room_id)
    sql += " ORDER BY s.start, r.id"
    conn = connect()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [_slot_from_row(row) for row in rows]


def counts_by_status() -> dict[str, int]:
    """Return how many slots sit in each status, for the build script's report."""
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM slots GROUP BY status"
        ).fetchall()
    finally:
        conn.close()
    return {row["status"]: row["n"] for row in rows}


DAYS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MONTHS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]


def format_slot(window: Slot | Offer) -> str:
    """Render a window the way the agent would say it out loud, in French."""
    start = window.start
    return (
        f"{DAYS_FR[start.weekday()]} {start.day} {MONTHS_FR[start.month - 1]} "
        f"à {start.strftime('%H:%M')} ({window.room_name}, {window.minutes} min)"
    )
