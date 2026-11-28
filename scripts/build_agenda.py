"""Build the simulated calendar: create and fill `data/processed/agenda.db`.

Creates the three tables, seeds the centre's rooms, generates two weeks of
back-to-back slots starting next Monday, and pre-books about 30% of them at
random with a fixed seed so every run of the demo looks the same.

Opening hours: Monday to Friday 8:00-18:00, Saturday 8:00-13:00, closed Sunday.
Slot length per modality is the most common ``duration_min`` for that modality
in the exam catalogue.

    python scripts/build_agenda.py
    python scripts/build_agenda.py --reset --seed 7 --prebook 0.3
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hermes.book import agenda  # noqa: E402

# --------------------------------------------------------------------------
# The centre
# --------------------------------------------------------------------------

#: How many rooms of each modality the centre has.
ROOM_PLAN: list[tuple[str, int, str]] = [
    # (modality, how many, display name template)
    ("IRM", 2, "IRM {n}"),
    ("scanner", 2, "Scanner {n}"),
    ("radiographie", 3, "Radiographie {n}"),
    ("échographie", 3, "Échographie {n}"),
    ("mammographie", 1, "Mammographie"),
    ("densitométrie", 1, "Densitométrie"),
    ("dentaire", 1, "Panoramique dentaire"),
]

#: Opening hours by weekday (Monday = 0). A missing day is closed.
OPENING_HOURS: dict[int, tuple[time, time]] = {
    0: (time(8, 0), time(18, 0)),
    1: (time(8, 0), time(18, 0)),
    2: (time(8, 0), time(18, 0)),
    3: (time(8, 0), time(18, 0)),
    4: (time(8, 0), time(18, 0)),
    5: (time(8, 0), time(13, 0)),
    # Sunday (6) is closed.
}

WEEKS = 2

#: Used only when a modality has no exam in the catalogue to measure.
FALLBACK_DURATION_MIN = 15


# --------------------------------------------------------------------------
# Rooms
# --------------------------------------------------------------------------

def build_rooms() -> list[agenda.Room]:
    """Expand :data:`ROOM_PLAN` into the centre's room list."""
    rooms: list[agenda.Room] = []
    for modality, count, template in ROOM_PLAN:
        key = agenda.normalise_modality(modality)
        for n in range(1, count + 1):
            rooms.append(
                agenda.Room(
                    id=f"{key}-{n}",
                    modality=key,
                    name=template.format(n=n),
                )
            )
    return rooms


def seed_rooms(conn, rooms: list[agenda.Room]) -> int:
    """Insert the rooms, replacing any that are already there."""
    conn.executemany(
        "INSERT OR REPLACE INTO rooms (id, modality, name) VALUES (?, ?, ?)",
        [(room.id, room.modality, room.name) for room in rooms],
    )
    return len(rooms)


# --------------------------------------------------------------------------
# Slots
# --------------------------------------------------------------------------

def next_monday(today: date | None = None) -> date:
    """Return the first Monday strictly after ``today``."""
    today = today or date.today()
    ahead = 7 - today.weekday()
    if ahead == 0:
        ahead = 7
    return today + timedelta(days=ahead)


def slot_lengths(exams_csv: Path | None = None) -> dict[str, int]:
    """Slot length per modality: the most common exam duration for that modality."""
    measured = agenda.modal_durations(exams_csv)
    lengths: dict[str, int] = {}
    for modality, _count, _template in ROOM_PLAN:
        key = agenda.normalise_modality(modality)
        lengths[key] = measured.get(key, FALLBACK_DURATION_MIN)
    return lengths


def day_slots(
    room: agenda.Room,
    day: date,
    length_min: int,
) -> list[tuple[str, str, datetime, datetime]]:
    """Return the back-to-back slots a room offers on one day.

    A slot is only generated if it fits entirely inside the opening hours, so
    the centre never closes mid-examination.
    """
    hours = OPENING_HOURS.get(day.weekday())
    if hours is None:
        return []

    opens, closes = hours
    cursor = datetime.combine(day, opens)
    closing = datetime.combine(day, closes)
    step = timedelta(minutes=length_min)

    slots: list[tuple[str, str, datetime, datetime]] = []
    while cursor + step <= closing:
        end = cursor + step
        slot_id = f"{room.id}-{cursor.strftime('%Y%m%d-%H%M')}"
        slots.append((slot_id, room.id, cursor, end))
        cursor = end
    return slots


def generate_slots(
    rooms: list[agenda.Room],
    lengths: dict[str, int],
    start: date,
    weeks: int = WEEKS,
) -> list[tuple[str, str, datetime, datetime]]:
    """Generate every slot for ``weeks`` weeks of calendar from ``start``."""
    rows: list[tuple[str, str, datetime, datetime]] = []
    for offset in range(weeks * 7):
        day = start + timedelta(days=offset)
        for room in rooms:
            length = lengths.get(room.modality, FALLBACK_DURATION_MIN)
            rows.extend(day_slots(room, day, length))
    return rows


def insert_slots(conn, rows, prebook: float, seed: int) -> tuple[int, int]:
    """Insert the slots, marking a fixed-seed random share of them as booked.

    The pre-booked slots stand for appointments the centre took through its
    other channels (counter, other phone lines), so they carry no row in
    ``appointments``: the voice agent only ever sees that they are taken.
    """
    rng = random.Random(seed)
    payload = []
    booked = 0
    for slot_id, room_id, start, end in rows:
        status = "booked" if rng.random() < prebook else "free"
        booked += status == "booked"
        payload.append(
            (
                slot_id,
                room_id,
                start.strftime(agenda.DT_FORMAT),
                end.strftime(agenda.DT_FORMAT),
                status,
            )
        )
    conn.executemany(
        'INSERT OR REPLACE INTO slots (id, room_id, start, "end", status) '
        "VALUES (?, ?, ?, ?, ?)",
        payload,
    )
    return len(payload), booked


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def report(conn, lengths: dict[str, int], start: date, weeks: int) -> None:
    """Print what was built, so the numbers can go straight into the report."""
    print(f"\ncalendar: {weeks} weeks from {start:%A %d %B %Y} "
          f"to {start + timedelta(days=weeks * 7 - 1):%A %d %B %Y}")
    print("hours:    Mon-Fri 08:00-18:00, Sat 08:00-13:00, Sun closed\n")

    rows = conn.execute(
        """
        SELECT r.modality,
               COUNT(DISTINCT r.id)                                   AS rooms,
               COUNT(s.id)                                            AS slots,
               SUM(s.status = 'free')                                 AS free,
               SUM(s.status = 'booked')                               AS booked,
               SUM(s.status = 'free'
                   AND CAST(substr(s.start, 12, 2) AS INTEGER) < 12)  AS free_morning
        FROM rooms r LEFT JOIN slots s ON s.room_id = r.id
        GROUP BY r.modality ORDER BY r.modality
        """
    ).fetchall()

    header = f"{'modality':<16}{'rooms':>6}{'min':>6}{'slots':>8}{'free':>8}{'booked':>8}{'free am':>9}"
    print(header)
    print("-" * len(header))
    totals = [0, 0, 0, 0, 0]
    for row in rows:
        length = lengths.get(row["modality"], FALLBACK_DURATION_MIN)
        print(
            f"{row['modality']:<16}{row['rooms']:>6}{length:>6}"
            f"{row['slots']:>8}{row['free']:>8}{row['booked']:>8}{row['free_morning']:>9}"
        )
        totals[0] += row["rooms"]
        totals[1] += row["slots"]
        totals[2] += row["free"]
        totals[3] += row["booked"]
        totals[4] += row["free_morning"]
    print("-" * len(header))
    print(
        f"{'total':<16}{totals[0]:>6}{'':>6}"
        f"{totals[1]:>8}{totals[2]:>8}{totals[3]:>8}{totals[4]:>9}"
    )
    if totals[1]:
        print(f"\npre-booked: {totals[3] / totals[1]:.1%} of slots")

    exams = agenda.load_exams(refresh=True)
    fasting = [e for e in exams.values() if e.requires_fasting]
    print(
        f"catalogue:  {len(exams)} exams, {len(fasting)} need fasting "
        f"(bookable before {agenda.FASTING_CUTOFF_HOUR}:00 only)"
    )

    multi_slot_report(exams, lengths)


def multi_slot_report(exams: dict[str, agenda.Exam], lengths: dict[str, int]) -> None:
    """Report the exams that need consecutive slots, and whether any run fits.

    An exam longer than its modality's slot length takes two or three slots in a
    row. Pre-booking fragments the calendar, so this checks that the longest
    exam of each modality can still actually be placed somewhere.
    """
    by_modality: dict[str, list[agenda.Exam]] = {}
    for exam in exams.values():
        by_modality.setdefault(exam.modality, []).append(exam)

    print("\nexams longer than one slot — these take consecutive slots:")
    header = (
        f"{'modality':<16}{'exams':>7}{'>1 slot':>9}"
        f"{'max slots':>11}{'longest exam':>14}{'windows free':>14}"
    )
    print(header)
    print("-" * len(header))

    for modality in sorted(by_modality):
        slot_len = lengths.get(modality, FALLBACK_DURATION_MIN)
        group = by_modality[modality]
        counted = [
            (agenda.slots_needed(e.duration_min, slot_len), e) for e in group
        ]
        over = [n for n, _e in counted if n > 1]
        worst_n, worst_exam = max(counted, key=lambda pair: pair[0])
        windows = len(agenda.find_slots(worst_exam.id, limit=100_000))
        print(
            f"{modality:<16}{len(group):>7}{len(over):>9}{worst_n:>11}"
            f"{worst_exam.duration_min:>11} min{windows:>14}"
        )
    print("-" * len(header))
    total_over = sum(
        1
        for e in exams.values()
        if agenda.slots_needed(
            e.duration_min, lengths.get(e.modality, FALLBACK_DURATION_MIN)
        )
        > 1
    )
    print(f"{total_over} of {len(exams)} exams need more than one slot")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the database path, catalogue path, start Monday, seed and share."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=agenda.DEFAULT_DB_PATH,
                        help="where to write the SQLite database")
    parser.add_argument("--exams", type=Path, default=agenda.DEFAULT_EXAMS_CSV,
                        help="the exam catalogue CSV")
    parser.add_argument("--start", type=str, default=None,
                        help="first day of the calendar, YYYY-MM-DD "
                             "(default: next Monday)")
    parser.add_argument("--weeks", type=int, default=WEEKS,
                        help="how many weeks of calendar to generate")
    parser.add_argument("--prebook", type=float, default=0.30,
                        help="share of slots to pre-book, 0 to 1")
    parser.add_argument("--seed", type=int, default=20261123,
                        help="seed for the pre-booking, so runs are repeatable")
    parser.add_argument("--reset", action="store_true",
                        help="drop the existing tables first")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Create the database, seed the rooms and slots, print the report."""
    args = parse_args(argv)

    if not args.exams.exists():
        print(f"error: exam catalogue not found: {args.exams}", file=sys.stderr)
        return 1
    if not 0.0 <= args.prebook <= 1.0:
        print("error: --prebook must be between 0 and 1", file=sys.stderr)
        return 1

    agenda.configure(db_path=args.db, exams_csv=args.exams)

    start = (
        datetime.strptime(args.start, agenda.DATE_FORMAT).date()
        if args.start
        else next_monday()
    )

    agenda.init_db(args.db, reset=args.reset)

    rooms = build_rooms()
    lengths = slot_lengths(args.exams)
    rows = generate_slots(rooms, lengths, start, args.weeks)

    conn = agenda.connect(args.db)
    try:
        with agenda.transaction(conn):
            n_rooms = seed_rooms(conn, rooms)
            n_slots, n_booked = insert_slots(conn, rows, args.prebook, args.seed)
        print(f"wrote {args.db}")
        print(f"  {n_rooms} rooms, {n_slots} slots, {n_booked} pre-booked")
        report(conn, lengths, start, args.weeks)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
