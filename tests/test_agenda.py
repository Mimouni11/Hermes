"""Tests for the appointment book: the booking rules, and the rules under race.

Every test runs against a database and a catalogue built in a temporary
directory, so nothing here touches `data/`.

The fixture lays out genuinely contiguous slots — a morning block and an
afternoon block per room — because an exam longer than one slot can only be
placed if the slots actually abut.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from hermes.book import agenda

# --------------------------------------------------------------------------
# Fixture data
# --------------------------------------------------------------------------

# id, modality, duration_min, name_fr, preparation_fr
EXAM_ROWS = [
    ("EX001", "IRM", 25, "IRM cérébrale",
     "Retirer tout objet métallique. Signaler un pacemaker."),
    ("EX002", "IRM", 25, "IRM abdominale",
     "Venir à jeun depuis 6 heures."),
    ("EX003", "scanner", 15, "Scanner thoracique",
     "Aucune préparation particulière."),
    ("EX004", "échographie", 15, "Échographie abdominale",
     "Se présenter à jeun depuis 6 heures."),
    # Longer than one slot: the cases the reviewer caught.
    ("EX005", "IRM", 50, "IRM cardiaque",
     "Aucune préparation particulière."),
    ("EX006", "IRM", 45, "IRM du rachis entier",
     "Retirer tout objet métallique."),
    ("EX007", "scanner", 45, "Uroscanner",
     "Aucune préparation particulière."),
    ("EX008", "IRM", 50, "IRM hépatique longue",
     "Venir à jeun depuis 6 heures."),
]

ROOM_ROWS = [
    ("irm-1", "irm", "IRM 1"),
    ("irm-2", "irm", "IRM 2"),
    ("scanner-1", "scanner", "Scanner 1"),
    ("echographie-1", "echographie", "Échographie 1"),
]

#: Slot length per room, mirroring what the build script would produce.
ROOM_SLOT_MIN = {"irm-1": 25, "irm-2": 25, "scanner-1": 15, "echographie-1": 15}

#: Four contiguous slots from each of these times, in every room.
BLOCK_STARTS = [(9, 0), (14, 0)]

#: Slots per contiguous block.
BLOCK_LENGTH = 4


@dataclass
class Book:
    """Handles on the temporary agenda a test works against."""

    db: Path
    exams_csv: Path
    day: date

    def slot(self, room_id: str, hour: int, minute: int = 0) -> str:
        """The id of the slot starting at that time in that room."""
        return f"{room_id}-{self.day:%Y%m%d}-{hour:02d}{minute:02d}"

    def nth(self, room_id: str, block: int, index: int) -> str:
        """The ``index``-th slot of a contiguous block (0 = morning, 1 = afternoon)."""
        hour, minute = BLOCK_STARTS[block]
        start = datetime.combine(self.day, datetime.min.time()).replace(
            hour=hour, minute=minute
        ) + timedelta(minutes=index * ROOM_SLOT_MIN[room_id])
        return self.slot(room_id, start.hour, start.minute)


def _write_exams(path: Path) -> None:
    lines = ["id,modality,duration_min,name_fr,preparation_fr"]
    for exam_id, modality, duration, name, preparation in EXAM_ROWS:
        lines.append(f'{exam_id},{modality},{duration},"{name}","{preparation}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _insert_slot(conn, day: date, room_id: str, start: datetime, minutes: int) -> None:
    end = start + timedelta(minutes=minutes)
    conn.execute(
        'INSERT INTO slots (id, room_id, start, "end", status) '
        "VALUES (?, ?, ?, ?, 'free')",
        (
            f"{room_id}-{day:%Y%m%d}-{start.hour:02d}{start.minute:02d}",
            room_id,
            start.strftime(agenda.DT_FORMAT),
            end.strftime(agenda.DT_FORMAT),
        ),
    )


def _add_slots(conn, day: date, room_id: str) -> None:
    """Add two contiguous blocks of slots, one each side of 12:00."""
    minutes = ROOM_SLOT_MIN[room_id]
    midnight = datetime.combine(day, datetime.min.time())
    for hour, minute in BLOCK_STARTS:
        first = midnight.replace(hour=hour, minute=minute)
        for i in range(BLOCK_LENGTH):
            _insert_slot(conn, day, room_id, first + timedelta(minutes=i * minutes), minutes)


@pytest.fixture
def book(tmp_path: Path) -> Book:
    """Build a small agenda in a temporary directory and point the module at it."""
    exams_csv = tmp_path / "exams.csv"
    db = tmp_path / "agenda.db"
    _write_exams(exams_csv)

    agenda.configure(db_path=db, exams_csv=exams_csv)
    agenda.init_db(db, reset=True)

    # A week ahead, so every slot is in the future whatever day the tests run.
    day = date.today() + timedelta(days=7)

    conn = agenda.connect(db)
    try:
        with agenda.transaction(conn):
            conn.executemany(
                "INSERT INTO rooms (id, modality, name) VALUES (?, ?, ?)", ROOM_ROWS
            )
            for room_id, _modality, _name in ROOM_ROWS:
                _add_slots(conn, day, room_id)
            # A slot with nothing after it: a long exam cannot start here.
            _insert_slot(
                conn,
                day,
                "irm-1",
                datetime.combine(day, datetime.min.time()).replace(hour=11),
                25,
            )
    finally:
        conn.close()

    return Book(db=db, exams_csv=exams_csv, day=day)


def slot_ids(offers) -> set[str]:
    """The first-slot ids of a list of offers."""
    return {offer.slot_id for offer in offers}


def _slot_id(day: date, room_id: str, start: datetime) -> str:
    """The id `_insert_slot` gives a slot, so a test can name one it added."""
    return f"{room_id}-{day:%Y%m%d}-{start.hour:02d}{start.minute:02d}"


#: An IRM room whose slots are 10 minutes, where the others run 25.
SHORT_ROOM = ("irm-3", "irm", "IRM 3")
SHORT_SLOT_MIN = 10
SHORT_SLOT_COUNT = 6


def _add_short_slot_room(book: Book) -> list[str]:
    """Add a second IRM room with a *different* slot length, and return its slots.

    This is the layout that catches a slot length read from one room and applied
    to another: a 50-minute exam needs two slots here and five there.
    """
    midnight = datetime.combine(book.day, datetime.min.time())
    first = midnight.replace(hour=9, minute=0)
    ids: list[str] = []
    conn = agenda.connect()
    try:
        with agenda.transaction(conn):
            conn.execute(
                "INSERT INTO rooms (id, modality, name) VALUES (?, ?, ?)", SHORT_ROOM
            )
            for i in range(SHORT_SLOT_COUNT):
                start = first + timedelta(minutes=i * SHORT_SLOT_MIN)
                _insert_slot(conn, book.day, SHORT_ROOM[0], start, SHORT_SLOT_MIN)
                ids.append(_slot_id(book.day, SHORT_ROOM[0], start))
    finally:
        conn.close()
    return ids


def _add_pair_straddling_noon(book: Book) -> list[str]:
    """Two contiguous 25-minute IRM slots at 11:45 and 12:10, so 11:45–12:35 exists."""
    midnight = datetime.combine(book.day, datetime.min.time())
    ids: list[str] = []
    conn = agenda.connect()
    try:
        with agenda.transaction(conn):
            for i in range(2):
                start = midnight.replace(hour=11, minute=45) + timedelta(minutes=i * 25)
                _insert_slot(conn, book.day, "irm-2", start, 25)
                ids.append(_slot_id(book.day, "irm-2", start))
    finally:
        conn.close()
    return ids


def _expire_every_hold(minutes: int = 1) -> None:
    """Push every live hold into the past, as if the caller had hung up."""
    past = (datetime.now() - timedelta(minutes=minutes)).strftime(agenda.DT_FORMAT)
    conn = agenda.connect()
    try:
        with agenda.transaction(conn):
            conn.execute(
                "UPDATE slots SET held_until = ? WHERE status = 'held'", (past,)
            )
    finally:
        conn.close()


# --------------------------------------------------------------------------
# The happy path: book, change, cancel
# --------------------------------------------------------------------------

def test_book_returns_a_ref_code_and_takes_the_slot(book: Book) -> None:
    """Booking returns a 6-character code and the slot is no longer free."""
    slot_id = book.nth("irm-1", 0, 0)

    ref = agenda.confirm_booking(slot_id, "EX001", "patient-01", "06 12 34 56 78")

    assert len(ref) == agenda.REF_LENGTH
    assert set(ref) <= set(agenda.REF_ALPHABET)
    assert agenda.get_slot(slot_id).status == "booked"

    appointment = agenda.get_appointment(ref)
    assert appointment.exam_id == "EX001"
    assert appointment.slot_id == slot_id
    assert appointment.status == "confirmed"
    assert appointment.patient_alias == "patient-01"
    assert appointment.slot_ids == (slot_id,)


def test_booked_slot_is_no_longer_offered(book: Book) -> None:
    """A slot that has been booked drops out of the availability search."""
    slot_id = book.nth("irm-1", 0, 0)
    assert slot_id in slot_ids(agenda.find_slots("EX001", limit=50))

    agenda.confirm_booking(slot_id, "EX001", "patient-01")

    assert slot_id not in slot_ids(agenda.find_slots("EX001", limit=50))


def test_change_moves_the_appointment_and_frees_the_old_slot(book: Book) -> None:
    """Rescheduling keeps the ref code, takes the new slot and releases the old."""
    old_slot = book.nth("irm-1", 0, 0)
    new_slot = book.nth("irm-1", 1, 0)

    ref = agenda.confirm_booking(old_slot, "EX001", "patient-02")
    moved = agenda.change_appointment(ref, new_slot)

    assert moved.ref_code == ref
    assert moved.slot_id == new_slot
    assert agenda.get_slot(new_slot).status == "booked"
    assert agenda.get_slot(old_slot).status == "free"


def test_change_to_another_room_is_allowed(book: Book) -> None:
    """A move to a different room of the same modality is fine."""
    ref = agenda.confirm_booking(book.nth("irm-1", 0, 0), "EX001", "patient-03")

    moved = agenda.change_appointment(ref, book.nth("irm-2", 0, 1))

    assert moved.slot_id == book.nth("irm-2", 0, 1)


def test_cancel_frees_the_slot_and_closes_the_appointment(book: Book) -> None:
    """Cancelling puts the slot back on offer and the code stops resolving."""
    slot_id = book.nth("scanner-1", 0, 0)
    ref = agenda.confirm_booking(slot_id, "EX003", "patient-04")

    cancelled = agenda.cancel_appointment(ref)

    assert cancelled.status == "cancelled"
    assert agenda.get_slot(slot_id).status == "free"
    with pytest.raises(agenda.UnknownRefCode):
        agenda.get_appointment(ref)


def test_cancelled_slot_can_be_booked_again(book: Book) -> None:
    """The freed slot is genuinely bookable, not just marked free."""
    slot_id = book.nth("scanner-1", 0, 0)
    first = agenda.confirm_booking(slot_id, "EX003", "patient-05")
    agenda.cancel_appointment(first)

    second = agenda.confirm_booking(slot_id, "EX003", "patient-06")

    assert second != first
    assert agenda.get_appointment(second).slot_id == slot_id


# --------------------------------------------------------------------------
# Exams longer than one slot
# --------------------------------------------------------------------------

def test_slots_needed_rounds_up() -> None:
    """A 45-minute exam in a 25-minute room takes two slots, not one and a bit."""
    assert agenda.slots_needed(25, 25) == 1
    assert agenda.slots_needed(30, 25) == 2
    assert agenda.slots_needed(45, 25) == 2
    assert agenda.slots_needed(50, 25) == 2
    assert agenda.slots_needed(45, 15) == 3
    assert agenda.slots_needed(0, 25) == 1


def test_long_exam_is_offered_a_long_window(book: Book) -> None:
    """A 50-minute IRM is offered a 50-minute window of two consecutive slots."""
    offers = agenda.find_slots("EX005", limit=5)

    assert offers
    for offer in offers:
        assert offer.n_slots == 2
        assert offer.minutes == 50
        assert offer.end - offer.start == timedelta(minutes=50)


def test_long_exam_claims_every_slot_it_needs(book: Book) -> None:
    """Booking a 50-minute IRM takes both 25-minute slots, not just the first."""
    first = book.nth("irm-1", 0, 0)
    second = book.nth("irm-1", 0, 1)

    ref = agenda.confirm_booking(first, "EX005", "patient-30")

    appointment = agenda.get_appointment(ref)
    assert appointment.slot_ids == (first, second)
    assert agenda.get_slot(first).status == "booked"
    assert agenda.get_slot(second).status == "booked"
    assert appointment.end is not None
    assert appointment.end - appointment.start == timedelta(minutes=50)


def test_the_second_slot_of_a_long_exam_is_not_resold(book: Book) -> None:
    """Nobody else is offered the slot a long exam runs into."""
    first = book.nth("irm-1", 0, 0)
    second = book.nth("irm-1", 0, 1)

    agenda.confirm_booking(first, "EX005", "patient-31")

    short = slot_ids(agenda.find_slots("EX001", limit=50))
    assert first not in short
    assert second not in short


def test_three_slot_exam_claims_three_slots(book: Book) -> None:
    """A 45-minute scanner in a 15-minute room takes three consecutive slots."""
    ref = agenda.confirm_booking(book.nth("scanner-1", 0, 0), "EX007", "patient-32")

    appointment = agenda.get_appointment(ref)
    assert len(appointment.slot_ids) == 3
    assert appointment.slot_ids == tuple(
        book.nth("scanner-1", 0, i) for i in range(3)
    )
    assert all(agenda.get_slot(s).status == "booked" for s in appointment.slot_ids)


def test_long_exam_cannot_start_where_the_run_is_too_short(book: Book) -> None:
    """An isolated slot is never offered to, nor accepted for, a long exam."""
    lonely = book.slot("irm-1", 11)
    assert agenda.get_slot(lonely).status == "free"
    assert lonely in slot_ids(agenda.find_slots("EX001", limit=50))

    assert lonely not in slot_ids(agenda.find_slots("EX005", limit=50))
    with pytest.raises(agenda.NotEnoughConsecutiveSlots):
        agenda.confirm_booking(lonely, "EX005", "patient-33")


def test_long_exam_cannot_start_at_the_end_of_a_block(book: Book) -> None:
    """The last slot of the morning block starts no two-slot window."""
    last = book.nth("irm-1", 0, BLOCK_LENGTH - 1)

    assert last not in slot_ids(agenda.find_slots("EX005", limit=50))
    with pytest.raises(agenda.NotEnoughConsecutiveSlots):
        agenda.confirm_booking(last, "EX005", "patient-34")


def test_a_short_booking_blocks_a_long_exam_behind_it(book: Book) -> None:
    """A single booked slot breaks the run, so the long exam moves elsewhere."""
    agenda.confirm_booking(book.nth("irm-1", 0, 1), "EX001", "patient-35")

    offers = agenda.find_slots("EX005", limit=50)

    assert book.nth("irm-1", 0, 0) not in slot_ids(offers)
    assert offers, "the long exam should still fit somewhere else"


def test_cancelling_a_long_exam_frees_every_slot(book: Book) -> None:
    """Cancelling a two-slot appointment puts both slots back on offer."""
    first = book.nth("irm-1", 0, 0)
    second = book.nth("irm-1", 0, 1)
    ref = agenda.confirm_booking(first, "EX005", "patient-36")

    agenda.cancel_appointment(ref)

    assert agenda.get_slot(first).status == "free"
    assert agenda.get_slot(second).status == "free"
    assert agenda.get_slot(second).appointment_id is None
    assert first in slot_ids(agenda.find_slots("EX005", limit=50))


def test_changing_a_long_exam_moves_the_whole_window(book: Book) -> None:
    """Rescheduling a two-slot appointment frees both old slots and claims both new."""
    old_first, old_second = book.nth("irm-1", 0, 0), book.nth("irm-1", 0, 1)
    new_first, new_second = book.nth("irm-1", 1, 0), book.nth("irm-1", 1, 1)
    ref = agenda.confirm_booking(old_first, "EX005", "patient-37")

    moved = agenda.change_appointment(ref, new_first)

    assert moved.slot_ids == (new_first, new_second)
    assert agenda.get_slot(old_first).status == "free"
    assert agenda.get_slot(old_second).status == "free"
    assert agenda.get_slot(new_second).status == "booked"


def test_changing_a_long_exam_into_a_short_gap_is_refused(book: Book) -> None:
    """A move needs room for the whole exam, and fails without touching the old one."""
    ref = agenda.confirm_booking(book.nth("irm-1", 0, 0), "EX005", "patient-38")

    with pytest.raises(agenda.NotEnoughConsecutiveSlots):
        agenda.change_appointment(ref, book.slot("irm-1", 11))

    assert agenda.get_appointment(ref).slot_ids == (
        book.nth("irm-1", 0, 0),
        book.nth("irm-1", 0, 1),
    )
    assert agenda.get_slot(book.nth("irm-1", 0, 1)).status == "booked"


def test_a_long_exam_can_shift_onto_its_own_slots(book: Book) -> None:
    """Moving a two-slot booking one slot later reuses the slot it already holds."""
    ref = agenda.confirm_booking(book.nth("irm-1", 0, 0), "EX005", "patient-39")

    moved = agenda.change_appointment(ref, book.nth("irm-1", 0, 1))

    assert moved.slot_ids == (book.nth("irm-1", 0, 1), book.nth("irm-1", 0, 2))
    assert agenda.get_slot(book.nth("irm-1", 0, 0)).status == "free"


def test_a_long_booking_does_not_double_book_under_a_second_attempt(book: Book) -> None:
    """Two callers after overlapping long windows: one wins, the other gets nothing."""
    first = agenda.confirm_booking(book.nth("irm-1", 0, 0), "EX005", "patient-40")

    with pytest.raises(agenda.SlotUnavailable):
        agenda.confirm_booking(book.nth("irm-1", 0, 1), "EX005", "patient-41")

    conn = agenda.connect()
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE status = 'confirmed'"
        ).fetchone()[0]
        booked = conn.execute(
            "SELECT COUNT(*) FROM slots WHERE status = 'booked'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert n == 1
    assert booked == 2
    assert agenda.get_appointment(first).slot_ids == (
        book.nth("irm-1", 0, 0),
        book.nth("irm-1", 0, 1),
    )


def test_offers_use_each_rooms_own_slot_length(book: Book) -> None:
    """Two IRM rooms with different slot lengths: each window is sized by its room.

    The same 50-minute exam needs two 25-minute slots in `irm-1` and five
    10-minute ones in `irm-3`. Sizing every room by the first room's slot length
    would offer a 20-minute window for a 50-minute exam.
    """
    short = _add_short_slot_room(book)

    offers = agenda.find_slots("EX005", limit=200)

    by_room: dict[str, list] = {}
    for offer in offers:
        by_room.setdefault(offer.room_id, []).append(offer)

    assert "irm-3" in by_room, "the 10-minute room should still be offered"
    assert all(o.minutes >= 50 for o in offers)
    assert {o.n_slots for o in by_room["irm-1"]} == {2}
    assert {o.n_slots for o in by_room["irm-3"]} == {5}
    assert by_room["irm-3"][0].slot_ids == tuple(short[:5])
    assert by_room["irm-3"][0].minutes == 50


def test_booking_in_a_short_slot_room_claims_the_whole_run(book: Book) -> None:
    """A 50-minute exam in a 10-minute room claims five slots, not two."""
    short = _add_short_slot_room(book)

    ref = agenda.confirm_booking(short[0], "EX005", "patient-60")

    appointment = agenda.get_appointment(ref)
    assert appointment.slot_ids == tuple(short[:5])
    assert appointment.end is not None
    assert appointment.end - appointment.start == timedelta(minutes=50)
    assert all(agenda.get_slot(s).status == "booked" for s in short[:5])
    assert agenda.get_slot(short[5]).status == "free"


def test_a_short_slot_room_refuses_a_run_that_does_not_fit(book: Book) -> None:
    """Starting a 50-minute exam one slot from the end leaves only 40 minutes."""
    short = _add_short_slot_room(book)

    with pytest.raises(agenda.NotEnoughConsecutiveSlots):
        agenda.confirm_booking(short[2], "EX005", "patient-61")

    assert all(agenda.get_slot(s).status == "free" for s in short)


def test_a_fasting_long_exam_still_respects_the_cutoff(book: Book) -> None:
    """A 50-minute fasting exam is morning-only, and the window is still whole."""
    offers = agenda.find_slots("EX008", limit=50)

    assert offers
    assert all(o.start.hour < agenda.FASTING_CUTOFF_HOUR for o in offers)
    assert all(o.n_slots == 2 for o in offers)
    with pytest.raises(agenda.FastingRuleViolation):
        agenda.confirm_booking(book.nth("irm-1", 1, 0), "EX008", "patient-42")


# --------------------------------------------------------------------------
# Double booking
# --------------------------------------------------------------------------

def test_double_booking_is_refused(book: Book) -> None:
    """The second booking of the same slot raises and changes nothing."""
    slot_id = book.nth("irm-1", 0, 0)
    first = agenda.confirm_booking(slot_id, "EX001", "patient-07")

    with pytest.raises(agenda.SlotUnavailable):
        agenda.confirm_booking(slot_id, "EX001", "patient-08")

    assert agenda.get_appointment(first).slot_id == slot_id
    conn = agenda.connect()
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE slot_id = ? AND status = 'confirmed'",
            (slot_id,),
        ).fetchone()[0]
    finally:
        conn.close()
    assert n == 1


def test_change_into_a_booked_slot_is_refused(book: Book) -> None:
    """A move onto someone else's slot fails and leaves both bookings intact."""
    taken = book.nth("irm-1", 0, 0)
    mine = book.nth("irm-1", 0, 2)
    agenda.confirm_booking(taken, "EX001", "patient-09")
    ref = agenda.confirm_booking(mine, "EX001", "patient-10")

    with pytest.raises(agenda.SlotUnavailable):
        agenda.change_appointment(ref, taken)

    assert agenda.get_appointment(ref).slot_id == mine
    assert agenda.get_slot(mine).status == "booked"


def test_a_hold_keeps_the_slot_then_gives_it_back(book: Book) -> None:
    """A held slot is off the market, and release_hold puts it back."""
    slot_id = book.nth("irm-2", 0, 0)

    held = agenda.hold_slot(slot_id, "call-01")
    assert [s.status for s in held] == ["held"]
    assert held[0].call_id == "call-01"
    assert slot_id not in slot_ids(agenda.find_slots("EX001", limit=50))
    with pytest.raises(agenda.HoldBelongsToAnotherCall):
        agenda.hold_slot(slot_id, "call-02")

    assert agenda.release_hold(slot_id) == 1
    assert agenda.get_slot(slot_id).status == "free"
    assert agenda.get_slot(slot_id).call_id is None


def test_a_hold_for_a_long_exam_covers_the_whole_window(book: Book) -> None:
    """Holding an offer's whole window holds both slots, and releasing frees both."""
    first = book.nth("irm-2", 0, 0)
    second = book.nth("irm-2", 0, 1)
    offer = next(o for o in agenda.find_slots("EX005", limit=50) if o.slot_id == first)

    held = agenda.hold_slot(offer.slot_ids, "call-03")

    assert [s.id for s in held] == [first, second]
    assert all(s.status == "held" for s in held)
    assert all(s.call_id == "call-03" for s in held)
    assert first not in slot_ids(agenda.find_slots("EX005", limit=50))

    assert agenda.release_hold(first) == 2
    assert agenda.get_slot(second).status == "free"


def test_a_held_slot_can_still_be_confirmed(book: Book) -> None:
    """The call that holds a slot can book it: its own hold is not in its way."""
    slot_id = book.nth("irm-2", 0, 0)
    agenda.hold_slot(slot_id, "call-04")

    ref = agenda.confirm_booking(slot_id, "EX001", "patient-11", call_id="call-04")

    assert agenda.get_appointment(ref).slot_id == slot_id
    assert agenda.get_slot(slot_id).status == "booked"
    assert agenda.get_slot(slot_id).call_id is None


def test_a_held_long_window_can_be_confirmed(book: Book) -> None:
    """A held two-slot window confirms into a two-slot booking."""
    first = book.nth("irm-2", 0, 0)
    offer = next(o for o in agenda.find_slots("EX005", limit=50) if o.slot_id == first)
    agenda.hold_slot(offer.slot_ids, "call-05")

    ref = agenda.confirm_booking(first, "EX005", "patient-43", call_id="call-05")

    assert len(agenda.get_appointment(ref).slot_ids) == 2


# --------------------------------------------------------------------------
# Who owns a hold
# --------------------------------------------------------------------------

def test_a_hold_carries_its_call_and_an_expiry(book: Book) -> None:
    """A hold records which call took it, and expires HOLD_TTL_MINUTES later."""
    slot_id = book.nth("irm-2", 0, 0)

    before = datetime.now()
    held = agenda.hold_slot(slot_id, "call-10")

    assert held[0].call_id == "call-10"
    assert held[0].held_until is not None
    ttl = timedelta(minutes=agenda.HOLD_TTL_MINUTES)
    assert before + ttl - timedelta(minutes=1) <= held[0].held_until <= before + ttl


def test_another_calls_hold_is_refused(book: Book) -> None:
    """A live hold belongs to the call that took it, and blocks every other."""
    slot_id = book.nth("irm-2", 0, 0)
    agenda.hold_slot(slot_id, "call-caller-a")

    with pytest.raises(agenda.HoldBelongsToAnotherCall):
        agenda.confirm_booking(slot_id, "EX001", "patient-50", call_id="call-caller-b")
    with pytest.raises(agenda.HoldBelongsToAnotherCall):
        agenda.confirm_booking(slot_id, "EX001", "patient-50")

    assert agenda.get_slot(slot_id).status == "held"
    assert agenda.get_slot(slot_id).call_id == "call-caller-a"


def test_another_calls_hold_blocks_the_rest_of_a_long_window(book: Book) -> None:
    """A foreign hold on the *second* slot stops a long booking just as well."""
    first = book.nth("irm-2", 0, 0)
    second = book.nth("irm-2", 0, 1)
    agenda.hold_slot(second, "call-caller-a")

    with pytest.raises(agenda.HoldBelongsToAnotherCall):
        agenda.confirm_booking(first, "EX005", "patient-51", call_id="call-caller-b")

    assert agenda.get_slot(first).status == "free"
    assert agenda.get_slot(second).status == "held"


def test_an_expired_hold_is_bookable_by_anyone(book: Book) -> None:
    """Once a hold has expired the slot is free again, whoever asks for it."""
    slot_id = book.nth("irm-2", 0, 0)
    agenda.hold_slot(slot_id, "call-caller-a")
    _expire_every_hold()

    assert slot_id in slot_ids(agenda.find_slots("EX001", limit=50))

    ref = agenda.confirm_booking(slot_id, "EX001", "patient-52", call_id="call-caller-b")

    assert agenda.get_appointment(ref).slot_id == slot_id
    assert agenda.get_slot(slot_id).status == "booked"


def test_an_expired_hold_can_be_retaken_by_another_call(book: Book) -> None:
    """A stale hold does not park the slot: the next caller can hold it."""
    slot_id = book.nth("irm-2", 0, 0)
    agenda.hold_slot(slot_id, "call-caller-a")
    _expire_every_hold()

    held = agenda.hold_slot(slot_id, "call-caller-b")

    assert held[0].call_id == "call-caller-b"
    assert held[0].hold_is_live() is True


def test_a_hold_needs_a_call_id(book: Book) -> None:
    """A hold with no owner is refused: it could never be confirmed or expired."""
    with pytest.raises(agenda.AgendaError):
        agenda.hold_slot(book.nth("irm-2", 0, 0), "   ")


def test_releasing_a_free_slot_does_nothing(book: Book) -> None:
    """release_hold on a slot nobody held is a no-op, not an error."""
    assert agenda.release_hold(book.nth("irm-1", 0, 0)) == 0


def test_a_booked_slot_cannot_be_released(book: Book) -> None:
    """release_hold refuses to undo a real booking."""
    slot_id = book.nth("irm-1", 0, 0)
    agenda.confirm_booking(slot_id, "EX001", "patient-12")

    with pytest.raises(agenda.SlotUnavailable):
        agenda.release_hold(slot_id)


# --------------------------------------------------------------------------
# Modality
# --------------------------------------------------------------------------

def test_wrong_modality_is_refused(book: Book) -> None:
    """An IRM exam cannot be booked into a scanner room."""
    scanner = book.nth("scanner-1", 0, 0)

    with pytest.raises(agenda.ModalityMismatch):
        agenda.confirm_booking(scanner, "EX001", "patient-13")

    assert agenda.get_slot(scanner).status == "free"


def test_find_slots_only_offers_the_right_modality(book: Book) -> None:
    """Availability for an IRM exam never includes another modality's rooms."""
    offers = agenda.find_slots("EX001", limit=50)

    assert offers
    assert {o.modality for o in offers} == {"irm"}


def test_change_to_the_wrong_modality_is_refused(book: Book) -> None:
    """Rescheduling re-checks the modality against the original exam."""
    mine = book.nth("irm-1", 0, 0)
    ref = agenda.confirm_booking(mine, "EX001", "patient-14")

    with pytest.raises(agenda.ModalityMismatch):
        agenda.change_appointment(ref, book.nth("scanner-1", 0, 0))

    assert agenda.get_appointment(ref).slot_id == mine


# --------------------------------------------------------------------------
# Fasting
# --------------------------------------------------------------------------

def test_fasting_exam_in_the_afternoon_is_refused(book: Book) -> None:
    """An exam needing fasting cannot be booked after 12:00."""
    afternoon = book.nth("irm-1", 1, 0)

    with pytest.raises(agenda.FastingRuleViolation):
        agenda.confirm_booking(afternoon, "EX002", "patient-15")

    assert agenda.get_slot(afternoon).status == "free"


def test_fasting_exam_in_the_morning_is_accepted(book: Book) -> None:
    """The same exam books fine before 12:00."""
    ref = agenda.confirm_booking(book.nth("irm-1", 0, 0), "EX002", "patient-16")

    assert agenda.get_appointment(ref).exam_id == "EX002"


def test_find_slots_offers_no_afternoon_for_a_fasting_exam(book: Book) -> None:
    """Availability for a fasting exam is morning-only, so the rule never surprises."""
    offers = agenda.find_slots("EX002", limit=50)

    assert offers
    assert all(o.start.hour < agenda.FASTING_CUTOFF_HOUR for o in offers)


def test_afternoon_period_for_a_fasting_exam_returns_nothing(book: Book) -> None:
    """Asking for an afternoon slot for a fasting exam returns an empty list."""
    assert agenda.find_slots("EX002", period="après-midi", limit=50) == []


def test_change_of_a_fasting_exam_into_the_afternoon_is_refused(book: Book) -> None:
    """Rescheduling re-checks the fasting rule too."""
    morning = book.nth("echographie-1", 0, 0)
    ref = agenda.confirm_booking(morning, "EX004", "patient-17")

    with pytest.raises(agenda.FastingRuleViolation):
        agenda.change_appointment(ref, book.nth("echographie-1", 1, 0))

    assert agenda.get_appointment(ref).slot_id == morning


def test_a_fasting_exam_may_run_past_noon(book: Book) -> None:
    """The fasting rule is on the start: 11:45 to 12:35 is a valid booking.

    The patient's fast ends when the exam begins, so a window that starts before
    12:00 and finishes after it breaks no rule.
    """
    straddling = _add_pair_straddling_noon(book)

    assert straddling[0] in slot_ids(agenda.find_slots("EX008", limit=200))

    ref = agenda.confirm_booking(straddling[0], "EX008", "patient-70")

    appointment = agenda.get_appointment(ref)
    assert appointment.slot_ids == tuple(straddling)
    assert appointment.start is not None and appointment.end is not None
    assert appointment.start.strftime("%H:%M") == "11:45"
    assert appointment.end.strftime("%H:%M") == "12:35"
    assert appointment.start.hour < agenda.FASTING_CUTOFF_HOUR
    assert appointment.end.hour >= agenda.FASTING_CUTOFF_HOUR


def test_fasting_is_detected_whatever_the_accent() -> None:
    """"à jeun" and "a jeun" both mean fasting."""
    assert agenda.needs_fasting("Venir à jeun depuis 6 heures.") is True
    assert agenda.needs_fasting("venir a jeun") is True
    assert agenda.needs_fasting("Aucune préparation particulière.") is False


# --------------------------------------------------------------------------
# Unknown identifiers
# --------------------------------------------------------------------------

def test_unknown_ref_code_raises(book: Book) -> None:
    """A code nobody holds raises UnknownRefCode, on every lookup path."""
    with pytest.raises(agenda.UnknownRefCode):
        agenda.get_appointment("ZZZZZZ")
    with pytest.raises(agenda.UnknownRefCode):
        agenda.cancel_appointment("ZZZZZZ")
    with pytest.raises(agenda.UnknownRefCode):
        agenda.change_appointment("ZZZZZZ", book.nth("irm-1", 0, 0))


def test_unknown_slot_raises(book: Book) -> None:
    """A slot id that does not exist raises SlotNotFound."""
    with pytest.raises(agenda.SlotNotFound):
        agenda.confirm_booking("no-such-slot", "EX001", "patient-18")
    with pytest.raises(agenda.SlotNotFound):
        agenda.hold_slot("no-such-slot", "call-99")


def test_unknown_exam_raises(book: Book) -> None:
    """An exam id outside the catalogue raises UnknownExam."""
    with pytest.raises(agenda.UnknownExam):
        agenda.find_slots("EX999")
    with pytest.raises(agenda.UnknownExam):
        agenda.confirm_booking(book.nth("irm-1", 0, 0), "EX999", "patient-19")


# --------------------------------------------------------------------------
# Availability filters
# --------------------------------------------------------------------------

def test_find_slots_respects_the_limit(book: Book) -> None:
    """The search returns at most ``limit`` windows, soonest first."""
    offers = agenda.find_slots("EX001", limit=3)

    assert len(offers) == 3
    assert [o.start for o in offers] == sorted(o.start for o in offers)


def test_find_slots_filters_by_day(book: Book) -> None:
    """A day with no slots yields nothing; the fixture's day yields some."""
    assert agenda.find_slots("EX001", day=book.day, limit=50)
    assert agenda.find_slots("EX001", day=book.day + timedelta(days=1), limit=50) == []


def test_find_slots_filters_by_period(book: Book) -> None:
    """Morning and afternoon split at 12:00."""
    morning = agenda.find_slots("EX001", period="matin", limit=50)
    afternoon = agenda.find_slots("EX001", period="apres-midi", limit=50)

    assert morning and afternoon
    assert all(o.start.hour < 12 for o in morning)
    assert all(o.start.hour >= 12 for o in afternoon)


def test_find_slots_ignores_the_past(book: Book) -> None:
    """A slot that has already started is never offered."""
    conn = agenda.connect()
    try:
        yesterday = datetime.now() - timedelta(days=1)
        with agenda.transaction(conn):
            conn.execute(
                'INSERT INTO slots (id, room_id, start, "end", status) '
                "VALUES ('past-slot', 'irm-1', ?, ?, 'free')",
                (
                    yesterday.strftime(agenda.DT_FORMAT),
                    (yesterday + timedelta(minutes=25)).strftime(agenda.DT_FORMAT),
                ),
            )
    finally:
        conn.close()

    assert "past-slot" not in slot_ids(agenda.find_slots("EX001", limit=50))


def test_unknown_period_is_rejected(book: Book) -> None:
    """A period the agent did not understand is an error, not a silent full search."""
    with pytest.raises(ValueError):
        agenda.find_slots("EX001", period="vers minuit")


# --------------------------------------------------------------------------
# No personal data
# --------------------------------------------------------------------------

def test_phone_is_stored_masked(book: Book) -> None:
    """The stored phone keeps only its last two digits."""
    ref = agenda.confirm_booking(
        book.nth("irm-1", 0, 0), "EX001", "patient-20", "06 12 34 56 78"
    )

    stored = agenda.get_appointment(ref).phone
    assert stored == "********78"
    assert "0612345678" not in (stored or "")


def test_booking_without_a_phone_is_allowed(book: Book) -> None:
    """A caller who gives no number can still be booked."""
    ref = agenda.confirm_booking(book.nth("irm-1", 0, 0), "EX001", "patient-21")

    assert agenda.get_appointment(ref).phone is None


def test_an_alias_is_required(book: Book) -> None:
    """A booking with no alias at all is refused."""
    with pytest.raises(agenda.AgendaError):
        agenda.confirm_booking(book.nth("irm-1", 0, 0), "EX001", "   ")


def test_ref_codes_avoid_ambiguous_characters() -> None:
    """Codes read aloud over the phone contain no I, O, 0 or 1."""
    codes = [agenda.new_ref_code() for _ in range(200)]

    assert all(len(c) == agenda.REF_LENGTH for c in codes)
    assert not set("".join(codes)) & set("IO01")
