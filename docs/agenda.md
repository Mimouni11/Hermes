# The appointment book

`src/hermes/agenda.py` is the only module that writes to the calendar. It holds
no LLM call and no network access: the model fills a JSON form, Python does the
searching, holding and booking, and every rule is checked here.

SQLite, three tables, built by `scripts/build_agenda.py`.

## Schema

```sql
rooms         (id, modality, name)
slots         (id, room_id, start, end, status, appointment_id, call_id, held_until)
appointments  (id, ref_code, exam_id, slot_id, patient_alias, phone, status, created_at)
```

Two columns go beyond the plan's sketch, and both earn their place:

- **`slots.appointment_id`** — which booking owns this slot. Without it,
  cancelling a three-slot appointment has to *recompute* which slots to free
  from the exam's duration in the CSV, which breaks silently the day a duration
  changes. It also separates agent bookings from the pre-booked 30 % the build
  script creates, which carry `NULL`.
- **`slots.call_id` and `slots.held_until`** — who holds a slot, and until when.

`appointments.phone` is stored masked to its last two digits, and
`patient_alias` is whatever the caller chooses. No real identity is kept.

## The rules

Every booking path enforces all six:

1. the slot exists,
2. it is free, or held by *this* call — an expired hold counts as free,
3. the room's modality matches the exam's,
4. a fasting exam must **start** before 12:00,
5. an exam longer than one slot takes consecutive slots in the same room,
6. a slot can never be booked twice.

### Exams longer than one slot

Each modality has one slot length (IRM 25 min, scanner 10, échographie 15…),
but **49 of the 173 catalogue exams are longer than their modality's slot**, up
to three slots. This is the normal case, not an edge case.

| modality | slot | exams | > 1 slot | worst |
|---|---|---|---|---|
| IRM | 25 min | 39 | 14 | 2 |
| scanner | 10 min | 35 | 10 | 3 |
| échographie | 15 min | 41 | 14 | 2 |
| mammographie | 10 min | 6 | 4 | 3 |
| radiographie | 5 min | 46 | 4 | 2 |
| dentaire | 5 min | 4 | 2 | 3 |
| densitométrie | 15 min | 2 | 1 | 2 |

`find_slots()` returns `Offer` objects — a window of one *or more* consecutive
slots long enough for the whole exam. Runs are grown from **each slot's own
length**, never one room's length applied to another, so two rooms of the same
modality may run different slot lengths. Contiguity is checked on
`end == start`, so a booked slot mid-block breaks the run by itself.

`confirm_booking(slot_id, …)` takes only the first slot and works the rest out
from the exam's duration.

### Holds

`hold_slot(slot_ids, call_id)` reserves a whole window for one call and stamps
`held_until = now + 5 min`. Only `confirm_booking` with the same `call_id` can
book it; once it expires, the slots are on offer again. A caller who hangs up
mid-booking cannot park a slot for the rest of the day.

### Never twice

Claiming a run is **one** statement inside `BEGIN IMMEDIATE`:

```sql
UPDATE slots SET status = 'booked'
 WHERE id IN (…) AND (free OR own hold OR expired hold OR own appointment)
```

with the row count checked against the run length. Either every slot was
available and the appointment is whole, or nothing moved. `IMMEDIATE` takes the
write lock up front, so two callers cannot both read a slot as free.

## Known deviation

The fasting rule reads the window's **start** only. A 50-minute fasting exam may
begin at 11:45 and finish at 12:35. This is deliberate — the patient's fast ends
when the exam begins — and it is pinned by a test.

## Tests

59 tests in `tests/test_agenda.py`, against a temporary database and catalogue
built per test, so nothing touches `data/`. The fixture lays out genuinely
contiguous blocks, because an exam longer than one slot can only be placed if
the slots actually abut — the first version of that fixture had gaps, and no
multi-slot run could have existed to test.
