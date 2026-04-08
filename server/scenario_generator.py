"""Random scenario generator for the meeting-scheduling RL environment.

Produces solvable scenarios with controlled difficulty so agents cannot
memorise fixed answers.  Difficulty is governed by a parameter dict that
controls attendee count, calendar density, and preference strictness.

Usage:
    from server.scenario_generator import generate_scenario
    scenario = generate_scenario("random_medium")
    scenario = generate_scenario("random_easy", seed=42)   # reproducible
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from .scheduling_logic import (
    find_conflicts,
    is_slot_free,
    within_collective_hours,
    calculate_collective_hours,
)

# ── Difficulty presets ────────────────────────────────────────────────

DIFFICULTY_PARAMS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "num_attendees": (2, 2),          # (min, max)
        "meetings_per_user": (1, 3),
        "meeting_duration": (30, 30),     # requested meeting duration in min
        "request_priority": (3, 4),       # lower number = higher priority
        "existing_priority_range": (2, 5),
        "pref_start_range": (8, 10),
        "pref_end_range": (16, 18),
        "max_meetings_day_range": (5, 8),
        "avoid_btb_prob": 0.1,
        "buffer_range": (0, 10),
        "calendar_slot_min": 30,
        "calendar_slot_max": 60,
        "guarantee_free_slot": True,      # easy always has a free slot
    },
    "medium": {
        "num_attendees": (3, 4),
        "meetings_per_user": (3, 5),
        "meeting_duration": (30, 60),
        "request_priority": (2, 3),
        "existing_priority_range": (2, 5),
        "pref_start_range": (9, 10),
        "pref_end_range": (15, 17),
        "max_meetings_day_range": (4, 6),
        "avoid_btb_prob": 0.5,
        "buffer_range": (10, 20),
        "calendar_slot_min": 30,
        "calendar_slot_max": 90,
        "guarantee_free_slot": False,
    },
    "hard": {
        "num_attendees": (4, 6),
        "meetings_per_user": (4, 7),
        "meeting_duration": (30, 60),
        "request_priority": (1, 2),
        "existing_priority_range": (2, 5),
        "pref_start_range": (9, 11),
        "pref_end_range": (15, 16),
        "max_meetings_day_range": (3, 5),
        "avoid_btb_prob": 0.7,
        "buffer_range": (10, 25),
        "calendar_slot_min": 30,
        "calendar_slot_max": 90,
        "guarantee_free_slot": False,
    },
}

MEETING_SUMMARIES = [
    "Standup", "Sprint planning", "Design review", "Code review",
    "Client call", "1-on-1", "Team sync", "Project checkpoint",
    "Lunch meeting", "Strategy session", "Architecture review",
    "Product demo", "Budget meeting", "Office hours", "Workshop",
    "Training session", "Coffee chat", "Retrospective",
    "Performance review", "Brainstorming", "Board meeting",
    "Vendor call", "Onboarding session", "Knowledge sharing",
]

# ── Helpers ───────────────────────────────────────────────────────────

def _rand_int(lo: int, hi: int, rng: random.Random) -> int:
    return rng.randint(lo, hi)


def _rand_range(r: Tuple[int, int], rng: random.Random) -> int:
    return rng.randint(r[0], r[1])


def _random_weekday(rng: random.Random) -> date:
    """Pick a random weekday within the next 30 days."""
    base = date(2025, 4, 7)  # fixed base so TZ stays consistent
    offset = rng.randint(0, 29)
    d = base + timedelta(days=offset)
    # shift to nearest weekday
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _generate_calendar(
    user_id: str,
    target_date: date,
    num_meetings: int,
    params: Dict[str, Any],
    rng: random.Random,
    reserved_slots: Optional[List[Tuple[datetime, datetime]]] = None,
) -> List[List]:
    """Generate random non-overlapping calendar entries for one user."""
    tz = timezone.utc
    day_start_hour = _rand_range(params["pref_start_range"], rng)
    day_end_hour = _rand_range(params["pref_end_range"], rng)
    if day_end_hour <= day_start_hour + 2:
        day_end_hour = day_start_hour + 6

    entries: List[List] = []
    occupied: List[Tuple[datetime, datetime]] = []
    if reserved_slots:
        occupied.extend(reserved_slots)

    attempts = 0
    while len(entries) < num_meetings and attempts < 80:
        attempts += 1
        dur = _rand_range(
            (params["calendar_slot_min"], params["calendar_slot_max"]),
            rng,
        )
        # round to 15-min
        dur = max(15, (dur // 15) * 15)

        hour = rng.randint(day_start_hour, max(day_start_hour, day_end_hour - 1))
        minute = rng.choice([0, 15, 30, 45])
        start = datetime(target_date.year, target_date.month, target_date.day,
                         hour, minute, 0, tzinfo=tz)
        end = start + timedelta(minutes=dur)

        boundary = datetime(target_date.year, target_date.month, target_date.day,
                            day_end_hour, 0, 0, tzinfo=tz)
        if end > boundary:
            continue

        # check overlap with already placed meetings
        overlap = False
        for occ_s, occ_e in occupied:
            if start < occ_e and occ_s < end:
                overlap = True
                break
        if overlap:
            continue

        priority = _rand_range(params["existing_priority_range"], rng)
        summary = rng.choice(MEETING_SUMMARIES)
        entries.append([start.isoformat(), end.isoformat(), priority, summary])
        occupied.append((start, end))

    # sort by start time
    entries.sort(key=lambda e: e[0])
    return entries


def _generate_preferences(
    user_id: str,
    params: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    """Generate random but realistic preferences for one user."""
    start_h = _rand_range(params["pref_start_range"], rng)
    end_h = _rand_range(params["pref_end_range"], rng)
    if end_h <= start_h + 4:
        end_h = start_h + 6
    if end_h > 18:
        end_h = 18

    avoid_btb = rng.random() < params["avoid_btb_prob"]
    buffer = _rand_range(params["buffer_range"], rng) if avoid_btb else 0
    # round buffer to 5
    buffer = (buffer // 5) * 5

    return {
        "preferred_hours": {"start": start_h, "end": end_h},
        "max_meetings_per_day": _rand_range(params["max_meetings_day_range"], rng),
        "avoid_back_to_back": avoid_btb,
        "buffer_minutes": buffer,
    }


# ── Solvability check ────────────────────────────────────────────────

def _find_solvable_slot(
    calendars: Dict[str, List[List]],
    attendees: List[str],
    duration: int,
    request_priority: int,
    collective_hours: Dict[str, int],
    target_date: date,
    allow_rescheduling: bool = True,
) -> Optional[str]:
    """Check if at least one slot exists (possibly after rescheduling).

    Returns the ISO start time of a viable slot, or None.
    """
    tz = timezone.utc
    min_h = collective_hours["min_start_hour"]
    max_h = collective_hours["max_end_hour"]

    candidate = datetime(target_date.year, target_date.month, target_date.day,
                         min_h, 0, 0, tzinfo=tz)
    end_boundary = datetime(target_date.year, target_date.month, target_date.day,
                            max_h, 0, 0, tzinfo=tz)
    step = timedelta(minutes=15)

    while candidate + timedelta(minutes=duration) <= end_boundary:
        c_start = candidate.isoformat()
        c_end = (candidate + timedelta(minutes=duration)).isoformat()

        conflicts = find_conflicts(calendars, c_start, c_end, attendees)

        if len(conflicts) == 0:
            return c_start

        if allow_rescheduling:
            # solvable if ALL conflicts have strictly lower priority (higher number)
            all_reschedulable = all(c["priority"] > request_priority for c in conflicts)
            if all_reschedulable:
                return c_start

        candidate += step

    return None


# ── Plant a guaranteed free slot (easy mode) ──────────────────────────

def _plant_free_slot(
    calendars: Dict[str, List[List]],
    attendees: List[str],
    duration: int,
    collective_hours: Dict[str, int],
    target_date: date,
    rng: random.Random,
) -> Optional[str]:
    """Remove conflicts from a random viable slot to guarantee a free one.

    Returns the ISO start of the planted slot.
    """
    tz = timezone.utc
    min_h = collective_hours["min_start_hour"]
    max_h = collective_hours["max_end_hour"]

    # collect all possible starts
    candidates = []
    t = datetime(target_date.year, target_date.month, target_date.day,
                 min_h, 0, 0, tzinfo=tz)
    end_boundary = datetime(target_date.year, target_date.month, target_date.day,
                            max_h, 0, 0, tzinfo=tz)
    step = timedelta(minutes=15)
    while t + timedelta(minutes=duration) <= end_boundary:
        candidates.append(t)
        t += step

    rng.shuffle(candidates)

    for candidate in candidates:
        c_start = candidate.isoformat()
        c_end = (candidate + timedelta(minutes=duration)).isoformat()

        # remove any overlapping entries for all attendees
        for att in attendees:
            calendars[att] = [
                e for e in calendars[att]
                if not (candidate < datetime.fromisoformat(e[1])
                        and datetime.fromisoformat(e[0]) < candidate + timedelta(minutes=duration))
            ]

        # verify it's now free
        conflicts = find_conflicts(calendars, c_start, c_end, attendees)
        if len(conflicts) == 0:
            return c_start

    return None


# ── Main generator ────────────────────────────────────────────────────

def generate_scenario(
    difficulty: str = "medium",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate a random solvable scheduling scenario.

    Args:
        difficulty: One of "easy", "medium", "hard".
        seed: Optional RNG seed for reproducibility.

    Returns:
        A scenario dict with the same structure as the static JSON files.
    """
    if difficulty not in DIFFICULTY_PARAMS:
        raise ValueError(f"Unknown difficulty: {difficulty}. Use easy/medium/hard.")

    params = DIFFICULTY_PARAMS[difficulty]
    rng = random.Random(seed)

    target_date = _random_weekday(rng)
    num_attendees = _rand_range(params["num_attendees"], rng)
    attendees = [f"user{i+1}" for i in range(num_attendees)]

    duration = _rand_range(params["meeting_duration"], rng)
    # round to 15
    duration = max(15, (duration // 15) * 15)

    request_priority = _rand_range(params["request_priority"], rng)

    # generate preferences first (needed for collective hours)
    preferences: Dict[str, Dict] = {}
    for att in attendees:
        preferences[att] = _generate_preferences(att, params, rng)

    collective_hours = calculate_collective_hours(preferences)
    # safety: ensure window is wide enough for the meeting
    window = collective_hours["max_end_hour"] - collective_hours["min_start_hour"]
    if window * 60 < duration:
        collective_hours["max_end_hour"] = collective_hours["min_start_hour"] + (duration // 60) + 2
        # also widen preferences
        for att in attendees:
            preferences[att]["preferred_hours"]["end"] = max(
                preferences[att]["preferred_hours"]["end"],
                collective_hours["max_end_hour"],
            )

    # generate calendars
    calendars: Dict[str, List[List]] = {}
    for att in attendees:
        n_meetings = _rand_range(params["meetings_per_user"], rng)
        calendars[att] = _generate_calendar(att, target_date, n_meetings, params, rng)

    # ensure solvability
    if params["guarantee_free_slot"]:
        _plant_free_slot(calendars, attendees, duration, collective_hours, target_date, rng)

    # verify at least one solution exists (with rescheduling allowed)
    max_retries = 10
    for attempt in range(max_retries):
        viable = _find_solvable_slot(
            calendars, attendees, duration, request_priority,
            collective_hours, target_date, allow_rescheduling=True,
        )
        if viable is not None:
            break

        # regenerate calendars with fewer meetings to open up space
        for att in attendees:
            reduced = max(1, _rand_range(params["meetings_per_user"], rng) - 1)
            calendars[att] = _generate_calendar(att, target_date, reduced, params, rng)
    else:
        # last resort: plant a free slot
        _plant_free_slot(calendars, attendees, duration, collective_hours, target_date, rng)

    # find solution info for metadata
    free_slot = _find_solvable_slot(
        calendars, attendees, duration, request_priority,
        collective_hours, target_date, allow_rescheduling=False,
    )
    needs_rescheduling = free_slot is None
    best_slot = free_slot or _find_solvable_slot(
        calendars, attendees, duration, request_priority,
        collective_hours, target_date, allow_rescheduling=True,
    )

    task_id = f"random_{difficulty}"
    description_map = {
        "easy": f"Schedule a {duration}-min meeting with {num_attendees} attendees (random easy)",
        "medium": f"Schedule a {duration}-min meeting with {num_attendees} attendees (random medium)",
        "hard": f"Schedule a {duration}-min meeting with {num_attendees} attendees (random hard)",
    }

    return {
        "task_id": task_id,
        "description": description_map[difficulty],
        "difficulty": difficulty,
        "meeting_request": {
            "duration": duration,
            "priority": request_priority,
            "attendees": attendees,
            "summary": rng.choice([
                "Team Sync", "Planning Session", "Design Review",
                "Sprint Review", "Cross-Team Standup", "Strategy Meeting",
            ]),
        },
        "calendars": calendars,
        "preferences": preferences,
        "expected_solution": {
            "optimal_slot": best_slot,
            "requires_rescheduling": needs_rescheduling,
            "generated": True,
        },
    }
