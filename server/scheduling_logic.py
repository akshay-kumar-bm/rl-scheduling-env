"""Pure utility functions for the meeting-scheduling RL environment.

Calendar format: Dict[str, List[List]]
  Each entry is [start_iso, end_iso, priority_int, summary_str].

All datetimes are timezone-aware ISO 8601 strings.
"""
from __future__ import annotations

import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional


def parse_iso(s: str) -> datetime:
    """Parse an ISO 8601 string into a datetime object."""
    return datetime.fromisoformat(s)


def load_scenario(scenario_path: str) -> dict:
    """Load a scenario JSON file and return the parsed dict."""
    with open(scenario_path, "r") as f:
        return json.load(f)


def find_conflicts(
    calendars: Dict[str, List[List]],
    proposed_start_iso: str,
    proposed_end_iso: str,
    attendee_ids: List[str],
) -> List[Dict]:
    """Find calendar conflicts between a proposed slot and existing meetings.

    Two intervals overlap when start1 < end2 and start2 < end1.

    Returns:
        List of conflict dicts with keys: attendee, start, end, priority,
        summary, meeting_id.
    """
    proposed_start = parse_iso(proposed_start_iso)
    proposed_end = parse_iso(proposed_end_iso)
    conflicts: List[Dict] = []

    for attendee in attendee_ids:
        entries = calendars.get(attendee, [])
        for entry in entries:
            entry_start_iso, entry_end_iso, priority, summary = entry
            entry_start = parse_iso(entry_start_iso)
            entry_end = parse_iso(entry_end_iso)

            if proposed_start < entry_end and entry_start < proposed_end:
                conflicts.append({
                    "attendee": attendee,
                    "start": entry_start_iso,
                    "end": entry_end_iso,
                    "priority": priority,
                    "summary": summary,
                    "meeting_id": f"{attendee}_{entry_start_iso}",
                })

    return conflicts


def calculate_collective_hours(preferences: Dict) -> Dict[str, int]:
    """Find the intersection of all users' preferred working hours.

    Each user preference has 'preferred_hours': {'start': int, 'end': int}.

    Returns:
        {"min_start_hour": <max of all starts>, "max_end_hour": <min of all ends>}
    """
    start_hours = [p.get("preferred_hours", {}).get("start", 9) for p in preferences.values()]
    end_hours = [p.get("preferred_hours", {}).get("end", 17) for p in preferences.values()]

    return {
        "min_start_hour": max(start_hours),
        "max_end_hour": min(end_hours),
    }


def within_collective_hours(
    start_iso: str,
    end_iso: str,
    collective_hours: Dict[str, int],
) -> bool:
    """Check if a proposed slot falls within collective working hours.

    The start hour must be >= min_start_hour and the end hour must be
    <= max_end_hour (exact hour boundary is allowed).
    """
    start = parse_iso(start_iso)
    end = parse_iso(end_iso)
    min_start = collective_hours["min_start_hour"]
    max_end = collective_hours["max_end_hour"]

    if start.hour < min_start:
        return False

    # Handle end at exact hour boundary (minute == 0) vs. mid-hour
    if end.minute == 0 and end.second == 0:
        if end.hour > max_end:
            return False
    else:
        if end.hour >= max_end:
            return False

    return True


def count_meetings_on_date(calendar_entries: List[List], target_date: date) -> int:
    """Count how many meetings a user has on a given date."""
    count = 0
    for entry in calendar_entries:
        entry_start = parse_iso(entry[0])
        if entry_start.date() == target_date:
            count += 1
    return count


def check_back_to_back(
    calendar_entries: List[List],
    proposed_start_iso: str,
    proposed_end_iso: str,
    buffer_minutes: int,
) -> bool:
    """Check if a proposed meeting would be back-to-back with an existing one.

    Returns True if any existing meeting ends within buffer_minutes before
    the proposed start, or starts within buffer_minutes after the proposed end.
    """
    proposed_start = parse_iso(proposed_start_iso)
    proposed_end = parse_iso(proposed_end_iso)
    buffer = timedelta(minutes=buffer_minutes)

    for entry in calendar_entries:
        entry_start = parse_iso(entry[0])
        entry_end = parse_iso(entry[1])

        # Existing meeting ends close before proposed start
        gap_before = proposed_start - entry_end
        if timedelta(0) <= gap_before < buffer:
            return True

        # Existing meeting starts close after proposed end
        gap_after = entry_start - proposed_end
        if timedelta(0) <= gap_after < buffer:
            return True

    return False


def calculate_preference_score(
    proposed_start_iso: str,
    duration_minutes: int,
    participant_preferences: Dict,
    calendars: Dict[str, List[List]],
) -> float:
    """Calculate penalty points for scheduling preference violations.

    Penalty rules:
        - Outside preferred hours: +50 per participant
        - Exceeds max meetings per day: +30 per participant
        - Back-to-back without buffer: +20 per participant

    Returns:
        Total penalty sum (float).
    """
    proposed_start = parse_iso(proposed_start_iso)
    proposed_end = proposed_start + timedelta(minutes=duration_minutes)
    proposed_end_iso = proposed_end.isoformat()
    proposed_date = proposed_start.date()

    total_penalty = 0.0

    for participant, prefs in participant_preferences.items():
        pref_hours = prefs.get("preferred_hours", {})
        pref_start = pref_hours.get("start", 9)
        pref_end = pref_hours.get("end", 17)
        max_meetings = prefs.get("max_meetings_per_day", 8)
        avoid_btb = prefs.get("avoid_back_to_back", False)
        buffer_mins = prefs.get("buffer_minutes", 0)

        # Outside preferred hours
        collective = {"min_start_hour": pref_start, "max_end_hour": pref_end}
        if not within_collective_hours(proposed_start_iso, proposed_end_iso, collective):
            total_penalty += 50

        # Exceeds max meetings per day
        entries = calendars.get(participant, [])
        existing_count = count_meetings_on_date(entries, proposed_date)
        if existing_count + 1 > max_meetings:
            total_penalty += 30

        # Back-to-back without buffer (only if user cares about it)
        if avoid_btb and buffer_mins > 0:
            if check_back_to_back(entries, proposed_start_iso, proposed_end_iso, buffer_mins):
                total_penalty += 20

    return total_penalty


def is_slot_free(
    attendee: str,
    start_iso: str,
    end_iso: str,
    calendars: Dict[str, List[List]],
) -> bool:
    """Check if a time slot is free for a specific attendee (no overlaps)."""
    start = parse_iso(start_iso)
    end = parse_iso(end_iso)

    for entry in calendars.get(attendee, []):
        entry_start = parse_iso(entry[0])
        entry_end = parse_iso(entry[1])
        if start < entry_end and entry_start < end:
            return False

    return True


def calculate_final_reward(
    preference_penalty: float,
    num_rescheduled: int,
    steps_taken: int,
    success: bool = True,
) -> float:
    """Compute the multi-component reward for an episode, clamped to [0.0, 1.0].

    Components (deducted from 1.0):
        - Preference deduction: min(0.75, (preference_penalty ** 1.2) / 200.0)
        - Rescheduling deduction: min(0.30, 0.05 * (1.8 ** num_rescheduled))
          (only applied when num_rescheduled > 0)
        - Time deduction: steps_taken * 0.015

    Returns 0.0 if the episode was not successful.
    """
    if not success:
        return 0.0

    reward = 1.0

    # Preference deduction
    pref_deduction = min(0.75, (preference_penalty ** 1.2) / 200.0)
    reward -= pref_deduction

    # Rescheduling deduction (exponential)
    if num_rescheduled > 0:
        reschedule_deduction = min(0.30, 0.05 * (1.8 ** num_rescheduled))
        reward -= reschedule_deduction

    # Time deduction
    time_deduction = steps_taken * 0.015
    reward -= time_deduction

    return max(0.0, min(1.0, reward))


def build_busy_slots(
    calendars: Dict[str, List[List]],
    attendee_ids: List[str],
) -> List[Dict]:
    """Convert calendar data to observation-friendly busy_slots format.

    Returns:
        List of dicts with keys: start, end, priority, summary, attendee.
    """
    busy_slots: List[Dict] = []

    for attendee in attendee_ids:
        for entry in calendars.get(attendee, []):
            start_iso, end_iso, priority, summary = entry
            busy_slots.append({
                "start": start_iso,
                "end": end_iso,
                "priority": priority,
                "summary": summary,
                "attendee": attendee,
            })

    return busy_slots


def find_earliest_free_slot(
    calendars: Dict[str, List[List]],
    attendees: List[str],
    duration_minutes: int,
    search_date_iso: str,
    collective_hours: Dict[str, int],
) -> Optional[str]:
    """Find the earliest free slot on a given date for all attendees.

    Iterates from min_start_hour to max_end_hour in 15-minute increments.
    Returns the ISO 8601 string of the first conflict-free slot, or None.
    """
    search_date = parse_iso(search_date_iso)
    base_date = search_date.date()
    tz = search_date.tzinfo

    min_start = collective_hours["min_start_hour"]
    max_end = collective_hours["max_end_hour"]

    candidate = datetime(base_date.year, base_date.month, base_date.day,
                         min_start, 0, 0, tzinfo=tz)
    end_boundary = datetime(base_date.year, base_date.month, base_date.day,
                            max_end, 0, 0, tzinfo=tz)
    step = timedelta(minutes=15)

    while candidate + timedelta(minutes=duration_minutes) <= end_boundary:
        candidate_iso = candidate.isoformat()
        candidate_end_iso = (candidate + timedelta(minutes=duration_minutes)).isoformat()

        all_free = True
        for attendee in attendees:
            if not is_slot_free(attendee, candidate_iso, candidate_end_iso, calendars):
                all_free = False
                break

        if all_free:
            return candidate_iso

        candidate += step

    return None
