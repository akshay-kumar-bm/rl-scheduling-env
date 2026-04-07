"""Graders for the meeting-scheduling RL environment.

Provides programmatic scoring (0.0–1.0) per episode and validation
that graders produce diverse scores across different agent trajectories.
"""

from __future__ import annotations

import logging
from typing import List

from .scheduling_logic import (
    calculate_collective_hours,
    calculate_final_reward,
    find_conflicts,
    parse_iso,
)

logger = logging.getLogger(__name__)


class SchedulingGrader:
    """Programmatic grader for scheduling tasks."""

    def grade_episode(self, final_state, final_observation) -> float:
        """Score an episode in [0.0, 1.0].

        Returns ``final_state.final_reward`` when the episode completed
        successfully, with a 50 % penalty applied if any hard constraint
        violations are detected.
        """
        if not final_state.completed or not final_observation.success:
            return 0.0

        score = final_state.final_reward

        violations = self._check_violations(final_state)
        if violations:
            score *= 0.5
            logger.warning("Constraint violations: %s", violations)

        return max(0.0, min(1.0, score))

    def _check_violations(self, state) -> List[str]:
        """Detect hard constraint violations in the final state."""
        violations: List[str] = []
        req_priority = state.meeting_request.get("priority", 99)

        # Violation 1: Rescheduled equal-or-higher priority meeting
        for rm in state.rescheduled_meetings:
            attendee = rm["attendee"]
            old_start = rm["old_start"]
            for entry in state.calendars.get(attendee, []):
                if entry[0] == old_start and entry[2] <= req_priority:
                    violations.append(
                        f"Rescheduled higher priority meeting: "
                        f"{attendee} {old_start}"
                    )

        # Violation 2: Proposed slot outside collective working hours
        if state.proposed_slot:
            collective = calculate_collective_hours(state.participant_preferences)
            start = parse_iso(state.proposed_slot[0])
            end = parse_iso(state.proposed_slot[1])
            if start.hour < collective["min_start_hour"]:
                violations.append(
                    f"Slot starts before working hours: {state.proposed_slot[0]}"
                )
            if end.hour > collective["max_end_hour"] or (
                end.hour == collective["max_end_hour"] and end.minute > 0
            ):
                violations.append(
                    f"Slot ends after working hours: {state.proposed_slot[1]}"
                )

        # Violation 3: Overlapping meetings after rescheduling
        for user_id, calendar in state.calendars.items():
            sorted_cal = sorted(calendar, key=lambda e: e[0])
            for i in range(len(sorted_cal) - 1):
                end_i = parse_iso(sorted_cal[i][1])
                start_next = parse_iso(sorted_cal[i + 1][0])
                if end_i > start_next:
                    violations.append(
                        f"Overlap for {user_id}: {sorted_cal[i][3]} / {sorted_cal[i+1][3]}"
                    )

        return violations
