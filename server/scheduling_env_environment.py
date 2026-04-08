# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Meeting Scheduling RL Environment.

Teaches agents to optimally schedule meetings across multiple attendees
by proposing time slots, rescheduling lower-priority conflicts, and
balancing participant preferences.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import SchedulingAction, SchedulingObservation, SchedulingState
except ImportError:
    from models import SchedulingAction, SchedulingObservation, SchedulingState

from .scheduling_logic import (
    build_busy_slots,
    calculate_collective_hours,
    calculate_final_reward,
    calculate_preference_score,
    find_conflicts,
    is_slot_free,
    parse_iso,
    within_collective_hours,
)
from .scenario_generator import generate_scenario

logger = logging.getLogger(__name__)

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
MAX_STEPS = 20


class SchedulingEnvironment(Environment):
    """RL environment for intelligent meeting scheduling.

    The agent must learn to:
    1. Propose valid time slots satisfying hard constraints
    2. Minimize preference violations
    3. Handle cascading rescheduling when conflicts exist
    4. Balance speed vs. quality
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = SchedulingState(episode_id=str(uuid4()), step_count=0)
        self._scenario: dict = {}
        self._collective_hours: dict = {}

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> SchedulingObservation:
        """Reset environment for a new episode.

        Accepts ``task_id`` kwarg.  Static tasks (``"task1_easy"`` etc.) load
        from JSON.  Random tasks (``"random_easy"``, ``"random_medium"``,
        ``"random_hard"``) generate a fresh scenario every call.  An optional
        ``seed`` kwarg makes random generation reproducible.
        """
        task_id = kwargs.get("task_id", "task1_easy")

        # ── random scenario generation ──
        if task_id.startswith("random_"):
            difficulty = task_id.split("_", 1)[1]
            seed = kwargs.get("seed", None)
            try:
                self._scenario = generate_scenario(difficulty, seed=seed)
            except ValueError:
                return SchedulingObservation(
                    error_message=f"Unknown difficulty in task_id: {task_id}",
                    done=True,
                    reward=0.0,
                )
        else:
            # ── static JSON scenario ──
            scenario_path = SCENARIOS_DIR / f"{task_id}.json"
            if not scenario_path.exists():
                return SchedulingObservation(
                    error_message=f"Unknown task_id: {task_id}",
                    done=True,
                    reward=0.0,
                )
            with open(scenario_path) as f:
                self._scenario = json.load(f)

        req = self._scenario["meeting_request"]
        prefs = self._scenario["preferences"]
        self._collective_hours = calculate_collective_hours(prefs)

        self._state = SchedulingState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task_id,
            scenario_name=self._scenario.get("description", task_id),
            meeting_request=req,
            calendars=copy.deepcopy(self._scenario["calendars"]),
            participant_preferences=prefs,
            proposed_slot=None,
            rescheduled_meetings=[],
            total_preference_penalty=0.0,
            total_steps=0,
            final_reward=0.0,
            completed=False,
        )

        attendees = req["attendees"]
        return SchedulingObservation(
            requested_duration=req["duration"],
            requested_priority=req["priority"],
            attendee_ids=attendees,
            busy_slots=build_busy_slots(self._state.calendars, attendees),
            collective_work_hours=self._collective_hours,
            preference_constraints=self._aggregate_preferences(prefs),
            current_proposal=None,
            conflicts=[],
            preference_penalty=0.0,
            num_rescheduled=0,
            steps_taken=0,
            max_steps=MAX_STEPS,
            success=False,
            error_message=None,
            done=False,
            reward=0.0,
        )

    def step(self, action: SchedulingAction) -> SchedulingObservation:  # type: ignore[override]
        """Process one agent action and return an observation."""
        if self._state.completed:
            return self._obs(error_message="Episode already completed", done=True, reward=0.0)

        self._state.step_count += 1
        self._state.total_steps += 1

        # Timeout check
        if self._state.step_count >= MAX_STEPS:
            return self._handle_timeout()

        action_type = action.action_type

        if action_type == "propose_slot":
            return self._process_propose_slot(action)
        elif action_type == "reschedule_meeting":
            return self._process_reschedule_meeting(action)
        elif action_type == "finalize":
            return self._process_finalize()
        elif action_type == "reject":
            return self._process_reject()
        else:
            return self._obs(error_message=f"Unknown action_type: {action_type}", reward=-0.1)

    @property
    def state(self) -> SchedulingState:
        return self._state

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _process_propose_slot(self, action: SchedulingAction) -> SchedulingObservation:
        if not action.proposed_start or not action.proposed_duration:
            return self._obs(
                error_message="propose_slot requires proposed_start and proposed_duration",
                reward=-0.1,
            )

        try:
            start = parse_iso(action.proposed_start)
        except (ValueError, TypeError):
            return self._obs(error_message="Invalid proposed_start format", reward=-0.1)

        end = start + timedelta(minutes=action.proposed_duration)
        start_iso = start.isoformat()
        end_iso = end.isoformat()
        attendees = self._state.meeting_request["attendees"]
        req_priority = self._state.meeting_request["priority"]

        # Validate working hours
        if not within_collective_hours(start_iso, end_iso, self._collective_hours):
            return self._obs(
                error_message="Proposed slot outside working hours",
                reward=-0.2,
            )

        # Find conflicts
        conflicts = find_conflicts(
            self._state.calendars, start_iso, end_iso, attendees
        )

        # Calculate preference penalty
        pref_penalty = calculate_preference_score(
            start_iso,
            action.proposed_duration,
            self._state.participant_preferences,
            self._state.calendars,
        )

        # Update state
        self._state.proposed_slot = [start_iso, end_iso]
        self._state.total_preference_penalty = pref_penalty

        # Step reward
        if len(conflicts) == 0 and pref_penalty < 100:
            step_reward = 0.5
        elif len(conflicts) > 0:
            if all(c["priority"] > req_priority for c in conflicts):
                step_reward = 0.2
            else:
                step_reward = -0.3
        else:
            step_reward = 0.0

        return self._obs(
            current_proposal={"start": start_iso, "end": end_iso},
            conflicts=conflicts,
            preference_penalty=pref_penalty,
            reward=step_reward,
        )

    def _process_reschedule_meeting(self, action: SchedulingAction) -> SchedulingObservation:
        if not action.meeting_id_to_move or not action.new_start_time:
            return self._obs(
                error_message="reschedule_meeting requires meeting_id_to_move and new_start_time",
                reward=-0.1,
            )

        if self._state.proposed_slot is None:
            return self._obs(
                error_message="Must propose a slot before rescheduling",
                reward=-0.2,
            )

        # Find the meeting to move
        meeting = self._find_meeting(action.meeting_id_to_move)
        if meeting is None:
            return self._obs(
                error_message=f"Meeting not found: {action.meeting_id_to_move}",
                reward=-0.2,
            )

        req_priority = self._state.meeting_request["priority"]
        if meeting["priority"] <= req_priority:
            return self._obs(
                error_message="Cannot reschedule equal or higher priority meeting",
                reward=-0.5,
            )

        # Validate new slot
        try:
            new_start = parse_iso(action.new_start_time)
        except (ValueError, TypeError):
            return self._obs(error_message="Invalid new_start_time format", reward=-0.1)

        old_start = parse_iso(meeting["start"])
        old_end = parse_iso(meeting["end"])
        duration = old_end - old_start
        new_end = new_start + duration
        new_start_iso = new_start.isoformat()
        new_end_iso = new_end.isoformat()
        attendee = meeting["attendee"]

        if not is_slot_free(attendee, new_start_iso, new_end_iso, self._state.calendars):
            return self._obs(error_message="New slot not free for attendee", reward=-0.2)

        # Update calendar: remove old, add new
        cal = self._state.calendars[attendee]
        self._state.calendars[attendee] = [
            e for e in cal if e[0] != meeting["start"]
        ]
        self._state.calendars[attendee].append(
            [new_start_iso, new_end_iso, meeting["priority"], meeting["summary"]]
        )

        self._state.rescheduled_meetings.append({
            "meeting_id": action.meeting_id_to_move,
            "old_start": meeting["start"],
            "new_start": new_start_iso,
            "attendee": attendee,
        })

        # Recalculate conflicts for current proposal
        attendees = self._state.meeting_request["attendees"]
        new_conflicts = find_conflicts(
            self._state.calendars,
            self._state.proposed_slot[0],
            self._state.proposed_slot[1],
            attendees,
        )

        num_rescheduled = len(self._state.rescheduled_meetings)
        step_reward = 0.5 if len(new_conflicts) == 0 else 0.3

        return self._obs(
            conflicts=new_conflicts,
            num_rescheduled=num_rescheduled,
            reward=step_reward,
        )

    def _process_finalize(self) -> SchedulingObservation:
        if self._state.proposed_slot is None:
            self._state.completed = True
            return self._obs(
                error_message="No slot proposed",
                success=False,
                reward=0.0,
                done=True,
            )

        attendees = self._state.meeting_request["attendees"]
        conflicts = find_conflicts(
            self._state.calendars,
            self._state.proposed_slot[0],
            self._state.proposed_slot[1],
            attendees,
        )

        if len(conflicts) > 0:
            self._state.completed = True
            return self._obs(
                error_message=f"Unresolved conflicts: {len(conflicts)} meetings",
                conflicts=conflicts,
                success=False,
                reward=0.0,
                done=True,
            )

        final_reward = calculate_final_reward(
            preference_penalty=self._state.total_preference_penalty,
            num_rescheduled=len(self._state.rescheduled_meetings),
            steps_taken=self._state.step_count,
            success=True,
        )

        self._state.completed = True
        self._state.final_reward = final_reward

        return self._obs(
            success=True,
            reward=final_reward,
            done=True,
        )

    def _process_reject(self) -> SchedulingObservation:
        self._state.completed = True
        return self._obs(
            success=False,
            reward=0.0,
            done=True,
            error_message="Agent rejected scheduling task",
        )

    def _handle_timeout(self) -> SchedulingObservation:
        """Give partial credit when max steps reached."""
        self._state.completed = True

        if self._state.proposed_slot is None:
            return self._obs(
                success=False,
                reward=0.0,
                done=True,
                error_message="Timeout: No slot proposed",
            )

        attendees = self._state.meeting_request["attendees"]
        conflicts = find_conflicts(
            self._state.calendars,
            self._state.proposed_slot[0],
            self._state.proposed_slot[1],
            attendees,
        )

        if len(conflicts) == 0:
            theoretical = calculate_final_reward(
                self._state.total_preference_penalty,
                len(self._state.rescheduled_meetings),
                self._state.step_count,
            )
            partial = theoretical * 0.7
        else:
            progress = 1.0 - (len(conflicts) / max(1, len(attendees)))
            partial = 0.2 * progress

        self._state.final_reward = partial
        return self._obs(
            success=False,
            reward=partial,
            done=True,
            error_message=f"Timeout after {self._state.step_count} steps (partial credit: {partial:.2f})",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs(self, **overrides) -> SchedulingObservation:
        """Build an observation from current state, applying overrides."""
        req = self._state.meeting_request
        attendees = req.get("attendees", [])

        defaults = dict(
            requested_duration=req.get("duration", 0),
            requested_priority=req.get("priority", 3),
            attendee_ids=attendees,
            busy_slots=build_busy_slots(self._state.calendars, attendees),
            collective_work_hours=self._collective_hours,
            preference_constraints=self._aggregate_preferences(
                self._state.participant_preferences
            ),
            current_proposal=(
                {"start": self._state.proposed_slot[0], "end": self._state.proposed_slot[1]}
                if self._state.proposed_slot
                else None
            ),
            conflicts=[],
            preference_penalty=self._state.total_preference_penalty,
            num_rescheduled=len(self._state.rescheduled_meetings),
            steps_taken=self._state.step_count,
            max_steps=MAX_STEPS,
            success=False,
            error_message=None,
            done=False,
            reward=0.0,
        )
        defaults.update(overrides)
        return SchedulingObservation(**defaults)

    def _find_meeting(self, meeting_id: str) -> dict | None:
        """Look up a meeting by its id (format: attendee_startiso)."""
        parts = meeting_id.split("_", 1)
        if len(parts) != 2:
            return None
        attendee, start_iso = parts
        for entry in self._state.calendars.get(attendee, []):
            if entry[0] == start_iso:
                return {
                    "attendee": attendee,
                    "start": entry[0],
                    "end": entry[1],
                    "priority": entry[2],
                    "summary": entry[3],
                }
        return None

    @staticmethod
    def _aggregate_preferences(prefs: dict) -> dict:
        """Summarize preferences for the observation."""
        if not prefs:
            return {}
        max_meetings = min(p.get("max_meetings_per_day", 99) for p in prefs.values())
        any_buffer = any(p.get("avoid_back_to_back", False) for p in prefs.values())
        buffer_mins = max(
            (p.get("buffer_minutes", 0) for p in prefs.values() if p.get("avoid_back_to_back")),
            default=0,
        )
        return {
            "max_meetings_per_day": max_meetings,
            "requires_buffer": any_buffer,
            "buffer_minutes": buffer_mins,
        }
