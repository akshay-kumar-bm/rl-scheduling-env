# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Meeting Scheduling RL Environment.

Defines the Action, Observation, and State Pydantic models used by the
scheduling environment to coordinate meeting proposals, rescheduling,
and conflict resolution across multiple attendees.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class SchedulingAction(Action):
    """Action the agent can take in the scheduling environment."""

    action_type: Literal["propose_slot", "reschedule_meeting", "finalize", "reject"] = Field(
        default="propose_slot",
        description="Type of scheduling action to perform.",
    )
    proposed_start: Optional[str] = Field(
        default=None,
        description="ISO8601 datetime string for the proposed meeting start (used with propose_slot).",
    )
    proposed_duration: Optional[int] = Field(
        default=None,
        description="Duration in minutes for the proposed meeting (used with propose_slot).",
    )
    meeting_id_to_move: Optional[str] = Field(
        default=None,
        description="Identifier of an existing meeting to reschedule (used with reschedule_meeting).",
    )
    new_start_time: Optional[str] = Field(
        default=None,
        description="ISO8601 datetime string for the new start time of a rescheduled meeting.",
    )


class SchedulingObservation(Observation):
    """Observation returned to the agent after each step."""

    requested_duration: int = Field(
        default=0,
        description="Requested meeting duration in minutes.",
    )
    requested_priority: int = Field(
        default=3,
        description="Priority of the meeting request (1=highest, 5=lowest).",
    )
    attendee_ids: List[str] = Field(
        default_factory=list,
        description="List of attendee user IDs required for the meeting.",
    )
    busy_slots: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Busy time slots: [{start, end, priority, summary, attendee}].",
    )
    collective_work_hours: Dict[str, int] = Field(
        default_factory=dict,
        description="Shared working hours window: {min_start_hour, max_end_hour}.",
    )
    preference_constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Attendee preference constraints (e.g. preferred times, avoid windows).",
    )
    current_proposal: Optional[Dict[str, str]] = Field(
        default=None,
        description="Currently proposed slot: {start, end} as ISO8601 strings.",
    )
    conflicts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of conflicts with the current proposal.",
    )
    preference_penalty: float = Field(
        default=0.0,
        description="Accumulated penalty from violating attendee preferences.",
    )
    num_rescheduled: int = Field(
        default=0,
        description="Number of existing meetings rescheduled so far.",
    )
    steps_taken: int = Field(
        default=0,
        description="Number of steps taken in the current episode.",
    )
    max_steps: int = Field(
        default=20,
        description="Maximum number of steps allowed in the episode.",
    )
    success: bool = Field(
        default=False,
        description="Whether the meeting was successfully scheduled.",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if the last action was invalid.",
    )


class SchedulingState(State):
    """Internal environment state tracking the full scheduling episode."""

    task_id: str = Field(
        default="",
        description="Unique identifier for the current task.",
    )
    scenario_name: str = Field(
        default="",
        description="Human-readable name of the scheduling scenario.",
    )
    meeting_request: Dict[str, Any] = Field(
        default_factory=dict,
        description="The incoming meeting request details.",
    )
    calendars: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Per-user calendars: {user_id: [[start, end, priority, summary], ...]}.",
    )
    participant_preferences: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-participant scheduling preferences.",
    )
    proposed_slot: Optional[List[str]] = Field(
        default=None,
        description="Currently proposed slot as [start_iso, end_iso].",
    )
    rescheduled_meetings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of meetings that have been rescheduled during this episode.",
    )
    total_preference_penalty: float = Field(
        default=0.0,
        description="Cumulative penalty from preference violations.",
    )
    total_steps: int = Field(
        default=0,
        description="Total steps taken so far in the episode.",
    )
    final_reward: float = Field(
        default=0.0,
        description="Final computed reward for the episode.",
    )
    completed: bool = Field(
        default=False,
        description="Whether the episode has ended.",
    )
